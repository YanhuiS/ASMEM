"""FastAPI wrapper for the challenge protocol: ``/add`` and ``/search``.

Failure policy (mirrors the runner's degradation chain):

* ``/add``  — durable before 200.  Same ``request_id`` + same payload is
  idempotent; same id + different payload is 409.  A formatting error drops
  the bad message instead of failing the whole batch, so the response stays a
  protocol-shaped ``success``.
* ``/search`` — never 5xx.  A dead model endpoint, an interrupted run or a
  client disconnect degrades to coarse evidence (``data: []`` only when the
  user really has no memory).  With ``request_id`` a retry reuses the cached
  result or resumes the interrupted workspace.
* shutdown — in-flight searches are drained (bounded) before the process
  exits; a search whose client already disconnected keeps running in the
  background so its result is cached for the retry.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import multimodal as multimodal_mod
from ..io import PROJECT_ROOT
from ..log import Logger
from ..search import runner
from . import store as store_mod
from .models import (AddRequest, AddResponse, MemoryItem, SearchRequest,
                     SearchResponse, normalize_content)

CONFIG_FILE = PROJECT_ROOT / "configs" / "asmem.yaml"
#: The HTTP service prefers its own config when one exists (remote generator and
#: embedding endpoints), so ``python -m asmem.api`` matches ``--config``.
API_CONFIG_FILE = PROJECT_ROOT / "configs" / "api.yaml"
DEFAULT_CONFIG: dict[str, Any] = {
    "store_dir": "data/api_store",
    "concurrency": 32,
    "add_concurrency": 32,
    "top_k": 100,
    # Whole-search wall-clock budget; the runner stops the agent and returns
    # the evidence it already has.  0 disables the budget.
    "search_timeout": 300,
    # Last-resort cap around run_question (0 = search_timeout + 60).
    "search_hard_timeout": 0,
    # How long a shutdown waits for running searches to finish.
    "shutdown_timeout": 20,
    # /health aggregates per-user meta/memory line counts no more often.
    "health_stats_ttl": 10,
    # Enforced only when non-empty; otherwise the protocol stays open.
    "api_key": "",
}


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    target = Path(path).expanduser() if path else None
    if target is not None and not target.is_absolute() and not target.exists():
        # Relative --config paths are relative to the project root, so the
        # service behaves the same wherever it is launched from.
        target = PROJECT_ROOT / target
    if target is None:
        target = API_CONFIG_FILE if API_CONFIG_FILE.exists() else CONFIG_FILE
    config = runner.load_config(target)
    for key, value in DEFAULT_CONFIG.items():
        config.setdefault(key, value)
    return config


def build_search_items(
    evidence: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    *,
    multimodal: bool = False,
    top_k: int = 100,
    media_root: Path | str | None = None,
    config: dict[str, Any] | None = None,
) -> list[MemoryItem]:
    """Map runner evidence to the competition's Search response contract.

    Guarantees, in order:

    * ``content`` is non-empty (string for text/code, text ``ContentPart[]``
      for multimodal);
    * multimodal results may append stored image parts after the text, within
      the protocol's per-image (10 MiB) and per-response (30 MiB) budgets;
    * ``id`` is unique and stable (source record id + content hash, or
      ``mem_<hash>`` for query-derived evidence);
    * ``score`` is strictly non-increasing along the returned rank order, as
      the contract requires "higher score == more relevant";
    * optional fields are left as ``None`` so the route's
      ``response_model_exclude_none`` omits them instead of sending ``null``.

    Images are only attached when the caller supplies ``media_root``; memory
    text never contains base64, so a text/code request is unaffected.
    """
    if top_k == 0:
        return []
    mm = multimodal_mod.mm_config(config)
    media_index = (multimodal_mod.media_by_key(records.values())
                   if multimodal else {})
    image_store = (multimodal_mod.ImageStore(Path(media_root))
                   if media_root is not None else None)
    max_images = max(0, int(mm.get("max_response_images", 8) or 0))
    max_image_bytes = max(0, int(mm.get("max_image_bytes", 10 * 1024 * 1024) or 0))
    max_response_bytes = max(0, int(mm.get("max_response_bytes",
                                           30 * 1024 * 1024) or 0))
    return_images = bool(mm.get("return_images", True))
    image_count = 0
    image_bytes = 0
    seen_media: set[str] = set()
    uri_cache: dict[str, str] = {}

    output: list[MemoryItem] = []
    seen_content: set[str] = set()
    for item in evidence:
        if top_k > 0 and len(output) >= top_k:
            break
        content = str(item.get("content") or "").strip()
        if not content:
            continue    # content is required and must be non-empty
        fingerprint = " ".join(content.split()).lower()
        if fingerprint in seen_content:
            continue    # one data entry per distinct memory text
        seen_content.add(fingerprint)
        metadata = item.get("metadata") or {}
        sources = metadata.get("source") or []
        if isinstance(sources, str):
            sources = [sources]
        anchor = str(metadata.get("anchor_id") or (sources[0] if sources else ""))
        source = records.get(anchor, {})
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
        item_id = f"{anchor}_{digest[:8]}" if source else f"mem_{digest[:16]}"
        created_at = source.get("time") if source else None
        rendered: Any = content
        if multimodal:
            parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
            if (return_images and image_store is not None
                    and image_count < max_images and max_image_bytes > 0):
                media = next((media_index[key] for key in [anchor, *sources]
                              if key and key in media_index), [])
                image_parts, added, used = multimodal_mod.image_parts(
                    media, image_store,
                    max_images=max_images - image_count,
                    max_image_bytes=max_image_bytes,
                    remaining_bytes=max_response_bytes - image_bytes,
                    seen=seen_media, uri_cache=uri_cache)
                if added:
                    parts.extend(image_parts)
                    image_count += added
                    image_bytes += used
            rendered = parts
        output.append(MemoryItem(
            id=item_id,
            content=rendered,
            score=None,     # assigned from the final rank below
            created_at=str(created_at) if created_at else None,
        ))

    # The evidence order is the fused retrieval rank.  Do not re-sort by the
    # evidence's own scores: agent claims and coarse seeds use different
    # scales, and the runner already curated this order.
    total = len(output)
    for rank, item in enumerate(output):
        item.score = round((total - rank) / total, 6) if total else 0.0
    return output[:top_k] if top_k > 0 else []


class ApiState:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        root = Path(str(config.get("store_dir") or "data/api_store"))
        self.store_root = root if root.is_absolute() else PROJECT_ROOT / root
        self.search_sem = asyncio.Semaphore(max(1, int(config.get("concurrency", 32))))
        self.add_sem = asyncio.Semaphore(max(1, int(config.get("add_concurrency", 32))))
        self.log = Logger.from_config(config, prefix="api")
        # One shared vision client; it is only used when an image arrives
        # without a caption and multimodal.caption is enabled.
        self.captioner = multimodal_mod.Captioner(config)
        self.api_key = str(config.get("api_key")
                            or os.environ.get("ASMEM_API_KEY") or "")
        self.search_timeout = max(0.0, float(config.get("search_timeout", 300) or 0))
        hard = max(0.0, float(config.get("search_hard_timeout", 0) or 0))
        self.hard_timeout = hard or (
            self.search_timeout + 60.0 if self.search_timeout > 0 else 0.0)
        self.shutdown_timeout = max(0.0, float(config.get("shutdown_timeout", 20) or 0))
        self.health_stats_ttl = max(0.0, float(config.get("health_stats_ttl", 10) or 0))
        self.started_at = time.time()
        self.accepting = True
        self.inflight = 0
        self.tasks: set[asyncio.Task[Any]] = set()
        self.last_error = ""
        self._stats_cache: tuple[float, dict[str, int]] | None = None

    def store(self, user_id: str) -> store_mod.UserStore:
        return store_mod.UserStore(user_id, self.store_root, config=self.config)

    # -- observability ------------------------------------------------------
    def note_error(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.last_error = f"{stamp} {message}"

    def stats(self) -> dict[str, int]:
        """Aggregate cheap per-user counters (cached for ``health_stats_ttl``)."""
        now = time.monotonic()
        if self._stats_cache and now - self._stats_cache[0] <= self.health_stats_ttl:
            return dict(self._stats_cache[1])
        users = records = pending = sessions = 0
        if self.store_root.exists():
            for child in store_mod.iter_user_dirs(self.store_root):
                if not ((child / store_mod.META_NAME).exists()
                        or (child / store_mod.MEMORY_NAME).exists()):
                    continue    # e.g. the shared index cache, not a user
                try:
                    store = store_mod.UserStore(child.name, self.store_root)
                    store.root = child     # exact directory, not a re-derived name
                    stats = store.stats()
                except OSError:
                    continue
                users += 1
                records += int(stats.get("records") or 0)
                pending += int(stats.get("pending") or 0)
                sessions += int(stats.get("sessions") or 0)
        snapshot = {"users": users, "records": records,
                    "pending": pending, "sessions": sessions}
        self._stats_cache = (now, snapshot)
        return dict(snapshot)

    # -- request lifecycle --------------------------------------------------
    async def run_tracked(self, coro: Any) -> Any:
        """Run ``coro`` so a client disconnect does not cancel the search.

        The inner task keeps running (and writes ``summary.json``), so the
        platform's retry with the same ``request_id`` hits the cached result
        instead of paying for a second agent run.
        """
        self.inflight += 1
        task = asyncio.create_task(coro)
        self.tasks.add(task)

        def _done(finished: asyncio.Task[Any]) -> None:
            self.tasks.discard(finished)
            self.inflight = max(0, self.inflight - 1)
            if not finished.cancelled():
                exc = finished.exception()
                if exc is not None:
                    self.note_error(
                        f"background search failed: {type(exc).__name__}: {exc}")

        task.add_done_callback(_done)
        return await asyncio.shield(task)

    async def drain(self) -> None:
        deadline = time.monotonic() + self.shutdown_timeout
        while self.inflight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)

    async def aclose(self) -> None:
        self.accepting = False
        if self.inflight:
            self.log.info(f"draining {self.inflight} search(es) before shutdown")
        await self.drain()
        for task in list(self.tasks):
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.captioner.aclose()
        self.log.close()


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    config = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store_mod.ensure_layout(app.state.api.store_root)
        settled = await store_mod.recover_all(app.state.api.store_root)
        if settled:
            app.state.api.log.info(f"recovered {settled} interrupted add(s)")
        yield
        await app.state.api.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.api = ApiState(config)
    state: ApiState = app.state.api

    def require_api_key(
        x_api_key: str | None = Header(default=None),
        authorization: str | None = Header(default=None),
    ) -> None:
        """Optional shared-secret auth; disabled unless ``api_key`` is set."""
        if not state.api_key:
            return
        provided = x_api_key
        if not provided and authorization:
            scheme, _, credential = authorization.partition(" ")
            if scheme.lower() in {"bearer", "token"}:
                provided = credential
        if not provided or not hmac.compare_digest(provided, state.api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    auth = [Depends(require_api_key)]

    @app.exception_handler(RequestValidationError)
    async def validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        safe_errors = [
            {"loc": list(error.get("loc", ())), "type": error.get("type"),
             "msg": error.get("msg")}
            for error in exc.errors()
        ]
        state.log.warn(
            f"request validation failed path={request.url.path} errors={safe_errors}")
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    @app.get("/health")
    async def health() -> dict[str, Any]:
        stats = state.stats()
        retrieval = state.config.get("retrieval") or {}
        generator = state.config.get("generator") or {}
        return {
            "status": "ok" if state.accepting else "draining",
            **stats,
            "inflight": state.inflight,
            "mode": str(retrieval.get("mode") or ""),
            "model": str(generator.get("model") or ""),
            "uptime_seconds": int(time.time() - state.started_at),
            "last_error": state.last_error,
        }

    @app.post("/add", response_model=AddResponse, dependencies=auth)
    async def add(request: AddRequest) -> AddResponse:
        store = state.store(request.user_id)
        # Idempotency is keyed on the RAW payload (including image bytes), not
        # on generated captions: a retry after a caption-endpoint blip must not
        # be rejected as "same request_id, different payload".
        raw_messages = [
            {"role": item.role, "content": item.content,
             "timestamp": item.timestamp}
            for item in request.messages
        ]
        payload_hash = store_mod.payload_hash(
            request.user_id, request.session_id, raw_messages)
        messages: list[dict[str, Any]] = []
        media: list[dict[str, Any]] = []
        image_budget = {"bytes": 0}      # whole request, not per message
        for item in request.messages:
            plan = await multimodal_mod.build_content(
                item.content,
                image_store=multimodal_mod.ImageStore(store.images_dir),
                captioner=state.captioner,
                config=state.config,
                budget=image_budget)
            messages.append({"role": item.role, "content": plan.text,
                             "timestamp": item.timestamp})
            media.extend(plan.images)
        async with state.add_sem:
            try:
                await store.append(
                    request.request_id, request.session_id, messages,
                    media=media, payload_hash=payload_hash)
                success = True
                retrieval = state.config.get("retrieval") or {}
                if retrieval.get("index_on_add"):
                    try:
                        await store.warm_index(state.config)
                    except Exception as exc:  # index warmup is optional
                        state.log.warn(
                            f"index warmup failed: {type(exc).__name__}: {exc}")
            except store_mod.RequestConflictError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:  # never fail the protocol on a formatting error
                state.note_error(f"add failed: {type(exc).__name__}: {exc}")
                state.log.error(
                    f"add failed user={request.user_id} "
                    f"request={request.request_id}: {type(exc).__name__}: {exc}")
                success = False
        return AddResponse(success=success, request_id=request.request_id,
                           user_id=request.user_id, session_id=request.session_id)

    @app.post("/search", response_model=SearchResponse,
              response_model_exclude_none=True, dependencies=auth)
    async def search(request: SearchRequest) -> SearchResponse:
        if not state.accepting:
            raise HTTPException(status_code=503, detail="server is shutting down")
        multimodal_query = isinstance(request.query, list)
        query = ("" if multimodal_query
                 else normalize_content(request.query)).strip()
        query_images: list[str] = []
        options = [str(option) for option in (request.options or []) if str(option).strip()]
        if not query and not multimodal_query:
            return SearchResponse(data=[])
        store = state.store(request.user_id)
        try:
            # Settle an add that was interrupted between the memory append and
            # the metadata update; otherwise its records stay invisible.
            await store.recover()
        except Exception as exc:  # noqa: BLE001 - recovery must not block search
            state.log.warn(f"recover failed user={request.user_id}: "
                           f"{type(exc).__name__}: {exc}")
        if not store.has_memory():
            return SearchResponse(data=[])
        if multimodal_query:
            # Retrieval is caption-based: the query image contributes a short
            # caption, and at most one inline image is kept for the bounded
            # gate call.  The base64 never becomes part of the question text.
            query, query_images = await multimodal_mod.query_content(
                request.query, captioner=state.captioner,
                config=state.config)
            if not query.strip():
                return SearchResponse(data=[])

        request_id = (request.request_id or "").strip()

        async def _run_search() -> dict[str, Any]:
            async with state.search_sem:
                coro = runner.run_question(
                    memory_path=store.memory_path, question=query,
                    config=state.config, run_root=store.runs_path, log=state.log,
                    options=options, cache_dir=store.cache_dir,
                    shared_dir=store.shared_dir,
                    request_id=request_id, user_id=request.user_id,
                    query_images=query_images)
                if state.hard_timeout <= 0:
                    return await coro
                return await asyncio.wait_for(coro, timeout=state.hard_timeout)

        started = time.perf_counter()
        try:
            result = await state.run_tracked(_run_search())
        except asyncio.CancelledError:
            # The client went away.  The inner task is shielded and keeps
            # running; a retry with the same request_id gets its cached
            # result instead of starting over.
            state.log.warn(
                f"search client disconnected user={request.user_id} "
                f"request={request_id or '-'}; keeping the run alive")
            raise
        except asyncio.TimeoutError:
            message = ("search hard timeout "
                       f"({state.hard_timeout:.0f}s) user={request.user_id}")
            state.note_error(message)
            state.log.error(message)
            return SearchResponse(data=[])
        except Exception as exc:  # noqa: BLE001 - protocol never sees a 5xx
            message = (f"search failed user={request.user_id} "
                       f"request={request_id or '-'}: {type(exc).__name__}: {exc}")
            state.note_error(message)
            state.log.error(message)
            return SearchResponse(data=[])

        summary = result.get("summary") or {}
        state.log.info(
            f"search user={request.user_id} request={request_id or '-'} "
            f"evidence={len(result.get('evidence') or [])} "
            f"degraded={bool(summary.get('degraded'))} "
            f"cached={bool(result.get('cached'))} "
            f"stop={summary.get('stop_reason') or '-'} "
            f"seconds={time.perf_counter() - started:.2f}")

        records = {str(item.get("id")): item for item in store.records()}
        limit = (request.top_k if request.top_k is not None
                 else int(state.config.get("top_k", 100)))
        data = build_search_items(
            result.get("evidence") or [], records,
            multimodal=multimodal_query, top_k=limit,
            media_root=store.images_dir, config=state.config)
        return SearchResponse(data=data)

    return app


#: ``uvicorn asmem.api.server:app``
app = create_app()
