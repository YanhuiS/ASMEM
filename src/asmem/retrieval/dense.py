"""Optional dense channel + persistent embedding cache.

Dense is optional.  ``build_embedder`` returns ``None`` when no encoder is
configured or available, in which case the caller simply skips this channel.
The cache is per-user and keyed by ``sha256(identity + text)`` so changing the
model/prefix never silently mixes vectors from different encoders.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import struct
import tempfile
import threading
from pathlib import Path
from typing import Any, Sequence


def _l2(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [float(value) / norm for value in vector]


class HashingEmbedder:
    """Deterministic feature hashing; used for tests and offline fallbacks."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = max(8, int(dim))
        self.identity = f"hash:v1:dim={self.dim}"

    def usage_snapshot(self) -> dict[str, int]:
        return {}

    async def embed(self, texts: Sequence[str], *, kind: str = "document",
                    usage_out: dict[str, int] | None = None) -> list[list[float]]:
        from .tokenize import tokenize

        out: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for term in tokenize(text):
                digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            out.append(_l2(vector))
        return out


class OpenAIEmbedder:
    """OpenAI-compatible ``/embeddings`` endpoint (vLLM, SGLang, TEI, OpenAI).

    Batches are issued concurrently.  A process-wide semaphore keyed by
    ``(base_url, model)`` caps in-flight requests across embedder instances, so
    a user fan-out cannot flood the endpoint.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.model = str(config.get("model") or "")
        self.base_url = str(config.get("base_url") or "")
        self.api_key = str(config.get("api_key") or "EMPTY")
        self.batch_size = max(1, int(config.get("batch_size", 16)))
        self.concurrency = max(1, int(config.get("concurrency", 8)))
        self.max_retries = max(0, int(config.get("max_retries", 3)))
        self.trust_env = config.get("trust_env")
        self.query_prefix = str(config.get("query_prefix") or "")
        self.document_prefix = str(config.get("document_prefix") or "")
        self.timeout = float(config.get("timeout", 300))
        # Optional provider-side dimension (Bailian text-embedding-v4 accepts
        # 64/128/256/512/768/1024/1536/2048/3072).  0 means "provider default".
        self.dimensions = int(config.get("dimensions", 0) or 0)
        self.identity = (
            f"openai:{self.model}|{self.base_url}|q={self.query_prefix}|d={self.document_prefix}"
            f"|dim={self.dimensions}"
        )
        self._client: Any = None
        self._sem_key = (self.base_url, self.model, self.concurrency)
        # Set when the instance is shared process-wide; the owner must not
        # close the HTTP client after one search.
        self.shared = False
        # Provider-reported usage.  Snapshots around a build/query give the
        # per-search delta even though the client is shared.
        self.usage: dict[str, int] = {
            "prompt_tokens": 0, "total_tokens": 0, "requests": 0, "inputs": 0,
        }

    def usage_snapshot(self) -> dict[str, int]:
        return dict(self.usage)

    def _get_client(self) -> Any:
        if self._client is None:
            from .. import llm

            # Reuse the LLM transport factory so the OpenAI 3.x httpx2 backend
            # and proxy handling stay consistent across chat and embeddings.
            self._client = llm.make_client(
                {
                    "base_url": self.base_url or None,
                    "api_key": self.api_key or "EMPTY",
                    "timeout": self.timeout,
                    "concurrency": self.concurrency,
                    "trust_env": self.trust_env,
                },
                sdk_retries=self.max_retries,
            )
        return self._client

    async def embed(self, texts: Sequence[str], *, kind: str = "document",
                    usage_out: dict[str, int] | None = None) -> list[list[float]]:
        if not texts:
            return []
        prefix = self.query_prefix if kind == "query" else self.document_prefix
        client = self._get_client()
        sem = embed_semaphore(self._sem_key, self.concurrency)
        batches = [
            (start, [prefix + text for text in texts[start:start + self.batch_size]])
            for start in range(0, len(texts), self.batch_size)
        ]

        async def one(start: int, batch: list[str]) -> tuple[int, list[list[float]], dict[str, int]]:
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            async with sem:
                response = await client.embeddings.create(**kwargs)
            usage = getattr(response, "usage", None)
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            total = int(getattr(usage, "total_tokens", 0) or 0) if usage else 0
            rows = sorted(response.data, key=lambda item: item.index)
            return start, [_l2(row.embedding) for row in rows], {
                "prompt_tokens": prompt, "total_tokens": total,
                "requests": 1, "inputs": len(batch),
            }

        results = await asyncio.gather(*(one(start, batch) for start, batch in batches))
        out: list[list[float]] = [[] for _ in texts]
        call_usage = {"prompt_tokens": 0, "total_tokens": 0,
                      "requests": 0, "inputs": 0}
        for start, vectors, usage in results:
            out[start:start + len(vectors)] = vectors
            for key, value in usage.items():
                call_usage[key] = call_usage.get(key, 0) + value
        for key, value in call_usage.items():
            self.usage[key] = self.usage.get(key, 0) + value
            if usage_out is not None:
                usage_out[key] = usage_out.get(key, 0) + value
        return out

    async def aclose(self) -> None:
        if self.shared:
            return
        if self._client is not None:
            await self._client.close()
            self._client = None


#: One semaphore per (base_url, model, limit) shared by every embedder instance
#: in the process.  This is the global cap that keeps concurrent users from
#: multiplying the per-request batch concurrency.
_EMBED_SEMAPHORES: dict[tuple[Any, ...], asyncio.Semaphore] = {}


def embed_semaphore(key: tuple[str, str, int], limit: int) -> asyncio.Semaphore:
    # Locks/semaphores are bound to the running loop on first use; include the
    # loop id so a second ``asyncio.run`` in the same process starts clean.
    full = (id(asyncio.get_running_loop()), *key)
    semaphore = _EMBED_SEMAPHORES.get(full)
    if semaphore is None:
        semaphore = asyncio.Semaphore(limit)
        _EMBED_SEMAPHORES[full] = semaphore
    return semaphore


#: SentenceTransformer is not safe to call concurrently from many threads; one
#: lock per model keeps a local backend predictable.
_LOCAL_LOCKS: dict[str, threading.Lock] = {}


class LocalEmbedder:
    """SentenceTransformer backend; only active when the package is installed."""

    def __init__(self, config: dict[str, Any]) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = str(config.get("model") or "")
        self.device = str(config.get("device") or "")
        self.batch_size = max(1, int(config.get("batch_size", 16)))
        self.query_prefix = str(config.get("query_prefix") or "")
        kwargs = {"device": self.device} if self.device else {}
        self.model = SentenceTransformer(self.model_name, **kwargs)
        self.identity = f"local:{self.model_name}|q={self.query_prefix}"
        self.usage: dict[str, int] = {
            "prompt_tokens": 0, "total_tokens": 0, "requests": 0, "inputs": 0,
        }

    def usage_snapshot(self) -> dict[str, int]:
        return dict(self.usage)

    async def embed(self, texts: Sequence[str], *, kind: str = "document",
                    usage_out: dict[str, int] | None = None) -> list[list[float]]:
        if not texts:
            return []
        prefix = self.query_prefix if kind == "query" else ""
        inputs = [prefix + text for text in texts]
        lock = _LOCAL_LOCKS.setdefault(self.identity, threading.Lock())

        def run() -> list[list[float]]:
            with lock:
                vectors = self.model.encode(
                    inputs, batch_size=self.batch_size, normalize_embeddings=True,
                    show_progress_bar=False,
                )
            self.usage["requests"] += 1
            self.usage["inputs"] += len(inputs)
            if usage_out is not None:
                usage_out["requests"] = usage_out.get("requests", 0) + 1
                usage_out["inputs"] = usage_out.get("inputs", 0) + len(inputs)
            return [list(map(float, row)) for row in vectors]

        return await asyncio.to_thread(run)


_EMBEDDER_CACHE: dict[tuple[Any, ...], Any] = {}


def build_embedder(config: dict[str, Any]) -> Any | None:
    mode = str(config.get("mode") or "auto").lower()
    if mode in {"off", "none", "false", "0"}:
        return None
    if mode in {"hash", "hashing"}:
        key = ("hash", int(config.get("dim", 128)))
        if key not in _EMBEDDER_CACHE:
            _EMBEDDER_CACHE[key] = HashingEmbedder(dim=key[1])
        return _EMBEDDER_CACHE[key]
    base_url = str(config.get("base_url") or "")
    model = str(config.get("model") or "")
    if base_url:
        # One client per endpoint/model/config for the whole process.  Query
        # fan-out is bounded by the global semaphore, and avoiding a new
        # AsyncOpenAI client per search keeps connection churn down.
        key = (
            "openai", model, base_url, str(config.get("api_key") or ""),
            str(config.get("query_prefix") or ""),
            str(config.get("document_prefix") or ""),
            int(config.get("dimensions", 0) or 0),
            int(config.get("batch_size", 16)),
            int(config.get("concurrency", 8)),
            int(config.get("max_retries", 3)),
            float(config.get("timeout", 300)),
        )
        embedder = _EMBEDDER_CACHE.get(key)
        if embedder is None:
            embedder = OpenAIEmbedder(config)
            embedder.shared = True
            _EMBEDDER_CACHE[key] = embedder
        return embedder
    if model:
        key = ("local", model, str(config.get("device") or ""),
               str(config.get("query_prefix") or ""),
               int(config.get("batch_size", 16)))
        if key in _EMBEDDER_CACHE:
            return _EMBEDDER_CACHE[key]
        try:
            embedder = LocalEmbedder(config)
        except Exception:  # noqa: BLE001 - optional dependency / missing weights
            if mode in {"on", "true", "local"}:
                raise
            return None
        _EMBEDDER_CACHE[key] = embedder
        return embedder
    return None


class EmbeddingCache:
    """Append-only float32 vector store + JSON offset index.

    ``embeddings.bin`` is append-only; ``embeddings.json`` maps
    ``sha256(identity + text)`` -> ``[offset, dim]``.  A crash between the two
    writes only loses the newest entries and is repaired by re-embedding them.
    """

    def __init__(self, root: Path, identity: str) -> None:
        self.root = Path(root)
        self.identity = identity
        self.index_path = self.root / "embeddings.json"
        self.bin_path = self.root / "embeddings.bin"
        self.root.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, list[int]] = {}
        self._handle: Any = None
        self._data: bytearray | None = None
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(value, dict):
            self._index = {
                str(key): [int(item[0]), int(item[1])]
                for key, item in value.items()
                if isinstance(item, list) and len(item) == 2
            }

    def key(self, text: str) -> str:
        material = f"{self.identity}\0{text}".encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    def get(self, text: str) -> list[float] | None:
        entry = self._index.get(self.key(text))
        if not entry or not self.bin_path.exists():
            self.misses += 1
            return None
        offset, dim = entry
        try:
            if self._data is None:
                self._data = bytearray(self.bin_path.read_bytes())
            raw = self._data[offset:offset + dim * 4]
            if len(raw) != dim * 4:
                self.misses += 1
                return None
            self.hits += 1
            return list(struct.unpack(f"<{dim}f", raw))
        except OSError:
            self.misses += 1
            return None

    def put(self, text: str, vector: Sequence[float]) -> None:
        if not vector:
            return
        key = self.key(text)
        if key in self._index:
            return
        packed = struct.pack(f"<{len(vector)}f", *vector)
        if self._handle is None:
            self._handle = self.bin_path.open("ab")
        offset = self._handle.tell()
        dim = len(vector)
        self._handle.write(packed)
        # Flushing once per save (not per vector) keeps bulk fills fast; a
        # crash only loses the newest unindexed entries, which are re-embedded.
        if self._data is not None:
            self._data.extend(packed)
        self._index[key] = [offset, dim]

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
        fd, temp = tempfile.mkstemp(dir=str(self.root), prefix=".embeddings.", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._index, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.index_path)
        except BaseException:
            try:
                os.unlink(temp)
            except OSError:
                pass
            raise

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
            except OSError:
                pass
            self._handle.close()
            self._handle = None


#: One async lock per cache directory.  Concurrent searches for the same user
#: must not all embed the same documents: the first one fills the cache and the
#: rest read it after the lock is released.
_CACHE_LOCKS: dict[tuple[Any, ...], asyncio.Lock] = {}


def cache_lock(root: Path) -> asyncio.Lock:
    key = (id(asyncio.get_running_loop()), str(Path(root).resolve()))
    lock = _CACHE_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _CACHE_LOCKS[key] = lock
    return lock
