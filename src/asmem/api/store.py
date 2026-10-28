"""Per-user storage: one directory, one generalized ``memory.jsonl``.

Layout (version 2)::

    <store_dir>/
      store.json                 # layout version + creation time
      users/
        <user_id>/               # readable, path-safe user id
          memory.jsonl           # durable records
          state.jsonl            # derived state index
          meta.json / .bak       # add WAL + counters
          cache/                 # embedding cache (one dir per identity)
          runs/
            <run id>/            # one search request, flat
              request.json evidence.jsonl summary.json ...
              inputs/            # hardlinked memory/state snapshot + coarse files
              bin -> ../../../../shared/bin
      shared/
        bin/                     # helper CLIs shared by every run

Add is pure CPU: normalize -> append.  Idempotency is by ``request_id``; a
small write-ahead log in ``meta.json`` makes an interrupted add recoverable.

Stores written by version 1 (``<store_dir>/u_<slug>_<hash>/`` with
``index/`` and ``runs/<id>/qa/``) are still read and written in place; a user
directory is migrated only when it is first created under the new layout.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator

from ..io import (DATA_DIR, atomic_write_text, now_iso, read_jsonl,
                  write_jsonl)
from ..schema import chunking_config, normalize_messages
from ..state import STATE_NAME, derive_state, write_state

META_NAME = "meta.json"
#: Last known-good copy.  ``meta.json`` is small but it is the WAL: a torn
#: write (power loss, container kill) must not erase ``pending``/``applied``.
META_BACKUP_NAME = "meta.json.bak"
MEMORY_NAME = "memory.jsonl"

#: Layout v2 names.
LAYOUT_VERSION = 2
STORE_META_NAME = "store.json"
USERS_DIR = "users"
SHARED_DIR = "shared"
RUNS_NAME = "runs"
CACHE_NAME = "cache"
#: Version-1 names, kept for in-place compatibility.
LEGACY_CACHE_NAME = "index"

#: ``[image: <id>]`` markers written by the multimodal content planner.  They
#: let the store attach image assets to every chunk of the same parent.
_IMAGE_MARKER = re.compile(r"\[image:\s*([A-Za-z0-9_.-]+)\]")

_SLUG = re.compile(r"[^A-Za-z0-9._-]+")
#: Characters that cannot appear verbatim in a path component.  ``%`` is
#: escaped too, so the mapping stays injective (a literal ``%20`` cannot
#: collide with an escaped space).
_UNSAFE = re.compile(r"[%/\s\\\x00-\x1f\x7f]")

_locks: dict[str, asyncio.Lock] = {}


class RequestConflictError(ValueError):
    """Same ``request_id`` with a different payload."""


def _payload_hash(user_id: str, session_id: str, messages: list[dict[str, Any]],
                  media: list[dict[str, Any]] | None = None) -> str:
    material = json.dumps(
        {"user_id": user_id, "session_id": session_id, "messages": messages,
         "media": media or []},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def payload_hash(user_id: str, session_id: str,
                 messages: list[dict[str, Any]]) -> str:
    """Public idempotency hash for the raw Add payload.

    The API computes this *before* captioning so a retry whose generated
    caption differs slightly still maps to the same ``request_id`` instead of
    raising a spurious conflict.
    """
    return _payload_hash(user_id, session_id, messages)


def _attach_media(records: list[dict[str, Any]],
                  media: list[dict[str, Any]]) -> None:
    """Attach image assets to records that reference them.

    A message may be split into several chunks; only the chunk that contains
    the ``[image: id]`` marker knows about it.  Media is therefore assigned per
    parent, so every sibling chunk can return the same image.
    """
    by_id = {str(item.get("id")): item for item in media if item.get("id")}
    if not by_id:
        return
    parent_media: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        found: list[str] = []
        for match in _IMAGE_MARKER.finditer(str(record.get("text") or "")):
            image_id = match.group(1)
            if image_id in by_id and image_id not in found:
                found.append(image_id)
        if not found:
            continue
        meta = record.setdefault("meta", {})
        parent = str(meta.get("parent_id") or record.get("id") or "")
        bucket = parent_media.setdefault(parent, [])
        for image_id in found:
            if not any(item.get("id") == image_id for item in bucket):
                bucket.append(by_id[image_id])
    for record in records:
        meta = record.setdefault("meta", {})
        parent = str(meta.get("parent_id") or record.get("id") or "")
        if parent in parent_media:
            meta["media"] = [dict(item) for item in parent_media[parent]]


def slug(user_id: str) -> str:
    """Version-1 directory name (kept for reading existing stores)."""
    clean = _SLUG.sub("_", user_id).strip("._-") or "user"
    digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:8]
    return f"{clean[:80]}_{digest}"


def dirname(user_id: str) -> str:
    """Readable, path-safe directory name for ``user_id``.

    ``eval:run:user-0`` stays literally readable; only characters that cannot
    appear in a path component are percent-escaped.  Over-long ids are
    truncated with a hash suffix.  The mapping is injective, so two different
    user ids can never share a directory.
    """
    raw = user_id or "user"
    clean = _UNSAFE.sub(lambda match: f"%{ord(match.group()):02x}", raw)
    if not clean:
        clean = "%empty"
    if clean.startswith("."):
        # Keep the directory visible and avoid "." / "..".
        clean = "%2e" + clean[1:]
    if len(clean.encode("utf-8")) > 150:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        clean = clean[:120] + "~" + digest
    return clean


def resolve_user_dir(root: Path, user_id: str) -> Path:
    """Pick the existing directory for ``user_id``, or the v2 location.

    A store created by version 1 keeps working in place: no rename, no
    re-embedding, no lost run history.
    """
    new = root / USERS_DIR / dirname(user_id)
    if new.exists():
        return new
    legacy = root / slug(user_id)
    if legacy.exists():
        return legacy
    return new


def iter_user_dirs(root: Path) -> Iterator[Path]:
    """Yield user directories in both layouts, skipping ``users``/``shared``."""
    seen: set[Path] = set()
    users = root / USERS_DIR
    if users.is_dir():
        for child in sorted(users.iterdir()):
            if child.is_dir():
                seen.add(child.resolve())
                yield child
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in {USERS_DIR, SHARED_DIR}:
                continue
            if child.resolve() in seen:
                continue
            yield child


def ensure_layout(root: Path) -> None:
    """Create ``users/``, ``shared/bin/`` and the store manifest."""
    (root / USERS_DIR).mkdir(parents=True, exist_ok=True)
    (root / SHARED_DIR / "bin").mkdir(parents=True, exist_ok=True)
    meta_path = root / STORE_META_NAME
    if meta_path.exists():
        return
    atomic_write_text(meta_path, json.dumps({
        "version": LAYOUT_VERSION,
        "created_at": now_iso(),
        "layout": {
            "users": USERS_DIR,
            "shared": SHARED_DIR,
            "user_files": [MEMORY_NAME, STATE_NAME, META_NAME,
                           META_BACKUP_NAME, RUNS_NAME, CACHE_NAME],
            "run_files": ["request.json", "evidence.jsonl", "summary.json",
                          "steps.jsonl", "agent_history.jsonl", "inputs/", "bin"],
        },
        "notes": "runs/*/inputs/memory.jsonl and state.jsonl are hardlinks "
                 "to the user's live files; runs/*/bin links to shared/bin.",
    }, ensure_ascii=False, indent=2))


def _lock(user_id: str) -> asyncio.Lock:
    lock = _locks.get(user_id)
    if lock is None:
        lock = asyncio.Lock()
        _locks[user_id] = lock
    return lock


class UserStore:
    def __init__(self, user_id: str, root: Path | None = None,
                 config: dict[str, Any] | None = None) -> None:
        self.user_id = user_id
        self.store_root = (Path(root) if root else (DATA_DIR / "api_store"))
        self.root = resolve_user_dir(self.store_root, user_id)
        self.config = config or {}
        self.chunking = chunking_config(self.config)

    @property
    def meta_path(self) -> Path:
        return self.root / META_NAME

    @property
    def memory_path(self) -> Path:
        return self.root / MEMORY_NAME

    @property
    def runs_path(self) -> Path:
        return self.root / RUNS_NAME

    @property
    def images_dir(self) -> Path:
        """Content-addressed image assets referenced by ``meta.media``."""
        return self.root / "images"

    @property
    def cache_dir(self) -> Path:
        """Embedding cache; reuse a v1 ``index/`` directory when present."""
        cache = self.root / CACHE_NAME
        if not cache.exists() and (self.root / LEGACY_CACHE_NAME).exists():
            return self.root / LEGACY_CACHE_NAME
        return cache

    #: Backwards-compatible alias for callers written before the rename.
    @property
    def index_dir(self) -> Path:
        return self.cache_dir

    @property
    def shared_dir(self) -> Path:
        return self.store_root / SHARED_DIR

    @property
    def shared_bin_dir(self) -> Path:
        return self.shared_dir / "bin"

    @property
    def state_path(self) -> Path:
        return self.root / STATE_NAME

    async def warm_index(self, config: dict[str, Any]) -> None:
        """Build BM25 + missing embeddings so the next search is cheap.

        Failures are intentionally non-fatal: the memory is already durable,
        and the search path rebuilds lazily when the index/cache is missing.
        """
        from ..retrieval import build_index

        index = await build_index(self.memory_path, config, cache_dir=self.cache_dir)
        await index.aclose()

    @staticmethod
    def _read_meta(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def _meta(self) -> dict[str, Any]:
        value = self._read_meta(self.meta_path) if self.meta_path.exists() else None
        if value is None:
            # A corrupt primary is a crash artifact, not an empty store.  Fall
            # back to the last good copy before giving up.
            backup = self.meta_path.with_name(META_BACKUP_NAME)
            if backup.exists():
                value = self._read_meta(backup)
        if value is not None:
            value.setdefault("user_id", self.user_id)
            value.setdefault("sessions", {})
            value.setdefault("next_index", 1)
            hashes = value.get("applied_hashes")
            if not isinstance(hashes, dict):
                hashes = {}
                value["applied_hashes"] = hashes
            # v1 kept a redundant ``applied`` list plus a hash map.  Fold the
            # list into the map (legacy ids get an empty hash and are adopted
            # on replay) and keep ``applied`` as an in-memory view only.
            legacy = value.get("applied")
            if isinstance(legacy, list):
                for request_id in legacy:
                    hashes.setdefault(str(request_id), "")
            value["applied"] = list(hashes)
            value.setdefault("pending", {})
            return value
        return {"user_id": self.user_id, "sessions": {}, "next_index": 1,
                "applied": [], "applied_hashes": {}, "pending": {}}

    def _save_meta(self, meta: dict[str, Any]) -> None:
        """Atomically replace ``meta.json`` and refresh the backup copy.

        The write-ahead log must survive an abrupt kill: a partial JSON file
        would make the store forget which requests were applied and which
        records are still pending.  ``applied`` is an in-memory index into
        ``applied_hashes`` and is not written; the map is the single source of
        truth (``request_id -> payload_hash``).
        """
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {key: value for key, value in meta.items() if key != "applied"},
            ensure_ascii=False, indent=2)
        atomic_write_text(self.meta_path, payload)
        try:
            atomic_write_text(self.meta_path.with_name(META_BACKUP_NAME), payload)
        except OSError:
            # The primary is already durable; a missing backup only narrows
            # the recovery window.
            pass

    def _append_records(self, records: list[dict[str, Any]]) -> int:
        existing = read_jsonl(self.memory_path)
        seen = {str(item.get("id")) for item in existing if item.get("id")}
        fresh = [item for item in records if str(item.get("id")) not in seen]
        if fresh:
            write_jsonl(self.memory_path, existing + fresh)
        return len(fresh)

    def _write_state(self) -> None:
        """Rebuild the derived state index from the current memory snapshot."""
        try:
            entries = derive_state(self.records())
            if entries:
                write_state(self.state_path, entries)
            elif self.state_path.exists():
                self.state_path.unlink()
        except OSError:
            # State is derived; a later search can rebuild it.
            pass

    def _settle(self, meta: dict[str, Any], request_id: str,
                pending: dict[str, Any]) -> int:
        added = self._append_records(pending.get("records") or [])
        if added or pending.get("records"):
            # Also rewrite state when the memory append already happened
            # before a crash: the derived index may never have been written.
            self._write_state()
        session = meta["sessions"].get(pending.get("session_id"))
        if session is not None:
            session["next"] = max(int(session.get("next", 1)),
                                  int(pending.get("next", session.get("next", 1))))
        if request_id not in meta["applied"]:
            meta["applied"].append(request_id)
        payload_hash = str(pending.get("payload_hash") or "")
        # Always write the map entry: it is the on-disk source of truth.
        meta.setdefault("applied_hashes", {})[request_id] = payload_hash
        meta["pending"].pop(request_id, None)
        return added

    def _recover(self, meta: dict[str, Any]) -> bool:
        changed = False
        for request_id, pending in list(meta["pending"].items()):
            self._settle(meta, request_id, pending)
            changed = True
        return changed

    async def append(self, request_id: str, session_id: str,
                     messages: list[dict[str, Any]],
                     *,
                     media: list[dict[str, Any]] | None = None,
                     payload_hash: str | None = None) -> int:
        ensure_layout(self.store_root)
        if payload_hash is None:
            payload_hash = _payload_hash(self.user_id, session_id, messages, media)
        media = media or []
        async with _lock(self.user_id):
            meta = self._meta()
            self._recover(meta)
            if request_id in meta["applied"]:
                stored = meta.setdefault("applied_hashes", {}).get(request_id)
                if stored and stored != payload_hash:
                    raise RequestConflictError(
                        f"request_id {request_id!r} was already applied with a "
                        "different payload")
                if not stored:
                    # Legacy entry from before hash tracking: adopt the hash.
                    meta["applied_hashes"][request_id] = payload_hash
                self._save_meta(meta)
                return 0

            session = meta["sessions"].get(session_id)
            if session is None:
                session = {"index": int(meta["next_index"]), "next": 1}
                meta["sessions"][session_id] = session
                meta["next_index"] = int(meta["next_index"]) + 1
            start = int(session.get("next", 1))
            _, records = normalize_messages(
                messages, session_id=session_id,
                session_index=int(session["index"]), start_seq=start,
                chunking=self.chunking)
            if media:
                _attach_media(records, media)
            next_seq = start + len(records)
            meta["pending"][request_id] = {
                "session_id": session_id,
                "next": next_seq,
                "records": records,
                "payload_hash": payload_hash,
            }
            self._save_meta(meta)
            added = self._settle(meta, request_id, meta["pending"][request_id])
            self._save_meta(meta)
            return added

    def records(self) -> list[dict[str, Any]]:
        return read_jsonl(self.memory_path)

    def has_memory(self) -> bool:
        """Non-empty memory without parsing the whole JSONL."""
        try:
            return self.memory_path.stat().st_size > 0
        except OSError:
            return False

    async def recover(self) -> int:
        """Settle this user's interrupted adds; returns how many were pending.

        Called at the start of a search so a crash between the memory append
        and the metadata update cannot make fresh records invisible until the
        next process start.
        """
        async with _lock(self.user_id):
            meta = self._meta()
            pending = list(meta["pending"])
            if not pending:
                return 0
            self._recover(meta)
            self._save_meta(meta)
            return len(pending)

    def stats(self) -> dict[str, Any]:
        """Cheap health counters; does not parse the memory JSONL."""
        meta = self._meta()
        records = 0
        if self.memory_path.exists():
            try:
                with self.memory_path.open("rb") as handle:
                    records = sum(1 for line in handle if line.strip())
            except OSError:
                records = 0
        return {
            "user_id": self.user_id,
            "records": records,
            "sessions": len(meta.get("sessions") or {}),
            "applied": len(meta.get("applied") or []),
            "pending": len(meta.get("pending") or {}),
        }


async def recover_all(root: Path | None = None) -> int:
    base = root or (DATA_DIR / "api_store")
    settled = 0
    if not base.exists():
        return 0
    for child in iter_user_dirs(base):
        if not (child / META_NAME).exists() and not (child / META_BACKUP_NAME).exists():
            continue
        store = UserStore(child.name, root=base)
        store.root = child
        store.store_root = base
        meta = store._meta()
        # Lock on the same key the request path uses (the real user_id, not
        # the directory slug).
        store.user_id = str(meta.get("user_id") or child.name)
        if not meta.get("pending"):
            continue
        async with _lock(store.user_id):
            meta = store._meta()
            if not meta.get("pending"):
                continue
            store._recover(meta)
            store._save_meta(meta)
        settled += 1
    return settled
