"""Content-addressed raw API request journal.

Every ``/add`` and ``/search`` payload is persisted exactly once, keyed by the
SHA-256 of its canonical JSON.  A retry (same request_id or same query) reuses
the same file instead of writing a second copy.

Inline image data URIs are **externalized**: the bytes go to the same
content-addressed :class:`asmem.multimodal.ImageStore` that memory uses, and the
payload keeps only ``asmem-image:<sha256>``.  This has two effects:

* a 30 MiB multimodal request does not become ~40 MiB of base64 in the journal;
* the same image in two users / two requests has exactly one physical copy
  (the store hardlinks the shared blob into each user directory).

The journal is an audit artifact only.  Nothing on the retrieval path reads it,
and a journal write failure never fails ``/add`` or ``/search``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..io import atomic_write_text, now_iso
from ..multimodal import (SUPPORTED_MIME, DEFAULT_MAX_IMAGE_BYTES,
                          ImageStore, parse_data_uri)

#: Marker written in place of an inline image URI.
IMAGE_REF_PREFIX = "asmem-image:"
UNSTORED_REF_PREFIX = "asmem-image-unstored:"

#: Kinds that get their own directory.  ``invalid`` keeps a request that
#: failed FastAPI validation (the exact payload the platform sent).
KINDS = ("add", "search", "invalid", "other")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _data_uri(value: str) -> bool:
    return isinstance(value, str) and value[:5].lower() == "data:"


def externalize_images(
    value: Any,
    image_store: ImageStore | None,
    *,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    media: list[dict[str, Any]] | None = None,
    seen: set[str] | None = None,
) -> Any:
    """Return a copy of ``value`` with inline image data URIs replaced.

    ``media`` collects the stored asset metadata (deduped by sha) so the
    journal record can resolve every ``asmem-image:<sha>`` marker.  Unparseable
    / unsupported / oversized images are replaced by a hash-only marker; the
    bytes are not copied into the journal.
    """
    if media is None:
        media = []
    if seen is None:
        seen = set()
    if isinstance(value, str):
        if not _data_uri(value):
            return value
        parsed = parse_data_uri(value)
        if parsed is None:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            return f"{UNSTORED_REF_PREFIX}{digest}"
        mime, data = parsed
        digest = hashlib.sha256(data).hexdigest()
        if mime not in SUPPORTED_MIME or len(data) > max_image_bytes:
            return f"{UNSTORED_REF_PREFIX}{digest}"
        if image_store is None:
            return f"{IMAGE_REF_PREFIX}{digest}"
        try:
            asset = image_store.save(data, mime)
        except (OSError, ValueError):
            return f"{UNSTORED_REF_PREFIX}{digest}"
        if asset["sha256"] not in seen:
            seen.add(asset["sha256"])
            media.append(dict(asset))
        return f"{IMAGE_REF_PREFIX}{asset['sha256']}"
    if isinstance(value, dict):
        return {key: externalize_images(item, image_store,
                                        max_image_bytes=max_image_bytes,
                                        media=media, seen=seen)
                for key, item in value.items()}
    if isinstance(value, list):
        return [externalize_images(item, image_store,
                                   max_image_bytes=max_image_bytes,
                                   media=media, seen=seen)
                for item in value]
    return value


class RequestJournal:
    """Write-once raw request/response files under ``<user>/requests/``.

    Layout::

        requests/
          add/<sha[:2]>/<sha>.json
          search/<sha[:2]>/<sha>.json
          search/responses/<request sha[:2]>/<request sha>/<response sha>.json
          invalid/<sha[:2]>/<sha>.json

    ``<sha>`` is the SHA-256 of the *externalized* payload, so a retry with the
    same content maps to the same file even if the base64 formatting differs.
    """

    def __init__(self, root: Path, image_store: ImageStore | None = None,
                 config: dict[str, Any] | None = None) -> None:
        self.root = Path(root)
        self.image_store = image_store
        config = config or {}
        mm = config.get("multimodal")
        mm = mm if isinstance(mm, dict) else {}
        self.max_image_bytes = max(
            1, int(mm.get("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES)
                   or DEFAULT_MAX_IMAGE_BYTES))
        self.keep_responses = bool(config.get("request_journal_responses", True))

    def _record_path(self, kind: str, digest: str) -> Path:
        kind = kind if kind in KINDS else "other"
        return self.root / kind / digest[:2] / f"{digest}.json"

    def record(self, kind: str, payload: Any, *,
               meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Persist one request payload; returns its content-addressed ref."""
        media: list[dict[str, Any]] = []
        clean = externalize_images(
            payload, self.image_store, max_image_bytes=self.max_image_bytes,
            media=media)
        material = {"kind": kind, "payload": clean}
        digest = _digest(material)
        path = self._record_path(kind, digest)
        created = False
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "kind": kind,
                "sha256": digest,
                "received_at": now_iso(),
                "meta": dict(meta or {}),
                "media": media,
                "payload": clean,
            }
            atomic_write_text(path, json.dumps(
                record, ensure_ascii=False, indent=2))
            created = True
        return {
            "sha256": digest,
            "path": str(path.relative_to(self.root)),
            "created": created,
            "media": len(media),
        }

    def response_path(self, kind: str, request_sha: str,
                      response_sha: str) -> Path:
        kind = kind if kind in KINDS else "other"
        return (self.root / kind / "responses" / request_sha[:2]
                / request_sha / f"{response_sha}.json")

    def record_response(self, kind: str, request_ref: dict[str, Any],
                        response: Any, *,
                        meta: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Persist one response body next to its request; write-once."""
        if not self.keep_responses:
            return None
        request_sha = str(request_ref.get("sha256") or "")
        if not request_sha:
            return None
        media: list[dict[str, Any]] = []
        clean = externalize_images(
            response, self.image_store, max_image_bytes=self.max_image_bytes,
            media=media)
        material = {"kind": kind, "request": request_sha, "response": clean}
        digest = _digest(material)
        path = self.response_path(kind, request_sha, digest)
        created = False
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "kind": kind,
                "request_sha256": request_sha,
                "response_sha256": digest,
                "received_at": now_iso(),
                "meta": dict(meta or {}),
                "media": media,
                "payload": clean,
            }
            atomic_write_text(path, json.dumps(
                record, ensure_ascii=False, indent=2))
            created = True
        return {
            "sha256": digest,
            "path": str(path.relative_to(self.root)),
            "created": created,
            "media": len(media),
        }
