"""Multimodal ContentPart handling: captions, image assets, bounded prompts.

The competition's multimodal track sends an ordered ``ContentPart[]`` (text
parts plus inline base64 images).  ASMEM retrieves text, so images are never
embedded: each image is stored as a content-addressed file and its **caption**
is the only searchable representation.  A caption comes from, in order:

1. an explicit ``caption`` / ``alt`` / ``description`` field on the part;
2. the text part that accompanies the image (the protocol calls this
   "caption or text evidence");
3. an optional vision-model caption (``multimodal.caption``), cached by the
   image hash so a retry does not pay for it twice.

Nothing in this module writes a data URI into memory, BM25, evidence or a
prompt.  Memory text carries only ``[image: <short id>]`` and
``[caption] ...``; the bytes live under ``<user>/images/``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import llm

#: MIME -> file extension.  The protocol accepts only these three.
SUPPORTED_MIME = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}
DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = 30 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 30 * 1024 * 1024

DEFAULT_MULTIMODAL: dict[str, Any] = {
    "enabled": True,
    # Generate a caption when no explicit caption/text accompanies an image.
    "caption": True,
    # Force a caption even when text is present.  Off by default: the text
    # part is the protocol's caption and a model call would only duplicate it.
    "caption_always": False,
    # A search query's text is a question, not a caption; caption its images
    # so image-to-image matching has searchable text.
    "caption_query": True,
    "caption_model": "",
    "caption_base_url": "",
    "caption_api_key": "",
    "caption_max_chars": 300,
    "caption_timeout": 20,
    "max_image_bytes": DEFAULT_MAX_IMAGE_BYTES,
    "max_request_bytes": DEFAULT_MAX_REQUEST_BYTES,
    "max_response_bytes": DEFAULT_MAX_RESPONSE_BYTES,
    "max_response_images": 8,
    "max_query_images": 1,
    "return_images": True,
    # "auto" sends a query image to the gate (one call) and the first evidence
    # loop turn; memory images are never sent to the postagent.  "never" keeps
    # every internal prompt text-only; "always" keeps the image on every turn.
    "postagent_images": "auto",
}

#: A short, stable reference written into memory text.  The full hash and
#: relative path live in ``meta.media``.
IMAGE_MARKER = re.compile(r"\[image:\s*([A-Za-z0-9_.-]+)\]")
_WS = re.compile(r"\s+")


def mm_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the top-level ``multimodal`` block with defaults."""
    raw = (config or {}).get("multimodal")
    merged = dict(DEFAULT_MULTIMODAL)
    if isinstance(raw, dict):
        merged.update({key: value for key, value in raw.items() if value is not None})
    return merged


def _as_list(content: Any) -> list[Any]:
    if isinstance(content, list):
        return content
    if isinstance(content, dict):
        return [content]
    return [content]


def _part_type(part: dict[str, Any]) -> str:
    return str(part.get("type") or "").strip().lower()


def part_text(part: Any) -> str:
    """The text of a text-like part, or ``""``.

    An image part may also carry a ``text`` field in some client encodings;
    that is a caption, not a separate text part, so it is handled by
    :func:`part_image` instead.
    """
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return ""
    if _part_type(part).startswith("image") or "image_url" in part:
        return ""
    text = part.get("text")
    if isinstance(text, str):
        return text
    return ""


def part_image(part: Any) -> tuple[str, str]:
    """Return ``(url, explicit_caption)`` for an image-like part.

    ``url`` is either an inline data URI, a remote URL, or ``""``.  The
    function is deliberately tolerant of the shapes seen in the wild
    (``image_url`` dict/str, ``source``, ``url``, ``data``).
    """
    if not isinstance(part, dict):
        return "", ""
    ptype = _part_type(part)
    caption = ""
    for key in ("caption", "alt", "description", "title", "text"):
        value = part.get(key)
        if isinstance(value, str) and value.strip():
            caption = value.strip()
            break

    url = ""
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        url = str(image_url.get("url") or "")
    elif isinstance(image_url, str):
        url = image_url
    for key in ("url", "file_name", "filename", "path", "name"):
        if not url and isinstance(part.get(key), str):
            url = str(part[key])
    if not url:
        source = part.get("source")
        if isinstance(source, dict):
            url = str(source.get("url") or source.get("data") or "")
        elif isinstance(source, str):
            url = source
    if not url and isinstance(part.get("data"), str):
        data = str(part["data"])
        url = data if data.startswith("data:") else f"data:image/*;base64,{data}"
    if ptype.startswith("image") or url or "image_url" in part:
        return url, caption
    return "", ""


def parse_data_uri(value: str) -> tuple[str, bytes] | None:
    """``data:image/png;base64,...`` -> ``(mime, bytes)`` or ``None``."""
    value = (value or "").strip()
    if not value.startswith("data:"):
        return None
    header, sep, payload = value.partition(",")
    if not sep or ";base64" not in header.lower():
        return None
    mime = header[5:].split(";", 1)[0].strip().lower()
    mime = {"image/jpg": "image/jpeg"}.get(mime, mime)
    if not mime.startswith("image/"):
        return None
    payload = "".join(payload.split())
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        try:
            data = base64.b64decode(payload)
        except (binascii.Error, ValueError):
            return None
    return mime, data


def data_uri_label(value: str) -> str:
    """A tiny, non-leaking description of an image URI."""
    value = (value or "").strip()
    if not value:
        return "[image]"
    if not value.startswith("data:"):
        return f"[image: {value[:120]}]"
    header, _, payload = value.partition(",")
    mime = header[5:].split(";", 1)[0].strip()
    size = (len(payload) * 3) // 4 if payload else 0
    suffix = f", ~{size} bytes" if size else ""
    return f"[image: {mime or 'inline'}{suffix}]"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class ImageStore:
    """Content-addressed ``<user>/images/`` directory.

    File name = SHA-256 of the decoded bytes, so the same image added twice
    occupies one file and the id is stable across retries.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(self, data: bytes, mime: str) -> dict[str, Any]:
        mime = {"image/jpg": "image/jpeg"}.get(mime, mime)
        ext = SUPPORTED_MIME.get(mime)
        if ext is None:
            raise ValueError(f"unsupported image mime: {mime}")
        sha = hashlib.sha256(data).hexdigest()
        path = self.root / sha[:2] / f"{sha}.{ext}"
        if not path.exists():
            _atomic_write_bytes(path, data)
        return {
            "id": sha[:16],
            "sha256": sha,
            "mime": mime,
            "bytes": len(data),
            "path": str(path.relative_to(self.root)),
        }

    def resolve(self, media: dict[str, Any]) -> Path | None:
        relative = str(media.get("path") or "")
        if not relative:
            return None
        try:
            root = self.root.resolve()
            path = (self.root / relative).resolve()
        except OSError:
            return None
        # Never let a hand-edited meta.media path escape the image directory.
        if path != root and root not in path.parents:
            return None
        return path if path.exists() else None

    def data_uri(self, media: dict[str, Any]) -> str:
        path = self.resolve(media)
        if path is None:
            return ""
        mime = str(media.get("mime") or "image/jpeg")
        try:
            data = path.read_bytes()
        except OSError:
            return ""
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    # -- caption cache ------------------------------------------------------
    def _caption_path(self, sha: str) -> Path:
        return self.root / "captions" / f"{sha}.txt"

    def read_caption(self, sha: str) -> str:
        if not sha:
            return ""
        try:
            return self._caption_path(sha).read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def write_caption(self, sha: str, caption: str) -> None:
        if not sha or not caption:
            return
        try:
            _atomic_write_bytes(self._caption_path(sha), caption.encode("utf-8"))
        except OSError:
            pass


def clean_caption(text: str, limit: int) -> str:
    text = _WS.sub(" ", str(text or "")).strip()
    text = text.replace("[", "(").replace("]", ")")
    if limit > 0 and len(text) > limit:
        text = text[:limit].rstrip()
    return text


class Captioner:
    """Optional vision-model captioner.  Failures are silent by design.

    Add must stay available even when the caption endpoint is down: a missing
    caption degrades to the text part / image marker, never to a failed add.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        mm = mm_config(self.config)
        generator = self.config.get("generator")
        generator = generator if isinstance(generator, dict) else {}
        self.enabled = bool(mm.get("caption", True))
        self.model = str(mm.get("caption_model") or generator.get("model") or "")
        self.base_url = str(mm.get("caption_base_url") or generator.get("base_url") or "")
        self.api_key = str(mm.get("caption_api_key") or generator.get("api_key") or "")
        self.max_chars = max(40, int(mm.get("caption_max_chars", 300) or 300))
        self.timeout = max(1.0, float(mm.get("caption_timeout", 20) or 20))
        self.max_image_bytes = max(1, int(mm.get("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES)))
        self.call_config = {
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "temperature": 0.0,
            "max_tokens": 200,
            "timeout": self.timeout,
            "connect_timeout": min(5.0, self.timeout),
            "max_retries": 1,
            "retry_truncated": False,
        }
        self.client: Any = None
        self.last_error = ""
        self.failures = 0

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.model and self.base_url)

    def _client(self) -> Any:
        if self.client is None:
            self.client = llm.make_client(self.call_config)
        return self.client

    async def caption(self, data_uri: str, *, hint: str = "") -> str:
        if not self.available:
            return ""
        parsed = parse_data_uri(data_uri)
        if parsed is None or parsed[0] not in SUPPORTED_MIME:
            return ""
        if len(parsed[1]) > self.max_image_bytes:
            return ""
        prompt = (
            "Describe this image factually for a memory-search index. Mention "
            "the visible people, objects, setting, colors, and any readable "
            "text. One or two sentences, no speculation.")
        hint = _WS.sub(" ", str(hint or "")).strip()
        if hint:
            prompt += f"\nAccompanying text: {hint[:400]}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }]
        try:
            text = await llm.chat_completion(
                self._client(), self.call_config, messages, role="caption")
        except Exception as exc:  # noqa: BLE001 - captioning is best-effort
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.failures += 1
            if self.failures >= 2:
                # A text-only endpoint fails every image; stop paying the
                # timeout on every Add/Search.  The text/caption fallback
                # keeps both endpoints working.
                self.enabled = False
            return ""
        self.failures = 0
        return clean_caption(text, self.max_chars)

    async def aclose(self) -> None:
        if self.client is None:
            return
        try:
            await self.client.close()
        except Exception:  # noqa: BLE001
            pass
        self.client = None


@dataclass
class ContentPlan:
    """The text a memory record gets, plus the image assets it references."""

    text: str = ""
    images: list[dict[str, Any]] = field(default_factory=list)


def _image_id(url: str) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"u{digest}"


async def build_content(
    content: Any,
    *,
    image_store: ImageStore | None = None,
    captioner: Captioner | None = None,
    config: dict[str, Any] | None = None,
    text_hint: str = "",
    budget: dict[str, int] | None = None,
) -> ContentPlan:
    """Normalize one message's content for Add.

    Inline images are stored, captioned, and referenced by a short id.  The
    returned ``images`` list is attached to the record metadata by the store.

    ``budget`` is a shared ``{"bytes": n}`` counter for a whole Add request:
    the protocol caps the request's decoded images at 30 MiB, not each message.
    """
    if isinstance(content, str):
        return ContentPlan(text=content.strip())
    if not isinstance(content, (list, dict)):
        return ContentPlan(text=str(content or "").strip())

    mm = mm_config(config)
    if not mm.get("enabled", True):
        from .api.models import normalize_content

        return ContentPlan(text=normalize_content(content))

    parts = _as_list(content)
    # part_text() is a pure scan of the part; compute it once and reuse it for
    # both the has_text probe and the loop below.
    part_texts = [part_text(part) for part in parts]
    has_text = any(text.strip() for text in part_texts)
    out: list[str] = []
    images: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    context_parts: list[str] = []
    total_bytes = 0
    used_before = int((budget or {}).get("bytes", 0) or 0)
    max_image = max(1, int(mm.get("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES)))
    max_request = max(1, int(mm.get("max_request_bytes", DEFAULT_MAX_REQUEST_BYTES)))
    generate = bool(captioner is not None and getattr(captioner, "available", False)
                    and mm.get("caption", True)
                    and (not has_text or mm.get("caption_always", False)))

    for part, text in zip(parts, part_texts):
        if text.strip():
            out.append(text)
            context_parts.append(text.strip())
            continue
        url, explicit = part_image(part)
        if not url and not _part_type(part).startswith("image"):
            continue
        caption = clean_caption(explicit, int(mm.get("caption_max_chars", 300) or 300))

        parsed = parse_data_uri(url) if url else None
        if parsed is not None:
            mime, data = parsed
            if mime not in SUPPORTED_MIME:
                out.append("[image omitted: unsupported media]")
                continue
            if len(data) > max_image:
                out.append("[image omitted: too large]")
                continue
            if used_before + total_bytes + len(data) > max_request:
                out.append("[image omitted: request budget exceeded]")
                continue
            if image_store is None:
                out.append(data_uri_label(url))
                continue
            try:
                media = image_store.save(data, mime)
            except (OSError, ValueError):
                out.append("[image omitted: store failed]")
                continue
            if not caption and generate:
                sha = str(media.get("sha256") or "")
                caption = image_store.read_caption(sha)
                if not caption:
                    hint = text_hint or " ".join(context_parts)
                    caption = await captioner.caption(url, hint=hint)
                    if caption:
                        image_store.write_caption(sha, caption)
            if media["id"] not in seen_ids:
                seen_ids.add(media["id"])
                media["caption"] = caption
                images.append(media)
            total_bytes += len(data)
            if budget is not None:
                budget["bytes"] = used_before + total_bytes
            if caption:
                out.append(f"[caption] {caption}")
            out.append(f"[image: {media['id']}]")
            continue

        if url and not url.startswith("data:"):
            # Remote URLs are outside the protocol; keep a short reference so
            # the record is still traceable, but never fetch or return it.
            media_id = _image_id(url)
            media = {"id": media_id, "url": url, "mime": "", "bytes": 0,
                     "caption": caption}
            if media_id not in seen_ids:
                seen_ids.add(media_id)
                images.append(media)
            if caption:
                out.append(f"[caption] {caption}")
            out.append(f"[image: {media_id}]")
            continue

        out.append("[image]" if not explicit else f"[caption] {caption}\n[image]")

    text_out = "\n".join(item.strip() for item in out if item and item.strip())
    return ContentPlan(text=text_out, images=images)


async def query_content(
    query: Any,
    *,
    captioner: Captioner | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    """Normalize a Search query to ``(retrieval_text, attachable_data_uris)``.

    Query images are not persisted.  Their captions make them searchable; at
    most ``max_query_images`` inline images are returned for the one bounded
    model call that needs to see them (the gate).
    """
    if isinstance(query, str):
        return query.strip(), []
    if not isinstance(query, (list, dict)):
        return str(query or "").strip(), []

    mm = mm_config(config)
    if not mm.get("enabled", True):
        from .api.models import normalize_content

        return normalize_content(query).strip(), []
    parts = _as_list(query)
    max_image = max(1, int(mm.get("max_image_bytes", DEFAULT_MAX_IMAGE_BYTES)))
    max_request = max(1, int(mm.get("max_request_bytes", DEFAULT_MAX_REQUEST_BYTES)))
    max_attach = max(0, int(mm.get("max_query_images", 1) or 0))
    generate = bool(captioner is not None and getattr(captioner, "available", False)
                    and mm.get("caption", True) and mm.get("caption_query", True))

    out: list[str] = []
    attached: list[str] = []
    total_bytes = 0
    text_so_far = ""
    for part in parts:
        text = part_text(part)
        if text.strip():
            text_so_far = text
            out.append(text)
            continue
        url, explicit = part_image(part)
        if not url and not _part_type(part).startswith("image"):
            continue
        caption = clean_caption(explicit, int(mm.get("caption_max_chars", 300) or 300))
        parsed = parse_data_uri(url) if url else None
        valid = (parsed is not None and parsed[0] in SUPPORTED_MIME
                 and len(parsed[1]) <= max_image
                 and total_bytes + len(parsed[1]) <= max_request)
        if not caption and generate and valid:
            caption = await captioner.caption(url, hint=text_so_far)
        if caption:
            out.append(f"[caption] {caption}")
        if valid and len(attached) < max_attach:
            attached.append(url)
            total_bytes += len(parsed[1])
        elif not valid:
            out.append(data_uri_label(url) if url else "[image]")
        else:
            out.append(f"[image: {_image_id(url)}]")
    return "\n".join(item.strip() for item in out if item and item.strip()), attached


def media_by_key(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Index record media by record id and parent id for Search lookups."""
    out: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        meta = record.get("meta")
        if not isinstance(meta, dict):
            continue
        media = meta.get("media")
        if not isinstance(media, list) or not media:
            continue
        clean = [item for item in media if isinstance(item, dict) and item.get("id")]
        if not clean:
            continue
        record_id = str(record.get("id") or "")
        parent_id = str(meta.get("parent_id") or "")
        if record_id:
            out[record_id] = clean
        if parent_id:
            out.setdefault(parent_id, clean)
    return out


def image_parts(
    media: Sequence[dict[str, Any]],
    store: ImageStore,
    *,
    max_images: int,
    max_image_bytes: int,
    remaining_bytes: int,
    seen: set[str],
    uri_cache: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    """Convert stored media to response ContentParts within the media budget.

    Returns ``(parts, images_added, bytes_used)``.  The protocol caps one
    decoded image at 10 MiB and a whole Search response at 30 MiB; an image
    that does not fit is silently skipped rather than truncating the response.
    """
    if max_images <= 0 or remaining_bytes <= 0:
        return [], 0, 0
    cache = uri_cache if uri_cache is not None else {}
    parts: list[dict[str, Any]] = []
    used = 0
    for item in media:
        if len(parts) >= max_images:
            break
        media_id = str(item.get("id") or "")
        if not media_id or media_id in seen:
            continue
        size = int(item.get("bytes") or 0)
        if size <= 0 or size > max_image_bytes:
            continue
        if used + size > remaining_bytes:
            continue
        uri = cache.get(media_id)
        if uri is None:
            uri = store.data_uri(item)
            if uri:
                cache[media_id] = uri
        if not uri:
            continue
        parts.append({"type": "image_url", "image_url": {"url": uri}})
        seen.add(media_id)
        used += size
    return parts, len(parts), used
