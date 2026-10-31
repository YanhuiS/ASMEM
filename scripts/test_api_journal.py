"""Pure-CPU tests for the raw request journal and multimodal dedupe.

    python scripts/test_api_journal.py

No network, no model calls.  Covers:

* raw Add/Search payload saved once (content-addressed), images externalized;
* the same image across two users occupies one physical shared blob;
* a query image is stored once and its caption is reused across searches;
* multimodal-track responses are ContentPart[] even for a text-only query;
* an end-to-end /add + /search through FastAPI writes request + response
  journals without putting base64 into them.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem import multimodal  # noqa: E402
from asmem.api import store as store_mod  # noqa: E402
from asmem.api.server import (build_search_items, create_app,  # noqa: E402
                              response_uses_parts)
from asmem.api.store import UserStore  # noqa: E402
from asmem.schema import normalize_messages  # noqa: E402


def _png(size: int = 64) -> str:
    data = b"\x89PNG\r\n\x1a\n" + b"x" * size
    return base64.b64encode(data).decode("ascii")


def _image_part(size: int = 64) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_png(size)}"}}


def _journal_files(store: UserStore, kind: str) -> list[Path]:
    # requests/<kind>/<sha[:2]>/<sha>.json (responses live one level deeper)
    return sorted((store.root / "requests" / kind).glob("*/*.json"))


def test_add_journal_dedup() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = UserStore("alice", Path(tmp))
        payload = {
            "request_id": "r1", "user_id": "alice", "session_id": "s1",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "a red car"},
                _image_part(),
            ]}],
        }
        first = store.journal.record("add", payload, meta={"request_id": "r1"})
        assert first["created"] and first["media"] == 1, first
        text = (store.root / "requests" / first["path"]).read_text(
            encoding="utf-8")
        assert "base64" not in text, text
        assert "asmem-image:" in text
        record = json.loads(text)
        assert record["payload"]["messages"][0]["content"][1]["image_url"][
            "url"].startswith("asmem-image:")

        # Writing the same payload again is a no-op (same file, same bytes).
        second = store.journal.record("add", payload, meta={"request_id": "r1"})
        assert second["sha256"] == first["sha256"]
        assert second["created"] is False
        assert len(_journal_files(store, "add")) == 1

        # Exactly one image file exists (shared blob, no per-user copy), and
        # the journal can resolve it.
        shared = [path for path in (store.shared_images_dir).rglob("*.*")
                  if path.suffix in {".png", ".jpg", ".webp"}]
        user = [path for path in (store.images_dir).rglob("*.*")
                if path.suffix in {".png", ".jpg", ".webp"}]
        assert len(shared) == 1, shared
        assert not user, user
        asset = record["media"][0]
        assert (store.shared_images_dir / asset["path"]).exists()
        assert asset.get("storage") == "shared"


def test_shared_image_and_caption_dedupe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        alice = UserStore("alice", root)
        bob = UserStore("bob", root)
        data = b"\x89PNG\r\n\x1a\n" + b"same" * 20
        a = alice.image_store.save(data, "image/png")
        b = bob.image_store.save(data, "image/png")
        assert a["sha256"] == b["sha256"]
        assert a.get("storage") == b.get("storage") == "shared"
        # One physical file under shared/; no per-user directory entries.
        shared = [path for path in (root / "shared" / "images").rglob("*.png")]
        assert len(shared) == 1, shared
        assert not list((alice.images_dir).rglob("*.png"))
        assert not list((bob.images_dir).rglob("*.png"))
        assert alice.image_store.resolve(a) == shared[0].resolve()
        assert bob.image_store.resolve(b) == shared[0].resolve()

        alice.image_store.write_caption(a["sha256"], "a shared caption")
        assert bob.image_store.read_caption(a["sha256"]) == "a shared caption"
        assert (root / "shared" / "images" / "captions"
                / f"{a['sha256']}.txt").exists()
        assert not (alice.images_dir / "captions"
                    / f"{a['sha256']}.txt").exists()


def test_query_image_reuses_store_and_caption() -> None:
    class FakeCaptioner:
        available = True

        def __init__(self) -> None:
            self.calls = 0

        async def caption(self, uri: str, *, hint: str = "") -> str:
            self.calls += 1
            return "a red car in paris"

    async def main() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = UserStore("alice", Path(tmp))
            captioner = FakeCaptioner()
            query = [{"type": "text", "text": "Which car?"}, _image_part()]
            media1: list[dict] = []
            media2: list[dict] = []
            text1, attached1 = await multimodal.query_content(
                query, captioner=captioner, config={},
                image_store=store.image_store, media_out=media1)
            text2, attached2 = await multimodal.query_content(
                query, captioner=captioner, config={},
                image_store=store.image_store, media_out=media2)
            assert "[caption] a red car in paris" in text1, text1
            assert text1 == text2
            assert captioner.calls == 1, captioner.calls
            assert len(media1) == 1 and len(media2) == 1
            assert media1[0]["sha256"] == media2[0]["sha256"]
            assert media1[0]["path"] == media2[0]["path"]
            assert len(attached1) == 1 and len(attached2) == 1
            images = [path for path in (store.shared_images_dir).rglob("*.*")
                      if path.suffix in {".png", ".jpg", ".webp"}]
            assert len(images) == 1, images
            assert not [path for path in (store.images_dir).rglob("*.*")
                        if path.suffix in {".png", ".jpg", ".webp"}]

    asyncio.run(main())


def test_response_parts_track() -> None:
    assert response_uses_parts(
        {"multimodal": {"response_parts": "always"}}, False) is True
    assert response_uses_parts({}, False) is False
    assert response_uses_parts({}, True) is True

    evidence = [{"content": "claim A",
                 "metadata": {"source": ["s1_00001"],
                              "anchor_id": "s1_00001"}}]
    records = {"s1_00001": {"id": "s1_00001"}}
    items = build_search_items(evidence, records, multimodal=True, top_k=10)
    assert isinstance(items[0].content, list)
    assert items[0].content[0] == {"type": "text", "text": "claim A"}


def test_api_end_to_end_journal() -> None:
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as tmp:
        config = {
            "store_dir": str(Path(tmp) / "store"),
            "retrieval": {"mode": "coarse", "dense": {"mode": "off"}},
            "multimodal": {"enabled": True, "caption": False,
                           "response_parts": "always"},
            "request_journal": True,
            "request_journal_responses": True,
            "search_timeout": 60,
        }
        app = create_app(config)
        with TestClient(app) as client:
            add_body = {
                "request_id": "r1", "user_id": "alice", "session_id": "s1",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "A red car in Paris"},
                    _image_part(),
                ]}],
            }
            added = client.post("/add", json=add_body)
            assert added.status_code == 200, added.text
            assert added.json()["success"] is True

            searched = client.post("/search", json={
                "query": "What color is the car in Paris?",
                "user_id": "alice", "top_k": 10})
            assert searched.status_code == 200, searched.text
            body = searched.json()
            assert body["data"], body
            item = body["data"][0]
            assert isinstance(item["content"], list), item
            assert item["content"][0]["type"] == "text"
            assert item["id"] and item["content"][0]["text"].strip()
            assert "score" in item and item["score"] is not None
            # The image lives only in shared/images; the response must resolve
            # it through shared_media_root.
            assert any(part.get("type") == "image_url"
                       for part in item["content"]), item["content"]

        root = Path(config["store_dir"])
        user_dir = store_mod.resolve_user_dir(root, "alice")
        adds = sorted((user_dir / "requests" / "add").glob("*/*.json"))
        searches = sorted((user_dir / "requests" / "search").glob("*/*.json"))
        responses = sorted(
            (user_dir / "requests" / "search" / "responses").rglob("*.json"))
        assert len(adds) == 1, adds
        assert len(searches) == 1, searches
        assert len(responses) >= 1, responses
        for path in adds + searches + responses:
            text = path.read_text(encoding="utf-8")
            assert "base64" not in text, path
        # Exactly one image: Add stored it in the shared blob store, Search
        # reused the same hash, and no per-user copy was created.
        images = [path for path in (root / "shared" / "images").rglob("*.*")
                  if path.suffix in {".png", ".jpg", ".webp"}]
        assert len(images) == 1, images
        assert not [path for path in (user_dir / "images").rglob("*.*")
                    if path.suffix in {".png", ".jpg", ".webp"}]


def main() -> None:
    test_add_journal_dedup()
    test_shared_image_and_caption_dedupe()
    test_query_image_reuses_store_and_caption()
    test_response_parts_track()
    test_api_end_to_end_journal()
    print("test_api_journal: OK")


if __name__ == "__main__":
    main()
