"""Pure-CPU multimodal tests: caption text, image assets, bounded responses.

    python scripts/test_multimodal.py
"""

from __future__ import annotations

import asyncio
import base64
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asmem import llm, multimodal  # noqa: E402
from asmem.api.models import normalize_content  # noqa: E402
from asmem.api.server import build_search_items  # noqa: E402
from asmem.api.store import UserStore, payload_hash  # noqa: E402
from asmem.schema import normalize_messages  # noqa: E402


def _png(size: int = 100) -> str:
    data = b"\x89PNG\r\n\x1a\n" + b"x" * size
    return base64.b64encode(data).decode("ascii")


def _image_part(size: int = 100, **extra) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_png(size)}"}, **extra}


def test_content_plan() -> None:
    async def main() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = multimodal.ImageStore(Path(tmp) / "images")
            plan = await multimodal.build_content(
                [{"type": "text", "text": "A red car in Paris"},
                 _image_part()],
                image_store=store, config={})
            assert plan.text.startswith("A red car in Paris"), plan.text
            assert "[caption]" not in plan.text     # text is the caption
            assert "[image: " in plan.text
            assert "base64" not in plan.text
            assert len(plan.images) == 1
            media = plan.images[0]
            assert media["mime"] == "image/png" and media["bytes"] > 0
            assert store.resolve(media) is not None
            assert store.data_uri(media).startswith("data:image/png;base64,")

            explicit = await multimodal.build_content(
                [{"type": "image_url", "caption": "a blue sky",
                  "image_url": {"url": f"data:image/png;base64,{_png(10)}"}}],
                image_store=store, config={})
            assert "[caption] a blue sky" in explicit.text, explicit.text
            assert "[image: " in explicit.text
    asyncio.run(main())


def test_image_limits() -> None:
    async def main() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = multimodal.ImageStore(Path(tmp) / "images")
            config = {"multimodal": {"max_image_bytes": 10}}
            plan = await multimodal.build_content(
                [_image_part(100)], image_store=store, config=config)
            assert "too large" in plan.text and not plan.images, plan

            unsupported = {
                "type": "image_url",
                "image_url": {"url": "data:image/gif;base64," + _png(10)},
            }
            plan = await multimodal.build_content(
                [unsupported], image_store=store, config={})
            assert "unsupported" in plan.text and not plan.images, plan

            # The 30 MiB cap is per Add request, not per message.
            budget = {"bytes": 0}
            config = {"multimodal": {"max_request_bytes": 150,
                                     "max_image_bytes": 1000}}
            first = await multimodal.build_content(
                [_image_part(100)], image_store=store, config=config,
                budget=budget)
            second = await multimodal.build_content(
                [_image_part(100)], image_store=store, config=config,
                budget=budget)
            assert first.images and not second.images, (first, second)
            assert "request budget exceeded" in second.text, second.text
    asyncio.run(main())


def test_caption_cache() -> None:
    class FakeCaptioner:
        available = True

        def __init__(self) -> None:
            self.calls = 0

        async def caption(self, uri: str, *, hint: str = "") -> str:
            self.calls += 1
            return "a green field with a tree"

    async def main() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = multimodal.ImageStore(Path(tmp) / "images")
            captioner = FakeCaptioner()
            plan = await multimodal.build_content(
                [_image_part()], image_store=store, captioner=captioner,
                config={})
            assert "[caption] a green field with a tree" in plan.text, plan.text
            assert plan.images[0]["caption"] == "a green field with a tree"
            assert captioner.calls == 1

            # The caption is cached by the image hash; a retry is free.
            again = await multimodal.build_content(
                [_image_part()], image_store=store, captioner=captioner,
                config={})
            assert "[caption] a green field with a tree" in again.text
            assert captioner.calls == 1

            # A query image is captioned for retrieval as well.
            text, images = await multimodal.query_content(
                [{"type": "text", "text": "Which answer matches?"}, _image_part()],
                captioner=captioner, config={})
            assert "[caption] a green field with a tree" in text, text
            assert len(images) == 1

    asyncio.run(main())


def test_query_caption() -> None:
    async def main() -> None:
        text, images = await multimodal.query_content(
            [{"type": "text", "text": "Which answer matches this image?"},
             _image_part()], config={})
        assert text.startswith("Which answer matches this image?"), text
        assert "base64" not in text
        assert len(images) == 1 and images[0].startswith("data:image/png;base64,")

        text, images = await multimodal.query_content(
            [_image_part(10, caption="a small tree")], config={})
        assert text == "[caption] a small tree", text
        assert len(images) == 1
    asyncio.run(main())


def test_store_media_and_chunks() -> None:
    async def main() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = UserStore("alice", Path(tmp))
            plan = await multimodal.build_content(
                [{"type": "text", "text": "short text"}, _image_part()],
                image_store=multimodal.ImageStore(store.images_dir), config={})
            messages = [{"role": "user", "content": plan.text}]
            raw = payload_hash("alice", "s1", messages)
            await store.append("r1", "s1", messages, media=plan.images,
                               payload_hash=raw)
            records = store.records()
            assert records and records[0]["meta"].get("media"), records
            assert records[0]["meta"]["media"][0]["id"] == plan.images[0]["id"]

            # A retry that produces a slightly different caption is still the
            # same request because the hash is computed from the raw payload.
            await store.append("r1", "s1",
                               [{"role": "user", "content": "a different caption"}],
                               media=[], payload_hash=raw)
            assert len(store.records()) == len(records)

            # Chunking is store-level config; build a tiny-window store.
            long_text = " ".join(["context"] * 120) + "\n" + plan.text
            store3 = UserStore("carol", Path(tmp), config={
                "retrieval": {"chunking": {"message_tokens": 20,
                                           "min_tokens": 5,
                                           "overlap_tokens": 0}}})
            await store3.append(
                "r3", "s3", [{"role": "user", "content": long_text}],
                media=plan.images)
            chunks = store3.records()
            assert len(chunks) > 1, len(chunks)
            parents = {chunk["meta"]["parent_id"] for chunk in chunks}
            assert len(parents) == 1, parents
            assert all(chunk["meta"].get("media") for chunk in chunks), chunks
    asyncio.run(main())


def test_build_search_items_images() -> None:
    async def main() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = UserStore("alice", Path(tmp))
            plan = await multimodal.build_content(
                [{"type": "text", "text": "a photo caption"}, _image_part()],
                image_store=multimodal.ImageStore(store.images_dir), config={})
            await store.append("r1", "s1",
                               [{"role": "user", "content": plan.text}],
                               media=plan.images)
            records = {item["id"]: item for item in store.records()}
            record = next(iter(records.values()))
            evidence = [{
                "content": record["text"],
                "metadata": {"source": [record["id"]],
                             "anchor_id": record["id"]},
            }]
            items = build_search_items(
                evidence, records, multimodal=True,
                media_root=store.images_dir, config={})
            parts = items[0].content
            assert parts[0]["type"] == "text" and parts[1]["type"] == "image_url"
            assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")

            # Budget zero -> text part only, never a giant URI.
            items = build_search_items(
                evidence, records, multimodal=True, media_root=store.images_dir,
                config={"multimodal": {"max_response_images": 0}})
            assert [part["type"] for part in items[0].content] == ["text"]

            # Text/code track is untouched.
            items = build_search_items(evidence, records, top_k=10)
            assert isinstance(items[0].content, str)
    asyncio.run(main())


def test_no_base64_in_memory() -> None:
    content = [{"type": "text", "text": "hello"}, _image_part()]
    # The pure fallback used outside the API store.
    text = normalize_content(content)
    assert "base64" not in text and _png(40) not in text, text

    # And the schema's own fallback when a ContentPart[] reaches it directly.
    _, records = normalize_messages(
        [{"role": "user", "content": content}], session_id="s1", session_index=1)
    joined = "\n".join(record["text"] for record in records)
    assert "base64" not in joined and _png(40) not in joined, joined


def test_fit_messages_content_parts() -> None:
    uri = f"data:image/png;base64,{_png(1000)}"
    messages = [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": [
            {"type": "text", "text": "question"},
            {"type": "image_url", "image_url": {"url": uri}},
        ]},
    ]
    fitted = llm.fit_messages(messages, 1000)
    rendered = str(fitted)
    assert "base64" not in rendered and uri not in rendered
    # A generous budget keeps the ordered parts and the image.
    fitted = llm.fit_messages(messages, 24000)
    assert isinstance(fitted[-1]["content"], list)
    assert fitted[-1]["content"][1]["type"] == "image_url"


def main() -> None:
    test_content_plan()
    test_image_limits()
    test_caption_cache()
    test_query_caption()
    test_store_media_and_chunks()
    test_build_search_items_images()
    test_no_base64_in_memory()
    test_fit_messages_content_parts()
    print("test_multimodal: OK")


if __name__ == "__main__":
    main()
