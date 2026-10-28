#!/usr/bin/env python3
"""Check relative Markdown links and heading anchors under docs/.

    python scripts/check_docs.py

External URLs are ignored.  Link fragments are matched against GitHub-style
heading slugs (lowercase, punctuation removed, spaces converted to hyphens).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
# ``docs/archive`` is historical material whose relative links refer to the
# pre-consolidation layout; it is intentionally excluded from the live check.
CANDIDATES = [
    ROOT / "README.md",
    *sorted(path for path in (ROOT / "docs").glob("*.md")),
]
LINK_RE = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"```.*?```", re.S)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)
INLINE_LINK_RE = re.compile(r"\[([^]]+)\]\([^)]+\)")


def slug(text: str) -> str:
    text = INLINE_LINK_RE.sub(r"\1", text).strip().lower()
    text = text.replace("`", "").replace("*", "")
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"[\s-]+", "-", text).strip("-")


def anchors(path: Path) -> set[str]:
    text = FENCE_RE.sub("", path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    result: set[str] = set()
    for match in HEADING_RE.finditer(text):
        base = slug(match.group(2))
        occurrence = counts.get(base, 0)
        result.add(base if occurrence == 0 else f"{base}-{occurrence}")
        counts[base] = occurrence + 1
    return result


def main() -> int:
    broken: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    for source in CANDIDATES:
        text = FENCE_RE.sub("", source.read_text(encoding="utf-8"))
        for raw in LINK_RE.findall(text):
            target = raw.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue
            file_part, _, fragment = target.partition("#")
            target_path = source if not file_part else (source.parent / unquote(file_part)).resolve()
            if not target_path.exists():
                broken.append(f"{source.relative_to(ROOT)} -> {target} (missing file)")
                continue
            if fragment:
                if target_path not in anchor_cache:
                    anchor_cache[target_path] = anchors(target_path)
                fragment = unquote(fragment).lower()
                if fragment not in anchor_cache[target_path]:
                    broken.append(
                        f"{source.relative_to(ROOT)} -> {target} (missing anchor)")
    if broken:
        print(f"broken docs links: {len(broken)}")
        for item in broken:
            print(item)
        return 1
    print(f"docs links: OK ({len(CANDIDATES)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
