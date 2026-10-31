"""Move legacy per-user images into the shared content-addressed store.

    python scripts/dedupe_store_images.py --root data/api_store_mm
    python scripts/dedupe_store_images.py --root data/api_store_mm --confirm

The API now writes each image once to ``<store>/shared/images/``.  Stores
created by the older hardlink layout (or before it) still carry per-user
directory entries, so the same image appears at both paths.  This tool:

1. moves a user-only image into ``shared/images/`` (hash verified);
2. removes a user entry that is a hardlink to the shared file;
3. removes a user copy only when its SHA-256 matches the shared file;
4. does the same for ``images/captions/*.txt``.

Default is a dry run; ``--confirm`` actually modifies files.  Stop the API
first: a move has a brief window where neither path exists.

The file name must be its SHA-256 (``<sha>.<ext>``); a mismatch is reported and
left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _user_dirs(root: Path):
    seen: set[Path] = set()
    users = root / "users"
    if users.is_dir():
        for child in sorted(users.iterdir()):
            if child.is_dir():
                seen.add(child.resolve())
                yield child
    if root.is_dir():
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name in {"users", "shared"}:
                continue
            if child.resolve() not in seen:
                yield child


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_content(left: Path, right: Path, verify: bool) -> str:
    try:
        ls, rs = left.stat(), right.stat()
    except OSError:
        return "missing"
    if (ls.st_dev, ls.st_ino) == (rs.st_dev, rs.st_ino):
        return "same_inode"
    if ls.st_size != rs.st_size:
        return "different"
    if verify and _sha256(left) != _sha256(right):
        return "different"
    return "same_content"


def _expected_sha(path: Path) -> str:
    return path.stem.lower()


def _move_into_shared(source: Path, target: Path, *, verify: bool,
                      report: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify and _sha256(source) != _expected_sha(source):
        report["kept"].append({"path": str(source), "reason": "sha_mismatch"})
        return
    try:
        os.replace(source, target)
        report["moved"].append(str(source))
        return
    except OSError:
        pass
    try:
        shutil.copy2(source, target)
        if verify and _sha256(target) != _sha256(source):
            target.unlink(missing_ok=True)
            report["errors"].append({"path": str(source), "reason": "copy_verify"})
            return
        source.unlink()
        report["moved"].append(str(source))
    except OSError as exc:
        report["errors"].append({"path": str(source), "reason": str(exc)})


def _remove_duplicate(source: Path, target: Path, *, verify: bool,
                      report: dict) -> None:
    relation = _same_content(source, target, verify)
    if relation in {"same_inode", "same_content"}:
        try:
            size = source.stat().st_size
            source.unlink()
            report["removed"].append(str(source))
            if relation == "same_content":
                # A real copy was removed; a hardlink only loses a directory
                # entry (the inode stays in shared/).
                report["bytes_saved"] += size
        except OSError as exc:
            report["errors"].append({"path": str(source), "reason": str(exc)})
        return
    report["kept"].append({"path": str(source), "reason": relation})


def _cleanup_empty(root: Path) -> None:
    if not root.is_dir():
        return
    for child in sorted(root.rglob("*"), key=lambda path: len(path.parts),
                        reverse=True):
        if child.is_dir():
            try:
                child.rmdir()
            except OSError:
                pass


def dedupe(root: Path, *, confirm: bool, verify: bool) -> dict:
    shared = root / "shared" / "images"
    report = {"root": str(root), "confirm": confirm, "verify": verify,
              "moved": [], "removed": [], "kept": [], "errors": [],
              "bytes_saved": 0}
    for user_dir in _user_dirs(root) or ():
        images = user_dir / "images"
        if not images.is_dir():
            continue
        for source in sorted(images.rglob("*")):
            if source.is_symlink() or not source.is_file():
                continue
            relative = source.relative_to(images)
            if relative.parts and relative.parts[0] == "captions":
                if source.suffix != ".txt":
                    continue
            elif source.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if verify and source.suffix.lower() in IMAGE_SUFFIXES:
                if _sha256(source) != _expected_sha(source):
                    report["kept"].append(
                        {"path": str(source), "reason": "sha_mismatch"})
                    continue
            target = shared / relative
            if target.exists():
                if confirm:
                    _remove_duplicate(source, target, verify=verify,
                                      report=report)
                else:
                    relation = _same_content(source, target, verify)
                    if relation in {"same_inode", "same_content"}:
                        report["removed"].append(str(source))
                    else:
                        report["kept"].append(
                            {"path": str(source), "reason": relation})
            elif confirm:
                _move_into_shared(source, target, verify=verify, report=report)
            else:
                report["moved"].append(str(source))
    if confirm:
        for user_dir in _user_dirs(root) or ():
            _cleanup_empty(user_dir / "images")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="store root (contains users/ and shared/)")
    parser.add_argument("--confirm", action="store_true",
                        help="actually modify files (default: dry run)")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip SHA-256 verification (faster, less safe)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"store root not found: {root}", file=sys.stderr)
        return 2
    report = dedupe(root, confirm=args.confirm, verify=not args.no_verify)
    report["moved_count"] = len(report["moved"])
    report["removed_count"] = len(report["removed"])
    report["kept_count"] = len(report["kept"])
    report["error_count"] = len(report["errors"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
