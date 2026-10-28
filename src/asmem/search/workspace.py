"""Per-request workspace, helper installation, and the candidate brief."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from ..io import atomic_write_text, link_or_copy, now_iso


def _run_id(question: str, request_key: str = "", payload_key: str = "") -> str:
    if request_key:
        # Deterministic: a retry of the same request resumes the same
        # workspace instead of starting a second agent run.
        digest = hashlib.sha1(
            f"{request_key}\0{payload_key}".encode("utf-8")).hexdigest()[:20]
        return f"req_{digest}"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha1(
        f"{question}\0{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
    return f"{stamp}_{digest}"

def _install_helper(source: Path, target: Path, *, executable: bool = False,
                    shebang: str | None = None) -> bool:
    """Install a workspace helper; optionally pin the interpreter and +x."""
    if not source.exists():
        return False
    try:
        text = source.read_text(encoding="utf-8")
        if shebang and text.startswith("#!"):
            _, _, body = text.partition("\n")
            text = f"{shebang}\n{body}"
        try:
            if (target.exists() and not target.is_symlink()
                    and target.read_text(encoding="utf-8") == text):
                if executable:
                    target.chmod(0o755)
                return True
        except OSError:
            pass
        atomic_write_text(target, text)
        if executable:
            target.chmod(0o755)
        return True
    except OSError:
        return False


#: Repo helper -> workspace command name.  ``coarse`` is a global PATH command
#: (``scripts/coarse`` / the console script), so it is not copied per run.
_HELPERS = (
    ("append_matches.py", "append_matches.py"),
    ("query_tables.py", "query_tables.py"),
)


def _install_helpers(bin_dir: Path) -> None:
    """Install the three bash-tool helpers, all directly executable.

    The same files are installed once in ``shared/bin`` and every run links to
    that directory, so a store with thousands of runs does not carry thousands
    of copies of the same three scripts.
    """
    here = Path(__file__).with_name
    shebang = f"#!{sys.executable}"
    for source_name, target_name in _HELPERS:
        _install_helper(here(source_name), bin_dir / target_name,
                        executable=True, shebang=shebang)


def _link_dir(source: Path, target: Path) -> bool:
    """Point ``target`` at ``source`` with a relative symlink.

    Returns True when the link was created (or already correct).  An existing
    real directory is kept: it is a self-contained workspace from an older run
    and rewriting it would discard the exact helpers that run used.
    """
    if target.is_symlink():
        try:
            if target.resolve() == source.resolve():
                return True
        except OSError:
            pass
        try:
            target.unlink()
        except OSError:
            return False
    if target.exists():
        return False
    try:
        target.symlink_to(os.path.relpath(source, target.parent),
                          target_is_directory=True)
        return True
    except OSError:
        try:
            shutil.copytree(source, target, dirs_exist_ok=True)
        except OSError:
            pass
        return False


def _snapshot_info(path: Path, link: str) -> dict[str, Any]:
    info: dict[str, Any] = {"source": str(path), "link": link}
    try:
        stat = path.stat()
        info.update({"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    except OSError:
        pass
    return info


def prepare_workspace(memory_path: Path, question: str,
                      run_root: Path, options: list[str] | None = None,
                      config: dict[str, Any] | None = None,
                      cache_dir: Path | None = None,
                      request_key: str = "",
                      payload_key: str = "",
                      shared_dir: Path | None = None) -> tuple[Path, Path, bool]:
    """Create (or reopen) the per-request workspace.

    With a ``request_key`` the run directory is deterministic, so a retried
    HTTP request lands in the same place.  ``resumed`` reports whether an
    interrupted attempt left evidence behind: the caller then keeps that file
    as the agent's starting point instead of reseeding it.

    Runs are flat (``runs/<id>/evidence.jsonl``).  Version-1 workspaces nested
    under ``runs/<id>/qa/`` are still reopened in place when a retry finds no
    flat ``request.json``.
    """
    run_base = run_root / _run_id(question, request_key, payload_key)
    legacy_qa = run_base / "qa"
    qa = (legacy_qa if legacy_qa.is_dir()
          and not (run_base / "request.json").exists() else run_base)
    inputs = qa / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    memory_link = link_or_copy(memory_path, inputs / "memory.jsonl")
    state_path = memory_path.parent / "state.jsonl"
    state_link = ""
    if state_path.exists() and state_path != memory_path:
        state_link = link_or_copy(state_path, inputs / "state.jsonl")
    bin_dir = qa / "bin"
    if shared_dir is not None:
        shared_bin = Path(shared_dir) / "bin"
        shared_bin.mkdir(parents=True, exist_ok=True)
        _install_helpers(shared_bin)
        _link_dir(shared_bin, bin_dir)
    else:
        bin_dir.mkdir(parents=True, exist_ok=True)
        _install_helpers(bin_dir)
    retrieval = (config or {}).get("retrieval")
    workspace_config = dict(retrieval) if isinstance(retrieval, dict) else {}
    # The CLI must share the main coarse stage's embedding cache; otherwise
    # every rewritten query would re-embed the whole snapshot.
    workspace_config["_cache_dir"] = str(cache_dir) if cache_dir else ""
    atomic_write_text(inputs / "retrieval.json",
                      json.dumps(workspace_config, ensure_ascii=False, indent=2))
    request_path = qa / "request.json"
    previous: dict[str, Any] = {}
    if request_path.exists():
        try:
            loaded = json.loads(request_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (json.JSONDecodeError, OSError):
            previous = {}
    evidence_path = qa / "evidence.jsonl"
    resumed = bool(
        request_key and previous
        and (not previous.get("question") or previous.get("question") == question)
        and evidence_path.exists()
        and evidence_path.stat().st_size > 0)
    atomic_write_text(request_path, json.dumps({
        "question": question, "options": options or [],
        "memory": str(memory_path),
        "snapshot": {
            "memory": _snapshot_info(memory_path, memory_link),
            "state": _snapshot_info(state_path, state_link) if state_link else None,
        },
        "request_key": request_key,
        "payload_key": payload_key,
        "created_at": previous.get("created_at") or now_iso(),
        "updated_at": now_iso(),
        "resume_count": int(previous.get("resume_count") or 0)
        + (1 if resumed else 0),
    }, ensure_ascii=False, indent=2))
    return qa, inputs, resumed


def _shell_prefix() -> str:
    """Expose ``coarse`` and the package modules inside every agent command."""
    repo_root = Path(__file__).resolve().parents[3]
    src_dir = repo_root / "src"
    scripts_dir = repo_root / "scripts"
    return (f'export LC_ALL=C; export PATH="{scripts_dir}:$PATH"; '
            f'export PYTHONPATH="{src_dir}:${{PYTHONPATH:-}}"')


def _brief(candidates: list[dict[str, Any]], question: str, intent: str,
           *, limit: int = 10) -> str:
    """A compact, model-readable summary of the shortlist.

    The agent used to spend its first step inspecting ``candidates.jsonl`` with
    jq, often degenerating into a giant filter.  The runner now renders the
    decision-relevant part (id, time, role, matched terms, a short snippet) so
    the first model step can be a real decision or a targeted search.
    """
    lines = [
        f"INTENT: {intent}",
        f"QUESTION: {question}",
        "",
        "TOP CANDIDATES (pre-ranked; verify these first, do not bulk-copy):",
    ]
    for candidate in list(candidates)[:limit]:
        text = " ".join(str(candidate.get("text") or "").split())[:220]
        why = ",".join(str(item) for item in (candidate.get("why") or [])[:6])
        lines.append(
            f"- #{candidate.get('rank')} {candidate.get('id')} | "
            f"{candidate.get('time') or 'time?'} | {candidate.get('role') or '-'} | "
            f"why: {why} | {text}")
    if len(candidates) > limit:
        lines.append(f"({len(candidates) - limit} more candidates in inputs/candidates.jsonl)")
    return "\n".join(lines)


