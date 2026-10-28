"""The postagent loop: read, search, write evidence, curate, finish.

The postagent runs after deterministic coarse recall has seeded
``evidence.jsonl``.  It has four tools: ``read`` / ``write`` / ``edit`` /
``bash``.  ``bash`` exposes shell grep/jq and the global ``coarse`` command
(which re-runs the deterministic first-stage retriever with a rewritten query).
This is the iterative, multi-hop loop: read -> name the missing slot -> search
-> write evidence -> curate -> finish.

Only the guards that survive simplification are kept: observation truncation,
repeated-call rejection, a no-progress stop, a nudge when the evidence file
stays empty, append-size limits, and context compaction.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..llm import TruncatedCompletion
from .. import llm
from . import append_matches, evidence, prompts, toolcfg
from .evidence import EVIDENCE_NAME
from .toolcfg import ToolCall
from .tools import ToolError, ToolResult

ModelCall = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None],
                     Awaitable["ModelReply"]]
AuditCall = Callable[[list[dict[str, Any]]], Awaitable[list[str]]]


@dataclass
class ModelReply:
    """One model response: either tool calls or a plain-text finish."""

    text: str = ""
    calls: list[ToolCall] = field(default_factory=list)
    assistant_entry: dict[str, Any] | None = None
    raw: Any = None

    @property
    def done(self) -> bool:
        return not self.calls and bool(self.text.strip()) and not toolcfg.looks_like_call(self.text)


@dataclass
class PostagentResult:
    steps: int = 0
    stop_reason: str = ""
    final_text: str = ""
    evidence_lines: int = 0
    audited: bool = False
    audit_missing: list[str] = field(default_factory=list)
    audit_sufficient: bool | None = None
    audit_at_finish: bool = False
    audit_fingerprint: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    #: Structured tool log (written to ``steps.jsonl`` by the runner).
    step_log: list[dict[str, Any]] = field(default_factory=list)


def _observation(result: ToolResult, limit: int) -> str:
    text = result.text or "(no output)"
    if limit and len(text) > limit:
        text = text[:limit] + f"\n... [truncated, {len(text)} chars total]"
    return text


def _implicit_evidence(text: str) -> list[dict[str, Any]]:
    """Accept an evidence record that the model replied with directly.

    Small models sometimes answer with ``{"content": ..., "metadata": ...}``
    instead of wrapping it in a bash heredoc.  The harness appends those
    records (the same schema ``edit``/bash would write) and keeps going, so a
    formatting habit cannot throw away a correct fact.
    """
    out: list[dict[str, Any]] = []
    for record in evidence.parse_text(text or ""):
        if not isinstance(record, dict) or "content" not in record:
            continue
        if "tool" in record or "sufficient" in record:
            continue
        normalized, _ = evidence.normalize(record)
        if normalized is not None:
            out.append(normalized)
    return out


def _append_implicit(evidence_path: Path, records: list[dict[str, Any]]) -> tuple[int, int]:
    """Append only records whose content is not already in the file.

    The model sometimes repeats the same evidence object across steps.  A
    duplicate append is not just wasted tokens: it stalls the loop and hides
    the fact that the missing slot was never searched.
    """
    current = evidence.read_file(evidence_path)
    seen = {" ".join(str(item.get("content") or "").split()).lower()
            for item in current}
    fresh: list[dict[str, Any]] = []
    for record in records:
        key = " ".join(str(record.get("content") or "").split()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        fresh.append(record)
    if fresh:
        evidence.write_file(evidence_path, current + fresh)
    return len(fresh), len(current) + len(fresh)


def _evidence_call(call: ToolCall) -> bool:
    """Whether this call can change ``evidence.jsonl``."""
    if call.tool in {"write", "edit"}:
        raw = call.args.get("path")
        return raw is None or Path(str(raw)).name == "evidence.jsonl"
    if call.tool == "bash":
        return "evidence.jsonl" in str(call.args.get("command") or "")
    return False


def _append_call(call: ToolCall) -> bool:
    """Whether the call is an append rather than a replace/curation."""
    if call.tool == "write":
        return str(call.args.get("mode") or "append").strip().lower() != "replace"
    if call.tool == "edit":
        return call.args.get("append") is not None and call.args.get("edits") is None
    if call.tool == "bash":
        command = str(call.args.get("command") or "")
        return ">>" in command or "append_matches" in command
    return False


def _is_coarse_command(command: str) -> bool:
    return bool(re.search(
        r"(?:^|[\s;|&])(?:bin/)?coarse(?:\.py|_search\.py)?(?:\s|$)",
        command or ""))


def _normalize_shell_call(call: ToolCall) -> ToolCall:
    """Accept ``coarse``/``grep`` as pseudo-tools and route them through bash.

    The model sometimes replies ``{"tool":"coarse","args":{"query":...}}``
    even though the prompt says these are shell commands.  Converting the call
    is cheaper than spending a step on an ``unknown tool`` observation.
    """
    name = str(call.tool or "").strip().lower()
    if name == "coarse":
        args = call.args or {}
        query = str(args.get("query") or args.get("q") or "").strip()
        if not query:
            return call
        command = f"coarse --query {shlex.quote(query)}"
        if args.get("top") is not None:
            try:
                command += f" --top {int(args['top'])}"
            except (TypeError, ValueError):
                pass
        if args.get("mode"):
            command += f" --mode {shlex.quote(str(args['mode']))}"
        for key, flag in (("date_from", "--date-from"), ("date_to", "--date-to")):
            if args.get(key):
                command += f" {flag} {shlex.quote(str(args[key]))}"
        return ToolCall("bash", {"command": command}, id=call.id)
    if name == "grep":
        args = call.args or {}
        pattern = str(args.get("pattern") or args.get("regex")
                      or args.get("query") or args.get("term") or "").strip()
        if not pattern:
            return call
        path = str(args.get("path") or "inputs/memory.jsonl")
        command = f"grep -in {shlex.quote(pattern)} {shlex.quote(path)}"
        return ToolCall("bash", {"command": command}, id=call.id)
    return call


def _grep_cli_pattern(command: str) -> str:
    """Extract the first quoted pattern from a real ``grep`` command."""
    match = re.search(r"\bgrep\b(?:\s+-[A-Za-z]+)*\s+(['\"])(.*?)\1", command or "")
    if not match:
        return ""
    # Basic-regex alternation uses ``\|``; Python's re wants ``|``.
    return match.group(2).replace("\\|", "|").strip()


def _append_matches(workspace: Path, evidence_path: Path, pattern: str,
                    limit: int, *, ids: bool) -> int:
    """Run ``append_matches`` over the run's memory snapshot."""
    memory = workspace / "inputs" / "memory.jsonl"
    if not pattern or not memory.exists():
        return 0
    before = evidence.count_lines(evidence_path)
    argv = ["--memory", str(memory), "--out", str(evidence_path),
            "--limit", str(max(1, int(limit)))]
    if ids:
        argv.append("--ids")
    argv.extend(["--", pattern])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = append_matches.main(argv)
    except (re.error, OSError, ValueError):
        return 0
    if code != 0:
        return 0
    return max(0, evidence.count_lines(evidence_path) - before)


def _capture_grep_hits(workspace: Path, evidence_path: Path, pattern: str,
                       limit: int) -> int:
    """Append the records a grep just surfaced, if the agent did not.

    Observed failure mode: the postagent runs
    ``grep -in 'Sweden' inputs/memory.jsonl``, the observation contains the answer,
    but the model never appends it and finishes; the answer model then sees
    only the coarse seeds.  Capturing the hit deterministically keeps the
    search from being wasted.  ``append_matches`` already dedupes by content
    and source, so a model that *did* append is unaffected.
    """
    return _append_matches(workspace, evidence_path, pattern, limit, ids=False)


_HIT_ID_RE = re.compile(r"(?m)^#\d+\s+([A-Za-z0-9_.:-]+)")


def _capture_query_hits(workspace: Path, evidence_path: Path,
                        observation: str, limit: int) -> int:
    """Append the top records a rewritten ``--query`` search returned.

    Same rationale as :func:`_capture_grep_hits`, for the other search verb:
    a query rewrite can surface a record the shortlist missed (e.g. the
    "home country, Sweden" record when the question asks where Caroline
    moved from) while the model keeps re-posting its own paraphrase.  The
    observation starts every hit with ``#<rank> <record id>``, so the ids are
    unambiguous.  Only the first ``limit`` hits are captured.
    """
    ids: list[str] = []
    for match in _HIT_ID_RE.finditer(observation or ""):
        record_id = match.group(1).rstrip(":")
        if record_id and record_id not in ids:
            ids.append(record_id)
        if len(ids) >= max(1, int(limit)):
            break
    if not ids:
        return 0
    pattern = "|".join(re.escape(record_id) for record_id in ids)
    return _append_matches(workspace, evidence_path, pattern, limit, ids=True)


def _compact(messages: list[dict[str, Any]], max_chars: int, keep_recent: int) -> list[dict[str, Any]]:
    total = sum(len(str(item.get("content") or "")) for item in messages)
    if max_chars <= 0 or total <= max_chars or len(messages) <= keep_recent + 2:
        return messages
    head = messages[:2]
    tail = messages[-keep_recent:]
    dropped = len(messages) - 2 - keep_recent
    note = {"role": "user",
            "content": f"[{dropped} earlier steps were compacted. Use grep again if you need them.]"}
    return head + [note] + tail


async def run_postagent(
    *,
    workspace: Path,
    tools: dict[str, Any],
    call_model: ModelCall,
    schemas: list[dict[str, Any]] | None,
    config: dict[str, Any],
    hints: list[str] | None = None,
    audit: AuditCall | None = None,
) -> PostagentResult:
    evidence_path = workspace / EVIDENCE_NAME
    question = str(config.get("question") or "")
    system = prompts.system_prompt()
    user = prompts.user_prompt(
        question, config.get("options"),
        brief=str(config.get("brief") or ""),
        intent=str(config.get("intent") or ""))
    mode = str(config.get("claim_mode") or "")
    if mode:
        user += "\n\n" + prompts.claim_mode_prompt(mode)
    if hints:
        user += "\n\n" + prompts.hints_prompt(hints)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    max_steps = int(config.get("max_steps", 6))
    observation_limit = int(config.get("observation_max_chars", 8000))
    patience = int(config.get("no_progress_patience", 6))
    repeat_limit = int(config.get("repeat_limit", 1))
    nudge_after = int(config.get("write_nudge_after", 5))
    max_empty_refusals = int(config.get("max_empty_finish_refusals", 2))
    max_parse_retries = int(config.get("max_parse_retries", 2))
    context_max = int(config.get("context_max_chars", 120000))
    keep_recent = int(config.get("context_keep_recent", 8))
    agent_context = int(config.get("agent_max_context_tokens", 0) or 0)
    max_append = max(1, int(config.get("max_append_per_call", 8)))
    max_command = max(0, int(config.get("max_command_chars", 800)))
    audit_after = max(0, int(config.get("audit_after_step", 4)))
    max_model_errors = max(1, int(config.get("max_model_errors", 2)))
    auto_capture = bool(config.get("auto_capture_hits",
                                   config.get("auto_capture_grep", True)))
    auto_capture_limit = max(1, int(config.get("auto_capture_limit", 2)))
    auto_capture_total = max(0, int(config.get("auto_capture_total", 6)))
    captured_total = 0
    # Absolute ``time.monotonic()`` deadline for the whole HTTP search.  When
    # it passes we stop before spending another model call; the evidence
    # written so far is still returned.
    raw_deadline = config.get("deadline")
    deadline = float(raw_deadline) if raw_deadline else None

    seen_calls: dict[str, int] = {}
    last_fingerprint = ""
    last_observation = ""
    stale = 0
    empty_refusals = 0
    parse_retries = 0
    model_errors = 0
    audited = False
    nudged = False
    # Set when a mid-loop audit says the evidence is sufficient.  The model is
    # allowed at most one final check, then the loop stops; otherwise a small
    # model keeps searching until `steps_exhausted` even though it is done.
    force_finish = False
    result = PostagentResult()

    for step in range(1, max_steps + 1):
        result.steps = step
        if deadline is not None and time.monotonic() >= deadline:
            result.stop_reason = "deadline"
            result.step_log.append({"step": step, "tool": "deadline"})
            break
        clean = evidence.read_file(evidence_path)

        if not clean and nudge_after and step >= nudge_after and not nudged:
            nudged = True
            messages.append({"role": "user", "content":
                             "You have not written any evidence yet. Append what you have found "
                             "so far to evidence.jsonl before continuing."})

        messages = _compact(messages, context_max, keep_recent)
        if agent_context > 0:
            messages = llm.fit_messages(
                messages, agent_context, reserve_tokens=8192)
        try:
            reply = await call_model(messages, schemas)
        except Exception as exc:  # noqa: BLE001 - one bad reply must not kill the QA
            model_errors += 1
            truncated = isinstance(exc, TruncatedCompletion) or "truncat" in type(exc).__name__.lower()
            if truncated:
                note = (
                    "Your last reply was cut off because it was too long. Do not "
                    "enumerate synonyms or build a long filter. Reply with ONE short "
                    "bash command (<= 800 characters) or write evidence and finish."
                )
            else:
                note = (
                    f"The last model call failed ({type(exc).__name__}). Reply with one "
                    "short bash command, or reply with plain text to finish."
                )
            messages.append({"role": "user", "content": note})
            result.stop_reason = f"model_error:{type(exc).__name__}"
            result.step_log.append({"step": step, "tool": "model_error",
                                    "error": type(exc).__name__})
            if model_errors >= max_model_errors:
                break
            continue

        if not reply.calls:
            implicit = _implicit_evidence(reply.text)
            if implicit:
                added, total = _append_implicit(evidence_path, implicit)
                messages.append({"role": "assistant", "content": reply.text})
                if added:
                    note = (f"Appended {added} new evidence record(s) directly. "
                            f"Total: {total}. Continue searching, or reply with plain "
                            "text to finish.")
                else:
                    note = ("That evidence is already in evidence.jsonl. Do not repeat it. "
                            "Search for the missing slot with a NEW term, or reply with "
                            "plain text to finish.")
                messages.append({"role": "user", "content": note})
                result.evidence_lines = total
                result.step_log.append({
                    "step": step, "tool": "implicit_evidence",
                    "added": added, "evidence_lines": total})
                # Claims written without a tool call still need the mid-loop
                # audit; otherwise an agent that only appends evidence never
                # sees the verifier's missing slot.
                current = evidence.read_file(evidence_path)
                fingerprint_now = evidence.fingerprint(current)
                if added > 0 or fingerprint_now != last_fingerprint:
                    stale = 0
                else:
                    stale += 1
                last_fingerprint = fingerprint_now
                if (audit and not audited and audit_after and step >= audit_after
                        and step < max_steps and current):
                    audited = True
                    result.audited = True
                    try:
                        missing = await audit(current)
                    except Exception:  # noqa: BLE001
                        missing = []
                    result.audit_missing = list(missing)
                    result.audit_sufficient = not missing
                    if missing:
                        messages.append({"role": "user",
                                         "content": prompts.hints_prompt(missing)})
                    else:
                        messages.append({"role": "user",
                                         "content": prompts.sufficient_prompt()})
                        force_finish = True
                if force_finish:
                    result.evidence_lines = len(evidence.read_file(evidence_path))
                    result.audit_at_finish = True
                    result.audit_fingerprint = evidence.fingerprint(current)
                    result.stop_reason = "sufficient_after_check"
                    break
                if patience and stale >= patience:
                    result.stop_reason = f"no_progress({stale})"
                    break
                continue

        if not reply.calls:
            if parse_retries < max_parse_retries and toolcfg.looks_like_call(reply.text):
                parse_retries += 1
                messages.append({"role": "assistant", "content": reply.text or "{"})
                messages.append({"role": "user", "content":
                                 "That was not a valid tool call. Reply with one JSON object: "
                                 '{"tool":"bash","args":{"command":"..."}} or reply with plain text to finish.'})
                continue
            parse_retries = 0
            if not reply.done:
                messages.append({"role": "assistant", "content": reply.text})
                messages.append({"role": "user", "content":
                                 "Empty reply. Call a tool, or reply with plain text if you are done."})
                continue
            # Do not accept "done" before the evidence has been audited.  This
            # is what catches a yes/no answer that found only the negative
            # statement while the shortlist contains the positive one.
            if audit and not audited and clean:
                audited = True
                result.audited = True
                try:
                    missing = await audit(clean)
                except Exception:  # noqa: BLE001
                    missing = []
                result.audit_missing = list(missing)
                result.audit_sufficient = not missing
                if missing:
                    messages.append({"role": "assistant", "content": reply.text})
                    messages.append({"role": "user", "content": prompts.hints_prompt(missing)})
                    continue
                # The evidence will not change: the runner can reuse this
                # verdict instead of paying for a second verifier call.
                result.audit_at_finish = True
                result.audit_fingerprint = evidence.fingerprint(clean)
                messages.append({"role": "assistant", "content": reply.text})
                messages.append({"role": "user", "content": prompts.sufficient_prompt()})
            if not clean and empty_refusals < max_empty_refusals:
                empty_refusals += 1
                messages.append({"role": "assistant", "content": reply.text})
                messages.append({"role": "user", "content":
                                 "evidence.jsonl is still empty. Append at least the most relevant "
                                 "record you have already read, then continue."})
                continue
            result.stop_reason = "sufficient"
            result.final_text = reply.text
            result.step_log.append({"step": step, "tool": "finish",
                                    "text": reply.text[:500]})
            break

        messages.append(reply.assistant_entry or {"role": "assistant", "content": reply.text})
        observations: list[str] = []
        for call in reply.calls:
            call = _normalize_shell_call(call)
            key = call.key()
            seen_calls[key] = seen_calls.get(key, 0) + 1
            command = (str(call.args.get("command") or "")
                       if call.tool == "bash" else "")
            evidence_touched = _evidence_call(call)
            before_records = (evidence.read_file(evidence_path)
                              if evidence_touched else [])
            details: dict[str, Any] = {}
            if (call.tool == "bash" and "candidates.jsonl" in command and any(
                    tool in command for tool in ("jq", "cat", "sed", "head", "grep"))):
                observation = (
                    "The candidate shortlist is already summarized in the brief and "
                    "seeded into evidence.jsonl. Do not inspect inputs/candidates.jsonl; "
                    "read evidence.jsonl (`read` or `sed -n '1,8p' evidence.jsonl`) or "
                    "search inputs/memory.jsonl."
                )
                ok = False
            elif call.tool == "bash" and max_command and len(command) > max_command:
                observation = (
                    f"ERROR: the command is {len(command)} characters; the limit is "
                    f"{max_command}. Use `coarse ... --top 8` or "
                    "`python bin/append_matches.py 'term1|term2' --limit 5`, or a "
                    "short `write` call. Never enumerate synonyms."
                )
                ok = False
            elif seen_calls[key] > repeat_limit:
                observation = (
                    f"ERROR: this exact {call.tool} call was already run. It cannot change the "
                    "result. Try a different word, file slice, or command."
                )
                ok = False
            else:
                try:
                    tool = tools[call.tool]
                except KeyError:
                    observation, ok = f"ERROR: unknown tool {call.tool!r}", False
                else:
                    try:
                        tool_result = await tool.run(call.args)
                        details = dict(tool_result.details or {})
                        observation, ok = _observation(tool_result, observation_limit), tool_result.ok
                    except ToolError as exc:
                        observation, ok = f"ERROR: {exc}", False
                    except Exception as exc:  # a failed step must not kill the QA
                        observation, ok = f"ERROR: {type(exc).__name__}: {exc}", False
                if ("already in evidence.jsonl" in observation
                        or "already present" in observation):
                    observation += (
                        "\n\n[note: those records are already in the file. Read them "
                        "(`read` evidence.jsonl or `sed -n '1,10p' evidence.jsonl`) and "
                        "rewrite the key fact as a claim; do not repeat this search.]"
                    )
                if evidence_touched:
                    after_records = evidence.read_file(evidence_path)
                    after_count = len(after_records)
                    added = after_count - len(before_records)
                    if _append_call(call) and added > max_append:
                        # Reject an oversized append and let the model retry with
                        # a narrower term / smaller limit.  An explicit replace is
                        # a curation operation and is allowed to change many lines.
                        evidence.write_file(evidence_path, before_records)
                        observation += (
                            f"\n\n[note: that call appended {added} records; the limit per "
                            f"call is {max_append}. The append was rolled back. Use a more "
                            "specific term and `--limit 5`, or filter the output.]"
                        )
                        after_count = len(before_records)
                    elif added < 0:
                        # Deleting/seeded curation is allowed: the file the agent
                        # leaves is the final evidence.
                        observation += (
                            f"\n\n[note: removed {-added} record(s); evidence.jsonl now has "
                            f"{after_count}. Keep enough evidence to answer.]"
                        )
                    elif evidence.fingerprint(after_records) == evidence.fingerprint(before_records):
                        observation += (
                            "\n\n[note: evidence.jsonl did not change. Use `write` with "
                            "content + source (the harness wraps it), or use "
                            "`python bin/append_matches.py 'term1|term2' --limit 5`. "
                            "To curate, use `edit` or a short bash command.]"
                        )
                    if call.tool == "bash" and (
                            "grep -n" in command or "grep -in" in command) and "jq" in command:
                        observation += (
                            "\n\n[note: grep -n/-in prefixes line numbers; the `jq` pipeline "
                            "then sees invalid JSON. Drop -n when piping to jq.]"
                        )
                if (auto_capture and ok and call.tool == "bash"
                        and captured_total < auto_capture_total):
                    room = auto_capture_total - captured_total
                    limit = min(auto_capture_limit, room)
                    captured = 0
                    if _is_coarse_command(command) and "--query" in command and limit > 0:
                        captured = _capture_query_hits(
                            workspace, evidence_path, observation, limit)
                    elif ("grep" in command
                          and "inputs/memory.jsonl" in command
                          and "evidence.jsonl" not in command
                          and limit > 0):
                        pattern = _grep_cli_pattern(command)
                        if pattern:
                            captured = _capture_grep_hits(
                                workspace, evidence_path, pattern, limit)
                    if captured:
                        captured_total += captured
                        observation += (
                            f"\n\n[note: that search surfaced a record that was not in "
                            f"evidence.jsonl; appended {captured} raw record(s) so the hit "
                            "is not lost. Read it (`read` the file or `tail -n 3 "
                            "evidence.jsonl`) and keep it, or delete it if it is unrelated.]"
                        )
                if evidence_touched:
                    observation += "\n\n" + evidence.digest(evidence_path)
            observations.append(observation)
            log_command = (command[:800] if command
                           else json.dumps(call.args, ensure_ascii=False)[:800])
            result.step_log.append({
                "step": step,
                "tool": call.tool,
                "command": log_command,
                "ok": bool(ok),
                "observation": observation[:1000],
                "duration_seconds": details.get("duration_seconds"),
                "evidence_lines": evidence.count_lines(evidence_path),
            })
            if reply.assistant_entry and reply.assistant_entry.get("tool_calls"):
                messages.append({"role": "tool", "tool_call_id": call.id or key,
                                 "content": observation})
            else:
                messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        current = evidence.read_file(evidence_path)
        if (audit and not audited and audit_after and step >= audit_after
                and step < max_steps and current):
            audited = True
            result.audited = True
            try:
                missing = await audit(current)
            except Exception:  # noqa: BLE001 - an audit failure is not a hard stop
                missing = []
            result.audit_missing = list(missing)
            result.audit_sufficient = not missing
            if missing:
                messages.append({"role": "user", "content": prompts.hints_prompt(missing)})
            else:
                messages.append({"role": "user", "content": prompts.sufficient_prompt()})
                force_finish = True
        if force_finish:
            # The audit already passed.  The tool call above (if any) was the
            # one allowed final check; stop instead of consuming the rest of
            # the budget.
            result.evidence_lines = len(evidence.read_file(evidence_path))
            result.audit_at_finish = True
            result.audit_fingerprint = evidence.fingerprint(current)
            result.stop_reason = "sufficient_after_check"
            break
        fingerprint = evidence.fingerprint(current)
        observation_signature = "\n".join(observations)[:2000]
        if fingerprint != last_fingerprint or observation_signature != last_observation:
            stale, last_fingerprint, last_observation = 0, fingerprint, observation_signature
        else:
            stale += 1
        if patience and stale >= patience:
            result.stop_reason = f"no_progress({stale})"
            break
        result.evidence_lines = len(current)
    else:
        result.stop_reason = "steps_exhausted"

    result.evidence_lines = len(evidence.read_file(evidence_path))
    result.history = messages
    return result
