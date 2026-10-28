"""ASMEM: a minimal code-agent memory system.

    add     data -> generalized memory.jsonl (format only, no model)
    search  question + memory.jsonl -> evidence.jsonl (code agent)
    api     /add and /search over a per-user memory directory

The old three-layer add (msgmem / summem / atommem) is intentionally gone.
"""

__all__ = ["adapters", "io", "llm", "log", "schema"]


def main() -> None:
    print("ASMEM\n  python -m asmem.add --input FILE --user USER\n"
          "  python -m asmem.search --memory FILE --query QUESTION\n"
          "  python -m asmem.api")
