"""Small, dependency-free tokenizer for BM25 and query analysis.

The design is deliberately language-agnostic enough for the datasets ASMEM
sees (English dialogue, Chinese notes, mixed tables):

- ASCII words: lowercase, stopword filter, light suffix stemming;
- CJK runs: unigrams + adjacent bigrams (no jieba dependency);
- punctuation/numbers are kept as tokens when they carry meaning.

It is not a linguistic tokenizer.  It is a deterministic indexer that makes
the same text produce the same terms on every machine.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

_WORD_OR_CJK = re.compile(
    r"[A-Za-z0-9_]+|[㐀-䶿一-鿿豈-﫿]+"
)
_QUOTED = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,80})[\"“”'‘’]")
_CAPITAL = re.compile(r"\b(?:[A-Z][A-Za-z0-9_.\-]*)(?:\s+[A-Z][A-Za-z0-9_.\-]*){1,4}\b")
_NUMBER = re.compile(r"\b\d+(?:[.,:/\-]\d+)*\b")
_ID = re.compile(r"\b[A-Za-z]+[-_]?\d+(?:[-_.]\d+)*\b")

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "by", "for", "with", "about", "as", "from", "into",
    "over", "under", "again", "further", "once", "here", "there", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "should", "now", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "i", "me", "my", "we", "our", "you", "your", "he",
    "him", "his", "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "whose", "when", "where", "why", "how", "please", "would", "could",
    "may", "might", "must", "shall",
})

# A deliberately small Porter-like stemmer.  It covers the common English
# plurals/gerunds in dialogue memory and, importantly, restores the silent
# "e" for verbs like moved -> move (otherwise "move" and "moved" never match).
_VOWELS = set("aeiou")


def _post_stem(base: str) -> str:
    if base.endswith(("at", "bl", "iz")):
        return base + "e"
    if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in "lsz":
        return base[:-1]
    if (len(base) >= 3 and base[-3] not in _VOWELS
            and base[-2] in _VOWELS and base[-1] not in "aeiouwxy"):
        return base + "e"
    return base


@lru_cache(maxsize=1 << 16)
def _stem_cached(word: str) -> str:
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) > 5:
        base = word[:-3]
        if any(ch in _VOWELS for ch in base):
            return _post_stem(base)
    if word.endswith("ed") and len(word) > 4:
        base = word[:-2]
        if any(ch in _VOWELS for ch in base):
            return _post_stem(base)
    if word.endswith("ly") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _stem(word: str) -> str:
    # Short and non-ASCII tokens skip the suffix rules; caching those would
    # only waste memory (CJK unigrams/bigrams are unbounded).  Repeating
    # vocabulary in a large memory is what the cache is for.
    if len(word) <= 3 or not word.isascii():
        return word
    return _stem_cached(word)


def tokenize(text: str) -> list[str]:
    """Return deterministic terms; duplicates are preserved for BM25 tf."""
    out: list[str] = []
    # findall() returns the matched strings directly; finditer() + Match.group()
    # allocates a Match object per token, which dominates on the hot path.
    for token in _WORD_OR_CJK.findall(text or ""):
        # _WORD_OR_CJK only matches ASCII word runs or CJK runs, so a
        # non-ASCII token is exactly a CJK run.  isascii() is a C-level check
        # and avoids a regex match per token on this hot path.
        if not token.isascii():
            chars = list(token)
            out.extend(chars)
            out.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
            continue
        lowered = token.lower()
        if lowered in STOPWORDS:
            continue
        out.append(_stem(lowered))
    return out


def terms(text: str) -> set[str]:
    return set(tokenize(text))


def raw_terms(text: str) -> list[str]:
    """Surface terms for grep hints (no stemming, no stopword removal)."""
    found: list[str] = []
    for match in _WORD_OR_CJK.finditer(text or ""):
        token = match.group(0).strip()
        if token and token not in found:
            found.append(token)
    return found


def extract_phrases(text: str) -> list[str]:
    """Quoted phrases, capitalized names, numbers, and ids."""
    found: list[str] = []
    for regex in (_QUOTED, _CAPITAL, _NUMBER, _ID):
        for match in regex.finditer(text or ""):
            value = (match.group(1) if match.groups() else match.group(0)).strip()
            if len(value) < 2:
                continue
            if value.lower() in STOPWORDS:
                continue
            if value not in found:
                found.append(value)
    return found


def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    # Already-materialized sets (the common case in the dedup loop) are used
    # as-is; set operations do not mutate their operands.
    a = left if isinstance(left, (set, frozenset)) else set(left)
    b = right if isinstance(right, (set, frozenset)) else set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
