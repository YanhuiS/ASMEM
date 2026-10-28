"""Pure-Python BM25 over per-user records.

Statistics are always local to one user's memory.  The implementation keeps a
posting list (term -> [(doc_index, tf)]) so a query only touches documents that
contain at least one query term.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

from .tokenize import tokenize


class BM25Index:
    def __init__(self, texts: Sequence[str], *, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.n = len(texts)
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.df: Counter[str] = Counter()
        for index, text in enumerate(texts):
            counts = Counter(tokenize(text))
            length = sum(counts.values())
            self.doc_len.append(length)
            for term, tf in counts.items():
                self.postings[term].append((index, tf))
                self.df[term] += 1
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df <= 0:
            return 0.0
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def score(self, doc_index: int, query_terms: Iterable[str]) -> float:
        dl = self.doc_len[doc_index] if doc_index < len(self.doc_len) else 0
        norm = 1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0)
        total = 0.0
        for term in set(query_terms):
            idf = self.idf(term)
            if idf <= 0:
                continue
            tf = 0
            for index, count in self.postings.get(term, ()):
                if index == doc_index:
                    tf = count
                    break
            if tf:
                total += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        return total

    def search(self, query_terms: Sequence[str], *, limit: int | None = None
               ) -> list[tuple[int, float]]:
        """Return ``[(doc_index, score)]`` with positive scores, best first."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(query_terms):
            idf = self.idf(term)
            if idf <= 0:
                continue
            for index, tf in self.postings.get(term, ()):
                dl = self.doc_len[index]
                norm = 1.0 - self.b + self.b * (dl / self.avgdl if self.avgdl else 0.0)
                scores[index] += idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        ranked = sorted(
            ((index, score) for index, score in scores.items() if score > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:limit] if limit else ranked
