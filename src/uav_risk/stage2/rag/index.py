from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import re

from rank_bm25 import BM25Okapi

from .loader import DocChunk


def _tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-\_\.]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


@dataclass
class RAGIndex:
    bm25: BM25Okapi
    chunks: List[DocChunk]
    tokenized: List[List[str]]

    @classmethod
    def build(cls, chunks: List[DocChunk]) -> "RAGIndex":
        tokenized = [_tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized) if chunks else BM25Okapi([["empty"]])
        return cls(bm25=bm25, chunks=chunks, tokenized=tokenized)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[Tuple[DocChunk, float]]:
        if not self.chunks:
            return []
        q = _tokenize(query)
        scores = self.bm25.get_scores(q)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: max(top_k * 2, top_k)]
        out: List[Tuple[DocChunk, float]] = []
        for idx, score in ranked:
            if score >= min_score:
                out.append((self.chunks[idx], float(score)))
            if len(out) >= top_k:
                break
        return out
