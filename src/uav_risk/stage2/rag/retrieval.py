from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .index import RAGIndex
from .loader import load_knowledge_chunks, DocChunk


DEFAULT_SCOPES: Dict[str, List[str]] = {
    "weather": ["weather/", "regulations/"],
    "gnss": ["safety/", "regulations/", "company_sop/"],
    "sop": ["company_sop/", "regulations/"],
    "manual": ["uav_manual/"],
    "all": [],
}


@dataclass
class RAGStore:
    indexes: Dict[str, RAGIndex]

    @classmethod
    def build(cls, knowledge_dir: str) -> "RAGStore":
        indexes: Dict[str, RAGIndex] = {}
        for scope, prefixes in DEFAULT_SCOPES.items():
            allow = prefixes if prefixes else None
            chunks = load_knowledge_chunks(knowledge_dir, allow_prefixes=allow)
            indexes[scope] = RAGIndex.build(chunks)
        return cls(indexes=indexes)

    def retrieve(self, scope: str, query: str, top_k: int = 5, min_score: float = 0.0) -> List[DocChunk]:
        idx = self.indexes.get(scope) or self.indexes["all"]
        hits = idx.search(query=query, top_k=top_k, min_score=min_score)
        return [c for (c, _score) in hits]
