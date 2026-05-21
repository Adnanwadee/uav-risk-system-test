"""
Module: src/uav_risk/stage2/rag/schemas.py
Author: Elite Technical Partner
Description: Encapsulates strict data contracts and structures for the Legislative RAG subsystem.
"""

from dataclasses import dataclass, field
import hashlib

@dataclass(frozen=True)
class RetrievedChunk:
    """Represents a unique, atomic slice of regulatory document retrieved from Vector DB."""
    content: str
    source_file: str
    page_number: int
    relevance_score: float
    reranker_score: float = 0.0
    chunk_id: str = field(init=False)

    def __post_init__(self):
        # توليد معرف فريد وثابت بناءً على محتوى النص والمصدر لمنع التكرار
        hash_input = f"{self.content[:100]}_{self.source_file}_{self.page_number}".encode('utf-8')
        unique_id = hashlib.sha256(hash_input).hexdigest()
        # تعيين القيمة لحقل frozen عبر الأب البديل لمنع كسر الحصانة
        object.__setattr__(self, 'chunk_id', unique_id)

    def to_citation_text(self) -> str:
        """Formats the chunk into a standardized legislative citation text block."""
        return f"Source: {self.source_file}, Page {self.page_number}\nContent:\n{self.content.strip()}"


@dataclass(frozen=True)
class LegalCitation:
    """Represents a clean, validated reference structure used for the final compliance report."""
    source_file: str
    page_number: int
    full_text: str


@dataclass(frozen=True)
class LegalAnswer:
    """The final legal context structure returned by the RAG core engine to the Agent layer."""
    query: str
    answer: str
    citations: list[LegalCitation] = field(default_factory=list)
    confidence_score: float = 0.0
    rag_available: bool = True

# =====================================================================
# Stage 2 Architectural Dependency Comment Block:
# This file defines core data contracts.
# Dependencies: None (Leaf Node Contract)
# Dependent Files: src/uav_risk/stage2/rag/enhanced_retriever.py, rag_core.py
# =====================================================================