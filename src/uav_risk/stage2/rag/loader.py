from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict


@dataclass
class DocChunk:
    evidence_id: str
    source: str
    title: str
    text: str


def _chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunk = text[i:j].strip()
        if chunk:
            chunks.append(chunk)
        i = max(i + chunk_size - overlap, j)
    return chunks


def load_knowledge_chunks(knowledge_dir: str) -> List[DocChunk]:
    base = Path(knowledge_dir)
    if not base.exists():
        return []

    chunks: List[DocChunk] = []
    md_files = sorted(base.rglob("*.md"))

    for fp in md_files:
        rel = fp.relative_to(base).as_posix()
        title = fp.stem.replace("_", " ")
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        parts = _chunk_text(raw)

        for k, part in enumerate(parts):
            evidence_id = f"{rel}::chunk{k:03d}"
            chunks.append(
                DocChunk(
                    evidence_id=evidence_id,
                    source=rel,
                    title=title,
                    text=part,
                )
            )
    return chunks
