from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re


@dataclass
class DocChunk:
    evidence_id: str       # unique id
    source: str            # relative file path under knowledge/
    title: str             # file title
    anchor: str            # markdown heading anchor (or "root")
    citation: str          # source#anchor
    text: str              # chunk text


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s\-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s or "section"


def _split_by_headings(md: str) -> List[tuple[str, str]]:
    md = md.strip()
    if not md:
        return []

    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [("root", md)]

    sections: List[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        heading_text = m.group(2)
        anchor = _slugify(heading_text)
        section_text = md[start:end].strip()
        sections.append((anchor, section_text))
    return sections


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = text.strip()
    if not text:
        return []
    out: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_size)
        chunk = text[i:j].strip()
        if chunk:
            out.append(chunk)
        if j >= len(text):
            break
        i = max(i + chunk_size - overlap, j)
    return out


def load_knowledge_chunks(
    knowledge_dir: str,
    allow_prefixes: Optional[List[str]] = None,
) -> List[DocChunk]:
    base = Path(knowledge_dir)
    if not base.exists():
        return []

    chunks: List[DocChunk] = []
    md_files = sorted(base.rglob("*.md"))

    for fp in md_files:
        rel = fp.relative_to(base).as_posix()

        if allow_prefixes:
            if not any(rel.startswith(pfx) for pfx in allow_prefixes):
                continue

        title = fp.stem.replace("_", " ")
        raw = fp.read_text(encoding="utf-8", errors="ignore")

        for anchor, section_text in _split_by_headings(raw):
            parts = _chunk_text(section_text)

            for k, part in enumerate(parts):
                evidence_id = f"{rel}#{anchor}::chunk{k:03d}"
                citation = f"{rel}#{anchor}"
                chunks.append(
                    DocChunk(
                        evidence_id=evidence_id,
                        source=rel,
                        title=title,
                        anchor=anchor,
                        citation=citation,
                        text=part,
                    )
                )
    return chunks
