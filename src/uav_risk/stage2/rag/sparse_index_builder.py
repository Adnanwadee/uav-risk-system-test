"""
Sparse Index Builder - Secure, Rebuildable, Corpus-aware
Builds sparse index (TF-IDF/BM25) from corpus with state tracking.
"""
import json
import pickle
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CorpusState:
    """Tracks corpus state for incremental updates"""
    total_documents: int = 0
    total_terms: int = 0
    avg_doc_length: float = 0.0
    last_updated: str = ""
    document_hashes: List[str] = None

    def __post_init__(self):
        if self.document_hashes is None:
            self.document_hashes = []

    def to_dict(self) -> Dict:
        return {
            "total_documents": self.total_documents,
            "total_terms": self.total_terms,
            "avg_doc_length": self.avg_doc_length,
            "last_updated": self.last_updated,
            "document_hashes": self.document_hashes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CorpusState":
        return cls(**data)

class SparseIndexBuilder:
    """
    Builds and maintains sparse retrieval indices (BM25 + TF-IDF).
    Rebuilds from corpus when state changes.
    """

    def __init__(self, corpus_dir: Path, output_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.corpus_state_path = self.output_dir / "corpus_state.json"
        self.sparse_index_path = self.output_dir / "sparse_index.pkl"

        self.corpus_state: Optional[CorpusState] = None
        self.sparse_index: Optional[Dict] = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple but effective tokenization"""
        import re
        # Lowercase, remove special chars, split
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = [t for t in text.split() if len(t) > 2]
        return tokens

    def _compute_hash(self, text: str) -> str:
        """Compute document hash for change detection"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _load_corpus(self) -> List[Tuple[str, str, str]]:
        """
        Load corpus documents.
        Returns: List of (doc_id, text, source)
        """
        documents = []

        if not self.corpus_dir.exists():
            logger.warning(f"Corpus directory not found: {self.corpus_dir}")
            return documents

        for file_path in self.corpus_dir.glob("**/*.txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                doc_id = file_path.stem
                source = str(file_path.relative_to(self.corpus_dir))
                documents.append((doc_id, text, source))
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        # Also support JSONL format
        for file_path in self.corpus_dir.glob("**/*.jsonl"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        data = json.loads(line)
                        text = data.get("text", "")
                        doc_id = data.get("id", f"{file_path.stem}_{i}")
                        source = str(file_path.relative_to(self.corpus_dir))
                        documents.append((doc_id, text, source))
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        return documents

    def _build_bm25(self, documents: List[Tuple[str, str, str]]) -> Dict:
        """
        Build BM25 index.
        Returns: {
            "doc_ids": [...],
            "doc_texts": [...],
            "doc_sources": [...],
            "term_doc_freq": {term: {doc_idx: freq}},
            "idf": {term: idf_score},
            "doc_lengths": [...],
            "avg_doc_length": float,
            "k1": 1.5,
            "b": 0.75
        }
        """
        N = len(documents)
        if N == 0:
            return {}

        doc_ids = []
        doc_texts = []
        doc_sources = []
        doc_tokens_list = []
        doc_lengths = []

        # Tokenize all documents
        for doc_id, text, source in documents:
            doc_ids.append(doc_id)
            doc_texts.append(text)
            doc_sources.append(source)
            tokens = self._tokenize(text)
            doc_tokens_list.append(tokens)
            doc_lengths.append(len(tokens))

        avg_doc_length = sum(doc_lengths) / N if N > 0 else 0

        # Build inverted index
        term_doc_freq = defaultdict(lambda: defaultdict(int))
        for doc_idx, tokens in enumerate(doc_tokens_list):
            for token in tokens:
                term_doc_freq[token][doc_idx] += 1

        # Compute IDF
        idf = {}
        for term, doc_freqs in term_doc_freq.items():
            df = len(doc_freqs)
            idf[term] = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

        sparse_index = {
            "doc_ids": doc_ids,
            "doc_texts": doc_texts,
            "doc_sources": doc_sources,
            "term_doc_freq": dict(term_doc_freq),
            "idf": idf,
            "doc_lengths": doc_lengths,
            "avg_doc_length": avg_doc_length,
            "k1": 1.5,
            "b": 0.75,
            "N": N
        }

        logger.info(f"Built BM25 index: {N} docs, {len(idf)} terms, avg_len={avg_doc_length:.1f}")
        return sparse_index

    def _save_index(self, index: Dict, state: CorpusState):
        """Save index and state atomically"""
        # Save to temp first
        temp_index = self.sparse_index_path.with_suffix(".tmp")
        temp_state = self.corpus_state_path.with_suffix(".tmp")

        with open(temp_index, "wb") as f:
            pickle.dump(index, f)

        with open(temp_state, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

        # Atomic rename
        temp_index.replace(self.sparse_index_path)
        temp_state.replace(self.corpus_state_path)

        logger.info(f"Saved sparse index to {self.sparse_index_path}")

    def _load_existing(self) -> Tuple[Optional[Dict], Optional[CorpusState]]:
        """Load existing index and state if valid"""
        if not self.sparse_index_path.exists() or not self.corpus_state_path.exists():
            return None, None

        try:
            with open(self.sparse_index_path, "rb") as f:
                index = pickle.load(f)

            with open(self.corpus_state_path, "r") as f:
                state = CorpusState.from_dict(json.load(f))

            return index, state
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            return None, None

    def _needs_rebuild(self, documents: List[Tuple[str, str, str]], 
                      state: CorpusState) -> bool:
        """Check if corpus changed and needs rebuild"""
        current_hashes = [self._compute_hash(text) for _, text, _ in documents]

        if len(current_hashes) != len(state.document_hashes):
            return True

        # Check for any new or changed documents
        existing_set = set(state.document_hashes)
        current_set = set(current_hashes)

        if existing_set != current_set:
            return True

        return False

    async def build_or_load(self, force_rebuild: bool = False) -> Dict:
        """
        Build index if needed, otherwise load existing.
        Thread-safe and async-friendly.
        """
        # Load existing
        existing_index, existing_state = self._load_existing()

        # Load current corpus
        documents = self._load_corpus()

        if not documents:
            logger.error("No documents found in corpus!")
            if existing_index:
                return existing_index
            return {}

        # Check if rebuild needed
        needs_rebuild = force_rebuild
        if existing_state and self._needs_rebuild(documents, existing_state):
            needs_rebuild = True
            logger.info("Corpus changed, rebuilding sparse index...")

        if not needs_rebuild and existing_index:
            logger.info("Using existing sparse index (corpus unchanged)")
            self.sparse_index = existing_index
            self.corpus_state = existing_state
            return existing_index

        # Build new index
        logger.info(f"Building sparse index from {len(documents)} documents...")

        # Run heavy computation in thread pool
        loop = asyncio.get_event_loop()
        index = await loop.run_in_executor(None, self._build_bm25, documents)

        # Update state
        current_hashes = [self._compute_hash(text) for _, text, _ in documents]
        state = CorpusState(
            total_documents=len(documents),
            total_terms=len(index.get("idf", {})),
            avg_doc_length=index.get("avg_doc_length", 0),
            last_updated=__import__("datetime").datetime.now().isoformat(),
            document_hashes=current_hashes
        )

        # Save
        await loop.run_in_executor(None, self._save_index, index, state)

        self.sparse_index = index
        self.corpus_state = state

        return index

    def get_index(self) -> Optional[Dict]:
        """Get current index (build if needed)"""
        if self.sparse_index is None:
            # Try to load existing
            existing_index, existing_state = self._load_existing()
            if existing_index:
                self.sparse_index = existing_index
                self.corpus_state = existing_state
        return self.sparse_index

    def get_stats(self) -> Dict:
        """Get index statistics"""
        if not self.corpus_state:
            return {"status": "not_built"}

        return {
            "status": "ready",
            "total_documents": self.corpus_state.total_documents,
            "total_terms": self.corpus_state.total_terms,
            "avg_doc_length": self.corpus_state.avg_doc_length,
            "last_updated": self.corpus_state.last_updated
        }