"""
Evidence Logger - Bounded Memory, Automatic Rotation
Prevents memory leaks with circular buffer and disk rotation.
"""
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class EvidenceEntry:
    """Single evidence log entry"""
    timestamp: str
    query: str
    scenario_type: str
    results: List[Dict]
    confidence_scores: List[float]
    retrieval_method: str  # "dense", "sparse", "hybrid", "hyde"
    latency_ms: float
    feature_count: int

    def to_dict(self) -> Dict:
        return asdict(self)

class BoundedEvidenceLog:
    """
    Evidence log with memory bounds and automatic rotation.
    Prevents memory leaks in long-running agents.
    """

    def __init__(self, 
                 max_entries: int = 1000,
                 rotation_size: int = 5000,
                 log_dir: Optional[Path] = None):
        self.max_entries = max_entries
        self.rotation_size = rotation_size
        self.log_dir = log_dir or Path("evidence_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer (circular)
        self._buffer: deque = deque(maxlen=max_entries)

        # Disk rotation tracking
        self._disk_count = 0
        self._current_disk_file: Optional[Path] = None

        # Async lock
        self._lock = asyncio.Lock()

        self._init_disk_log()

    def _init_disk_log(self):
        """Initialize current disk log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._current_disk_file = self.log_dir / f"evidence_{timestamp}.jsonl"
        self._disk_count = 0

    async def log(self, 
                 query: str,
                 scenario_type: str,
                 results: List[Dict],
                 confidence_scores: List[float],
                 retrieval_method: str,
                 latency_ms: float,
                 feature_count: int):
        """
        Log evidence with automatic rotation.
        """
        async with self._lock:
            entry = EvidenceEntry(
                timestamp=datetime.now().isoformat(),
                query=query,
                scenario_type=scenario_type,
                results=results,
                confidence_scores=confidence_scores,
                retrieval_method=retrieval_method,
                latency_ms=latency_ms,
                feature_count=feature_count
            )

            # Add to memory buffer
            self._buffer.append(entry)

            # Write to disk
            await self._write_to_disk(entry)

            # Check rotation
            if self._disk_count >= self.rotation_size:
                await self._rotate_disk_log()

    async def _write_to_disk(self, entry: EvidenceEntry):
        """Write entry to current disk file"""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._sync_write,
                entry
            )
            self._disk_count += 1
        except Exception as e:
            logger.warning(f"Failed to write evidence to disk: {e}")

    def _sync_write(self, entry: EvidenceEntry):
        """Synchronous disk write"""
        with open(self._current_disk_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    async def _rotate_disk_log(self):
        """Rotate to new disk file"""
        logger.info(f"Rotating evidence log after {self._disk_count} entries")
        self._init_disk_log()

    async def get_recent(self, count: int = 100) -> List[EvidenceEntry]:
        """Get recent entries from memory buffer"""
        async with self._lock:
            return list(self._buffer)[-count:]

    async def get_stats(self) -> Dict:
        """Get logging statistics"""
        async with self._lock:
            return {
                "memory_entries": len(self._buffer),
                "max_memory_entries": self.max_entries,
                "disk_entries_current_file": self._disk_count,
                "rotation_size": self.rotation_size,
                "current_file": str(self._current_disk_file)
            }

    async def search_history(self, query_pattern: str, 
                            scenario_type: Optional[str] = None,
                            limit: int = 50) -> List[EvidenceEntry]:
        """
        Search historical evidence.
        Only searches in-memory buffer for performance.
        """
        async with self._lock:
            results = []
            for entry in reversed(self._buffer):
                if query_pattern.lower() in entry.query.lower():
                    if scenario_type is None or entry.scenario_type == scenario_type:
                        results.append(entry)
                        if len(results) >= limit:
                            break
            return results

    async def clear_memory(self):
        """Clear memory buffer (disk logs preserved)"""
        async with self._lock:
            self._buffer.clear()
            logger.info("Evidence memory buffer cleared")

    async def export_to_file(self, output_path: Path, 
                            entries: Optional[List[EvidenceEntry]] = None):
        """Export entries to JSON file"""
        to_export = entries or list(self._buffer)

        data = {
            "exported_at": datetime.now().isoformat(),
            "entry_count": len(to_export),
            "entries": [e.to_dict() for e in to_export]
        }

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: Path(output_path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
        )