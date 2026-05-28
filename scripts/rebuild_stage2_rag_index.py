from __future__ import annotations

import argparse
import json
import sys

from uav_risk.stage2.rag.build_index import build_rag_index, repair_canonical_index_from_existing


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild canonical Stage2 RAG index.")
    parser.add_argument("--force", action="store_true", help="Overwrite canonical outputs if they exist.")
    parser.add_argument(
        "--repair-from-existing",
        action="store_true",
        help="Non-default compatibility mode: repair canonical outputs from existing legacy indices.",
    )
    args = parser.parse_args()

    try:
        if args.repair_from_existing:
            result = repair_canonical_index_from_existing(force=args.force)
        else:
            # Default: canonical rebuild from project-local docs into project-local vectdb.
            result = build_rag_index(force=args.force)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("status") == "success" else 1
    except Exception:
        print(json.dumps({"status": "failed", "error": "stage2_rag_rebuild_script_failed"}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
