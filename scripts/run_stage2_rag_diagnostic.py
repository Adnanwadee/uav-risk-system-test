from __future__ import annotations

import argparse
import asyncio
import json
import sys

from uav_risk.stage2.rag.runtime_diagnostics import run_rag_runtime_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage2 RAG runtime diagnostics.")
    parser.add_argument(
        "--run-quality",
        action="store_true",
        help="Run opt-in retrieval quality checks using runtime resources.",
    )
    args = parser.parse_args()

    try:
        result = asyncio.run(run_rag_runtime_diagnostic(run_quality=args.run_quality))
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        return 0
    except Exception:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": "stage2_rag_diagnostic_script_failed",
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
