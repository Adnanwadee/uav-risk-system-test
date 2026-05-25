#!/usr/bin/env python3
"""Generate a worklist of features that need vetted SAFE values.

Output files:
 - artifacts/safe_values_worklist.csv
 - artifacts/safe_values_worklist.json

This lists all features, whether they're core, current SAFE_VALUES_REGISTRY value
if present, feature safe_min/safe_max and description so domain experts can fill
`recommended_safe_value` and `rationale` columns.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, Any

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uav_risk.ml import feature_defs


def main() -> None:
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_defs = feature_defs.get_all_feature_definitions()
    safe_registry = feature_defs.SAFE_VALUES_REGISTRY
    core_features = set(feature_defs.get_core_features())

    rows = []
    for name, defn in all_defs.items():
        is_core = name in core_features
        current_safe = safe_registry.get(name)
        row = {
            "feature_name": name,
            "is_core": bool(is_core),
            "current_safe_value": None if current_safe is None else float(current_safe),
            "safe_min": defn.get("safe_min"),
            "safe_max": defn.get("safe_max"),
            "description": defn.get("description", ""),
            "recommended_safe_value": "",  # to be filled by SME
            "rationale": "",
        }
        rows.append(row)

    csv_path = out_dir / "safe_values_worklist.csv"
    json_path = out_dir / "safe_values_worklist.json"

    # Write CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} feature rows to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
