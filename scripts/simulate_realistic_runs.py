"""Legacy/demo/manual-only harness.

This script is not canonical runtime and should not be used as final readiness evidence.
Canonical readiness commands are listed in README.md.

Simple harness to simulate realistic API runs against the FastAPI app.
Uses TestClient to call /api/flight/assemble and /v2/evaluate.
"""
from pathlib import Path
import json
from fastapi.testclient import TestClient

# import the app
from uav_risk.api.main import app


# Load authoritative feature names from the bundle
import joblib
bundle = joblib.load(Path('artifacts/stage1_production_bundle.pkl'))
feature_names = list(bundle['feature_names'])

# helper to build payload with 40 core features (from feature_defs.get_core_features)
from uav_risk.ml.feature_defs import get_core_features, get_safe_value
core = get_core_features()
from uav_risk.ml.feature_defs import get_feature_definition

# Example 1: proper payload with all 40 cores provided
payload1 = {
    "flight_id": "sim-001",
    "profile": {"drone_id": "drone-sim-01", "operator_id": "op-1"},
    "uav_model_id": "sim-model-01",
    "uav_model_spec": {"manufacturer": "SimCo", "model": "S-1"},
    "features": {name: get_safe_value(name) for name in feature_names}
}
# force user-supplied core features to mid-safe-range (golden values)
for name in core:
    fd = get_feature_definition(name) or {}
    safe_min = fd.get('safe_min')
    safe_max = fd.get('safe_max')
    if safe_min is not None and safe_max is not None:
        try:
            payload1['features'][name] = float((safe_min + safe_max) / 2.0)
        except Exception:
            payload1['features'][name] = get_safe_value(name)
    else:
        payload1['features'][name] = get_safe_value(name)

# Example 2: missing some core features to trigger veto
payload2 = {
    "flight_id": "sim-002",
    "profile": {"drone_id": "drone-sim-02", "operator_id": "op-2"},
    "uav_model_id": "sim-model-02",
    "uav_model_spec": {"manufacturer": "SimCo", "model": "S-2"},
    "features": {name: get_safe_value(name) for name in feature_names}
}
# remove half of core features to simulate user omission
for name in core[:20]:
    payload2['features'].pop(name, None)


with TestClient(app) as client:
    def run_assemble(payload):
        r = client.post('/api/flight/assemble', json=payload)
        try:
            body = r.json()
        except Exception:
            body = r.text
        print('assemble', r.status_code, body)
        return r

    def run_evaluate(payload):
        r = client.post('/v2/evaluate', json=payload)
        try:
            body = r.json()
        except Exception:
            body = r.text
        print('evaluate', r.status_code, body)
        return r

    if __name__ == '__main__':
        print('Running simulation 1: complete payload')
        flat1 = dict(payload1)
        features1 = flat1.pop('features')
        flat1.update(features1)
        run_assemble({"payload": flat1})
        try:
            run_evaluate(flat1)
        except Exception:
            print('evaluate skipped due to payload shape or missing model')

        print('\nRunning simulation 2: missing cores (expect veto/failure)')
        flat2 = dict(payload2)
        features2 = flat2.pop('features')
        flat2.update(features2)
        run_assemble({"payload": flat2})
        try:
            run_evaluate(flat2)
        except Exception:
            print('evaluate skipped due to payload shape or missing model')
