# UAV Risk Stage-1 ML Pipeline - Delivery Package

## What is this?

Stage-1 ML model for UAV risk classification (High/Medium/Low Risk).
Trained on 57,248 UAVBench scenarios. Uses LightGBM with 198 features.

## Quick Start

```bash
pip install -r stage1_requirements.txt
```

```python
from stage1_inference import RiskPredictor
predictor = RiskPredictor('stage1_production_bundle.pkl')
result = predictor.predict('scenario.json')
print(result['risk_category'])
```

## Files

| File | Purpose |
| ------ | ------- |
| stage1_production_bundle.pkl | Model + Preprocessor + Encoder + SHAP |
| stage1_inference.py | Standalone inference script |
| stage1_requirements.txt | Python dependencies |
| uav_stage1_clean.parquet | Clean data (57,248 x 219) |
| processed_splits_final.npz | Train/Val/Test splits |
| model_card.json | Performance metrics |

## Model Performance

| Metric | Value |
| ------ | ------- |
| Test Accuracy | 94.22% |
| Test Macro F1 | 89.69% |
| Balanced Accuracy | 92.15% |
| Aviation Safety | 99.75% |
| Fatal Error Rate | 0.047% |

## Requirements

Python 3.9+, lightgbm, scikit-learn, shap, joblib
See stage1_requirements.txt for full list.
