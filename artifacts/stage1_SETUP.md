
UAV Risk Stage-1 ML Pipeline - Setup Guide
===========================================

Quick Start:

  1. Install: pip install -r stage1_requirements.txt
  2. Load:    bundle = joblib.load('stage1_production_bundle.pkl')
  3. Predict: result = predictor.predict('scenario.json')

Files in this package:
  stage1_production_bundle.pkl   - Model + Preprocessor + SHAP
  stage1_requirements.txt        - Python dependencies
  stage1_feature_mapping.json    - 198 feature names
  stage1_inference_config.json   - API documentation
  stage1_inference.py            - Standalone inference script
  stage1_SETUP.md                - This guide
