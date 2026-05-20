"""
Core Pipeline Tests (Gates 0, 1, 2, 3)
Comprehensive test suite validating the UAV Risk Assessment data pipeline.
"""

import pytest
import math
import numpy as np
import logging
import os
import json
from typing import Dict

# ==========================================
# Imports from our core modules
# ==========================================
from uav_risk.ml.feature_defs import (
    get_all_feature_names, 
    get_core_features, 
    get_feature_definition, 
    get_safe_value
)
from uav_risk.core.contracts import MasterFlightPayload, UAVSpecs
from uav_risk.core.data_validator import DataValidator
from uav_risk.core.imputation_strategy import ImputationStrategy
from uav_risk.core.feature_router import FeatureRouter
from uav_risk.verify_environment import run_all_checks

# ==========================================
# Fixtures (Setup before tests)
# ==========================================

@pytest.fixture
def validator():
    val = DataValidator()
    val.imputation_strategy = ImputationStrategy()
    return val

@pytest.fixture
def feature_mapping():
    mapping_path = "artifacts/stage1_feature_mapping.json"
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {str(i): f"feature_{i}" for i in range(198)}

@pytest.fixture
def router(feature_mapping):
    return FeatureRouter(feature_defs={}, feature_mapping=feature_mapping)

# ==========================================
# GATE 0: Environment & Readiness
# ==========================================
def test_gate0_environment_checks():
    result = run_all_checks()
    assert isinstance(result, bool)

# ==========================================
# GATE 1: Feature Definitions & Contracts
# ==========================================
def test_feature_definitions_integrity():
    all_names = get_all_feature_names()
    core_names = get_core_features()
    
    assert len(all_names) == 198, "Must be exactly 198 features"
    assert len(core_names) == 40, "Must be exactly 40 core features"
    assert all_names == get_all_feature_names(), "Feature list must be deterministic"
    
    defn = get_feature_definition("uav_mass_kg")
    assert defn is not None
    # [تعديل 1] تم تغيير type إلى unit لتطابق الدستور
    assert "unit" in defn 
    assert get_safe_value("uav_mass_kg") == 5.0

def test_contracts_flexible_parsing():
    payload = MasterFlightPayload(uav=UAVSpecs(mass_kg="N/A", max_speed_mps=" 15.5 "))
    assert payload.uav.mass_kg is None
    assert payload.uav.max_speed_mps == 15.5

def test_contracts_flattening():
    payload = MasterFlightPayload(uav=UAVSpecs(mass_kg=5.0))
    flat = payload.flatten_for_ml()
    assert "uav_mass_kg" in flat
    assert "uav_uav_mass_kg" not in flat
    assert flat["uav_mass_kg"] == 5.0

# ==========================================
# GATE 2: Data Validation & Imputation
# ==========================================
def test_validator_empty_input(validator):
    result = validator.validate_and_store({})
    assert len(result.validated_features) == 198
    assert result.has_critical_missing is True
    assert result.is_usable is False

def test_validator_out_of_range_clipping(validator):
    result = validator.validate_and_store({"uav_mass_kg": -10.0})
    val = result.validated_features["uav_mass_kg"]
    assert val >= 0.0
    record = next(r for r in result.validation_records if r.feature_name == "uav_mass_kg")
    assert record.was_out_of_range is True
    assert record.status == "CORRECTED"

def test_validator_nan_inf_handling(validator):
    result = validator.validate_and_store({
        "uav_mass_kg": float('nan'),
        "environment_weather_wind_mps": float('inf')
    })
    assert not any(math.isnan(v) for v in result.validated_features.values())
    assert not any(math.isinf(v) for v in result.validated_features.values())
    
def test_validator_physics_derivation(validator):
    result = validator.validate_and_store({
        "uav_battery_capacity_mah": 5000.0,
        "uav_battery_voltage_v": 20.0
    })
    assert result.validated_features["uav_battery_wh"] == 100.0

def test_validator_logs_changes(validator, caplog):
    with caplog.at_level(logging.INFO):
        validator.validate_and_store({})
        assert "Validation Complete" in caplog.text

# ==========================================
# GATE 3: Feature Routing & Vectorization
# ==========================================
def test_router_vector_shape_and_validity(validator, router):
    result = validator.validate_and_store({"uav_mass_kg": 7.5})
    vector = router.route_to_vector(result.validated_features)
    
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (198,)
    assert vector.dtype == np.float64
    assert not np.any(np.isnan(vector))
    assert not np.any(np.isinf(vector))
    
    is_valid, issues = router.validate_vector(vector)
    assert is_valid is True

def test_router_string_indices_protection(validator, feature_mapping):
    router = FeatureRouter(feature_defs={}, feature_mapping=feature_mapping)
    result = validator.validate_and_store({"uav_mass_kg": 5.0})
    vector = router.route_to_vector(result.validated_features)
    
    # [تعديل 2] سؤال الـ Router مباشرة عن مكان الميزة لتفادي أخطاء قراءة الـ JSON
    expected_index = router._index_map["uav_mass_kg"]
    assert vector[expected_index] == 5.0

def test_router_context_pool(validator, router):
    result = validator.validate_and_store({"uav_mass_kg": 5.0, "environment_weather_wind_mps": 3.0})
    pool = router.route_to_context_pool(result.validated_features)
    
    assert "aerodynamic" in pool
    assert "environmental" in pool
    assert pool["environmental"]["environment_weather_wind_mps"] == 3.0