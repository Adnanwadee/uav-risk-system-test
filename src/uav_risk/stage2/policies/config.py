# src/uav_risk/stage2/policies/config.py

from pydantic import BaseModel

class SafetyThresholds(BaseModel):
    """
    Aviation Regulatory & Physics Thresholds (Configurable)
    Default values aligned with general EASA/FAA guidelines for specific category.
    """
    max_altitude_agl_m: float = 120.0        # 400ft absolute legal ceiling
    critical_jamming_dbm: float = -65.0      # Severe interference threshold
    min_power_density_wkg: float = 150.0     # Minimum Watts per kg for safe maneuverability
    bvlos_max_wind_mps: float = 8.0          # Stricter wind limit for Beyond Visual Line of Sight
    dense_pop_max_alt_m: float = 10.0        # Max altitude over dense populations without specific permits

# Global instance to be used by the engine
THRESHOLDS = SafetyThresholds()