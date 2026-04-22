#canonicalize.py
from typing import Dict, Any, List
import pandas as pd
import numpy as np


EXPECTED_COLUMNS = [
    # ================= RAW INPUTS =================
    "uav.mass_kg",
    "uav.max_speed_mps",
    "uav.battery_model.hover_power_W",
    "airspace.altitude_agl_max_m",

    "environment.weather.wind_mps",
    "environment.weather.gust_mps",
    "environment.weather.visibility",

    "environment.gnss_jam_dbm",
    "environment.gnss_multipath",
    "environment.em_interference",

    "mission.type",
    "mission.pattern",
    "mission.runway_required",

    "daa.sep_threshold_m",
    "daa.ttc_threshold_s",

    "comms.uplink_ok",
    "comms.downlink_ok",

    # ================= DERIVED =================
    "feat_mission_dist_m",
    "feat_mission_climb_m",
    "feat_mission_tortuosity",
    "feat_power_to_weight",
    "feat_weather_score",
    "feat_airspace_area_m2",
    "feat_obstacle_density_per_km2",
    "feat_obstacle_avg_speed",

    # ================= DATA QUALITY =================
    "dq_core_present_pct",
    "dq_weather_present",
    "dq_uav_present",
    "dq_comms_present",
    "dq_sensors_present_pct",
    "dq_mission_present",

    # ================= SENSORS =================
    "has_gnss",
    "has_imu",
    "has_lidar",
    "has_radar",
    "has_camera_rgb",
    "has_camera_thermal",
]



def canonicalize_scenario(scenario: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert raw scenario dict into a canonical DataFrame
    with all expected columns present.
    """

    row = {}

    # Fill known values
    for col in EXPECTED_COLUMNS:
        row[col] = scenario.get(col, np.nan)

    df = pd.DataFrame([row])

    # ----------------------------
    # Missingness flags
    # ----------------------------
    for col in EXPECTED_COLUMNS:
        df[f"is_missing__{col}"] = df[col].isna().astype(int)

    # ----------------------------
    # Data Quality scores (simple v1)
    # ----------------------------
    df["dq_uav_present"] = int(
        not pd.isna(df["uav.mass_kg"].iloc[0])
        and not pd.isna(df["uav.max_speed_mps"].iloc[0])
    )

    df["dq_weather_present"] = int(
        not pd.isna(df["environment.weather.wind_mps"].iloc[0])
    )

    df["dq_core_present_pct"] = (
        1.0 - df.filter(like="is_missing__").mean(axis=1)
    ).iloc[0]

    df["dq_mission_present"] = int(
        not pd.isna(df["mission.type"].iloc[0])
    )

    df["dq_comms_present"] = 0
    df["dq_sensors_present_pct"] = (
        1.0 - df.filter(like="is_missing__has_").mean(axis=1)
    ).iloc[0]

    return df
