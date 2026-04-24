# src/uav_risk/stage1/utils.py
from __future__ import annotations
import numpy as np

def to_string_safe(x):
    try:
        return x.astype(str)
    except Exception:
        return x

def clip_value(val: float, min_val: float, max_val: float) -> float:
    """يمنع القيم المتطرفة من إفساد الحسابات الإحصائية للنموذج."""
    return float(np.clip(val, min_val, max_val))

def calc_power_to_weight(hover_power: float | None, mass: float | None) -> float:
    if hover_power is None or mass is None or mass <= 0:
        return 0.0
    # قص النسبة لتجنب القيم الفيزيائية المستحيلة (مثلاً محرك نفاث على كرتونة)
    ratio = float(hover_power / mass)
    return clip_value(ratio, 50.0, 800.0)

def calc_effective_wind_gust(wind: float | None, gust: float | None) -> float:
    w = clip_value(wind if wind is not None else 0.0, 0.0, 50.0)
    g = clip_value(gust if gust is not None else w * 1.2, 0.0, 70.0)
    return float(max(w, g))