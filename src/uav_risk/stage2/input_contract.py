"""
API Input Contract (V13.0 - Resilient Data Ingestion)
========================================================
Fixes applied:
1. Pydantic Expansion: Added ConfigDict(extra="allow") to allow 50+ ML dataset columns.
2. Tier-0 Dict Export: Now safely exports the FULL dictionary for the AI agents.
3. Pure Functions: Flattening logic refactored.

Author: Stage 2 — ACE System
"""

import math
import logging
from pydantic import BaseModel, Field, ConfigDict, BeforeValidator
from typing import Optional, List, Dict, Any
from typing_extensions import Annotated

logger = logging.getLogger("InputContract")

# ─── 1. The Auditable Validator Hook ───
def flexible_float(v: Any) -> Optional[float]:
    """يحول الأرقام بأمان، ولا ينهار إذا كانت القيمة مفقودة (للتعامل مع NaN في الداتا)."""
    if v is None or v == "": 
        return None
    try:
        val = float(v)
        if math.isfinite(val):
            return val
        else:
            logger.debug(f"Audit: Rejected non-finite value '{v}' in input contract.")
            return None
    except (ValueError, TypeError):
        logger.debug(f"Audit: Failed to parse '{v}' as float. Passed as None to Stage 1.")
        return None

# ─── 2. Base Models (Configured to accept extra Dataset columns) ───
class UAVPayload(BaseModel):
    model_config = ConfigDict(extra="allow") # [إصلاح]: السماح بكل الأعمدة الإضافية
    mass_kg: Optional[float] = None
    max_speed_mps: Optional[float] = None
    max_thrust_n: Optional[float] = None
    type: Optional[str] = "quadrotor"

class MasterFlightPayload(BaseModel):
    """
    العقد الرئيسي لاستقبال الطلب. مجهز لاستيعاب كل بيانات UAV_Dataset_v2
    """
    model_config = ConfigDict(extra="allow") # [إصلاح]: استيعاب جميع بيانات الرحلة
    
    uav: UAVPayload = Field(default_factory=UAVPayload)
    # يمكنك إضافة بقية النماذج الفرعية مثل environment, comms إذا كانت موجودة في نسختك
    
    def flatten_for_ml(self) -> Dict[str, Any]:
        """
        تسطيح دلالي ونقي (Pure Semantic Flattening).
        """
        dump = self.model_dump(exclude_none=False) # نحافظ على القيم المفقودة من أجل is_missing_
        
        def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
            items = []
            for k, v in d.items():
                if isinstance(v, list) or k == "sensors":
                    continue # نتخطى القوائم الخام لكي لا نكسر XGBoost
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        return flatten_dict(dump)

    def to_tier0_dict(self) -> Dict[str, Any]:
        """
        [إصلاح هندسي]: تمرير كل البيانات المتاحة بدلاً من فلترتها، لكي تستفيد منها البوابة والوكلاء.
        """
        flat_data = self.flatten_for_ml()
        
        # التأكد من وجود المتغيرات الأساسية بأسماء معيارية للطبقات الأدنى
        if "altitude_m" not in flat_data:
            flat_data["altitude_m"] = flat_data.get("airspace.altitude_agl_max_m", None)
            
        return flat_data