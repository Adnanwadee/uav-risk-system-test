import sys
import json
import structlog
from typing import Any

# إضافة مسار المشروع لضمان الاستيراد السليم
sys.path.insert(0, "./src")
sys.path.insert(0, ".")

from uav_risk.core.contracts import MasterFlightPayload

logger = structlog.get_logger()

def inspect_pydantic_schema():
    print("\n====== 🔍 ACE System Contracts Introspection 🔍 ======\n")
    
    # 1. فحص حقول كائن MasterFlightPayload الرئيسي
    if hasattr(MasterFlightPayload, "model_fields"):  # Pydantic V2
        fields = MasterFlightPayload.model_fields
        for name, field_info in fields.items():
            print(f"📦 Block Name: '{name}'")
            # فحص إذا كان الحقل يعتمد على Sub-Model متداخل
            sub_model = field_info.annotation
            if hasattr(sub_model, "model_fields"):
                print(f"   └── ├─ Type: Nested Model ({sub_model.__name__})")
                print(f"   └── └─ Expected Sub-Fields:")
                for sub_name in sub_model.model_fields.keys():
                    print(f"          ├── {sub_name}")
            else:
                print(f"   └── └─ Type: {sub_model}")
            print("-" * 50)
            
    elif hasattr(MasterFlightPayload, "__fields__"):  # Pydantic V1
        fields = MasterFlightPayload.__fields__
        for name, field_obj in fields.items():
            print(f"📦 Block Name: '{name}'")
            sub_model = field_obj.type_
            if hasattr(sub_model, "__fields__"):
                print(f"   └── ├─ Type: Nested Model ({sub_model.__name__})")
                print(f"   └── └─ Expected Sub-Fields:")
                for sub_name in sub_model.__fields__.keys():
                    print(f"          ├── {sub_name}")
            else:
                print(f"   └── └─ Type: {sub_model}")
            print("-" * 50)
            
if __name__ == "__main__":
    inspect_pydantic_schema()