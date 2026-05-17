#!/usr/bin/env python3
"""
Phase 0 Integration Smoke Test
Verifies: Contracts, Routing (198-dim), Imputation, Config, and Structlog.
Run from project root: python verify_phase0.py
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

# Add src/ to Python path dynamically
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

def run_verification() -> bool:
    print("🔍 بدء التحقق من مرحلة Phase 0...")
    print("-" * 50)

    try:
        # 1️⃣ اختبار السجلات (Structlog)
        from uav_risk.core.logging import setup_logging, get_logger, set_request_id
        setup_logging("INFO")
        logger = get_logger("phase0_verify")
        print("✅ structlog تم تهيئته بنجاح.")

        with set_request_id("verify-run-001"):
            logger.info("Phase0 verification started")

            # 2️⃣ اختبار الإعدادات (Config)
            from uav_risk.core.config import get_settings
            cfg = get_settings()
            assert abs(cfg.DECISION_WEIGHT_ML - 0.15) < 1e-9, "DECISION_WEIGHT_ML mismatch"
            assert cfg.RAG_MIN_CONFIDENCE == 0.55, "RAG_MIN_CONFIDENCE mismatch"
            print("✅ pydantic-settings تم تحميله من .env/defaults.")

            # 3️⃣ اختبار العقود (Contracts + Discriminator + Auto-Fallback)
            from uav_risk.core.contracts import FlightInput, Tier1Input, Tier4Input
            
            # أ) إنشاء Tier 1 صريح
            t1_payload = Tier1Input(speed=5.0, altitude=80.0, distance=300.0, flight_id="TEST-T1")
            t1 = FlightInput(payload=t1_payload)
            assert t1.tier == "1", "Tier 1 not detected"
            assert len(t1.validate_bounds()) == 0, "Unexpected bounds warning on valid T1"

            # ب) اختبار التوجيه التلقائي (Auto-Fallback) عبر Dictionary
            auto_input = FlightInput.model_validate({
                "payload": {
                    "speed": 28.0,      # >25 → pushes to Tier 4
                    "altitude": 150.0,  # >120 → pushes to Tier 4
                    "distance": 6000.0, # >5000 → pushes to Tier 4
                    "flight_id": "TEST-AUTO"
                }
            })
            assert auto_input.tier == "4", f"Auto-detect failed: expected 4, got {auto_input.tier}"
            print("✅ Contracts (Discriminator + Auto-Fallback) تعمل بدقة.")

            # 4️⃣ اختبار التوجيه (FeatureRouter → 198-dim)
            from uav_risk.core.feature_router import FeatureRouter
            router = FeatureRouter()
            ml_vector, context_pool = router.route_payload(auto_input)
            
            assert isinstance(ml_vector, list), "ml_vector is not a list"
            assert len(ml_vector) == 198, f"Expected 198 features, got {len(ml_vector)}"
            assert "flight_id" in context_pool, "context_pool missing flight_id"
            assert isinstance(context_pool.get("bounds_warnings"), list), "bounds_warnings malformed"
            
            nan_count = sum(1 for v in ml_vector if isinstance(v, float) and math.isnan(v))
            print(f"✅ FeatureRouter: متجه 198-بُعد تم إنشاؤه. القيم المفقودة (NaN): {nan_count}")

            # 5️⃣ اختبار الافتراضات (ImputationStrategy)
            from uav_risk.core.imputation_strategy import ImputationStrategy
            imputer = ImputationStrategy()
            final_vector, final_context = imputer.apply_imputation(ml_vector, context_pool)
            
            post_nan_count = sum(1 for v in final_vector if isinstance(v, float) and math.isnan(v))
            assert post_nan_count == 0, f"Imputation failed: {post_nan_count} NaNs remain"
            assert len(final_context["imputation_log"]) > 0, "Imputation log is empty"
            
            log_layers = set(entry["layer"] for entry in final_context["imputation_log"])
            print(f"✅ ImputationStrategy: تم ملء جميع القيم المفقودة. الطبقات المستخدمة: {log_layers}")
            print(f"   📝 عدد السجلات التوثيقية: {len(final_context['imputation_log'])}")

        logger.info("Phase0 verification PASSED")
        print("-" * 50)
        print("🎉 تم اجتياز جميع اختبارات Phase 0 بنجاح. النظام جاهز لـ Phase 1.")
        return True

    except ImportError as e:
        print(f"❌ خطأ في الاستيراد: {e}")
        print("💡 تأكد من تثبيت التبعيات: pip install structlog pydantic pydantic-settings")
        return False
    except AssertionError as e:
        print(f"❌ فشل التحقق المنطقي: {e}")
        return False
    except Exception as e:
        print(f"❌ خطأ غير متوقع: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)