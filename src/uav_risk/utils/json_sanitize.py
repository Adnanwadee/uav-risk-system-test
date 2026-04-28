"""
Aviation-Grade JSON Sanitizer (V14.0 - ACE Master Integration)
===========================================================
الدور: التطهير النهائي للبيانات قبل إرسالها للـ UI أو حفظها في قاعدة البيانات.
التحديثات لـ V14.0:
1. دعم NaN/Inf الشامل: التعامل مع sentinels المفقودة في الـ 50 عاموداً.
2. تطهير عودي عميق: حماية ضد الدوران اللا نهائي في حزم الأدلة الضخمة.
3. دعم ML & NumPy: تحويل أنواع بيانات الوكلاء (Physics/ML) إلى أنواع JSON قياسية.
"""

from __future__ import annotations
import math
import decimal
import datetime
import uuid
import logging
from typing import Any
import numpy as np

logger = logging.getLogger("DataSanitizer")

def sanitize_for_json(obj: Any, _seen: set[int] | None = None) -> Any:
    """
    تطهير البيانات بشكل عودي لضمان سلامة الـ JSON بنسبة 100% وفق معيار RFC-8259.
    يضمن عدم حدوث كراش (TypeError) عند إرسال الـ 50 عاموداً.
    """
    if _seen is None:
        _seen = set()

    if obj is None:
        return None

    # 1. حماية من الدوران اللا نهائي (Recursion Guard)
    # ضروري جداً لأن حزمة الأدلة (Evidence Pack) أصبحت ضخمة جداً
    obj_id = id(obj)
    if obj_id in _seen:
        return "[CIRCULAR_REFERENCE_REDACTED]"
    
    # نضيف الكائنات القابلة للتكرار فقط إلى set الحماية
    if isinstance(obj, (dict, list, set, tuple)):
        _seen.add(obj_id)

    # 2. معالجة الأرقام (الفيزياء والـ ML)
    if isinstance(obj, (float, np.floating)):
        # [تعديل حاسم]: تحويل NaN الناتج عن البيانات المفقودة في toolbox إلى null
        if not math.isfinite(obj):
            return None 
        return float(obj)

    if isinstance(obj, (int, np.integer)):
        return int(obj)
        
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)

    if isinstance(obj, decimal.Decimal):
        return float(obj)

    # 3. معالجة النصوص والبيانات الخام
    if isinstance(obj, str):
        return obj

    if isinstance(obj, bytes):
        # فك تشفير البيانات الخام لضمان عدم كسر الـ API
        return obj.decode("utf-8", errors="replace")

    # 4. المصفوفات والقوائم (نتائج Monte Carlo والـ RAG)
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(v, _seen) for v in obj.tolist()]
        
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v, _seen) for v in obj]

    # 5. القواميس (التدقيق الكامل للـ 50 عاموداً)
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v, _seen) for k, v in obj.items()}

    # 6. الكائنات الزمنية (Report Timestamps)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
        
    if isinstance(obj, uuid.UUID):
        return str(obj)

    # 7. التراجع الآمن (Fallback)
    # بدلاً من الانهيار، نحول الكائن المجهول إلى نص للتدقيق
    try:
        return str(obj)
    except:
        return "[UNSERIALIZABLE_DATA]"