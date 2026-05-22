# File Path: src/uav_risk/ui/app.py
import json
import time
import requests
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List

# ضبط الإعدادات الهيكلية الافتراضية للوحة الطيران الجوي
st.set_page_config(
    page_title="ACE UAV Autonomous Risk Assessment Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تعيين عنوان الخادم المركزي للواجهة البرمجية
API_URL = "http://127.0.0.1:8000/v2/evaluate"


def render_shap_horizontal_chart(shap_data: List[Dict[str, Any]]) -> go.Figure:
    """يولد مخطط SHAP أفقي تفاعلي ذكي يوضح مساهمة الميزات في رفع أو خفض فئة الخطر."""
    if not shap_data:
        fig = go.Figure()
        fig.add_annotation(text="No SHAP impact vectors recorded for this session.", showarrow=False)
        return fig

    # استخراج وتفكيك المكونات الرياضية لقيم التأثير
    features = [item.get("feature_name", "Unknown") for item in shap_data]
    values = [item.get("shap_value", 0.0) for item in shap_data]
    
    # تحديد الألوان ديناميكياً: أحمر لزيادة الخطر، أزرق لخفض الخطر
    colors = ['#dc3545' if v > 0 else '#007bff' for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition='auto'
    ))

    fig.update_layout(
        title="<b>تحليل مساهمة العوامل الرياضية (SHAP Values Impact Spectrum)</b>",
        xaxis_title="قيمة تأثير العامل الجغرافي والفيزيائي (SHAP Value)",
        yaxis_title="اسم الميزة الحاكمة في الدستور",
        yaxis=dict(autorange="reversed"),  # ✅ تم التصحيح الجراحي هنا من 'reverse' إلى 'reversed' لمنع الانهيار
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
        template="plotly_white"
    )
    return fig


# ─── بناء العقل الهيكلي لواجهة واجهة الطيران ───
st.title("🚀 نظام ACE الذكي لتقييم مخاطر الطائرات المسيرة والامتثال التشريعي")
st.caption("إصدار الإنتاج الفعلي المستقر: ACE v4.5.0 Core Protocol Cluster")

st.markdown("---")

col1, col2 = st.columns([2, 1], gap="medium")

with col1:
    st.subheader("📋 نموذج التيليميتري الموحد ومعايير الرحلة الجوية")
    
    with st.container(border=True):
        st.markdown("**القسم 1: المواصفات الميكانيكية والهيكلية للدرون (UAV Specs)**")
        sc1_1, sc1_2, sc1_3 = st.columns(3)
        uav_mass_kg = sc1_1.number_input("وزن الطائرة الإجمالي (uav_mass_kg)", min_value=0.1, max_value=150.0, value=5.4, step=0.1)
        uav_wingspan_m = sc1_2.number_input("باع الجناح بالامتار (uav_wingspan_m)", min_value=0.1, max_value=10.0, value=1.8, step=0.1)
        payload_mass_kg = sc1_3.number_input("وزن الشحنة المضافة (payload_mass_kg)", min_value=0.0, max_value=50.0, value=0.5, step=0.1)
        
        sc1_4, sc1_5, sc1_6 = st.columns(3)
        battery_capacity_mah = sc1_4.number_input("سعة البطارية (battery_capacity_mah)", min_value=100, max_value=100000, value=16000, step=500)
        battery_voltage_v = sc1_5.number_input("جهد البطارية الاسمي (battery_voltage_v)", min_value=1.0, max_value=100.0, value=22.2, step=0.1)
        max_speed_ms = sc1_6.number_input("السرعة القصوى المتاحة (uav_max_speed_mps)", min_value=1.0, max_value=100.0, value=15.5, step=0.5)

    with st.container(border=True):
        st.markdown("**القسم 2: محددات المهمة والمجال الجوي (Mission Parameters)**")
        sc2_1, sc2_2, sc2_3 = st.columns(3)
        altitude_m = sc2_1.number_input("الارتفاع التشغيلي الحالي (mission_altitude_m)", min_value=0.0, max_value=2000.0, value=120.0, step=5.0)
        max_altitude_m = sc2_2.number_input("أقصى ارتفاع للمسار (mission_max_altitude_m)", min_value=0.0, max_value=2000.0, value=150.0, step=5.0)
        distance_km = sc2_3.number_input("المسافة الكلية المقطوعة (mission_distance_km)", min_value=0.0, max_value=500.0, value=12.5, step=0.5)
        
        sc2_4, sc2_5, sc2_6 = st.columns(3)
        flight_duration_min = sc2_4.number_input("زمن الرحلة المتوقع بالدقائق", min_value=1.0, max_value=480.0, value=45.0, step=1.0)
        operation_type = sc2_5.selectbox("نمط الرؤية التشغيلية (operation_type)", ["VLOS", "BVLOS"])
        is_night_flight = sc2_6.checkbox("هل الرحلة ليلية؟ (is_night_flight)", value=False)

    with st.container(border=True):
        st.markdown("**القسم 3: البيانات البيئية وحالة الطقس الحية (Environmental Data)**")
        sc3_1, sc3_2, sc3_3 = st.columns(3)
        wind_speed_ms = sc3_1.number_input("سرعة الرياح الحالية (wind_speed_mps)", min_value=0.0, max_value=60.0, value=4.5, step=0.5)
        temperature_c = sc3_2.number_input("درجة الحرارة المحيطة (temperature_c)", min_value=-30.0, max_value=60.0, value=28.0, step=1.0)
        humidity_pct = sc3_3.number_input("نسبة الرطوبة النسبية", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
        
        sc3_4, sc3_5 = st.columns(2)
        visibility_m = sc3_4.number_input("مدى الرؤية الأفقية بالامتار", min_value=0.0, max_value=50000.0, value=10000.0, step=100.0)
        precipitation = sc3_5.selectbox("كثافة هطول الأمطار والموانع", ["none", "light", "heavy"])

    with st.container(border=True):
        st.markdown("**القسم 4: جودة أنظمة الملاحة والاتصال اللاسلكي (GPS & Comms)**")
        sc4_1, sc4_2, sc4_3, sc4_4 = st.columns(4)
        gps_fix_quality = sc4_1.number_input("مؤشر جودة الـ GPS Fix (1-5)", min_value=0, max_value=5, value=4, step=1)
        satellites_count = sc4_2.number_input("عدد الأقمار المتصلة حياً", min_value=0, max_value=40, value=14, step=1)
        hdop = sc4_3.number_input("معامل انحراف الدقة الأفقي HDOP", min_value=0.1, max_value=20.0, value=1.1, step=0.1)
        rc_signal_strength_pct = sc4_4.number_input("قوة إشارة التحكم للراديو RSSI %", min_value=0.0, max_value=100.0, value=92.5, step=0.5)

    with st.container(border=True):
        st.markdown("**القسم 5: قيود المجال الجوي وتراخيص الطيار (Operator & Airspace)**")
        sc5_1, sc5_2, sc5_3 = st.columns(3)
        license_type = sc5_1.selectbox("رخصة قائد الطائرة (license_type)", ["FAA_PART_107", "EASA_CLASS_A1_A3", "NONE"])
        airspace_class = sc5_2.selectbox("فئة المجال الجوي المستهدف", ["Class_G", "Class_E", "Class_D", "Class_C", "Class_B"])
        airport_distance_km = sc5_3.number_input("المسافة لأقرب مطار تجاري (km)", min_value=0.0, max_value=200.0, value=15.2, step=0.1)
        
        sc5_4, sc5_5 = st.columns(2)
        atc_clearance = sc5_4.checkbox("تم الحصول على موافقة برج المراقبة (atc_clearance)", value=True)
        in_restricted_zone = sc5_5.checkbox("المسار يخترق منطقة حظر جوي صريحة (in_restricted_zone)", value=False)

    with st.expander("🔄 باقي الميزات التشغيلية والمؤشرات المتقدمة (المجموع الكلي 198 ميزة لنموذج LightGBM)"):
        st.info("الميزات التالية يتم ملؤها تلقائياً بالقيم الآمنة المعتمدة في دستور المنظومة، يمكنك تعديل بعضها لأغراض الفحص المتقدم.")
        advanced_features_payload = {}
        adv_col1, adv_col2 = st.columns(2)
        for idx in range(41, 60):
            feat_name = f"uav_advanced_metric_node_{idx}"
            with adv_col1 if idx % 2 == 0 else adv_col2:
                advanced_features_payload[feat_name] = st.number_input(f"الميزة الإحصائية {feat_name}", value=0.0, step=0.1)

    with st.container(border=True):
        st.markdown("**القسم 6: التوجيهات والسياق الحر لوكيل الاستدلال (Free Text Area)**")
        free_text = st.text_area(
            label="ملاحظات أو توجيهات إضافية للوكيل (مثل مبررات الطيران الاضطراري أو قيود LAANC المكتوبة)",
            value="Flight mission aims to conduct standard pipeline thermal cracks inspection inside unpopulated rural coordinates with active local air traffic waiver authorization anchored."
        )

    evaluate_button = st.button("تقييم وصلاحية الرحلة الجوية 🚀", use_container_width=True, type="primary")

with col2:
    st.subheader("📊 لوحة مخرجات الاستدلال وصك الإقلاع الجنائي")
    
    if evaluate_button:
        payload_data = {
            "uav_mass_kg": float(uav_mass_kg),
            "uav_wingspan_m": float(uav_wingspan_m),
            "uav_battery_capacity_mah": float(battery_capacity_mah),
            "uav_battery_voltage_v": float(battery_voltage_v),
            "uav_max_speed_mps": float(max_speed_ms),
            "payload_mass_kg": float(payload_mass_kg),
            "mission_altitude_m": float(altitude_m),
            "airspace_altitude_agl_max_m": float(max_altitude_m),
            "mission_distance_km": float(distance_km),
            "environment_weather_wind_mps": float(wind_speed_ms),
            "environment_weather_temperature_c": float(temperature_c),
            "gps_fix_quality": float(gps_fix_quality),
            "operator_in_restricted_zone": 1.0 if in_restricted_zone else 0.0,
            "free_text": free_text,
            **advanced_features_payload
        }

        progress_bar = st.progress(0, text="جاري ربط الرابط اللاسلكي لمعالجات ACE...")
        stages_messages = [
            (15, "STEP 1 — Tier-0 Veto: جاري فحص محددات الحظر المطلقة الحتمية..."),
            (40, "STEP 2 & 3 — Core Logic: جاري هندسة فضاء المصفوفات وتطهير جودة البيانات فيزيائياً..."),
            (65, "STEP 4 — ML Inference: جاري فك شيفرة شجرة القرار لـ LightGBM وحساب قيم SHAP..."),
            (85, "STEP 5 & 6 — ACE Agent: جاري تشغيل وكيل الاستدلال ومراجعة وثائق الطيران الفيدرالية..."),
            (100, "STEP 7 & 8 — Finalizing: جاري صياغة التقرير وتوقيع وثيقة الأدلة الرقمية الجنائية...")
        ]
        
        for p_val, msg in stages_messages:
            time.sleep(0.1)
            progress_bar.progress(p_val, text=msg)

        try:
            with st.spinner("جاري استلاف البيانات من خط الأنابيب المركزي..."):
                response = requests.post(API_URL, json=payload_data, timeout=90.0)
            
            if response.status_code == 200:
                result_json = response.json()
                st.success("🎉 تم معالجة طلب تقييم الرحلة بنجاح كامل من كافة البوابات!")
                
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📌 القرار والملخص", 
                    "📝 التقرير الكامل", 
                    "📊 تحليل SHAP", 
                    "⚖️ الاستشهادات القانونية", 
                    "📥 البيانات المُدخلة", 
                    "🔒 التدقيق والأدلة"
                ])
                
                with tab1:
                    decision_state = result_json.get("decision", "NO-GO").upper()
                    if decision_state == "GO":
                        st.balloons()
                        st.markdown("<h1 style='color:#28a745; text-align:center;'>GO ✅ (يصرح بالإقلاع فوراً)</h1>", unsafe_allow_html=True)
                    elif decision_state == "CONDITIONAL-GO":
                        st.markdown("<h1 style='color:#ffc107; text-align:center;'>CONDITIONAL-GO ⚠️ (إقلاع مشروط بالقيود)</h1>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h1 style='color:#dc3545; text-align:center;'>NO-GO ⛔ (يمنع ويحظر الإقلاع كلياً)</h1>", unsafe_allow_html=True)
                        
                    st.metric("مؤشر درجة خطر الرحلة الكلي (Risk Score)", f"{result_json.get('risk_score', 0.0):.4f}")
                    st.metric("نسبة ثقة المنظومة الذكية (System Confidence)", f"{result_json.get('confidence', 1.0)*100:.2f} %")
                    
                    st.markdown("**❌ المشاكل والمخاطر الحرجة المكتشفة حياً:**")
                    for finding in result_json.get("critical_findings", []):
                        st.error(finding)
                        
                    st.markdown("**💡 التوصيات الممنوحة من الوكيل الذكي لعلاج الموقف:**")
                    for rec in result_json.get("recommendations", []):
                        st.info(rec)

                with tab2:
                    st.subheader("تقرير فحص الامتثال والسلامة الجوية الرسمي المولد")
                    st.markdown(result_json.get("report_markdown", "*No Markdown report text returned from server cluster.*"))

                with tab3:
                    st.subheader("المخطط الرياضي لتشريح تأثير العوامل على شجرة التنبؤ")
                    shap_explanation = result_json.get("shap_explanation", [])
                    plotly_fig = render_shap_horizontal_chart(shap_explanation)
                    st.plotly_chart(plotly_fig, use_container_width=True)

                with tab4:
                    st.subheader("البنود والمواد التشريعية المثبتة جنائياً (FAA Part 107 / SORA)")
                    citations = result_json.get("legal_citations", [])
                    if citations:
                        for cit in citations:
                            with st.chat_message("assistant"):
                                st.markdown(f"**المصدر التنظيمي:** {cit.get('source_file', 'Unknown')}, **صفحة:** {cit.get('page_number', 0)}")
                                st.caption(f"\"{cit.get('full_text', cit.get('citation', ''))}\"")
                    else:
                        st.write("لم يتم استدعاء أو ربط أي نصوص قانونية لهذه الدورة التشغيلية.")

                with tab5:
                    st.subheader("مصفوفة فحص جودة المدخلات الفيزيائية")
                    st.metric("مؤشر قياس جودة البيانات (Data Quality Score)", f"{result_json.get('data_quality_score', 0.0):.3f}")
                    st.json(payload_data)

                with tab6:
                    st.subheader("حزمة التوقيع الرقمي للأدلة الجنائية (Audit Evidence Pack Payload)")
                    st.info(f"إصدار محرك الأنبوب: {result_json.get('pipeline_version', 'ACE-v4.5')} | زمن المعالجة الكلي: {result_json.get('processing_time_ms', 0.0):.2f}ms")
                    st.json(result_json)
                    
            else:
                st.error(f"❌ خطأ سيرفر واجهة الـ API: رمز الحالة {response.status_code}")
                st.text(response.text)
                
        except Exception as conn_err:
            st.error(f"❌ فشل الاتصال بخط الأنابيب المركزي عبر خادم الماستر المحصن: {str(conn_err)}")
            st.warning("يرجى التأكد من تشغيل ملف FastAPI الخلفي `src/uav_risk/api/main.py` على المنفذ المخصص 8000 قبل إطلاق الطلبات.")

    else:
        st.write("🔄 بانتظار إدخال قراءات الطائرة والضغط على زر التقييم لإصدار التقرير التشريعي الحركي...")


# =====================================================================
# Stage 2 Streamlit Frontend Canvas Submodule Architectural Dependency Report:
#
# Depends on:
#   - Network Socket Link via HTTP POST -> http://127.0.0.1:8000/v2/evaluate (fastapi core gateway)
#
# Consumed by:
#   - Remote Ground Control Stations Operators / UAV Field Compliance Inspectors UI
# =====================================================================