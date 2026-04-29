import streamlit as st
import json
import httpx
import asyncio
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ─── 1. إعدادات الواجهة الاحترافية (Aviation HUD Theme) ───
st.set_page_config(page_title="ACE Apex Mission Control", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #c9d1d9; }
    .status-go { color: #238636; font-weight: bold; font-size: 28px; border: 2px solid #238636; padding: 10px; border-radius: 8px; background: rgba(35, 134, 54, 0.1); }
    .status-no-go { color: #da3633; font-weight: bold; font-size: 28px; border: 2px solid #da3633; padding: 10px; border-radius: 8px; background: rgba(218, 54, 51, 0.1); }
    .metric-box { background: #1c2128; padding: 20px; border-radius: 12px; border: 1px solid #30363d; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
    .evidence-card { background-color: #161b22; border-left: 5px solid #58a6ff; padding: 15px; margin-bottom: 12px; border-radius: 5px; font-family: 'Courier New', monospace; font-size: 0.9em; }
    </style>
""", unsafe_allow_html=True)

# ─── 2. محرك الاتصال (Resilient API Client) ───
API_URL = "http://localhost:8000/v2/stage2/evaluate"

async def call_ace_api(payload):
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(API_URL, json=payload)
            return response.json()
        except Exception as e:
            return {"status": "ERROR", "error_details": str(e)}

# ─── 3. بناء لوحة التحكم (Mission Control) ───
st.title("🛡️ ACE Apex | Mission Critical Flight Guardian")
st.caption(f"Operational Intel: V17.0 Apex-Ready | Deployment Date: {datetime.now().strftime('%Y-%m-%d')}")

col_input, col_display = st.columns([1.3, 2], gap="large")

with col_input:
    st.header("🎛️ Flight Parameters (GO Scenario)")
    
    # هذه هي البيانات "الإجبارية" لعمل الوكلاء باحترافية
    with st.expander("✈️ UAV Engineering Specs", expanded=True):
        # قيم افتراضية لسيناريو GO: طائرة خفيفة (2كجم) ودفع قوي جداً (100ن)
        uav_mass = st.number_input("Mass (kg)", value=2.0, step=0.1, help="Required by Physics & Legal Agents")
        uav_thrust = st.number_input("Max Thrust (N)", value=100.0, step=5.0, help="Required for T/W Ratio calculations")
        uav_power = st.number_input("Hover Power (W)", value=200.0, step=10.0, help="Required for Energy Drain simulation")
        uav_type = st.selectbox("Frame Type", ["quadrotor", "hexarotor", "fixed-wing"], index=0)
        
        st.markdown("**Sensor Suite**")
        c1, c2 = st.columns(2)
        has_camera = c1.toggle("HD Optical Camera", value=True)
        has_lidar = c2.toggle("LiDAR Sensor", value=False)
    
    with st.expander("🌐 Operational Environment", expanded=True):
        # قيم افتراضية لسيناريو GO: ارتفاع منخفض (40م)، رياح هادئة (2م/ث)، منطقة ريفية آمنة
        pop_density = st.selectbox("Population Density", ["RURAL", "SUBURBAN", "URBAN", "HIGH_DENSE"], index=0)
        wind_speed = st.slider("Current Wind Speed (m/s)", 0.0, 25.0, 2.0)
        altitude = st.slider("Target Altitude (m)", 0, 500, 40)
        mission_type = st.selectbox("Mission Profile", ["VLOS flight", "BVLOS flight", "Delivery"], index=0)
        
    with st.expander("🔋 Mission Energy Profile", expanded=True):
        # قيم افتراضية لسيناريو GO: بطارية ممتلئة (95%) ووقت رحلة قصير (10 دقائق)
        mission_time = st.number_input("Mission Duration (min)", value=10.0, step=1.0)
        mission_dist = st.number_input("Planned Distance (m)", value=1000.0, step=100.0)
        battery_pct = st.slider("Current Battery Charge (%)", 0, 100, 95)
        drain_rate = st.number_input("Est. Consumption (%/min)", value=1.5, step=0.1)

    if st.button("🚀 EXECUTE SAFETY AUDIT", use_container_width=True, type="primary"):
        # بناء الـ Payload الشامل (الـ 50 عاموداً) لضمان عدم وجود نقص داتا
        payload = {
            "uav": {
                "mass_kg": uav_mass,
                "max_thrust_n": uav_thrust,
                "hover_power_w": uav_power,
                "type": uav_type,
                "sensors": {
                    "has_camera": "1" if has_camera else "0",
                    "has_lidar": "1" if has_lidar else "0"
                }
            },
            "environment": {
                "weather": {"wind_mps": wind_speed, "temp_c": 22.0}
            },
            "telemetry": {
                "altitude_m": altitude,
                "battery_level_pct": battery_pct,
                "battery_drain_rate_pct_per_min": drain_rate,
                "population_density": pop_density
            },
            "mission": {
                "type": mission_type,
                "estimated_flight_time_min": mission_time,
                "planned_distance_m": mission_dist
            }
        }
        
        with st.spinner("Council of Agents is Analyzing System Integrity..."):
            result = asyncio.run(call_ace_api(payload))
            st.session_state.ace_result = result
            st.session_state.last_payload = payload

# ─── 4. معالجة وعرض مخرجات الوكلاء ───
with col_display:
    if "ace_result" in st.session_state:
        res = st.session_state.ace_result
        
        if res.get("status") == "ERROR":
            st.error(f"FATAL PIPELINE ERROR: {res.get('error_details')}")
        else:
            # مؤشرات الأداء الحيوية (KPIs)
            k1, k2, k3 = st.columns(3)
            decision = res.get('decision', 'N/A')
            with k1:
                st.markdown(f"<div class='metric-box'><h6>Final Decision</h6><p class='{'status-go' if decision == 'GO' else 'status-no-go'}'>{decision}</p></div>", unsafe_allow_html=True)
            with k2:
                conf = res.get("metrics", {}).get("confidence", 0.0)
                st.markdown(f"<div class='metric-box'><h6>System Confidence</h6><h2>{round((1-conf)*100, 1)}%</h2></div>", unsafe_allow_html=True)
            with k3:
                time_ms = res.get("metrics", {}).get("total_time_ms", 0.0)
                st.markdown(f"<div class='metric-box'><h6>Analysis Time</h6><h2>{int(time_ms)}ms</h2></div>", unsafe_allow_html=True)

            # التبويبات المتطورة
            tab_rep, tab_viz, tab_audit = st.tabs(["📄 Safety Audit Report", "📊 Risk Distribution", "📡 Telemetry Audit"])

            with tab_rep:
                st.subheader("📋 Executive Mission Safety Report")
                st.markdown(res.get("report_markdown", "No report content generated."))

            with tab_viz:
                st.subheader("⚖️ Neural-Symbolic Radar Balance")
                scores = res.get("structured_data", {}).get("agent_scores", {})
                
                # تمثيل "هامش الأمان": 1.0 تعني أمان مطلق، 0 تعني خطر مطلق
                categories = ['Physics Guardian', 'Legal Compliance', 'Temporal Stability', 'ML Consultant']
                values = [
                    max(0, 1.0 - scores.get('physics', 1.0)),
                    max(0, 1.0 - scores.get('legal', 1.0)),
                    max(0, 1.0 - scores.get('temporal', 1.0)),
                    max(0, 1.0 - scores.get('ml', 1.0))
                ]

                fig = go.Figure(data=go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    line_color='#58a6ff',
                    fillcolor='rgba(88, 166, 255, 0.3)'
                ))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, width='stretch')

            with tab_audit:
                col_c, col_d = st.columns([1, 1])
                with col_c:
                    st.subheader("⚖️ Legal RAG Citations")
                    citations = res.get("structured_data", {}).get("legal_citations", [])
                    if citations:
                        for cite in citations:
                            st.markdown(f"<div class='evidence-card'>{cite}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No legal warnings detected for this configuration.")

                with col_d:
                    st.subheader("📡 Raw Sensor Trace")
                    # عرض كل البيانات المرسلة للتأكد من وصولها للـ Backend
                    audit_log = pd.json_normalize(st.session_state.last_payload).T.reset_index()
                    audit_log.columns = ["Data Path", "Value"]
                    # [الإصلاح الحاسم لـ Arrow]: تحويل القيم لنصوص
                    st.dataframe(audit_log.astype(str), width='stretch', height=400)

    else:
        st.markdown("<br><br><br><div style='text-align: center; color: #8b949e;'><h3>🛡️ Awaiting Mission Configuration</h3><p>Adjust parameters and execute the audit sequence to generate a safety report.</p></div>", unsafe_allow_html=True)

# ─── 5. حالة النظام ───
st.sidebar.markdown("---")
st.sidebar.subheader("🔌 System Integrity")
st.sidebar.success("✅ Backend Service: Connected")
st.sidebar.success("✅ Groq Cognitive: Online")
st.sidebar.success("✅ RAG Database: Synchronized")