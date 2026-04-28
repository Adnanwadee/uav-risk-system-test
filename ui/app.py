"""
ACE Mission Control UI (V5.3 - Mission Ready)
=============================================
التعديلات:
1. حل الـ 404: تحديث رابط الـ API ليشمل البادئة الموحدة /v2/stage2/evaluate.
2. إصلاح التحذيرات: استبدال use_container_width بـ width='stretch' حسب تنبيهات Streamlit الأخيرة.
"""

import streamlit as st
import json
import httpx
import asyncio
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# 1. إعدادات الصفحة والهوية البصرية
st.set_page_config(page_title="ACE Mission Control", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# البيانات الافتراضية (الـ 50 عاموداً)
DEFAULT_PAYLOAD = {
    "uav": {
        "mass_kg": 45.0,
        "max_thrust_n": 150.0,
        "sensors": {"lidar": True, "radar": True, "gnss": True, "imu": True}
    },
    "environment": {
        "weather": {"wind_mps": 5.5, "temp_c": 35.0}, 
        "gnss_jam_dbm": -105.0
    },
    "telemetry": {
        "battery_level_pct": 88.0,
        "altitude_m": 120.0,
        "population_density": "HIGH_DENSE"
    }
}

def create_risk_gauge(score: float, title: str):
    """رسم عداد المخاطر الاحترافي."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        title = {'text': title, 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff4b4b" if score > 0.7 else "#238636"},
            'steps': [{'range': [0, 100], 'color': "rgba(255,255,255,0.1)"}]
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
    return fig

st.title("🛡️ ACE Mission Control")

tab_config, tab_analysis, tab_audit = st.tabs(["⚙️ Mission Config", "🧠 ACE Deliberation", "📜 Data Audit"])

with tab_config:
    col_input, col_json = st.columns([1, 1])
    with col_input:
        st.subheader("Flight Parameters")
        mass = st.number_input("UAV Mass (kg)", value=45.0)
        wind = st.slider("Wind Speed (m/s)", 0.0, 40.0, 5.5)
        
        DEFAULT_PAYLOAD["uav"]["mass_kg"] = mass
        DEFAULT_PAYLOAD["environment"]["weather"]["wind_mps"] = wind

    with col_json:
        st.subheader("Telemetry Snapshot")
        raw_json = st.text_area("JSON Editor", value=json.dumps(DEFAULT_PAYLOAD, indent=2), height=300)
        payload = json.loads(raw_json)

    # [إصلاح]: استخدام المعامل الجديد حسب تحذيرات Streamlit
    if st.button("🚀 EXECUTE SAFETY AUDIT", type="primary", width='stretch'): 
        async def call_ace():
            async with httpx.AsyncClient(timeout=60.0) as client:
                # [إصلاح الـ 404]: إضافة /v2 لتتوافق مع ملف main.py
                url = "http://127.0.0.1:8000/v2/stage2/evaluate"
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    return None
                return response.json()
        
        with st.spinner("Agents are deliberating..."):
            result = asyncio.run(call_ace())
            if result:
                st.session_state.ace_result = result

with tab_analysis:
    if "ace_result" in st.session_state:
        res = st.session_state.ace_result
        st.header(f"Final Verdict: {res.get('decision')}")
        
        col_phys, col_leg, col_temp, col_ml = st.columns(4)
        drivers = {d['agent']: d['score'] for d in res.get("structured_data", {}).get("forensic_drivers", [])}
        
        # [إصلاح]: استبدال use_container_width بـ width='stretch' لإسكات التحذيرات
        col_phys.plotly_chart(create_risk_gauge(drivers.get("PHYSICS", 0), "Physics"), width='stretch')
        col_leg.plotly_chart(create_risk_gauge(drivers.get("LEGAL", 0), "Legal (RAG)"), width='stretch')
        col_temp.plotly_chart(create_risk_gauge(drivers.get("TEMPORAL", 0), "Temporal"), width='stretch')
        col_ml.plotly_chart(create_risk_gauge(drivers.get("ML_CONSULTANT", 0), "ML (10%)"), width='stretch')

        st.markdown("---")
        st.markdown(res.get("report_markdown", "No report available."))

with tab_audit:
    if "ace_result" in st.session_state:
        st.subheader("Data Audit Trace")
        audit_data = res.get("structured_data", {}).get("raw_snapshot", {})
        # [إصلاح التحذير]: استخدام width='stretch'
        st.dataframe(pd.DataFrame(list(audit_data.items()), columns=["Field", "Value"]), width='stretch')