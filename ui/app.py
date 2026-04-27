import streamlit as st
import json
import httpx
import asyncio
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="ACE Mission Control | V5.1", layout="wide", page_icon="🛡️")

# تهيئة البيانات الافتراضية كاملة بدون أي اختصارات
DEFAULT_PAYLOAD = {
    "uav": {
        "mass_kg": 1.3,
        "max_speed_mps": 18.0,
        "max_thrust_n": 95.0,
        "sensors": {"lidar": True, "radar": True, "gnss": True, "imu": True}
    },
    "environment": {
        "weather": {"wind_mps": 2.5}, 
        "gnss_jam_dbm": -105.0
    },
    "telemetry": {
        "battery_level_pct": 95.0,
        "battery_drain_rate_pct_per_min": 0.5,
        "altitude_m": 45.0,
        "temperature_c": 22.0,
        "wind_speed_ms": 3.0,
        "wind_direction_deg": 45.0,
        "uav_heading_deg": 90.0,
        "planned_distance_m": 800.0,
        "speed_mps": 8.0,
        "estimated_flight_time_min": 10.0,
        "population_density": "SPARSE",
        "comms_uplink_status": "EXCELLENT",
        "environment_gnss_jam_dbm": -105.0
    }
}

# [FIX] يجب أن تكون التهيئة قبل أي استدعاء للسلايدر
if 'payload' not in st.session_state:
    st.session_state.payload = DEFAULT_PAYLOAD

# Professional CSS
st.markdown("""
    <style>
    .main { background-color: #0d1117; }
    .report-container { background-color: #161b22; padding: 20px; border-radius: 12px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ ACE Mission Control")

# 2. Layout
col_input, col_results = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📥 Mission Parameters")
    with st.container(border=True):
        st.write("**Real-time Telemetry Tuning**")
        # [FIX] توحيد الأسماء في السلايدر (ms بدلاً من mps)
        wind = st.slider("Wind Intensity (m/s)", 0.0, 25.0, float(st.session_state.payload['telemetry'].get('wind_speed_ms', 3.0)))
        batt = st.slider("Battery Reserve (%)", 0.0, 100.0, float(st.session_state.payload['telemetry'].get('battery_level_pct', 95.0)))
        
        # [FIX] تحديث الـ session_state بالأسماء الصحيحة
        st.session_state.payload['telemetry']['wind_speed_ms'] = wind
        st.session_state.payload['telemetry']['battery_level_pct'] = batt
        
        st.markdown("---")
        # [FIX] json.dumps سيعمل الآن لأننا تخلصنا من الـ sets
        raw_json = st.text_area("Full JSON Editor", 
                               value=json.dumps(st.session_state.payload, indent=2), 
                               height=400)
        
        try:
            st.session_state.payload = json.loads(raw_json)
        except:
            st.error("⚠️ Invalid JSON Structure")

    if st.button("🚀 EXECUTE EVALUATION", type="primary"):
        async def call_api():
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post("http://localhost:8000/v2/stage2/evaluate", json=st.session_state.payload)
                return resp.json()
        
        with st.spinner("Analyzing Safety..."):
            try:
                st.session_state.last_result = asyncio.run(call_api())
            except Exception as e:
                st.error(f"Connection Failed: {e}")

with col_results:
    if "last_result" in st.session_state:
        res = st.session_state.last_result
        st.subheader(f"Final Verdict: {res.get('decision', 'N/A')}")
        st.markdown(res.get("report_markdown", "No report available."))