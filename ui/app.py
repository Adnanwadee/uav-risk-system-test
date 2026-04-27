"""
ACE Mission Control Dashboard (V4.5 - Agentic UI)
=================================================
Role: Professional Interface for UAV Risk Evaluation.
Features: 
- Live API Integration (Testing the Fortress).
- Multi-Agent Feedback Visualization.
- Real-time Forensic Evidence Rendering.
- Telemetry Drift Analysis.

Author: Stage 2 — ACE System
"""

import streamlit as st
import json
import httpx
import asyncio
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ============================================================
# 1. Page Configuration & Branding
# ============================================================
st.set_page_config(page_title="ACE Mission Control", layout="wide", page_icon="🛡️")

# Custom Professional CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .agent-card { padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #238636; background: #0d1117; }
    .status-go { color: #238636; font-weight: bold; }
    .status-nogo { color: #da3633; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. State Management & Constants
# ============================================================
API_BASE_URL = "http://localhost:8000/v2/stage2"

# الهيكل الجديد المتوافق مع MasterFlightPayload V2.3
DEFAULT_PAYLOAD = {
    "uav": {
        "mass_kg": 2.5,
        "max_speed_mps": 15.0,
        "max_thrust_n": 80.0,
        "sensors": {"lidar": True, "radar": False, "gnss": True, "imu": True}
    },
    "environment": {
        "weather": {"wind_mps": 5.5, "gust_mps": 8.0},
        "gnss_jam_dbm": -95.0
    },
    "telemetry": {
        "battery_state_of_charge_pct": 85.0,
        "altitude_m": 120.0,
        "wind_speed_mps": 6.0,
        "distance_remaining_m": 1200.0,
        "speed_mps": 10.0,
        "estimated_flight_time_min": 12.0,
        "population_density": "SPARSE"
    }
}

# ============================================================
# 3. API Communication Logic
# ============================================================
async def call_ace_api(payload: dict):
    """يستدعي محرك ACE المصفح عبر الشبكة."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{API_BASE_URL}/evaluate", json=payload)
        return response.json()

# ============================================================
# 4. Sidebar Controls
# ============================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/579/579268.png", width=80)
    st.title("ACE Settings")
    endpoint = st.text_input("API Endpoint", value=API_BASE_URL)
    st.divider()
    if st.button("Reset Telemetry", use_container_width=True):
        st.session_state.payload = DEFAULT_PAYLOAD
        st.rerun()
    
    st.info("System Status: **FLIGHT-READY**")
    st.caption("ACE Generation 2 - Secure Control Logic")

# ============================================================
# 5. Main Layout
# ============================================================
st.title("🛡️ ACE Mission Control")
st.markdown("### Autonomous Control Engine — Risk & Physics Evaluation")

tab_input, tab_dashboard, tab_audit = st.tabs(["📥 Flight Input", "📊 Analytics Dashboard", "📑 Forensic Audit"])

with tab_input:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Payload Editor")
        raw_json = st.text_area("Edit JSON Mission Data", 
                               value=json.dumps(st.session_state.get('payload', DEFAULT_PAYLOAD), indent=2), 
                               height=500)
        
        try:
            current_payload = json.loads(raw_json)
            st.session_state.payload = current_payload
        except:
            st.error("Invalid JSON Format")

    with col2:
        st.subheader("Quick Simulation")
        st.slider("Simulate Wind Speed (m/s)", 0.0, 30.0, 5.0, key="wind_sim")
        st.progress(st.session_state.payload['telemetry']['battery_state_of_charge_pct']/100, text="Battery Level")
        
        if st.button("🚀 EXECUTE EVALUATION", type="primary", use_container_width=True):
            # تحديث الرياح في الـ JSON قبل الإرسال
            st.session_state.payload['telemetry']['wind_speed_mps'] = st.session_state.wind_sim
            
            with st.spinner("ACE Agents Deliberating..."):
                try:
                    result = asyncio.run(call_ace_api(st.session_state.payload))
                    st.session_state.last_result = result
                    st.success("Consensus Reached.")
                except Exception as e:
                    st.error(f"API Connection Failed: {e}")

# ============================================================
# 6. Dashboard & Results Visualization
# ============================================================
if "last_result" in st.session_state:
    res = st.session_state.last_result
    data = res.get("data", {})
    
    with tab_dashboard:
        # Top KPI Row
        k1, k2, k3, k4 = st.columns(4)
        decision = res.get("decision", "UNKNOWN")
        color = "green" if decision == "GO" else "orange" if decision == "CAUTION" else "red"
        
        k1.metric("Final Verdict", decision)
        k2.metric("Confidence Score", f"{res.get('data', {}).get('overall_confidence', 0)*100:.1f}%")
        k3.metric("Risk Level", res.get("data", {}).get("risk_level", "N/A"))
        k4.metric("Latency", f"{res.get('metrics', {}).get('total_pipeline_ms', 0):.0f}ms")

        # Visualization
        col_charts, col_agents = st.columns([1.5, 1])
        
        with col_charts:
            st.subheader("Risk Distribution")
            # Gauge Chart for Confidence
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = res.get('data', {}).get('overall_confidence', 0) * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Consensus Confidence (%)"},
                gauge = {'axis': {'range': [0, 100]},
                         'bar': {'color': color}}
            ))
            st.plotly_chart(fig, use_container_width=True)

        with col_agents:
            st.subheader("Agent Feedback")
            for driver in data.get("forensic_drivers", []):
                severity_color = "red" if driver['severity'] == "CRITICAL" else "orange"
                st.markdown(f"""
                <div class="agent-card" style="border-left-color: {severity_color}">
                    <strong>🤖 {driver['agent']} AGENT</strong><br>
                    <small>{driver['driver']}</small><br>
                    {driver['evidence_text']}
                </div>
                """, unsafe_allow_html=True)

    with tab_audit:
        st.subheader("Formal Safety Audit Report")
        st.info(f"Traceability ID: `{res.get('observability_thread')}`")
        
        col_report, col_raw = st.columns([2, 1])
        
        with col_report:
            st.markdown(res.get("report_markdown", "No report generated."))
            
        with col_raw:
            st.subheader("Legal Citations")
            for cite in data.get("legal_citations", []):
                st.code(cite, language="markdown")
            
            st.subheader("Data Quality")
            dq = data.get("quality_profile", {})
            st.write(f"Completeness: {dq.get('completeness_ratio', 0)*100:.1f}%")
            if dq.get("missing_critical_fields"):
                st.warning(f"Missing: {dq.get('missing_critical_fields')}")

else:
    with tab_dashboard:
        st.info("Waiting for execution... Please use the 'Flight Input' tab to run an evaluation.")

# Footer
st.divider()
st.caption(f"ACE V4.1 | Request ID: {st.session_state.get('last_result', {}).get('observability_thread', 'N/A')}")