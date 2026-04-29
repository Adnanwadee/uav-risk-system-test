"""
ACE Mission Control | V16.0 Apex Command Center
==============================================
التحديثات الهندسية الكبرى:
1. Multi-Agent Radar Chart: تمثيل مرئي لتوازن القوى بين (الفيزياء، القانون، الزمن، والـ ML).
2. Cognitive Evidence Vault: عرض الاستشهادات القانونية [المصدر | المادة] بشكل تفاعلي.
3. Resilience Engine: آلية انتظار ذكية (Smart Retries) عند تشغيل السيرفر.
4. Professional HUD: تصميم داكن مستوحى من أنظمة التحكم في الطائرات المسيرة.
"""

import streamlit as st
import json
import httpx
import asyncio
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

# ─── 1. الإعدادات البصرية المتقدمة (Aviation Theme) ───
st.set_page_config(page_title="ACE Apex Control", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #c9d1d9; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #161b22; border-radius: 5px 5px 0 0; padding: 10px 20px; }
    .status-go { color: #238636; font-weight: bold; border: 1px solid #238636; padding: 5px 15px; border-radius: 20px; }
    .status-nogo { color: #f85149; font-weight: bold; border: 1px solid #f85149; padding: 5px 15px; border-radius: 20px; }
    .evidence-card { background-color: #0d1117; border-right: 4px solid #1f6feb; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

# ─── 2. إدارة البيانات (الـ 50 عاموداً) ───
if 'payload' not in st.session_state:
    st.session_state.payload = {
        "uav": {
            "mass_kg": 45.0,
            "max_thrust_n": 150.0,
            "sensors": {"has_lidar": True, "has_camera": True, "has_radar": True}
        },
        "environment": {
            "weather": {"wind_mps": 8.0, "temp_c": 35.0}, 
            "gnss_jam_dbm": -105.0
        },
        "telemetry": {
            "battery_level_pct": 85.0,
            "altitude_m": 120.0,
            "population_density": "HIGH_DENSE",
            "comms_status": "STABLE"
        }
    }

# ─── 3. محرك الرسوم البيانية الهندسية ───
def create_radar_chart(drivers):
    """إنشاء مخطط راداري يوضح وزن كل وكيل في القرار."""
    agents = ['Physics', 'Legal (RAG)', 'Temporal', 'ML (10%)']
    scores = [drivers.get('PHYSICS', 0)*100, drivers.get('LEGAL', 0)*100, 
              drivers.get('TEMPORAL', 0)*100, drivers.get('ML_CONSULTANT', 0)*100]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=scores + [scores[0]],
        theta=agents + [agents[0]],
        fill='toself',
        fillcolor='rgba(31, 111, 235, 0.3)',
        line=dict(color='#1f6feb', width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#30363d"),
                   bgcolor="rgba(0,0,0,0)", angularaxis=dict(gridcolor="#30363d")),
        showlegend=False, paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(t=30, b=30)
    )
    return fig

# ─── 4. هيكل التطبيق الرئيسي ───
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2528/2528143.png", width=100)
st.sidebar.title("ACE v16.0 Apex")
st.sidebar.info("Aviation Compliance Engine is ONLINE.")

st.title("🛡️ Mission Command Center")
st.caption("Strategic Autonomous Risk Assessment & Compliance Framework")

tab_config, tab_deliberation, tab_audit = st.tabs(["🚀 Mission Setup", "🧠 ACE Deliberation", "📜 Evidence Audit"])

with tab_config:
    col_input, col_json = st.columns([1.2, 1])
    
    with col_input:
        st.subheader("Mission Telemetry Injection")
        with st.expander("✈️ Aircraft Configuration", expanded=True):
            mass = st.number_input("Payload Mass (kg)", value=st.session_state.payload['uav']['mass_kg'])
            st.session_state.payload['uav']['mass_kg'] = mass
            
        with st.expander("🌍 Environmental Context", expanded=True):
            wind = st.slider("Wind Intensity (m/s)", 0.0, 30.0, st.session_state.payload['environment']['weather']['wind_mps'])
            pop = st.selectbox("Area Density", ["SPARSE", "RURAL", "SUBURBAN", "URBAN", "HIGH_DENSE"], index=4)
            st.session_state.payload['environment']['weather']['wind_mps'] = wind
            st.session_state.payload['telemetry']['population_density'] = pop

    with col_json:
        st.subheader("Raw Data Stream (50+ Columns)")
        raw_json = st.text_area("Telemetry Buffer", value=json.dumps(st.session_state.payload, indent=2), height=350)
        try:
            st.session_state.payload = json.loads(raw_json)
        except:
            st.error("Invalid JSON")

    if st.button("⚡ EXECUTE CATASTROPHIC RISK ANALYSIS", type="primary", width="stretch"):
        async def call_ace():
            # [Resilience]: آلية انتظار للسيرفر مع 3 محاولات
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        url = "http://127.0.0.1:8000/v2/stage2/evaluate"
                        resp = await client.post(url, json=st.session_state.payload)
                        if resp.status_code == 200: return resp.json()
                        st.error(f"API Error {resp.status_code}")
                except httpx.ConnectError:
                    if attempt < 2: await asyncio.sleep(2); continue
                    st.error("Backend Offline.")
            return None

        with st.spinner("Council of Agents is deliberating on flight safety..."):
            res = asyncio.run(call_ace())
            if res: st.session_state.ace_result = res

# ─── 5. عرض نتائج "مجلس الوكلاء" ───
with tab_deliberation:
    if "ace_result" in st.session_state:
        res = st.session_state.ace_result
        decision = res.get('decision', 'UNKNOWN')
        
        # هيدر القرار النهائي
        st.markdown(f"### Final Verdict: <span class='{'status-go' if decision=='GO' else 'status-nogo'}'>{decision}</span>", unsafe_allow_html=True)
        
        col_chart, col_stats = st.columns([1, 1])
        
        with col_chart:
            st.write("**Agent Risk Distribution (NRS)**")
            drivers = {d['agent']: d['score'] for d in res.get("structured_data", {}).get("forensic_drivers", [])}
            st.plotly_chart(create_radar_chart(drivers), width="stretch")
            
        with col_stats:
            st.write("**Agent Sentiment Analysis**")
            st.progress(drivers.get('PHYSICS', 0), text=f"Physics Margin: {drivers.get('PHYSICS', 0)*100}%")
            st.progress(drivers.get('LEGAL', 0), text=f"Legal Risk: {drivers.get('LEGAL', 0)*100}%")
            st.progress(drivers.get('ML_CONSULTANT', 0), text=f"ML Stats (10% weight): {drivers.get('ML_CONSULTANT', 0)*100}%")

        st.markdown("---")
        st.subheader("📋 Official Safety Audit Report")
        st.markdown(res.get("report_markdown", "Report processing..."))

# ─── 6. تدقيق الأدلة (Evidence & RAG) ───
with tab_audit:
    if "ace_result" in st.session_state:
        col_rag, col_raw = st.columns([1, 1])
        
        with col_rag:
            st.subheader("⚖️ Legal RAG Evidence Vault")
            citations = res.get("structured_data", {}).get("legal_citations", [])
            if citations:
                for cite in citations:
                    st.markdown(f"<div class='evidence-card'><b>Citation:</b> {cite}</div>", unsafe_allow_html=True)
            else:
                st.info("No specific legal violations cited.")

        with col_raw:
            st.subheader("📡 Full Telemetry Trace")
            audit_log = res.get("structured_data", {}).get("raw_snapshot", {})
            st.dataframe(pd.DataFrame(list(audit_log.items()), columns=["Sensor", "Value"]), width="stretch", height=400)
            
            st.download_button("📥 Download Audit Pack (JSON)", data=json.dumps(res, indent=2), file_name="ACE_Evidence_Pack.json")
