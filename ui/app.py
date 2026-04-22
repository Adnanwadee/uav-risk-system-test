# ============================================================
# CRITICAL: pickle compatibility injection (MUST be first)
# ============================================================
import __main__

def to_string_safe(x):
    try:
        return x.astype(str)
    except Exception:
        return x

__main__.to_string_safe = to_string_safe

import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from uav_risk.stage2.pipeline import run_stage2_report


# =========================
# Page Config + Branding
# =========================
st.set_page_config(
    page_title="UAV Risk System",
    layout="wide",
    page_icon="🛰️",
)

APP_TITLE = "UAV Risk Assessment (Stage-2)"
APP_SUBTITLE = "واجهة تشغيل احترافية لتقييم مخاطر الطائرات بدون طيار — بدون curl وبدون uvicorn"

# Light CSS for “company-like” UI
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
      .stMetric { background: rgba(255,255,255,0.03); padding: 14px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }
      .card { background: rgba(255,255,255,0.03); padding: 14px 16px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.06); }
      .muted { opacity: 0.8; }
      .badge { display:inline-block; padding:6px 10px; border-radius: 999px; font-weight: 700; font-size: 13px; border: 1px solid rgba(255,255,255,0.08); }
      .badge-go { background: rgba(34,197,94,0.15); color: rgb(34,197,94); }
      .badge-caution { background: rgba(234,179,8,0.15); color: rgb(234,179,8); }
      .badge-nogo { background: rgba(239,68,68,0.15); color: rgb(239,68,68); }
      .badge-unk { background: rgba(148,163,184,0.12); color: rgb(148,163,184); }
      .small { font-size: 12px; opacity: 0.85; }
      textarea { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# Defaults / Helpers
# =========================
DEFAULT_PAYLOAD: Dict[str, Any] = {
    "scenario": {
        "uav.type": "quadcopter",
        "uav.mass_kg": 1.3,
        "uav.max_speed_mps": 9.0,
        "uav.battery_model.hover_power_W": 220.0,
        "environment.weather.wind_mps": 1.0,
        "environment.weather.gust_mps": 1.5,
        "environment.weather.visibility": 9000,
        "environment.gnss_jam_dbm": -120,
        "environment.gnss_multipath": False,
        "environment.em_interference": False,
        "daa.sep_threshold_m": 90,
        "daa.ttc_threshold_s": 60,
        "airspace.altitude_agl_max_m": 40,
        "comms.uplink_ok": True,
        "comms.downlink_ok": True,
        "mission.type": "survey",
        "mission.pattern": "vlos",
        "mission.runway_required": False,
        "feat_mission_dist_m": 500,
        "feat_mission_climb_m": 6,
        "feat_mission_tortuosity": 1.01,
        "feat_power_to_weight": 169.0,
        "feat_weather_score": 0.98,
        "feat_airspace_area_m2": 450000,
        "feat_obstacle_density_per_km2": 1.0,
        "feat_obstacle_avg_speed": 0.0,
        "dq_core_present_pct": 1.0,
        "dq_weather_present": 1,
        "dq_uav_present": 1,
        "dq_comms_present": 0,
        "dq_sensors_present_pct": 1.0,
        "dq_mission_present": 1,
        "has_gnss": True,
        "has_imu": True,
        "has_lidar": False,
        "has_radar": False,
        "has_camera_rgb": True,
        "has_camera_thermal": False,
    }
}


def _badge(decision: str) -> str:
    d = (decision or "").upper().strip()
    if d == "GO":
        return '<span class="badge badge-go">GO</span>'
    if d == "CAUTION":
        return '<span class="badge badge-caution">CAUTION</span>'
    if d in {"NO_GO", "NOGO", "NO-GO"}:
        return '<span class="badge badge-nogo">NO_GO</span>'
    if d == "INSUFFICIENT_DATA":
        return '<span class="badge badge-unk">INSUFFICIENT_DATA</span>'
    return f'<span class="badge badge-unk">{d or "UNKNOWN"}</span>'


def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def _extract_core_metrics(resp_json: Dict[str, Any]) -> Dict[str, Any]:
    decision = resp_json.get("decision", "UNKNOWN")
    s1 = _safe_get(resp_json, ["facts", "stage1"], {}) or {}
    rules = _safe_get(resp_json, ["facts", "rules"], {}) or {}
    quality = resp_json.get("quality", {}) or {}
    dq = _safe_get(quality, ["data_quality"], {}) or {}

    stage1_pred = s1.get("predicted_class")
    stage1_conf = s1.get("confidence")
    stage1_score = s1.get("risk_score")

    hard_count = len((rules.get("hard_violations") or []))
    adv_count = len((rules.get("advisories") or []))

    completeness = dq.get("completeness_ratio")

    return {
        "decision": decision,
        "stage1_pred": stage1_pred,
        "stage1_conf": stage1_conf,
        "stage1_score": stage1_score,
        "hard_count": hard_count,
        "adv_count": adv_count,
        "completeness": completeness,
    }


def _plot_probability_bar(probabilities: Dict[str, float]):
    if not probabilities:
        return None
    labels = list(probabilities.keys())
    values = [float(probabilities[k]) for k in labels]

    fig = plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Probability")
    plt.title("Stage-1 Class Probabilities")
    return fig


def _plot_history_table(history: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for h in history:
        m = h.get("metrics", {})
        rows.append({
            "time": h.get("time"),
            "decision": m.get("decision"),
            "stage1_pred": m.get("stage1_pred"),
            "risk_score": m.get("stage1_score"),
            "confidence": m.get("stage1_conf"),
            "hard": m.get("hard_count"),
            "advisories": m.get("adv_count"),
            "completeness": m.get("completeness"),
            "latency_s": h.get("latency_s"),
        })
    df = pd.DataFrame(rows)
    return df


# =========================
# Cache hooks (ready for future)
# =========================
@st.cache_resource
def _warmup_resources() -> Dict[str, Any]:
    """
    Resource cache placeholder.

    Today: returns empty dict.
    Tomorrow: once we expose RAG index build in pipeline (or separate function),
    we can build it once here and reuse across runs.
    """
    return {"ready": True}


# =========================
# Session State
# =========================
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts (each run)
if "saved_scenarios" not in st.session_state:
    st.session_state.saved_scenarios = {}  # name -> payload
if "active_payload" not in st.session_state:
    st.session_state.active_payload = json.dumps(DEFAULT_PAYLOAD, indent=2)


# =========================
# Sidebar (Brand + Controls)
# =========================
with st.sidebar:
    st.markdown(f"## {APP_TITLE}")
    st.markdown(f"<div class='muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### تشغيل")
    artifacts_dir = st.text_input("artifacts_dir", value="artifacts")
    st.caption("مكان ملفات النموذج/المخرجات. اتركه كما هو عادة.")

    st.markdown("### إدارة الذاكرة")
    if st.button("مسح سجل النتائج (History)"):
        st.session_state.history = []
        st.success("تم مسح السجل.")

    if st.button("إعادة ضبط السيناريو الافتراضي"):
        st.session_state.active_payload = json.dumps(DEFAULT_PAYLOAD, indent=2)
        st.success("تمت إعادة ضبط السيناريو.")

    st.divider()
    st.markdown("### سيناريوهات محفوظة")
    save_name = st.text_input("اسم لحفظ السيناريو الحالي", value="")
    if st.button("حفظ السيناريو الحالي"):
        name = (save_name or "").strip()
        if not name:
            st.warning("اكتب اسمًا للسيناريو.")
        else:
            try:
                st.session_state.saved_scenarios[name] = json.loads(st.session_state.active_payload)
                st.success(f"تم حفظ السيناريو: {name}")
            except Exception as e:
                st.error(f"فشل حفظ السيناريو: {e}")

    if st.session_state.saved_scenarios:
        chosen = st.selectbox("تحميل سيناريو محفوظ", options=["-"] + list(st.session_state.saved_scenarios.keys()))
        if chosen != "-" and st.button("تحميل"):
            st.session_state.active_payload = json.dumps(st.session_state.saved_scenarios[chosen], indent=2)
            st.success(f"تم تحميل: {chosen}")

    st.divider()
    st.markdown("### هوية الشركة")
    company = st.text_input("Company Name", value="UAV Safety Lab")
    tagline = st.text_input("Tagline", value="Operational Risk Intelligence for UAV Missions")


# =========================
# Header
# =========================
st.markdown(f"# {APP_TITLE}")
st.markdown(f"<div class='muted'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='small'>Brand: <b>{company}</b> — {tagline}</div>",
    unsafe_allow_html=True
)
st.divider()

# Warmup cached resources
_warmup_resources()


# =========================
# Layout
# =========================
col_left, col_right = st.columns([1.05, 1.0], gap="large")

with col_left:
    st.markdown("## Scenario JSON")
    st.caption("ألصق الـ payload (JSON) ثم اضغط Run. النظام سيستدعي Stage-2 مباشرة من بايثون بدون API.")

    raw = st.text_area(
        "Paste your payload هنا",
        value=st.session_state.active_payload,
        height=540,
    )

    # Keep in session
    st.session_state.active_payload = raw

    btn_cols = st.columns([1, 1, 2])
    with btn_cols[0]:
        run_now = st.button("Run Stage-2", type="primary")
    with btn_cols[1]:
        pretty = st.button("Format JSON")
    with btn_cols[2]:
        st.caption("Tip: استعمل Format إذا لصقت JSON غير منسق.")

    if pretty:
        try:
            parsed = json.loads(raw)
            st.session_state.active_payload = json.dumps(parsed, indent=2)
            st.rerun()
        except Exception as e:
            st.error(f"لا يمكن تنسيق JSON: {e}")

with col_right:
    st.markdown("## Result")

    # If no run yet, show placeholder dashboard from last history
    last_run = st.session_state.history[-1] if st.session_state.history else None

    if run_now:
        # 1) Parse JSON
        try:
            payload = json.loads(st.session_state.active_payload)
        except json.JSONDecodeError as e:
            st.error(f"JSON غير صحيح: {e}")
            st.stop()

        # 2) Validate schema minimally
        if not isinstance(payload, dict) or "scenario" not in payload or not isinstance(payload["scenario"], dict):
            st.error("الـ payload لازم يكون dict ويحتوي على مفتاح scenario كقاموس.")
            st.stop()

        # 3) Run Stage-2 directly (NO requests)
        t0 = time.time()
        try:
            resp = run_stage2_report(
                scenario=payload["scenario"],
                artifacts_dir=artifacts_dir,
            )
        except Exception as e:
            st.error("Stage-2 فشل أثناء التنفيذ. التفاصيل بالأسفل:")
            st.exception(e)
            st.stop()
        latency = time.time() - t0

        resp_json = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)  # safe
        metrics = _extract_core_metrics(resp_json)

        run_record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "latency_s": round(float(latency), 3),
            "payload": payload,
            "response": resp_json,
            "metrics": metrics,
        }
        st.session_state.history.append(run_record)
        last_run = run_record

    if not last_run:
        st.info("لم يتم تشغيل أي تقييم بعد. أدخل Scenario ثم اضغط Run Stage-2.")
    else:
        m = last_run["metrics"]
        decision = m.get("decision", "UNKNOWN")

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Stage-2 Decision", decision)
        k2.metric("Stage-1 Class", str(m.get("stage1_pred", "—")))
        k3.metric("Risk Score", f"{m.get('stage1_score', '—')}")
        k4.metric("Latency (s)", f"{last_run.get('latency_s', '—')}")

        st.markdown(
            f"<div class='card'>"
            f"<div class='muted'>Decision Badge</div>"
            f"<div style='margin-top:8px;font-size:18px'>{_badge(decision)}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Stage-1 probabilities chart (if available)
        probs = _safe_get(last_run["response"], ["facts", "stage1", "probabilities"], {}) or {}
        fig = _plot_probability_bar(probs)
        if fig is not None:
            st.pyplot(fig, clear_figure=True)

        # Expanders: JSON, Snapshot, Markdown
        with st.expander("Stage-1 Snapshot", expanded=True):
            s1 = _safe_get(last_run["response"], ["facts", "stage1"], {}) or {}
            st.json({
                "predicted_class": s1.get("predicted_class"),
                "risk_score": s1.get("risk_score"),
                "confidence": s1.get("confidence"),
                "stage1_decision": s1.get("decision"),
            })

        with st.expander("Rules (Hard / Advisories)", expanded=False):
            rules = _safe_get(last_run["response"], ["facts", "rules"], {}) or {}
            st.write("**Hard Violations**")
            st.json(rules.get("hard_violations") or [])
            st.write("**Advisories**")
            st.json(rules.get("advisories") or [])

        report_md = last_run["response"].get("report_md")
        if report_md:
            with st.expander("Report (Markdown)", expanded=True):
                st.markdown(report_md)
        else:
            st.warning("لا يوجد report_md في الاستجابة.")

        with st.expander("Full JSON Response", expanded=False):
            st.json(last_run["response"])


# =========================
# History Dashboard
# =========================
st.divider()
st.markdown("## Dashboard — Run History")

if not st.session_state.history:
    st.caption("لا يوجد سجل بعد.")
else:
    df = _plot_history_table(st.session_state.history)

    # Quick filters
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        decision_filter = st.selectbox("Filter by Decision", options=["ALL"] + sorted(df["decision"].dropna().unique().tolist()))
    with f2:
        last_n = st.number_input("Show last N runs", min_value=1, max_value=200, value=min(20, len(df)))
    with f3:
        st.caption("يمكنك مقارنة النتائج بين التشغيلات بدون إعادة لصق السيناريو كل مرة.")

    df_view = df.copy()
    if decision_filter != "ALL":
        df_view = df_view[df_view["decision"] == decision_filter]
    df_view = df_view.tail(int(last_n))

    st.dataframe(df_view, use_container_width=True)

    # Simple trends
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("### Risk Score Trend")
        fig = plt.figure()
        # plot risk score if numeric
        y = pd.to_numeric(df_view["risk_score"], errors="coerce")
        plt.plot(y.values)
        plt.ylabel("Risk Score")
        plt.xlabel("Run Index (filtered)")
        plt.title("Risk Score over Runs")
        st.pyplot(fig, clear_figure=True)

    with tcol2:
        st.markdown("### Confidence Trend")
        fig = plt.figure()
        y = pd.to_numeric(df_view["confidence"], errors="coerce")
        plt.plot(y.values)
        plt.ylabel("Confidence")
        plt.xlabel("Run Index (filtered)")
        plt.title("Stage-1 Confidence over Runs")
        st.pyplot(fig, clear_figure=True)

    # Compare two runs
    st.markdown("### Compare Two Runs")
    idx_options = list(range(len(st.session_state.history)))
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("Run A", options=idx_options, index=max(0, len(idx_options) - 1))
    with c2:
        b = st.selectbox("Run B", options=idx_options, index=max(0, len(idx_options) - 2) if len(idx_options) > 1 else 0)

    run_a = st.session_state.history[a]
    run_b = st.session_state.history[b]

    ma = run_a["metrics"]
    mb = run_b["metrics"]

    comp_df = pd.DataFrame([
        {"metric": "decision", "A": ma.get("decision"), "B": mb.get("decision")},
        {"metric": "stage1_pred", "A": ma.get("stage1_pred"), "B": mb.get("stage1_pred")},
        {"metric": "risk_score", "A": ma.get("stage1_score"), "B": mb.get("stage1_score")},
        {"metric": "confidence", "A": ma.get("stage1_conf"), "B": mb.get("stage1_conf")},
        {"metric": "hard_count", "A": ma.get("hard_count"), "B": mb.get("hard_count")},
        {"metric": "advisories", "A": ma.get("adv_count"), "B": mb.get("adv_count")},
        {"metric": "completeness", "A": ma.get("completeness"), "B": mb.get("completeness")},
        {"metric": "latency_s", "A": run_a.get("latency_s"), "B": run_b.get("latency_s")},
    ])

    st.dataframe(comp_df, use_container_width=True)

st.divider()
st.info(
    "ملاحظة: هذه الواجهة لا تحتاج API ولا uvicorn. "
    "تشغّل Stage-2 مباشرة من بايثون. "
    "الخطوة القادمة: سنبني run.py لتشغيلها بأمر واحد فقط."
)
