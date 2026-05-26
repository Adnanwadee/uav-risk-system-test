from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any, Dict, List

import plotly.graph_objects as go
import requests
import streamlit as st

from uav_risk.ml.feature_defs import get_core_features, get_feature_definition


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
VALIDATE_PATH = "/api/flight/validate"
EVALUATE_PATH = "/v2/evaluate"

CORE_FEATURES = get_core_features()
CORE_GROUP_ORDER = [
    "uav",
    "mission",
    "controls",
    "swarm",
    "sim",
    "environment",
    "airspace",
    "landing",
    "traffic",
    "moving",
    "daa",
    "comms",
    "faults",
    "autofix",
    "other",
]


def render_shap_horizontal_chart(shap_data: List[Dict[str, Any]]) -> go.Figure:
    if not shap_data:
        figure = go.Figure()
        figure.add_annotation(text="No SHAP impact vectors recorded for this session.", showarrow=False)
        return figure

    feature_names = [item.get("feature_name", "Unknown") for item in shap_data]
    values = [float(item.get("shap_value", 0.0)) for item in shap_data]
    colors = ["#d1495b" if value > 0 else "#1d4ed8" for value in values]

    figure = go.Figure(
        go.Bar(
            x=values,
            y=feature_names,
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.4f}" for value in values],
            textposition="auto",
        )
    )
    figure.update_layout(
        title="تحليل SHAP",
        xaxis_title="قيمة تأثير الميزة",
        yaxis_title="اسم الميزة",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
        template="plotly_white",
    )
    return figure


def _group_name(feature_name: str) -> str:
    if feature_name.startswith("spawn_xyz_first"):
        return "mission"
    prefix = feature_name.split("_", 1)[0]
    return prefix if prefix in CORE_GROUP_ORDER else "other"


def _group_core_features() -> OrderedDict[str, List[str]]:
    grouped: OrderedDict[str, List[str]] = OrderedDict((group, []) for group in CORE_GROUP_ORDER)
    for feature_name in CORE_FEATURES:
        grouped.setdefault(_group_name(feature_name), []).append(feature_name)
    return OrderedDict((group, names) for group, names in grouped.items() if names)


def _feature_help(feature_name: str) -> str:
    definition = get_feature_definition(feature_name)
    description = definition.get("description", "")
    unit = definition.get("unit", "")
    if unit:
        return f"{description} [unit: {unit}]"
    return description


def _default_numeric_value(feature_name: str) -> float:
    if feature_name.endswith("_count") or feature_name.endswith("_steps"):
        return 0.0
    if feature_name.endswith("_ok") or feature_name.endswith("_enabled"):
        return 0.0
    return 0.0


def _render_numeric_feature(feature_name: str) -> float:
    definition = get_feature_definition(feature_name)
    unit = definition.get("unit", "")
    step = 1.0 if unit in {"count", "boolean", "Hz"} else 0.1
    min_value = 0.0
    if feature_name.endswith("_deg"):
        min_value = -360.0
    elif feature_name.endswith("_dbm") or feature_name.endswith("_dbm_min"):
        min_value = -200.0

    return float(
        st.number_input(
            feature_name,
            value=float(_default_numeric_value(feature_name)),
            step=step,
            min_value=min_value,
            help=_feature_help(feature_name),
            key=f"core::{feature_name}",
        )
    )


def _render_spawn_xyz_first() -> List[float]:
    col1, col2, col3 = st.columns(3)
    x = float(col1.number_input("spawn_xyz_first.x", value=0.0, step=0.1, key="core::spawn_xyz_first::x"))
    y = float(col2.number_input("spawn_xyz_first.y", value=0.0, step=0.1, key="core::spawn_xyz_first::y"))
    z = float(col3.number_input("spawn_xyz_first.z", value=0.0, step=0.1, key="core::spawn_xyz_first::z"))
    return [x, y, z]


def _build_payload(core_values: Dict[str, Any], sidebar_values: Dict[str, Any], secondary_overrides: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(core_values)
    payload.update(secondary_overrides)
    payload.update(
        {
            "flight_id": sidebar_values.get("flight_id") or None,
            "drone_profile_id": sidebar_values.get("drone_profile_id") or None,
            "drone_profile_name": sidebar_values.get("drone_profile_name") or None,
            "uav_model_id": sidebar_values.get("uav_model_id") or None,
            "uav_model_spec": sidebar_values.get("uav_model_spec") or None,
            "free_text": sidebar_values.get("free_text") or "",
            "timestamp": sidebar_values.get("timestamp") or None,
        }
    )
    return payload


def _parse_json_dict(raw_text: str, field_name: str) -> Dict[str, Any]:
    raw_text = raw_text.strip()
    if not raw_text:
        return {}
    parsed = json.loads(raw_text)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return parsed


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 90.0) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def main() -> None:
    st.set_page_config(
        page_title="ACE UAV Autonomous Risk Assessment Suite",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ACE UAV Autonomous Risk Assessment Suite")
    st.caption("واجهة تشغيل مباشرة متوافقة مع عقدة 68 core + overrides")

    with st.sidebar:
        st.subheader("Connection")
        api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)
        st.subheader("Flight Metadata")
        flight_id = st.text_input("Flight ID", value="")
        drone_profile_id = st.text_input("Drone profile ID", value="")
        drone_profile_name = st.text_input("Drone profile name", value="")
        uav_model_id = st.text_input("UAV model ID", value="")
        timestamp = st.text_input("Timestamp", value="")
        st.markdown("---")
        st.subheader("Optional model spec")
        uav_model_spec_raw = st.text_area("uav_model_spec JSON", value="{}", height=120)
        st.subheader("Free text")
        free_text = st.text_area("Notes for the ReAct agent", value="Standard inspection mission.", height=120)
        st.subheader("Secondary overrides JSON")
        secondary_overrides_raw = st.text_area("Overrides", value="{}", height=160)

    st.info("أدخل جميع ميزات الـ core المطلوبة، ثم نفّذ التحقق والتقييم من نفس النموذج.")

    grouped_features = _group_core_features()

    with st.form("flight_payload_form"):
        core_values: Dict[str, Any] = {}
        for group_name, feature_names in grouped_features.items():
            with st.expander(f"{group_name.upper()} features ({len(feature_names)})", expanded=group_name in {"uav", "mission"}):
                if group_name == "mission" and "spawn_xyz_first" in feature_names:
                    for feature_name in feature_names:
                        if feature_name == "spawn_xyz_first":
                            core_values[feature_name] = _render_spawn_xyz_first()
                        else:
                            core_values[feature_name] = _render_numeric_feature(feature_name)
                else:
                    cols = st.columns(2)
                    for index, feature_name in enumerate(feature_names):
                        with cols[index % 2]:
                            core_values[feature_name] = _render_numeric_feature(feature_name)

        submitted = st.form_submit_button("Validate and evaluate", type="primary", use_container_width=True)

    col1, col2 = st.columns([2, 1], gap="medium")
    with col2:
        st.subheader("Contract Summary")
        st.metric("Core feature count", len(CORE_FEATURES))
        st.metric("Group count", len(grouped_features))
        st.caption("spawn_xyz_first is rendered as a triplet and passed as a list.")

    if not submitted:
        return

    try:
        uav_model_spec = _parse_json_dict(uav_model_spec_raw, "uav_model_spec")
    except Exception as exc:
        st.error(f"Invalid uav_model_spec JSON: {exc}")
        return

    try:
        secondary_overrides = _parse_json_dict(secondary_overrides_raw, "secondary overrides")
    except Exception as exc:
        st.error(f"Invalid secondary overrides JSON: {exc}")
        return

    payload = _build_payload(
        core_values,
        {
            "flight_id": flight_id,
            "drone_profile_id": drone_profile_id,
            "drone_profile_name": drone_profile_name,
            "uav_model_id": uav_model_id,
            "uav_model_spec": uav_model_spec,
            "free_text": free_text,
            "timestamp": timestamp,
        },
        secondary_overrides,
    )

    with col1:
        with st.expander("Payload preview", expanded=False):
            st.json(payload)

    validate_url = api_base_url.rstrip("/") + VALIDATE_PATH
    evaluate_url = api_base_url.rstrip("/") + EVALUATE_PATH

    progress = st.progress(0, text="Sending validation request...")
    try:
        progress.progress(25, text="Validating contract against backend...")
        validation_response = _post_json(validate_url, {"payload": payload}, timeout=30.0)
    except Exception as exc:
        st.error(f"Validation request failed: {exc}")
        return

    if validation_response.status_code != 200:
        st.error(f"Validation endpoint returned {validation_response.status_code}")
        st.text(validation_response.text)
        return

    validation_json = validation_response.json()
    with st.expander("Validation response", expanded=True):
        st.json(validation_json)

    if not validation_json.get("is_usable", False):
        st.warning("Backend rejected the payload before evaluation. Fix the listed issues and try again.")
        return

    try:
        progress.progress(65, text="Running end-to-end evaluation...")
        evaluation_response = _post_json(evaluate_url, payload, timeout=120.0)
    except Exception as exc:
        st.error(f"Evaluation request failed: {exc}")
        return

    progress.progress(100, text="Evaluation completed.")

    if evaluation_response.status_code != 200:
        st.error(f"Evaluation endpoint returned {evaluation_response.status_code}")
        st.text(evaluation_response.text)
        return

    result_json = evaluation_response.json()
    st.success("Request processed successfully.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Decision", "Report", "SHAP", "Evidence", "Raw JSON"])

    with tab1:
        decision_state = str(result_json.get("decision", "NO-GO")).upper()
        if decision_state == "GO":
            st.success("GO")
        elif decision_state == "CONDITIONAL-GO":
            st.warning("CONDITIONAL-GO")
        else:
            st.error("NO-GO")
        st.metric("Risk score", f"{float(result_json.get('risk_score', 0.0)):.4f}")
        st.metric("Confidence", f"{float(result_json.get('confidence', 1.0)) * 100:.2f}%")
        for finding in result_json.get("critical_findings", []):
            st.error(str(finding))
        for recommendation in result_json.get("recommendations", []):
            st.info(str(recommendation))

    with tab2:
        st.markdown(result_json.get("report_markdown", "*No report returned.*"))

    with tab3:
        figure = render_shap_horizontal_chart(result_json.get("shap_explanation", []))
        st.plotly_chart(figure, use_container_width=True)

    with tab4:
        st.metric("Data quality score", f"{float(result_json.get('data_quality_score', 0.0)):.3f}")
        st.write("Legal citations")
        for citation in result_json.get("legal_citations", []):
            st.markdown(
                f"- **{citation.get('source_file', 'Unknown')}** page {citation.get('page_number', 0)}: {citation.get('full_text', citation.get('citation', ''))}"
            )

    with tab5:
        st.json(result_json)


if __name__ == "__main__":
    main()