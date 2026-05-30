# STAGE6_CLEANUP_REVIEW:
# Classification: LEGACY_BROKEN_UI_ENTRYPOINT
# Runtime status: Streamlit entrypoint is not part of current backend/frontend API contract.
# Legacy signal: Imports uav_risk.ui.streamlit_app, which is not present in the current Python inventory.
# Replacement: Current frontend should consume documented FastAPI Stage2 assessment/profile endpoints.
# Action rule: Do not use as UI readiness evidence. Review together with Makefile streamlit target.
from uav_risk.ui.streamlit_app import main


main()