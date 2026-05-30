run-api:
	PYTHONPATH=src uvicorn uav_risk.api.main:app --reload --host 127.0.0.1 --port 8000

# STAGE6_CLEANUP_REVIEW:
# Classification: MAKEFILE_TARGET_LEGACY_REVIEW
# Runtime status: run-ui points to legacy ui/app.py and is not current backend readiness evidence.
# Replacement: Use documented FastAPI backend endpoints and frontend handoff contract.
# Action rule: Do not use run-ui/run as final demo validation until UI target is replaced or removed.
run-ui:
	PYTHONPATH=src streamlit run ui/app.py --server.port 8501

run:
	@echo "Starting API on http://127.0.0.1:8000"
	@echo "Starting UI  on http://127.0.0.1:8501"
	@make -j2 run-api run-ui
