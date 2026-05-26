run-api:
	PYTHONPATH=src uvicorn uav_risk.api.main:app --reload --host 127.0.0.1 --port 8000

run-ui:
	PYTHONPATH=src streamlit run ui/app.py --server.port 8501

run:
	@echo "Starting API on http://127.0.0.1:8000"
	@echo "Starting UI  on http://127.0.0.1:8501"
	@make -j2 run-api run-ui
