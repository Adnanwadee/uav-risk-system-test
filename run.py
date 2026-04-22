import subprocess
import sys
import time

def run_api():
    return subprocess.Popen([
        sys.executable,
        "-m",
        "uvicorn",
        "uav_risk.api.main:app",
        "--reload"
    ])

def run_ui():
    return subprocess.Popen([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "ui/app.py"
    ])

if __name__ == "__main__":
    print("🚀 Starting UAV Risk System...")

    api = run_api()
    time.sleep(2)  # نعطي API وقت يشتغل

    ui = run_ui()

    try:
        api.wait()
        ui.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        api.terminate()
        ui.terminate()
