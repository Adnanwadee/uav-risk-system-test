"""
ACE System Launcher (V2.0)
==========================
Orchestrates the API and UI processes simultaneously.
"""

import subprocess
import sys
import time
import os

# تكوين المنافذ (Ports)
API_PORT = 8000
UI_PORT = 8501

def run_api():
    print(f"📡 Launching ACE API on port {API_PORT}...")
    # نستخدم uvicorn لتشغيل محرك المرحلة الأولى والثانية
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "uav_risk.api.main:app", 
        "--host", "0.0.0.0", 
        "--port", str(API_PORT),
        "--reload"
    ])

def run_ui():
    print(f"🎨 Launching ACE Dashboard on port {UI_PORT}...")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "ui/app.py", 
        "--server.port", str(UI_PORT)
    ])

if __name__ == "__main__":
    # تنظيف المتغيرات البيئية (اختياري)
    # os.environ["GROQ_API_KEY"] = "your_key_here"

    print("🚀 Starting ACE (Autonomous Control Engine) Ecosystem...")
    
    api_process = run_api()
    time.sleep(3)  # انتهاء تحميل نماذج Stage 1 الثقيلة
    
    ui_process = run_ui()

    try:
        while True:
            time.sleep(1)
            # التأكد من أن العمليات لا تزال تعمل
            if api_process.poll() is not None:
                print("❌ API Process died. Restarting...")
                api_process = run_api()
            if ui_process.poll() is not None:
                print("❌ UI Process died. Restarting...")
                ui_process = run_ui()
                
    except KeyboardInterrupt:
        print("\n🛑 ACE System Shutdown Initiated...")
        api_process.terminate()
        ui_process.terminate()
        print("✅ All processes halted.")