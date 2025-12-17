# /root/run_all.py
import os
import subprocess
import time
from dotenv import load_dotenv

# -------- CONFIG ---------
PROJECT_DIR = "/root/chitraksh/stock_market_breakout_7"
VENV_ACTIVATE = os.path.join(PROJECT_DIR, "stock_market_env/bin/activate")
GET_ACCESS_SCRIPT = os.path.join(PROJECT_DIR, "access_token.py")
COLLECTOR_SCRIPT = os.path.join(PROJECT_DIR, "collector1.py")
SERVER_SCRIPT = os.path.join(PROJECT_DIR, "server30b.py")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# -------- SETUP ----------
os.makedirs(LOG_DIR, exist_ok=True)
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

print("📁 Changing directory to:", PROJECT_DIR)
os.chdir(PROJECT_DIR)

# -------- STEP 1: Get Access Token --------
print("\n🔑 Running get_access_token.py...")
subprocess.run(
    f"source {VENV_ACTIVATE} && python {GET_ACCESS_SCRIPT}",
    shell=True,
    executable="/bin/bash",
)

# -------- STEP 2: Start Collector & Server --------
print("\n🚀 Starting collector.py and server30b.py in background...")

collector_log = os.path.join(LOG_DIR, "collector.log")
server_log = os.path.join(LOG_DIR, "server.log")

collector_cmd = (
    f"source {VENV_ACTIVATE} && python {COLLECTOR_SCRIPT} >> {collector_log} 2>&1"
)
server_cmd = f"source {VENV_ACTIVATE} && python {SERVER_SCRIPT} >> {server_log} 2>&1"

collector_proc = subprocess.Popen(collector_cmd, shell=True, executable="/bin/bash")
server_proc = subprocess.Popen(server_cmd, shell=True, executable="/bin/bash")

print(f"✅ Collector running with PID {collector_proc.pid}")
print(f"✅ Server running with PID {server_proc.pid}")
print(f"🧾 Logs are being saved to {LOG_DIR}/")

try:
    while True:
        time.sleep(5)
except KeyboardInterrupt:
    print("\n🛑 Stopping processes...")
    collector_proc.terminate()
    server_proc.terminate()
    print("✅ All stopped.")
