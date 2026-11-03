"""
run.py — NETPULSE AI Application Launcher
=========================================
Run this file to launch the NETPULSE AI Web Application:
    python run.py

Server starts at: http://localhost:5000
"""

import os
import sys

# Configure UTF-8 encoding on Windows to prevent CP1252 crash
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 55)
    print("NETPULSE AI -- Network Performance Analysis System")
    print(f"Server Running at: http://localhost:{port}")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)

