"""
NETPULSE AI — Root Launcher
===========================
Launches the NETPULSE AI application on local or cloud environments (Render, Railway, Heroku).
"""

import os
import sys

# Add subfolder to sys.path
SUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "campus-network-optimizer")
sys.path.insert(0, SUB_DIR)

from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"NETPULSE AI Cloud Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
