"""
NETPULSE AI — Root Streamlit Entrypoint
=======================================
Redirects to or executes the campus-network-optimizer Streamlit app.
"""

import os
import sys

SUB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "campus-network-optimizer")
sys.path.insert(0, SUB_DIR)
os.chdir(SUB_DIR)

# Execute the inner app.py
with open(os.path.join(SUB_DIR, "app.py"), "r", encoding="utf-8") as f:
    exec(f.read())
