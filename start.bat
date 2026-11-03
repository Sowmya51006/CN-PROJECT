@echo off
title NETPULSE AI - Control Room Server
cd /d "%~dp0"
if exist campus-network-optimizer cd campus-network-optimizer
echo =======================================================
echo NETPULSE AI -- Digital Control Room Server
echo Opening application at http://localhost:5000 ...
echo =======================================================
start "" http://localhost:5000
python run.py
pause

