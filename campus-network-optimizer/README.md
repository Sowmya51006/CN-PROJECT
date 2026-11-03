# ⚡ NETPULSE AI — Network Performance Analysis System

> **"See Your Network. Predict Problems. Optimize Performance."**  
> Operational Network Intelligence Platform · Digital Control Room Theme  
> Built with Python · Streamlit · Scikit-learn · Plotly

---

## 🎯 Primary Operational Workflow

NETPULSE AI is designed as a direct, functional network engineering workspace:

```text
INPUT DATA ──> ANALYZE NETWORK (BUTTON) ──> ML PREDICTION ──> OUTPUT RESULT ──> RECOMMENDATIONS
```

A user can enter measurements or pick a benchmark dataset record, click **ANALYZE NETWORK**, and immediately receive an ML-based condition diagnosis alongside a maximum of 3 targeted engineering recommendations.

---

## 🎛️ 4-Page Streamlined Navigation

### 1. ⚡ Network Analyzer (Main Operational Screen)
- **Immediate Input Form:** No long introductions, no charts before analysis.
- **Option A (Manual Input):** Large numeric input boxes with dataset defaults and one-click presets (`🟢 Healthy Normal`, `🟠 High Latency`, `🔴 Packet Loss`, `🟣 High Jitter`, `🟡 Gateway Degraded`).
  - Average Latency / RTT (ms)
  - Packet Loss (%)
  - Jitter (ms)
  - Download Speed (Mbps)
  - Upload Speed (Mbps)
  - Gateway Router RTT (ms)
  - DNS Resolution Time (ms)
  - HTTP Response Time (ms)
  - Wi-Fi Signal Strength (%)
- **Option B (Use Dataset Record):** Select real telemetry samples directly from the 54,617-row dataset (`Normal`, `HighLatency`, `PacketLoss`, `HighJitter`, `GatewayDegraded`, or custom row index).
- **⚡ ANALYZE NETWORK Button:** Prominent, high-contrast action button that invokes the trained Random Forest classifier.
- **NETWORK ANALYSIS RESULT Panel:**
  - Kicker: `NETWORK CONDITION`
  - Large prediction card in 46px font (`NORMAL` in Green, `HIGH LATENCY` / `HIGH JITTER` / `GATEWAY DEGRADED` in Amber, `PACKET LOSS` / `OUTAGE` in Red).
  - Model confidence percentage.
  - Recapped telemetry measurements strip.
  - Plain-English operational impact explanation.
- **RECOMMENDED ACTIONS (Max 3):**
  - **Identified Issue**
  - **Why It Matters**
  - **What to Investigate or Improve**

### 2. 🗄️ Dataset
- Upload custom CSV or Parquet telemetry files.
- Dataset record count (`54,617`), telemetry features (`67`), and ground truth categories.
- Filter, search, and view records in a large readable table.
- **⬇️ Download Cleaned CSV** button.

### 3. 📡 Analytics (Only 2 Focused Charts)
- 4 top summary metrics (Average Latency, Packet Loss, Download Bandwidth, Normal Ratio).
- **Chart 1:** Longitudinal Network Latency (RTT) Trend (full-width interactive line chart).
- **Chart 2:** Network Condition Distribution (bar chart of normal vs anomalous events).
- One-sentence takeaway under each chart explaining its operational significance.

### 4. 📖 About
- Project overview, core objectives, and technology stack.
- Machine Learning evaluation details (**99.99% accuracy**, precision, recall, F1, and threshold learning explanation).
- Computer Networks 101 guide for beginners (Bandwidth, Latency, Throughput, Packet Loss, Jitter, QoS, VLAN, Bottlenecks with campus analogies).

---

## 🎨 Visual Design & Typography Standard

- **Theme Palette:**
  - Main Background: `#080D1F`
  - Sidebar: `#0D1730`
  - Panels & Cards: `#111F3A`
  - Secondary Panel: `#162747`
  - Electric Cyan: `#00D9FF`
  - Neon Blue: `#3685FF`
  - Aurora Violet: `#9B6DFF`
  - Text Main: `#F4F7FF`
  - Secondary Text: `#C0CBE0`
  - Normal Green: `#36D6A0`
  - Warning Amber: `#FFBD59`
  - Critical Red: `#FF5D75`
- **Typography Scale:**
  - Main Title: **46 px**
  - Section Headings: **28 px**
  - Input Labels: **19 px**
  - Input Values: **22 px**
  - Metric Numbers: **46 px**
  - Body Text: **18 px**
  - Navigation Labels: **18 px**
  - Chart Labels: **15–16 px**

---

## 🚀 How to Run the Application

Navigate to the `campus-network-optimizer` directory:

```bash
cd campus-network-optimizer
```

### Option 1: Custom Neon Network Studio (Flask REST API + HTML/CSS/JS) — Recommended
Runs the high-contrast custom frontend with clean, mutually exclusive page routing and real-time ML inference:

```bash
python run.py
```

Then open your browser at:
👉 **`http://localhost:5000`**

### Option 2: Streamlit Operational Edition
Runs the Streamlit dashboard edition with fixed navigation and congestion controls:

```bash
streamlit run app.py
```

Then open your browser at:
👉 **`http://localhost:8501`**

---

## ⚡ Congestion Control & Mitigation Techniques Displayed
Upon clicking **ANALYZE NETWORK**, the system evaluates the diagnosed network condition and immediately displays tailored, industry-standard Computer Networks Congestion Control mechanisms:
1. **Active Queue Management (AQM):** FQ-CoDel (Fair Queueing with Controlled Delay) & RED (Random Early Detection) to eliminate bufferbloat without starving throughput.
2. **Explicit Congestion Notification (ECN - RFC 3168):** Intermediate router CE marking to prompt sender window throttling without packet drops.
3. **TCP BBR (Bottleneck Bandwidth and RTT):** Model-based TCP congestion avoidance operating at Kleinrock's optimal point.
4. **Traffic Shaping & Rate Limiting:** Token Bucket Filtering (TBF) and Hierarchical Token Bucket (HTB) for per-department bandwidth partitioning.
5. **Quality of Service (QoS) & Queuing:** Strict Priority Queuing (PQ) + Weighted Fair Queuing (WFQ) with DiffServ DSCP marking (Expedited Forwarding EF 46 for VoIP).
