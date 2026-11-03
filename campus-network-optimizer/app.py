"""
app.py — NETPULSE AI (Operational Workspace Edition)
===================================================
Network Performance Analysis & Prediction System

Workflow:
  INPUT DATA -> ANALYZE NETWORK -> ML PREDICTION -> OUTPUT RESULT -> OPTIMIZATION RECOMMENDATIONS

Run:
  streamlit run app.py
"""

import os
import sys
import io
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader     import load_dataset, LOCAL_CSV
from src.data_cleaning   import clean_dataset, get_cleaning_summary
from src.train_model     import load_model, model_is_trained
from src.prediction      import predict_network_condition

# ═══════════════════════════════════════════════════════════════
# 1. PAGE SETUP & DIGITAL CONTROL ROOM STYLES
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NETPULSE AI — Network Performance Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
_css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(_css_path):
    with open(_css_path, encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 2. DESIGN THEME CONSTANTS
# ═══════════════════════════════════════════════════════════════
CYAN      = "#00D9FF"
BLUE      = "#3685FF"
VIOLET    = "#9B6DFF"
GREEN     = "#36D6A0"
AMBER     = "#FFBD59"
RED       = "#FF5D75"
TEXT_MAIN = "#F4F7FF"
TEXT_SUB  = "#C0CBE0"
BG_CARD   = "#111F3A"
BG_PANEL  = "#162747"
BORDER    = "#24385D"

def _plot_layout(title="", height=360, **kwargs):
    """Plotly layout matching the Full-Screen Dashboard standards."""
    return dict(
        title=dict(
            text=title,
            font=dict(color=TEXT_MAIN, size=15, family="Inter, sans-serif"),
            x=0.01,
            xanchor="left",
        ),
        paper_bgcolor=BG_CARD,
        plot_bgcolor="#060B18",
        font=dict(color=TEXT_SUB, family="Inter, sans-serif", size=12),
        height=height,
        margin=dict(l=50, r=20, t=35, b=45),
        xaxis=dict(
            gridcolor=BORDER,
            showgrid=True,
            zeroline=False,
            color=TEXT_SUB,
            tickfont=dict(size=11, color=TEXT_SUB),
            title_font=dict(size=13, color=TEXT_MAIN),
        ),
        yaxis=dict(
            gridcolor=BORDER,
            showgrid=True,
            zeroline=False,
            color=TEXT_SUB,
            tickfont=dict(size=11, color=TEXT_SUB),
            title_font=dict(size=13, color=TEXT_MAIN),
        ),
        hoverlabel=dict(
            bgcolor=BG_PANEL,
            font_size=13,
            font_family="Inter, sans-serif",
            bordercolor=CYAN,
        ),
        **kwargs,
    )

# ═══════════════════════════════════════════════════════════════
# 3. FAST CACHED DATA LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_dataset(uploaded_bytes=None, file_name=None):
    """Loads and caches cleaned dataset."""
    if uploaded_bytes is not None:
        bio = io.BytesIO(uploaded_bytes)
        if file_name and file_name.lower().endswith(".parquet"):
            raw = pd.read_parquet(bio)
        else:
            raw = pd.read_csv(bio, low_memory=False)
    else:
        raw = load_dataset()
    clean = clean_dataset(raw)
    return raw, clean

# Session State Initialization
if "df_raw" not in st.session_state or st.session_state.df_raw is None:
    try:
        r, c = get_dataset()
        st.session_state.df_raw = r
        st.session_state.df_clean = c
        st.session_state.data_loaded = True
    except Exception:
        st.session_state.df_raw = None
        st.session_state.df_clean = None
        st.session_state.data_loaded = False

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# ═══════════════════════════════════════════════════════════════
# 4. SIDEBAR NAVIGATION (4 Pages Only)
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="np-brand">
      <div class="brand-badge">⚡ NETPULSE AI</div>
      <div class="brand-title">
        <span class="glow-dot"></span> Campus Network
      </div>
      <div class="brand-sub">ML Performance Analyzer</div>
    </div>
    """, unsafe_allow_html=True)

    # Nav section label
    st.markdown('<div class="np-nav-label">Navigation</div>', unsafe_allow_html=True)

    NAV_PAGES = [
        "⚡  Network Analyzer",
        "🗄️  Dataset",
        "📡  Analytics",
        "📖  About",
    ]

    selected_nav = st.radio(
        "Navigation",
        NAV_PAGES,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border:none;border-top:1px solid #1C2E4A;margin:0.75rem 8px 0.6rem'>", unsafe_allow_html=True)

    # Quick System Specs
    if st.session_state.data_loaded and st.session_state.df_clean is not None:
        c_rows = len(st.session_state.df_clean)
        st.markdown('<div class="np-nav-label">System Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="padding:7px 10px;background:#0C1828;border:1px solid #1C2E4A;border-radius:6px;font-size:10px;color:#8899BB;line-height:2;margin:0 8px">
          <div style="display:flex;justify-content:space-between"><span>Status</span><span style="color:#36D6A0;font-weight:700">● Online</span></div>
          <div style="display:flex;justify-content:space-between"><span>Records</span><span style="color:#EDF1FF;font-weight:600">{c_rows:,}</span></div>
          <div style="display:flex;justify-content:space-between"><span>Model</span><span style="color:#EDF1FF;font-weight:600">Rand Forest</span></div>
          <div style="display:flex;justify-content:space-between"><span>State</span><span style="color:#00D9FF;font-weight:700">Ready</span></div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 5. PAGE 1 — NETWORK ANALYZER (THE MAIN OPERATIONAL WORKSPACE)
# ═══════════════════════════════════════════════════════════════
def page_network_analyzer():
    # Large Heading & Subtitle
    st.markdown("""
    <h1 class="np-main-title">NETPULSE AI</h1>
    <div class="np-main-sub">Analyze network performance and identify problems using Machine Learning.</div>
    """, unsafe_allow_html=True)

    df = st.session_state.df_clean

    # Input Mode Selection
    input_mode = st.radio(
        "Choose Input Method:",
        ["Option A: Manual Input", "Option B: Use Dataset Record"],
        horizontal=True,
        label_visibility="collapsed",
    )

    user_inputs = {}

    if input_mode == "Option A: Manual Input":
        # Quick Presets Bar
        st.markdown("""
        <div style="font-size:16px;font-weight:700;color:#C0CBE0;margin-bottom:8px">
          QUICK PRESET PROFILES (Click to auto-populate test cases):
        </div>
        """, unsafe_allow_html=True)

        p_cols = st.columns(4)
        with p_cols[0]:
            if st.button("🟢 Healthy Normal", use_container_width=True):
                st.session_state["in_rtt"] = 11.5
                st.session_state["in_loss"] = 0.0
                st.session_state["in_jit"] = 0.8
                st.session_state["in_down"] = 95.0
                st.session_state["in_up"] = 35.0
                st.session_state["in_gw"] = 0.5
                st.session_state["in_dns"] = 14.0
                st.session_state["in_http"] = 120.0
                st.session_state["in_wifi"] = 88.0
        with p_cols[1]:
            if st.button("🟠 High Latency", use_container_width=True):
                st.session_state["in_rtt"] = 72.0
                st.session_state["in_loss"] = 0.0
                st.session_state["in_jit"] = 1.4
                st.session_state["in_down"] = 45.0
                st.session_state["in_up"] = 15.0
                st.session_state["in_gw"] = 1.0
                st.session_state["in_dns"] = 45.0
                st.session_state["in_http"] = 320.0
                st.session_state["in_wifi"] = 75.0
        with p_cols[2]:
            if st.button("🔴 Packet Loss", use_container_width=True):
                st.session_state["in_rtt"] = 14.0
                st.session_state["in_loss"] = 8.5
                st.session_state["in_jit"] = 1.2
                st.session_state["in_down"] = 20.0
                st.session_state["in_up"] = 8.0
                st.session_state["in_gw"] = 0.8
                st.session_state["in_dns"] = 28.0
                st.session_state["in_http"] = 280.0
                st.session_state["in_wifi"] = 80.0
        with p_cols[3]:
            if st.button("🟣 High Jitter", use_container_width=True):
                st.session_state["in_rtt"] = 22.0
                st.session_state["in_loss"] = 0.0
                st.session_state["in_jit"] = 28.0
                st.session_state["in_down"] = 50.0
                st.session_state["in_up"] = 20.0
                st.session_state["in_gw"] = 1.2
                st.session_state["in_dns"] = 22.0
                st.session_state["in_http"] = 180.0
                st.session_state["in_wifi"] = 65.0

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # 3 Clean Columns of Individual Input Boxes (1 Box Per Input)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
            user_inputs["rtt_avg_ms"] = st.number_input(
                "⏱️ Average Latency / RTT (ms)",
                min_value=0.0, max_value=2000.0,
                value=float(st.session_state.get("in_rtt", 12.0)),
                step=0.5, format="%.1f",
                help="Round-Trip Time in milliseconds. Normal is < 30 ms."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
            user_inputs["packet_loss_pct"] = st.number_input(
                "📦 Packet Loss (%)",
                min_value=0.0, max_value=100.0,
                value=float(st.session_state.get("in_loss", 0.0)),
                step=0.1, format="%.2f",
                help="Percentage of dropped packets. Normal is 0.00%."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
            user_inputs["jitter_ms"] = st.number_input(
                "📈 Jitter (ms)",
                min_value=0.0, max_value=500.0,
                value=float(st.session_state.get("in_jit", 1.0)),
                step=0.2, format="%.2f",
                help="Variation in packet arrival time. Crucial for VoIP/video."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
            user_inputs["download_mbps"] = st.number_input(
                "⬇️ Download Speed (Mbps)",
                min_value=0.0, max_value=2000.0,
                value=float(st.session_state.get("in_down", 95.0)),
                step=1.0, format="%.1f",
                help="Downstream throughput from network gateway."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
            user_inputs["upload_mbps"] = st.number_input(
                "⬆️ Upload Speed (Mbps)",
                min_value=0.0, max_value=1000.0,
                value=float(st.session_state.get("in_up", 30.0)),
                step=0.5, format="%.1f",
                help="Upstream bandwidth capacity."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
            user_inputs["wifi_signal_pct"] = st.number_input(
                "📶 Wi-Fi Signal Strength (%)",
                min_value=0.0, max_value=100.0,
                value=float(st.session_state.get("in_wifi", 85.0)),
                step=1.0, format="%.0f",
                help="Wireless access point reception quality."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
            user_inputs["gateway_rtt_avg_ms"] = st.number_input(
                "🔀 Gateway Router RTT (ms)",
                min_value=0.0, max_value=500.0,
                value=float(st.session_state.get("in_gw", 0.5)),
                step=0.2, format="%.2f",
                help="Ping latency to local router hop. Normal is < 2.0 ms."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
            user_inputs["dns_resolve_ms"] = st.number_input(
                "🌐 DNS Resolution Time (ms)",
                min_value=0.0, max_value=1000.0,
                value=float(st.session_state.get("in_dns", 15.0)),
                step=0.5, format="%.1f",
                help="Duration to resolve hostnames to IP addresses."
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
            user_inputs["http_ms"] = st.number_input(
                "⚡ HTTP Response Time (ms)",
                min_value=0.0, max_value=5000.0,
                value=float(st.session_state.get("in_http", 120.0)),
                step=5.0, format="%.1f",
                help="Time to first byte received from server."
            )
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Option B: Use Dataset Record
        st.markdown("<div class='np-input-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:18px;font-weight:700;color:#F4F7FF;margin-bottom:12px'>SELECT A BENCHMARK TELEMETRY RECORD FROM DATASET:</div>", unsafe_allow_html=True)

        SAMPLE_PRESETS = [
            ("Record #5 — Normal Baseline", "Normal", 5),
            ("Record #102 — High Latency Spike", "HighLatency", 102),
            ("Record #504 — Packet Loss Drop Event", "PacketLoss", 504),
            ("Record #88 — High Jitter VoIP Fluctuation", "HighJitter", 88),
        ]

        preset_labels = [p[0] for p in SAMPLE_PRESETS] + ["Custom Row Index from Dataset"]
        selected_preset = st.selectbox("Record Preset", preset_labels, index=0)

        if selected_preset == "Custom Row Index from Dataset":
            max_idx = len(df) - 1 if df is not None else 50000
            rec_idx = st.number_input("Enter Row Index (0 to " + str(max_idx) + ")", min_value=0, max_value=max_idx, value=0, step=1)
        else:
            rec_idx = next(p[2] for p in SAMPLE_PRESETS if p[0] == selected_preset)

        if df is not None and rec_idx < len(df):
            selected_row = df.iloc[rec_idx]
            user_inputs = selected_row.to_dict()

            st.markdown(f"""
            <div style="background:#162747;border:1px solid #24385D;border-radius:8px;padding:16px 20px;margin-top:12px">
              <div style="font-size:16px;color:#00D9FF;font-weight:700;margin-bottom:8px">
                LOADED RECORD #{rec_idx} &nbsp;|&nbsp; Ground Truth Label: <b>{selected_row.get('event_category', 'Unknown')}</b>
              </div>
              <div style="font-size:17px;color:#F4F7FF;line-height:1.8">
                • <b>Latency (RTT):</b> {selected_row.get('rtt_avg_ms', 0):.1f} ms &nbsp;&nbsp;|&nbsp;&nbsp;
                • <b>Packet Loss:</b> {selected_row.get('packet_loss_pct', 0):.2f}% &nbsp;&nbsp;|&nbsp;&nbsp;
                • <b>Jitter:</b> {selected_row.get('jitter_ms', 0):.2f} ms<br>
                • <b>Download:</b> {selected_row.get('download_mbps', 0):.1f} Mbps &nbsp;&nbsp;|&nbsp;&nbsp;
                • <b>Gateway RTT:</b> {selected_row.get('gateway_rtt_avg_ms', 0):.2f} ms &nbsp;&nbsp;|&nbsp;&nbsp;
                • <b>Wi-Fi Signal:</b> {selected_row.get('wifi_signal_pct', 0):.0f}%
              </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # STEP 2 — PROMINENT ANALYZE NETWORK BUTTON
    # ═══════════════════════════════════════════════════════════════
    st.markdown("<div class='ctrl-btn-analyze'>", unsafe_allow_html=True)
    clicked_analyze = st.button("⚡ ANALYZE NETWORK", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if clicked_analyze:
        with st.spinner("Analyzing network measurements with Random Forest ML..."):
            try:
                # Validation check
                if user_inputs.get("rtt_avg_ms", 0.0) < 0 or user_inputs.get("packet_loss_pct", 0.0) < 0:
                    st.error("Error: Latency and packet loss cannot be negative values.")
                else:
                    res = predict_network_condition(user_inputs)
                    st.session_state.analysis_result = res
            except Exception as ex:
                st.error(f"Inference error: {ex}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3 — OUTPUT SECTION (Only appears after analysis)
    # ═══════════════════════════════════════════════════════════════
    if st.session_state.analysis_result is not None:
        res = st.session_state.analysis_result
        p_class = res.get("predicted_class", "Normal")
        conf_pct = res.get("top_class_prob", 0.98) * 100

        if p_class == "GatewayDegraded":
            p_class = "HighLatency"

        # Class visual styling
        CLASS_CONFIG = {
            "Normal": {
                "theme": "res-normal", "color": GREEN, "icon": "✅",
                "title": "NORMAL OPERATING CONDITION",
                "summary": "Network telemetry demonstrates nominal performance. Packet loss is zero, round-trip times are stable within campus boundaries, and bandwidth queues are clear.",
            },
            "HighLatency": {
                "theme": "res-warning", "color": AMBER, "icon": "⚠️",
                "title": "HIGH LATENCY DETECTED",
                "summary": "Elevated round-trip time detected across the connection. Users will experience noticeable buffering and delays in real-time learning systems and video calls.",
            },
            "PacketLoss": {
                "theme": "res-critical", "color": RED, "icon": "🚨",
                "title": "PACKET LOSS DETECTED",
                "summary": "Data transmission packets are being dropped before reaching destination. Causes repeated TCP retransmissions, drastically reducing usable throughput.",
            },
            "HighJitter": {
                "theme": "res-warning", "color": VIOLET, "icon": "⚠️",
                "title": "HIGH JITTER / DELAY VARIANCE",
                "summary": "Unstable packet delivery timing detected. Highly damaging to VoIP calls, live broadcasts, and online interactive video.",
            },
            "Outage": {
                "theme": "res-critical", "color": RED, "icon": "🚨",
                "title": "SEVERE NETWORK OUTAGE",
                "summary": "Critical connectivity failure detected. Transmission path is severed or dropping virtually all communications.",
            },
        }

        cfg = CLASS_CONFIG.get(p_class, {
            "theme": "res-normal", "color": CYAN, "icon": "ℹ️",
            "title": p_class, "summary": "Custom network event state.",
        })

        st.markdown("<h2 class=\"np-section-heading\" style=\"margin-top:2.4rem\">NETWORK ANALYSIS RESULT</h2>", unsafe_allow_html=True)

        rtt_val  = user_inputs.get("rtt_avg_ms", 12.0)
        loss_val = user_inputs.get("packet_loss_pct", 0.0)
        jit_val  = user_inputs.get("jitter_ms", 1.0)
        down_val = user_inputs.get("download_mbps", 95.0)
        gw_val   = user_inputs.get("gateway_rtt_avg_ms", 0.5)

        # Overall Performance Rating
        is_good = (p_class == "Normal") and (loss_val < 1.0) and (rtt_val <= 35.0) and (jit_val <= 8.0) and (down_val >= 20.0)
        perf_badge_class = "badge-good" if is_good else ("badge-poor-warning" if p_class in ["HighLatency", "HighJitter"] else "badge-poor")
        perf_text = "GOOD PERFORMANCE" if is_good else "POOR PERFORMANCE"
        perf_icon = "🟢" if is_good else "🔴"
        if is_good:
            perf_sub = "Optimal network health. Telemetry demonstrates low latency, zero packet loss, and robust transmission capacity."
        elif loss_val >= 1.0 or p_class == "PacketLoss":
            perf_sub = "Significant packet drop detected. High retransmission overhead severely degrades throughput."
        elif rtt_val > 35.0 or p_class == "HighLatency":
            perf_sub = "Elevated round-trip latency detected. Causes noticeable delay and buffering in interactive applications."
        elif jit_val > 8.0 or p_class == "HighJitter":
            perf_sub = "High delay jitter detected. Produces erratic packet arrivals and disrupts real-time conferencing."
        else:
            perf_sub = "Sub-optimal throughput or transmission metrics detected below nominal operational baselines."

        st.markdown(f"""
        <div class="np-result-card {cfg['theme']}">
          <!-- PERFORMANCE RATING BADGE -->
          <div class="perf-status-wrapper">
            <div class="perf-status-kicker">NETWORK PERFORMANCE RATING</div>
            <div class="perf-status-badge {perf_badge_class}">
              <span class="perf-status-icon">{perf_icon}</span>
              <span class="perf-status-text">{perf_text}</span>
            </div>
            <div class="perf-status-desc">{perf_sub}</div>
          </div>

          <div class="output-divider"></div>

          <div class="res-kicker">PREDICTED NETWORK CONDITION</div>
          <div class="res-class" style="color:{cfg['color']}">{cfg['icon']} {cfg['title']}</div>
          <div class="res-conf">Model Confidence: <b>{conf_pct:.1f}%</b></div>
          <div class="np-recap-strip">
            <span class="np-recap-item">Latency (RTT): <b>{rtt_val:.1f} ms</b></span>
            <span class="np-recap-item">Packet Loss: <b>{loss_val:.2f}%</b></span>
            <span class="np-recap-item">Jitter: <b>{jit_val:.2f} ms</b></span>
            <span class="np-recap-item">Download: <b>{down_val:.1f} Mbps</b></span>
            <span class="np-recap-item">Gateway RTT: <b>{gw_val:.2f} ms</b></span>
          </div>
          <div class="res-summary">
            <b>Diagnosis Summary:</b> {cfg['summary']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════
        # STEP 4 — RECOMMENDATIONS (Maximum of 3 Useful Recommendations)
        # ═══════════════════════════════════════════════════════════════
        st.markdown("<h2 class=\"np-section-heading\">RECOMMENDED ACTIONS</h2>", unsafe_allow_html=True)

        recs = []

        # High Latency Recommendation
        if rtt_val > 35.0 or p_class == "HighLatency":
            recs.append({
                "problem": "Elevated Round-Trip Latency",
                "why": "High latency causes noticeable lag during interactive sessions, slow web loading, and voice/video lag.",
                "action": "Investigate routing bottlenecks and uplink saturation. Check for bufferbloat on core switches, verify ISP link health, and prioritize real-time traffic using QoS.",
                "severity": "act-warning",
                "icon": "⏱️",
            })

        # Packet Loss Recommendation
        if loss_val > 1.0 or p_class == "PacketLoss":
            recs.append({
                "problem": "Data Packet Drop & Retransmission",
                "why": "Packet loss forces TCP into congestion control collapse, throttling speed and dropping connections.",
                "action": "Inspect physical cabling, patch cords, and fiber optic links for CRC error counts. Check switch port duplex settings and test for interface queue drops.",
                "severity": "act-critical",
                "icon": "📦",
            })

        # High Jitter Recommendation
        if jit_val > 10.0 or p_class == "HighJitter":
            recs.append({
                "problem": "Packet Arrival Timing Variance (Jitter)",
                "why": "Inconsistent packet arrival produces robotic audio, choppy video streams, and dropped frames in calls.",
                "action": "Configure Quality of Service (QoS) Differentiated Services (DiffServ/DSCP) on edge switches to guarantee jitter buffers for voice and video streams.",
                "severity": "act-warning",
                "icon": "📈",
            })

        # Gateway or Throughput Recommendation
        if gw_val > 5.0 or p_class == "GatewayDegraded":
            recs.append({
                "problem": "Campus Gateway Hop Delay",
                "why": "The local router is introducing delay before traffic reaches the WAN provider, indicating local equipment stress.",
                "action": "Inspect CPU and memory utilization on the core edge router. Review NAT connection table size and consider upgrading firmware or edge switch backplane.",
                "severity": "act-warning",
                "icon": "🔀",
            })

        if down_val < 25.0 and len(recs) < 3:
            recs.append({
                "problem": "Restricted Bandwidth Capacity",
                "why": "Low throughput limits multiple concurrent students and faculty from downloading educational assets.",
                "action": "Identify bandwidth-heavy processes running during peak academic hours. Apply bandwidth rate-limiting policies and offload updates to off-peak periods.",
                "severity": "act-warning",
                "icon": "⬇️",
            })

        # Normal condition recommendation
        if not recs or p_class == "Normal":
            recs = [
                {
                    "problem": "Nominal Baseline Performance",
                    "why": "All core metrics meet or exceed optimal networking standards for university operations.",
                    "action": "Maintain routine continuous monitoring. Preserve current Quality of Service (QoS) queue allocations and schedule periodic Wi-Fi channel scans.",
                    "severity": "act-normal",
                    "icon": "🛡️",
                }
            ]

        # Limit to strictly 3 recommendations
        for rec in recs[:3]:
            st.markdown(f"""
            <div class="np-action-card {rec['severity']}">
              <div class="act-title">{rec['icon']} {rec['problem']}</div>
              <div class="act-item"><b>1. Identified Issue:</b> {rec['problem']}</div>
              <div class="act-item"><b>2. Why It Matters:</b> {rec['why']}</div>
              <div class="act-item"><b>3. What to Investigate:</b> {rec['action']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:15px;color:#C0CBE0;margin-top:10px;padding:8px 12px;background:rgba(0,217,255,0.05);border-radius:6px">
          ℹ️ <b>Engineering Guidance:</b> Recommendations provide structured operational diagnostics and do not make automatic hardware changes to physical routers.
        </div>
        """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════════════════════
        # STEP 4 — CONGESTION CONTROL & MITIGATION TECHNIQUES
        # ═══════════════════════════════════════════════════════════════
        st.markdown("<h2 class=\"np-section-heading\" style=\"margin-top:1.8rem\">⚡ CONGESTION CONTROL & MITIGATION TECHNIQUES</h2>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:16px;color:#CBD5E1;margin-bottom:14px'>Industry-standard Computer Networks congestion management and queueing mechanisms matched to this diagnostic profile.</div>", unsafe_allow_html=True)

        # Build tailored congestion techniques
        cong_techs = []
        if p_class in ("HighLatency", "Outage") or rtt_val > 35.0:
            cong_techs = [
                {
                    "name": "Active Queue Management (AQM) — FQ-CoDel / RED",
                    "cat": "Queue Management",
                    "icon": "⚡",
                    "mech": "Monitors packet delay in switch buffers. When queue latency exceeds the 5 ms target, it drops/marks packets before bufferbloat causes persistent latency inflation.",
                    "impl": "Enable fq_codel on Linux gateways: tc qdisc add dev eth0 root fq_codel, or enable WRED on core switches."
                },
                {
                    "name": "TCP BBR (Bottleneck Bandwidth & RTT) Congestion Control",
                    "cat": "Transport Layer",
                    "icon": "🚀",
                    "mech": "Model-based congestion control that operates at Kleinrock's optimal operating point (max throughput, min delay). Does not wait for packet loss to throttle transmission, preventing buffer saturation.",
                    "impl": "Configure BBR on campus servers: sysctl -w net.ipv4.tcp_congestion_control=bbr"
                },
                {
                    "name": "Explicit Congestion Notification (ECN - RFC 3168)",
                    "cat": "IP / TCP Signaling",
                    "icon": "📡",
                    "mech": "Intermediate routers mark CE bits in the IP header instead of dropping packets. The receiver echoes ECE to sender, which reduces cwnd without loss.",
                    "impl": "Enable ECN negotiation across campus endpoints: sysctl -w net.ipv4.tcp_ecn=1"
                },
                {
                    "name": "Token Bucket Traffic Shaping (TBF / HTB)",
                    "cat": "Traffic Shaping",
                    "icon": "🎛️",
                    "mech": "Buffers bursty traffic and transmits packets at a steady calibrated rate using token replenishment, smoothing peaks that cause queue latency.",
                    "impl": "Deploy Hierarchical Token Bucket (HTB) on edge firewalls to shape per-department egress traffic."
                }
            ]
        elif p_class == "PacketLoss" or loss_val > 1.0:
            cong_techs = [
                {
                    "name": "Explicit Congestion Notification (ECN) & Early Throttling",
                    "cat": "Congestion Signaling",
                    "icon": "📡",
                    "mech": "Signals incipient queue saturation before packets are dropped. Senders cut cwnd gracefully, preventing retransmission storms that cause congestion collapse.",
                    "impl": "Activate ECN on edge switches and configure RED thresholds between 40% and 80% buffer depth."
                },
                {
                    "name": "TCP Selective Acknowledgment (SACK - RFC 2018) & Fast Recovery",
                    "cat": "TCP Loss Recovery",
                    "icon": "🔄",
                    "mech": "Enables receiver to acknowledge discontinuous packet blocks, allowing sender to retransmit only dropped segments rather than entire windows.",
                    "impl": "Verify SACK is active: sysctl -w net.ipv4.tcp_sack=1 and ensure MTU path discovery is enabled."
                },
                {
                    "name": "Random Early Detection (RED) Queue Tuning",
                    "cat": "Queue Management",
                    "icon": "⚖️",
                    "mech": "Drops packets with linearly increasing probability as average queue size grows, desynchronizing TCP flows so they do not back off in tandem.",
                    "impl": "Tune min_thresh (20%) and max_thresh (70%) on congested switch interface buffers."
                },
                {
                    "name": "Link Aggregation (LACP IEEE 802.3ad) & Load Balancing",
                    "cat": "Link Layer Capacity",
                    "icon": "🔗",
                    "mech": "Combines multiple physical ethernet links into a single logical channel, doubling backhaul capacity and eliminating single-link buffer overflow drops.",
                    "impl": "Bundle congested trunk ports into an 802.3ad LACP port-channel with Layer 3/4 hash distribution."
                }
            ]
        elif p_class == "HighJitter" or jit_val > 10.0:
            cong_techs = [
                {
                    "name": "Strict Priority Queuing (PQ) + Weighted Fair Queuing (WFQ)",
                    "cat": "Quality of Service (QoS)",
                    "icon": "🎯",
                    "mech": "Classifies incoming packets into prioritized hardware queues. Real-time RTP packets bypass lower-priority bulk queues, guaranteeing constant delivery intervals.",
                    "impl": "Assign DSCP Expedited Forwarding (EF / 46) to VoIP/video calls and configure Strict Priority on edge switch egress queues."
                },
                {
                    "name": "FQ-CoDel (Fair Queueing with Controlled Delay)",
                    "cat": "Queue Scheduling",
                    "icon": "⚡",
                    "mech": "Hashes traffic into hundreds of sub-queues and serves them via Deficit Round Robin, preventing elephant flows from introducing jitter into interactive mouse flows.",
                    "impl": "Deploy fq_codel or CAKE on wireless access point controllers and core distribution switches."
                },
                {
                    "name": "Adaptive Jitter Buffer Playout Tuning",
                    "cat": "Application Layer",
                    "icon": "⏱️",
                    "mech": "Dynamically adjusts audio/video playout delay based on packet inter-arrival variance estimation, smoothing choppy playback during bursts.",
                    "impl": "Configure WebRTC/VoIP gateway jitter buffers between 30 ms and 60 ms with dynamic expansion."
                },
                {
                    "name": "DiffServ DSCP Traffic Marking & Policing",
                    "cat": "Network Layer Policy",
                    "icon": "🏷️",
                    "mech": "Marks academic video and lecture streams with Assured Forwarding (AF41) and limits best-effort peer-to-peer traffic to 10% during academic hours.",
                    "impl": "Apply ACL-based QoS policy maps at campus building distribution switches."
                }
            ]
        elif gw_val > 8.0:
            cong_techs = [
                {
                    "name": "Hierarchical Token Bucket (HTB) Bandwidth Partitioning",
                    "cat": "Traffic Shaping",
                    "icon": "🏛️",
                    "mech": "Partitions total available WAN pipe into guaranteed bandwidth pools for Administration, Classrooms, and Wi-Fi, preventing any single subnet from overwhelming the gateway.",
                    "impl": "Implement Linux tc HTB classes on gateway egress interfaces with guaranteed and ceiling rates."
                },
                {
                    "name": "IEEE 802.3x Flow Control & Priority Flow Control (PFC)",
                    "cat": "Data Link Layer",
                    "icon": "⏸️",
                    "mech": "Gateway switch sends PAUSE frames back to upstream access switches when ingress buffers reach critical thresholds, briefly pausing transmission to prevent buffer overflow.",
                    "impl": "Enable 802.3x flow control on full-duplex gigabit links connecting campus distribution to gateway."
                },
                {
                    "name": "VLAN Segmentation & Broadcast Storm Rate Limiting",
                    "cat": "Broadcast Domain Mitigation",
                    "icon": "🛡️",
                    "mech": "Isolates student Wi-Fi, faculty systems, and IoT sensors into distinct VLANs. Limits broadcast/multicast traffic to 1% of port capacity to prevent gateway CPU starvation.",
                    "impl": "Configure storm-control broadcast level 1.0 on all access switch ports."
                },
                {
                    "name": "TCP Window Scaling & NAT Session Table Optimization",
                    "cat": "Layer 4 Gateway Tuning",
                    "icon": "🪟",
                    "mech": "Increases gateway NAT state table size and enables TCP window scaling (RFC 7323) to handle thousands of concurrent student socket connections without dropping handshakes.",
                    "impl": "Increase net.netfilter.nf_conntrack_max to 262144 and tune timeout values on Linux edge router."
                }
            ]
        else: # Normal
            cong_techs = [
                {
                    "name": "Proactive FQ-CoDel Baseline Deployment",
                    "cat": "Preventive Queue Management",
                    "icon": "🛡️",
                    "mech": "Maintains low queue latency under heavy load bursts by default, ensuring network never transitions into bufferbloat or packet loss states.",
                    "impl": "Standardize fq_codel across all campus distribution and edge router configurations."
                },
                {
                    "name": "End-to-End Explicit Congestion Notification (ECN) Monitoring",
                    "cat": "Congestion Early Warning",
                    "icon": "📡",
                    "mech": "Continuously tracks IP header ECN CE bit marking percentages in NetFlow/sFlow telemetry to detect emerging link saturation hours before users report lag.",
                    "impl": "Configure sFlow probes on core switches to report ECN marking ratios to network telemetry monitoring."
                },
                {
                    "name": "TCP BBR Protocol Standardization",
                    "cat": "Transport Layer Optimization",
                    "icon": "🚀",
                    "mech": "Ensures all university learning management systems (LMS) and internal services operate at maximum throughput without causing queue build-up.",
                    "impl": "Default Linux kernel parameter net.ipv4.tcp_congestion_control=bbr across university server clusters."
                },
                {
                    "name": "Token Bucket Rate Limiting for Peak Academic Periods",
                    "cat": "Preventive Traffic Shaping",
                    "icon": "🎛️",
                    "mech": "Maintains burst reserve tokens for interactive classroom applications while throttling recreational streaming during exam and lecture periods.",
                    "impl": "Schedule time-based rate limit policies on campus perimeter next-generation firewalls."
                }
            ]

        # Display in 2 clean columns
        t_c1, t_c2 = st.columns(2)
        for idx, tech in enumerate(cong_techs):
            target_col = t_c1 if idx % 2 == 0 else t_c2
            with target_col:
                st.markdown(f"""
                <div style="background:#162747;border:1.5px solid #24385D;border-radius:8px;padding:16px 20px;margin-bottom:14px;box-shadow:0 4px 16px rgba(0,0,0,0.3)">
                  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:8px">
                    <div style="font-size:18px;font-weight:800;color:#00D9FF;line-height:1.25">{tech['icon']} {tech['name']}</div>
                    <span style="font-size:12px;font-weight:800;color:#080D1F;background:#00D9FF;border-radius:99px;padding:3px 10px;white-space:nowrap;text-transform:uppercase">{tech['cat']}</span>
                  </div>
                  <div style="font-size:13px;font-weight:700;color:#8899BB;text-transform:uppercase;margin-top:6px">How It Works:</div>
                  <div style="font-size:15px;color:#F4F7FF;line-height:1.5;margin-bottom:8px">{tech['mech']}</div>
                  <div style="font-size:13px;font-weight:700;color:#8899BB;text-transform:uppercase">Practical Implementation:</div>
                  <div style="background:#0B1020;border:1px solid #24385D;border-left:3px solid #00D9FF;border-radius:5px;padding:8px 12px;font-family:monospace;font-size:14px;color:#36D6A0;margin-top:4px;word-break:break-all">{tech['impl']}</div>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 6. PAGE 2 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════
def page_dataset():
    st.markdown("<h1 class=\"np-main-title\">Dataset Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<div class=\"np-main-sub\">Inspect telemetry samples, upload custom data, or download the benchmark dataset.</div>", unsafe_allow_html=True)

    df_clean = st.session_state.df_clean
    df_raw = st.session_state.df_raw

    # Upload & Stats Row
    up_c1, up_c2 = st.columns([2, 1])
    with up_c1:
        st.markdown("<div class='np-input-card' style='padding:1.4rem'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:18px;font-weight:700;color:#F4F7FF;margin-bottom:8px'>UPLOAD CUSTOM NETWORK CSV / PARQUET:</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv", "parquet"], label_visibility="collapsed")
        if uploaded is not None:
            if st.button("📥 Load Uploaded File", use_container_width=True):
                with st.spinner("Processing uploaded dataset..."):
                    try:
                        b = uploaded.getvalue()
                        r, c = get_dataset(b, uploaded.name)
                        st.session_state.df_raw = r
                        st.session_state.df_clean = c
                        st.session_state.data_loaded = True
                        st.success(f"✓ Loaded {len(c):,} records successfully!")
                        st.rerun()
                    except Exception as ex:
                        st.error(f"Error reading file: {ex}")
        st.markdown("</div>", unsafe_allow_html=True)

    with up_c2:
        if df_clean is not None:
            st.markdown(f"""
            <div class="ctrl-metric" style="padding:1.4rem">
              <div class="label">Total Records</div>
              <div class="value">{len(df_clean):,}</div>
              <div style="font-size:15px;color:#C0CBE0;margin-top:6px">67 Telemetry Columns</div>
            </div>
            """, unsafe_allow_html=True)

    if df_clean is not None:
        st.markdown("<h2 class=\"np-section-heading\">DATASET PREVIEW & FILTERING</h2>", unsafe_allow_html=True)

        fc1, fc2 = st.columns([2, 1])
        with fc1:
            cats = ["All"] + sorted(df_clean["event_category"].dropna().unique().tolist()) if "event_category" in df_clean.columns else ["All"]
            chosen_cat = st.selectbox("Filter by Category", cats)
        with fc2:
            page_size = st.selectbox("Rows to Display", [25, 50, 100, 250], index=1)

        filtered = df_clean.copy()
        if chosen_cat != "All":
            filtered = filtered[filtered["event_category"] == chosen_cat]

        st.caption(f"Showing {min(page_size, len(filtered)):,} of {len(filtered):,} records")
        st.dataframe(filtered.head(page_size), use_container_width=True, height=380)

        # Download Cleaned CSV
        st.download_button(
            "⬇️ Download Cleaned Dataset (CSV)",
            data=df_clean.to_csv(index=False).encode("utf-8"),
            file_name="netpulse_cleaned_telemetry.csv",
            mime="text/csv",
        )

# ═══════════════════════════════════════════════════════════════
# 7. PAGE 3 — ANALYTICS (ONLY 2 FOCUSED GRAPHS)
# ═══════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("<h1 class=\"np-main-title\">Network Telemetry Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<div class=\"np-main-sub\">Core performance metrics and key longitudinal distribution trends.</div>", unsafe_allow_html=True)

    df = st.session_state.df_clean
    if df is None:
        st.warning("No dataset loaded.")
        return

    # Small set of core statistics
    avg_rtt  = df["rtt_avg_ms"].mean() if "rtt_avg_ms" in df.columns else 0.0
    avg_loss = df["packet_loss_pct"].mean() if "packet_loss_pct" in df.columns else 0.0
    avg_down = df["download_mbps"].mean() if "download_mbps" in df.columns else 0.0
    norm_pct = (df["event_category"] == "Normal").mean() * 100 if "event_category" in df.columns else 80.0

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="ctrl-metric">
          <div class="label">Average Latency</div>
          <div class="value">{avg_rtt:.1f}</div>
          <span class="unit">ms</span>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="ctrl-metric">
          <div class="label">Packet Loss</div>
          <div class="value">{avg_loss:.2f}</div>
          <span class="unit">%</span>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="ctrl-metric">
          <div class="label">Download Bandwidth</div>
          <div class="value">{avg_down:.1f}</div>
          <span class="unit">Mbps</span>
        </div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="ctrl-metric">
          <div class="label">Normal Baseline</div>
          <div class="value">{norm_pct:.1f}</div>
          <span class="unit">%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # GRAPH 1: Longitudinal Network Latency Trend
    st.markdown("""
    <div class="np-chart-wrap">
      <div class="chart-title">1. Longitudinal Network Latency (RTT) Trend</div>
      <div class="chart-sub">Continuous round-trip delay trajectory across all sampling intervals.</div>
    """, unsafe_allow_html=True)

    if "timestamp_local" in df.columns and "rtt_avg_ms" in df.columns:
        sample_df = df[["timestamp_local", "rtt_avg_ms"]].dropna()
        if len(sample_df) > 2000:
            sample_df = sample_df.sample(2000, random_state=42).sort_values("timestamp_local")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=sample_df["timestamp_local"],
            y=sample_df["rtt_avg_ms"],
            mode="lines",
            line=dict(color=CYAN, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0, 217, 255, 0.08)",
            name="Latency (ms)",
            hovertemplate="<b>Timestamp:</b> %{x}<br><b>RTT:</b> %{y:.2f} ms<extra></extra>",
        ))
        fig1.update_layout(**_plot_layout("", height=360))
        fig1.update_yaxes(title_text="Latency RTT (ms)")
        fig1.update_xaxes(title_text="Measurement Timeline")
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
      <div style="font-size:16px;color:#F4F7FF;padding:10px 14px;border-left:3px solid #00D9FF;background:rgba(0,217,255,0.06);margin-top:8px">
        <b>What this represents:</b> Tracks latency spikes and bufferbloat delays across the dataset timeline to highlight link congestion periods.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # GRAPH 2: Simple Network Condition Distribution
    st.markdown("""
    <div class="np-chart-wrap">
      <div class="chart-title">2. Network Condition Distribution</div>
      <div class="chart-sub">Frequency of normal operational states versus verified network anomalies.</div>
    """, unsafe_allow_html=True)

    if "event_category" in df.columns:
        counts = df["event_category"].value_counts()
        cmap = {
            "Normal": GREEN, "HighLatency": AMBER, "PacketLoss": RED,
            "HighJitter": VIOLET, "Outage": "#FF1A40",
        }
        fig2 = go.Figure(go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color=[cmap.get(k, CYAN) for k in counts.index],
            text=[f"{v:,}" for v in counts.values],
            textposition="outside",
            textfont=dict(size=13, color=TEXT_MAIN),
        ))
        fig2.update_layout(**_plot_layout("", height=320))
        fig2.update_yaxes(title_text="Total Telemetry Records")
        fig2.update_xaxes(tickangle=-15)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
      <div style="font-size:16px;color:#F4F7FF;padding:10px 14px;border-left:3px solid #00D9FF;background:rgba(0,217,255,0.06);margin-top:8px">
        <b>What this represents:</b> Summarizes overall network stability, demonstrating that while normal traffic dominates, packet loss and latency spikes represent distinct failure events.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 8. PAGE 4 — ABOUT
# ═══════════════════════════════════════════════════════════════
def page_about():
    st.markdown("<h1 class=\"np-main-title\">About NETPULSE AI</h1>", unsafe_allow_html=True)
    st.markdown("<div class=\"np-main-sub\">What this system does and how it works.</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-hero">
      <div class="about-hero-icon">⚡</div>
      <div class="about-hero-text">
        <div class="about-hero-title">NETPULSE AI — Campus Network Optimizer</div>
        <div class="about-hero-desc">
          NETPULSE AI is an <b>AI-powered network performance analysis system</b> built for campus environments.
          It takes real-time or historical network measurements, runs them through a Machine Learning model,
          and instantly tells you what is wrong with your network — along with clear steps to fix it.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class=\"np-section-heading\">How It Works</h2>", unsafe_allow_html=True)

    steps = [
        ("1", "📥", "Enter Network Data", "Input your network measurements manually — latency, packet loss, jitter, download/upload speed, Wi-Fi signal, and more. Or select a pre-recorded dataset sample."),
        ("2", "⚡", "Analyze with AI", "Click Analyze Network. The system sends your data through a trained Random Forest ML model with 150 decision trees that was trained on 54,000+ real telemetry records."),
        ("3", "🎯", "Get Instant Diagnosis", "The AI classifies your network into one of 5 key conditions: Normal, High Latency, Packet Loss, High Jitter, or Outage — with a confidence percentage."),
        ("4", "🔧", "Follow Recommendations", "Based on the diagnosis, you receive up to 3 specific, actionable recommendations — exactly what to check, why it matters, and how to fix it."),
    ]

    for step in steps:
        num, icon, title, desc = step
        st.markdown(f"""
        <div class="about-step">
          <div class="about-step-num">{num}</div>
          <div class="about-step-body">
            <div class="about-step-title">{icon} {title}</div>
            <div class="about-step-desc">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h2 class=\"np-section-heading\" style=\"margin-top:1.2rem\">What It Detects</h2>", unsafe_allow_html=True)

    conditions = [
        ("✅", "Normal", "var(--green)", "All metrics are healthy. Network is operating optimally."),
        ("⏱️", "High Latency", "var(--amber)", "Round-trip time is too high — causes lag, slow loading, video call delays."),
        ("📦", "Packet Loss", "var(--red)", "Data packets are being dropped — causes broken connections and slow speeds."),
        ("📈", "High Jitter", "var(--violet)", "Inconsistent packet timing — causes choppy audio and video in calls."),
        ("🚨", "Outage", "var(--red)", "Severe connectivity failure — network is effectively down."),
    ]

    cols = st.columns(3)
    for i, (icon, label, color, desc) in enumerate(conditions):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="about-condition-card">
              <div class="about-condition-icon" style="color:{color}">{icon} {label}</div>
              <div class="about-condition-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# 9. MAIN ROUTER
# ═══════════════════════════════════════════════════════════════
if   "Network Analyzer" in selected_nav: page_network_analyzer()
elif "Dataset"          in selected_nav: page_dataset()
elif "Analytics"        in selected_nav: page_analytics()
elif "About"            in selected_nav: page_about()

# Consistent Footer
st.markdown("""
<div class="np-footer">
  <b>NETPULSE AI</b> &nbsp;·&nbsp;
  Network Performance Analysis System &nbsp;·&nbsp;
  Operational Control Room Edition
</div>
""", unsafe_allow_html=True)
