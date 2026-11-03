"""
recommendations.py
------------------
This module generates intelligent network optimization recommendations
based on the ACTUAL measured values from the dataset or from user input.

Rules are based on common networking thresholds:
  - RTT > 100 ms    → High latency concern
  - Packet loss > 2% → Significant packet loss concern
  - Jitter > 30 ms  → Jitter / stability concern
  - Download < 5 Mbps → Low throughput concern
  - WiFi signal < 50% → Weak WiFi signal

Each recommendation includes:
  - A metric name
  - The observed value
  - A severity level: "Good", "Warning", or "Critical"
  - A plain-English explanation
  - Actionable suggestions

IMPORTANT: This module does NOT claim to automatically fix any network.
           It only provides educational/diagnostic suggestions.
"""

from typing import List, Dict


# ── Threshold definitions ──────────────────────────────────────────────────────
THRESHOLDS = {
    "rtt_avg_ms": {
        "good"    : 50,
        "warning" : 100,
        "unit"    : "ms",
        "label"   : "Average RTT (Latency)",
        "lower_is_better": True,
    },
    "jitter_ms": {
        "good"    : 15,
        "warning" : 30,
        "unit"    : "ms",
        "label"   : "Jitter",
        "lower_is_better": True,
    },
    "packet_loss_pct": {
        "good"    : 1.0,
        "warning" : 5.0,
        "unit"    : "%",
        "label"   : "Packet Loss",
        "lower_is_better": True,
    },
    "download_mbps": {
        "good"    : 25,
        "warning" : 10,
        "unit"    : "Mbps",
        "label"   : "Download Speed",
        "lower_is_better": False,   # higher is better
    },
    "upload_mbps": {
        "good"    : 10,
        "warning" : 5,
        "unit"    : "Mbps",
        "label"   : "Upload Speed",
        "lower_is_better": False,
    },
    "dns_resolve_ms": {
        "good"    : 20,
        "warning" : 80,
        "unit"    : "ms",
        "label"   : "DNS Resolution Time",
        "lower_is_better": True,
    },
    "http_ms": {
        "good"    : 300,
        "warning" : 1000,
        "unit"    : "ms",
        "label"   : "HTTP Response Time",
        "lower_is_better": True,
    },
    "wifi_signal_pct": {
        "good"    : 70,
        "warning" : 50,
        "unit"    : "%",
        "label"   : "Wi-Fi Signal Strength",
        "lower_is_better": False,
    },
    "gateway_rtt_avg_ms": {
        "good"    : 5,
        "warning" : 20,
        "unit"    : "ms",
        "label"   : "Gateway RTT",
        "lower_is_better": True,
    },
}


# ── Detailed recommendation text ───────────────────────────────────────────────
RECOMMENDATION_TEXT = {
    "rtt_avg_ms": {
        "warning" : {
            "title" : "⚠️ Elevated Latency Detected",
            "cause" : "Average RTT is above 100 ms. This may cause noticeable delays in web browsing, video calls, and online applications.",
            "fixes" : [
                "Check if the network is congested during peak hours (e.g., lunch time, evenings).",
                "Inspect network path: run a traceroute to find slow hops.",
                "Verify if the ISP link is saturated — compare with off-peak measurements.",
                "Consider QoS rules to prioritise real-time traffic (video, VoIP).",
            ],
        },
        "critical": {
            "title" : "🚨 Critical Latency — Network Severely Degraded",
            "cause" : "Average RTT exceeds 200 ms. Users will experience significant lag in all interactive applications.",
            "fixes" : [
                "Immediately investigate for routing loops or failed primary links.",
                "Check whether a failover/backup link is active with higher latency.",
                "Contact ISP if the problem is outside the local network.",
                "Restrict bandwidth-heavy background traffic (updates, backups) temporarily.",
            ],
        },
    },
    "jitter_ms": {
        "warning" : {
            "title" : "⚠️ High Jitter — Unstable Connection",
            "cause" : "Jitter above 30 ms indicates inconsistent packet delivery timing, which affects voice and video quality.",
            "fixes" : [
                "Enable QoS (Quality of Service) to prioritise VoIP and video traffic.",
                "Check for competing traffic using the same link (downloads, backups).",
                "Switch to a wired connection instead of Wi-Fi if possible.",
                "Investigate wireless interference if on Wi-Fi (other SSIDs, nearby devices).",
            ],
        },
        "critical": {
            "title" : "🚨 Critical Jitter — Real-Time Apps Unusable",
            "cause" : "Jitter exceeds 60 ms. Video calls and VoIP will experience choppy audio/video.",
            "fixes" : [
                "Immediately apply traffic shaping or QoS policies.",
                "Check for flooding or broadcast storms on the local network.",
                "Investigate wireless channel congestion — consider switching Wi-Fi channel.",
            ],
        },
    },
    "packet_loss_pct": {
        "warning" : {
            "title" : "⚠️ Packet Loss Detected",
            "cause" : "Packet loss between 2–10% will cause TCP retransmissions, slowing down all network activity.",
            "fixes" : [
                "Run a sustained ping test to determine if loss is constant or intermittent.",
                "Inspect cables and physical connections for damage.",
                "Check for duplex mismatch on wired connections.",
                "Review switch port error counters for CRC or collision errors.",
            ],
        },
        "critical": {
            "title" : "🚨 Severe Packet Loss — Network Unstable",
            "cause" : "Packet loss above 10% makes the network effectively unreliable. Downloads will fail, streams will buffer constantly.",
            "fixes" : [
                "Immediately check all physical links for damage.",
                "Verify that no routing loop exists.",
                "Restart network interface if the problem is software-related.",
                "Contact ISP if loss occurs at the WAN gateway.",
            ],
        },
    },
    "download_mbps": {
        "warning" : {
            "title" : "⚠️ Low Download Speed",
            "cause" : "Download speed is below 10 Mbps. Multiple users sharing this link will experience slow performance.",
            "fixes" : [
                "Test speed at different times to identify peak congestion periods.",
                "Limit or schedule large downloads (OS updates, backups) to off-peak hours.",
                "Check if the ISP plan matches the required bandwidth for campus usage.",
                "Investigate if the Wi-Fi band is congested — try switching to 5 GHz.",
            ],
        },
        "critical": {
            "title" : "🚨 Critical Throughput — Very Low Download Speed",
            "cause" : "Download speed is below 5 Mbps. Basic web usage will be very slow.",
            "fixes" : [
                "Check ISP link status — service outage may be occurring.",
                "Disconnect non-essential devices consuming bandwidth.",
                "Reboot modem and router as a first step.",
            ],
        },
    },
    "upload_mbps": {
        "warning" : {
            "title" : "⚠️ Low Upload Speed",
            "cause" : "Upload speed is below 5 Mbps, affecting video conferencing and cloud uploads.",
            "fixes" : [
                "Prioritise upload traffic for video conferencing using QoS.",
                "Limit background cloud sync during work hours.",
                "Check if upload and download speeds are asymmetrically limited by the ISP plan.",
            ],
        },
        "critical": {
            "title" : "🚨 Critical Upload Speed",
            "cause" : "Upload speed is critically low. Video calls and remote collaboration will fail.",
            "fixes" : [
                "Contact ISP to check upstream link health.",
                "Check for high upstream traffic from a single device (virus, misconfigured backup).",
            ],
        },
    },
    "dns_resolve_ms": {
        "warning" : {
            "title" : "⚠️ Slow DNS Resolution",
            "cause" : "DNS queries are taking above 80 ms. Every new web page load involves a DNS lookup, so slow DNS impacts all browsing.",
            "fixes" : [
                "Consider switching to a faster DNS server (e.g., Google 8.8.8.8 or Cloudflare 1.1.1.1).",
                "Enable DNS caching on the local resolver or router.",
                "Check if the current DNS server is overloaded.",
            ],
        },
        "critical": {
            "title" : "🚨 DNS Very Slow — Web Browsing Affected",
            "cause" : "DNS response time exceeds 200 ms. Web pages will load very slowly even on fast connections.",
            "fixes" : [
                "Immediately switch to a reliable public DNS server.",
                "Check DNS server availability and uptime.",
            ],
        },
    },
    "http_ms": {
        "warning" : {
            "title" : "⚠️ Slow HTTP Response",
            "cause" : "Web server response time is above 1 second. Users will notice slow page loads.",
            "fixes" : [
                "Check internet latency and throughput first — HTTP time depends on both.",
                "If hosting a local server, check server resource usage (CPU, memory).",
                "Enable content caching at the network edge if available.",
            ],
        },
        "critical": {
            "title" : "🚨 Very Slow HTTP — Web Access Degraded",
            "cause" : "HTTP response time exceeds 3 seconds. Web-based applications will be frustrating to use.",
            "fixes" : [
                "Investigate ISP or WAN link health.",
                "Check for proxy or firewall rules causing HTTP inspection delays.",
            ],
        },
    },
    "wifi_signal_pct": {
        "warning" : {
            "title" : "⚠️ Weak Wi-Fi Signal",
            "cause" : "Wi-Fi signal strength is below 50%. This will reduce speed and increase packet errors on wireless connections.",
            "fixes" : [
                "Move closer to the Wi-Fi access point.",
                "Check for physical obstructions (walls, metal surfaces).",
                "Consider adding a Wi-Fi repeater or access point in weak-signal areas.",
                "Ensure the 5 GHz band is used when close to the access point.",
            ],
        },
        "critical": {
            "title" : "🚨 Critical Wi-Fi Signal — Connection Dropping",
            "cause" : "Wi-Fi signal is critically weak. Frequent disconnections and very low speeds are expected.",
            "fixes" : [
                "Immediately move the device closer to an access point.",
                "Check if the access point is functioning correctly.",
                "Consider deploying additional access points for better coverage.",
            ],
        },
    },
    "gateway_rtt_avg_ms": {
        "warning" : {
            "title" : "⚠️ High Gateway RTT",
            "cause" : "Round-trip time to the local gateway is above 20 ms. This indicates local network congestion or a slow router.",
            "fixes" : [
                "Check router/gateway CPU and memory utilisation.",
                "Reduce the number of devices connected simultaneously.",
                "Update gateway firmware if outdated.",
            ],
        },
        "critical": {
            "title" : "🚨 Gateway RTT Critical — Router Under Stress",
            "cause" : "Gateway RTT exceeds 50 ms. The local router is heavily loaded or malfunctioning.",
            "fixes" : [
                "Reboot the gateway/router.",
                "Check for devices generating excessive ARP or broadcast traffic.",
                "Consider replacing an aged or underpowered router.",
            ],
        },
    },
}


def generate_recommendations(metrics: dict) -> List[Dict]:
    """
    Generate a list of recommendations based on observed metric values.

    Parameters:
        metrics (dict): { metric_name: observed_value }

    Returns:
        List of recommendation dicts, each containing:
          - metric, label, value, unit, severity, title, cause, fixes
    """
    recommendations = []

    for metric, value in metrics.items():
        if metric not in THRESHOLDS or value is None:
            continue

        thresh = THRESHOLDS[metric]
        lower_is_better = thresh["lower_is_better"]
        good_val    = thresh["good"]
        warning_val = thresh["warning"]

        if lower_is_better:
            # Low values are better (latency, packet loss, jitter)
            if value <= good_val:
                severity = "Good"
            elif value <= warning_val:
                severity = "Warning"
            else:
                severity = "Critical"
        else:
            # High values are better (speed, signal)
            if value >= good_val:
                severity = "Good"
            elif value >= warning_val:
                severity = "Warning"
            else:
                severity = "Critical"

        rec = {
            "metric"  : metric,
            "label"   : thresh["label"],
            "value"   : round(value, 2),
            "unit"    : thresh["unit"],
            "severity": severity,
        }

        # Add detailed text only for non-Good metrics
        if severity in ("Warning", "Critical") and metric in RECOMMENDATION_TEXT:
            level_key = "warning" if severity == "Warning" else "critical"
            text_key  = level_key if level_key in RECOMMENDATION_TEXT[metric] else "warning"
            text      = RECOMMENDATION_TEXT[metric].get(text_key, {})
            rec["title"] = text.get("title", f"{severity}: {thresh['label']}")
            rec["cause"] = text.get("cause", "Measurement is outside acceptable range.")
            rec["fixes"] = text.get("fixes", ["Investigate network conditions."])
        else:
            rec["title"] = f"✅ {thresh['label']} — Normal"
            rec["cause"] = "Measurement is within acceptable range."
            rec["fixes"] = []

        recommendations.append(rec)

    # Sort: Critical first, then Warning, then Good
    order = {"Critical": 0, "Warning": 1, "Good": 2}
    recommendations.sort(key=lambda r: order.get(r["severity"], 3))

    return recommendations


def get_overall_status(recommendations: List[Dict]) -> str:
    """
    Compute an overall network status from a list of recommendations.
    Returns "Critical", "Warning", or "Healthy".
    """
    severities = {r["severity"] for r in recommendations}
    if "Critical" in severities:
        return "Critical"
    if "Warning" in severities:
        return "Warning"
    return "Healthy"
