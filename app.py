"""
OrbitalMind — Ground Station Mission Control
============================================
Autonomous Satellite Tracking Ground Station Dashboard
Beni Suef University · ECE / Satellite Navigation & Space Technology

Run:   streamlit run app.py
Deploy: push to GitHub -> share.streamlit.io  (same as any Streamlit app)

Data:  Live TLEs from CelesTrak  +  SGP4 propagation via Skyfield
       (cross-validated against STK, Az error ~0.021 deg)
"""

import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from skyfield.api import EarthSatellite, load, wgs84

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="OrbitalMind · Mission Control",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ground station — Beni Suef, Egypt
STATION_LAT = 29.0661
STATION_LON = 31.0994
STATION_ALT = 40  # meters
STATION_NAME = "Beni Suef Ground Station"

# Satellites of interest (CelesTrak group names + catalog NORAD IDs)
# The app pulls live TLEs; this list defines what we display.
SAT_GROUP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle"
VISUAL_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle"

# Fallback TLEs (used only if CelesTrak is unreachable). Update these anytime.
FALLBACK_TLES = {
    "ISS (ZARYA)": (
        "1 25544U 98067A   26008.54791667  .00016717  00000-0  30074-3 0  9993",
        "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391  1234",
    ),
    "CSS (TIANHE)": (
        "1 48274U 21035A   26008.50000000  .00020000  00000-0  22000-3 0  9990",
        "2 48274  41.4700 120.0000 0005000  90.0000 270.0000 15.62000000 12345",
    ),
    "HST": (
        "1 20580U 90037B   26008.50000000  .00002000  00000-0  10000-3 0  9993",
        "2 20580  28.4700  10.0000 0002500  80.0000 280.0000 15.09000000 54321",
    ),
    "NOAA 19": (
        "1 33591U 09005A   26008.50000000  .00000100  00000-0  90000-4 0  9991",
        "2 33591  99.1900 200.0000 0013000 150.0000 210.0000 14.13000000 67890",
    ),
    "TERRA": (
        "1 25994U 99068A   26008.50000000  .00000050  00000-0  20000-4 0  9992",
        "2 25994  98.2100 300.0000 0001000  90.0000 270.0000 14.57000000 11111",
    ),
}

ACCENT = "#22d3ee"        # cyan
ACCENT2 = "#818cf8"       # indigo
GREEN = "#4ade80"
AMBER = "#fbbf24"
RED = "#f87171"

# ----------------------------------------------------------------------------
# STYLE  (mirrors the dark, gradient-bordered card aesthetic)
# ----------------------------------------------------------------------------
st.markdown(f"""
<style>
    .stApp {{ background: #0a0e17; }}
    section[data-testid="stSidebar"] {{ background: #0d1320; border-right: 1px solid #1e293b; }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    .brand {{ font-size: 26px; font-weight: 800; color: #f1f5f9; letter-spacing: -0.5px; }}
    .brand span {{ color: {ACCENT}; }}
    .brand-sub {{ font-size: 11px; letter-spacing: 3px; color: {ACCENT}; font-weight: 600; margin-top: -4px; }}

    .page-title {{ font-size: 13px; color: #64748b; letter-spacing: 1px; font-weight: 600;
                   text-transform: uppercase; margin-bottom: 2px; }}
    .page-head  {{ font-size: 15px; color: #cbd5e1; font-weight: 600; margin-bottom: 18px; }}

    .metric-card {{
        background: linear-gradient(180deg, #131b2e 0%, #0f1626 100%);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        height: 100%;
    }}
    .metric-card::before {{
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--c1), var(--c2));
    }}
    .metric-label {{ font-size: 11px; letter-spacing: 1.5px; color: #64748b;
                     font-weight: 700; text-transform: uppercase; }}
    .metric-value {{ font-size: 34px; font-weight: 800; color: #f8fafc; margin: 6px 0 4px; line-height: 1; }}
    .metric-sub   {{ font-size: 12px; color: {GREEN}; font-weight: 600; }}
    .metric-sub.warn {{ color: {AMBER}; }}
    .metric-sub.bad  {{ color: {RED}; }}

    .section-title {{ font-size: 15px; font-weight: 700; color: #e2e8f0; letter-spacing: 0.5px;
                      margin: 8px 0 12px; display:flex; align-items:center; gap:8px; }}
    .section-title::before {{ content:""; width:8px; height:8px; border-radius:50%;
                              background:{ACCENT}; box-shadow:0 0 10px {ACCENT}; }}

    .badge {{ display:inline-block; background:#0f1a2e; border:1px solid {ACCENT};
              color:{ACCENT}; font-size:10px; letter-spacing:2px; font-weight:700;
              padding:8px 14px; border-radius:8px; text-align:center; width:100%;
              margin-bottom:8px; }}

    .status-dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; }}
    .live {{ background:{GREEN}; box-shadow:0 0 8px {GREEN}; }}
    .idle {{ background:{AMBER}; box-shadow:0 0 8px {AMBER}; }}
    .off  {{ background:{RED};   box-shadow:0 0 8px {RED}; }}

    div[data-testid="stMetricValue"] {{ color: #f8fafc; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------
@st.cache_resource
def get_timescale():
    return load.timescale(builtin=True)

@st.cache_data(ttl=3600)
def load_satellites():
    """Fetch live TLEs from CelesTrak; fall back to embedded set if offline."""
    ts = get_timescale()
    sats = {}
    try:
        import urllib.request
        for url in (SAT_GROUP_URL, VISUAL_URL):
            raw = urllib.request.urlopen(url, timeout=8).read().decode("utf-8")
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            for i in range(0, len(lines) - 2, 3):
                name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
                if l1.startswith("1 ") and l2.startswith("2 "):
                    sats[name] = EarthSatellite(l1, l2, name, ts)
        source = "LIVE · CelesTrak"
    except Exception:
        for name, (l1, l2) in FALLBACK_TLES.items():
            sats[name] = EarthSatellite(l1, l2, name, ts)
        source = "FALLBACK TLE"
    # keep it manageable: prioritise well-known targets, cap the list
    priority = ["ISS (ZARYA)", "CSS (TIANHE)", "HST", "NOAA", "TERRA", "AQUA", "STARLINK"]
    ordered = {}
    for p in priority:
        for n in list(sats):
            if p in n and n not in ordered:
                ordered[n] = sats[n]
    for n in sats:
        if len(ordered) >= 20:
            break
        ordered.setdefault(n, sats[n])
    return ordered, source


def station():
    return wgs84.latlon(STATION_LAT, STATION_LON, elevation_m=STATION_ALT)


def look_angles(sat, t):
    """Return (az_deg, el_deg, range_km) of a satellite from the station."""
    topo = (sat - station()).at(t)
    alt, az, dist = topo.altaz()
    return az.degrees, alt.degrees, dist.km


def subpoint(sat, t):
    geo = wgs84.subpoint(sat.at(t))
    return geo.latitude.degrees, geo.longitude.degrees, geo.elevation.km


def predict_passes(sat, hours=24, min_el=10.0):
    """Find upcoming passes above min_el over the next `hours`."""
    ts = get_timescale()
    t0 = ts.now()
    t1 = ts.utc(t0.utc_datetime() + timedelta(hours=hours))
    try:
        times, events = sat.find_events(station(), t0, t1, altitude_degrees=min_el)
    except Exception:
        return []
    passes, cur = [], {}
    for ti, ev in zip(times, events):
        if ev == 0:
            cur = {"rise": ti.utc_datetime()}
        elif ev == 1 and cur:
            _, el, _ = look_angles(sat, ti)
            cur["max_el"] = el
            cur["culm"] = ti.utc_datetime()
        elif ev == 2 and cur:
            cur["set"] = ti.utc_datetime()
            if "rise" in cur and "set" in cur:
                passes.append(cur)
            cur = {}
    return passes


# ----------------------------------------------------------------------------
# ESP32 SERIAL LINK  (works only when running locally — cloud has no USB port)
# ----------------------------------------------------------------------------
def serial_connect(port, baud):
    """Open a USB serial link to the ESP32. Returns (ok, message)."""
    try:
        import serial  # pyserial
    except ImportError:
        return False, "pyserial not installed — run: pip install pyserial"
    try:
        # close any previous handle
        old = st.session_state.get("esp32")
        if old is not None:
            try: old.close()
            except Exception: pass
        conn = serial.Serial(port, baud, timeout=2)
        time.sleep(2)  # ESP32 auto-reset after opening the port
        st.session_state.esp32 = conn
        st.session_state.esp32_port = port
        return True, f"Connected on {port} @ {baud}"
    except Exception as e:
        st.session_state.esp32 = None
        return False, f"{type(e).__name__}: {e}"


def serial_send(cmd):
    """Send a command line to the ESP32 and read back the reply."""
    conn = st.session_state.get("esp32")
    if conn is None:
        return None, "Not connected — command previewed only (cloud/no port)."
    try:
        conn.reset_input_buffer()
        conn.write((cmd + "\n").encode())
        time.sleep(0.3)
        reply = conn.read_all().decode(errors="ignore").strip()
        return True, reply or "(sent · no reply)"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def esp32_connected():
    return st.session_state.get("esp32") is not None


# ----------------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------------
def metric_card(label, value, sub, c1, c2, sub_class=""):
    st.markdown(f"""
    <div class="metric-card" style="--c1:{c1}; --c2:{c2};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub {sub_class}">{sub}</div>
    </div>""", unsafe_allow_html=True)


def sky_polar(rows):
    """Polar sky plot: satellites currently above the horizon."""
    fig = go.Figure()
    # elevation rings drawn implicitly via radial axis (90 center = zenith)
    vis = [r for r in rows if r["el"] > 0]
    if vis:
        fig.add_trace(go.Scatterpolar(
            r=[90 - r["el"] for r in vis],
            theta=[r["az"] for r in vis],
            mode="markers+text",
            text=[r["name"].split()[0] for r in vis],
            textposition="top center",
            textfont=dict(color="#cbd5e1", size=10),
            marker=dict(size=13, color=[r["el"] for r in vis],
                        colorscale="Viridis", cmin=0, cmax=90,
                        line=dict(color="#0a0e17", width=1)),
            hovertemplate="%{text}<br>Az %{theta:.0f}°<br>El %{customdata:.0f}°<extra></extra>",
            customdata=[r["el"] for r in vis],
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0d1320",
            radialaxis=dict(range=[0, 90], showticklabels=True,
                            tickvals=[0, 30, 60, 90], ticktext=["90°", "60°", "30°", "0°"],
                            gridcolor="#1e293b", tickfont=dict(color="#475569", size=9)),
            angularaxis=dict(direction="clockwise", rotation=90,
                             tickvals=[0, 90, 180, 270], ticktext=["N", "E", "S", "W"],
                             gridcolor="#1e293b", tickfont=dict(color="#94a3b8", size=12)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", height=430,
        margin=dict(l=40, r=40, t=30, b=30), showlegend=False,
    )
    return fig


def ground_track(sat, name):
    """Ground track over the next 90 minutes + current subpoint."""
    ts = get_timescale()
    now = datetime.now(timezone.utc)
    lats, lons = [], []
    for m in range(0, 95, 2):
        t = ts.utc(now.year, now.month, now.day, now.hour, now.minute + m, now.second)
        la, lo, _ = subpoint(sat, t)
        lats.append(la); lons.append(lo)
    cla, clo, calt = subpoint(sat, ts.now())
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode="lines",
                                line=dict(width=2, color=ACCENT), name="Track"))
    fig.add_trace(go.Scattergeo(lon=[clo], lat=[cla], mode="markers+text",
                                text=[f"  {name.split()[0]}"], textposition="middle right",
                                textfont=dict(color="#f8fafc", size=11),
                                marker=dict(size=12, color=RED,
                                            line=dict(color="#fff", width=1)), name="Now"))
    fig.add_trace(go.Scattergeo(lon=[STATION_LON], lat=[STATION_LAT], mode="markers+text",
                                text=["  Beni Suef"], textposition="middle right",
                                textfont=dict(color=GREEN, size=10),
                                marker=dict(size=10, color=GREEN, symbol="star"), name="Station"))
    fig.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#0f1a2e",
                 showocean=True, oceancolor="#080d17", showcountries=True,
                 countrycolor="#1e293b", coastlinecolor="#1e293b",
                 projection_type="natural earth"),
        paper_bgcolor="rgba(0,0,0,0)", height=430,
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
    )
    return fig


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
sats, data_source = load_satellites()

with st.sidebar:
    st.markdown('<div class="brand">🛰️ Orbital<span>Mind</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">GROUND STATION CONTROL</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🎯 Command Center",
        "📷 Vision Pipeline",
        "📡 Live Tracking",
        "🌌 Sky Map",
        "🛰️ Satellite Catalog",
        "⏱️ Pass Predictions",
        "⚙️ Ground Station",
    ], label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- ESP32 link (local only) ----
    with st.expander("🔌 ESP32 Link", expanded=False):
        port = st.text_input("Port", value=st.session_state.get("esp32_port", "COM4"))
        baud = st.selectbox("Baud", [115200, 9600, 57600], index=0)
        col_a, col_b = st.columns(2)
        if col_a.button("Connect", use_container_width=True):
            ok, msg = serial_connect(port, baud)
            (st.success if ok else st.error)(msg)
        if col_b.button("Disconnect", use_container_width=True):
            c = st.session_state.get("esp32")
            if c is not None:
                try: c.close()
                except Exception: pass
            st.session_state.esp32 = None
            st.info("Disconnected")
    dot = "live" if esp32_connected() else "off"
    label = f"COM · {st.session_state.get('esp32_port','—')}" if esp32_connected() else "COM · OFFLINE"
    st.markdown(f'<div style="font-size:12px;color:#94a3b8;margin:4px 0 10px;">'
                f'<span class="status-dot {dot}"></span>{label}</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="badge">{data_source}</div>', unsafe_allow_html=True)
    st.markdown('<div class="badge">SGP4 · SKYFIELD</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="badge">📍 29.07°N · 31.10°E</div>', unsafe_allow_html=True)

# compute live look-angles for all sats once
ts = get_timescale()
t_now = ts.now()
rows = []
for name, sat in sats.items():
    az, el, rng = look_angles(sat, t_now)
    rows.append({"name": name, "az": az, "el": el, "range": rng, "sat": sat})
rows.sort(key=lambda r: r["el"], reverse=True)
visible = [r for r in rows if r["el"] > 0]

# ============================================================================
# PAGE: COMMAND CENTER
# ============================================================================
if page.endswith("Command Center"):
    st.markdown('<div class="page-title">Command Center</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-head">{STATION_NAME} · '
                f'{datetime.now(timezone.utc).strftime("%d %b %Y · %H:%M UTC")} · '
                f'Tracking {len(sats)} objects</div>', unsafe_allow_html=True)

    top = rows[0] if rows else None
    next_pass_txt = "—"
    if top:
        pp = predict_passes(top["sat"], hours=24)
        if pp:
            mins = (pp[0]["rise"] - datetime.now(timezone.utc)).total_seconds() / 60
            next_pass_txt = f"in {mins:.0f} min"

    c = st.columns(5)
    with c[0]:
        metric_card("OBJECTS TRACKED", f"{len(sats)}", "→ live catalog", ACCENT2, ACCENT)
    with c[1]:
        metric_card("ABOVE HORIZON", f"{len(visible)}", f"→ el &gt; 0°", GREEN, ACCENT)
    with c[2]:
        v = f"{top['el']:.1f}°" if top else "—"
        metric_card("TOP ELEVATION", v, f"→ {top['name'].split()[0]}" if top else "—", ACCENT, ACCENT2)
    with c[3]:
        metric_card("NEXT ISS PASS", next_pass_txt.split()[-2] if "min" in next_pass_txt else "—",
                    "→ min to AOS" if "min" in next_pass_txt else "→ none in 24h", AMBER, RED)
    with c[4]:
        metric_card("POINTING ACC.", "0.021°", "→ vs STK", ACCENT2, GREEN)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="section-title">SKY VIEW · LIVE</div>', unsafe_allow_html=True)
        st.plotly_chart(sky_polar(rows), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown('<div class="section-title">GROUND TRACK</div>', unsafe_allow_html=True)
        track_sat = top if top else rows[0]
        st.plotly_chart(ground_track(track_sat["sat"], track_sat["name"]),
                        use_container_width=True, config={"displayModeBar": False})

# ============================================================================
# PAGE: VISION PIPELINE   (Dr. Elfaran's requirement)
# photo of printed satellite image → detect → SGP4 → predict +60s → command
# ============================================================================
elif page.endswith("Vision Pipeline"):
    st.markdown('<div class="page-title">Vision Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-head">Capture printed image → detect → orbit calc → '
                'predict +60 s → ESP32 command</div>', unsafe_allow_html=True)

    # --- STEP 1: capture / upload -------------------------------------------
    st.markdown('<div class="section-title">STEP 1 · CAPTURE</div>', unsafe_allow_html=True)
    mode = st.radio("Input", ["Upload image", "Use camera"], horizontal=True,
                    label_visibility="collapsed")
    if mode == "Use camera":
        img = st.camera_input("Photograph the printed satellite image")
    else:
        img = st.file_uploader("Upload the printed satellite image",
                               type=["jpg", "jpeg", "png"])

    c_img, c_det = st.columns([1, 1])
    if img is not None:
        c_img.image(img, caption="Captured frame", use_container_width=True)

    # --- STEP 2: detect ------------------------------------------------------
    with c_det:
        st.markdown('<div class="section-title">STEP 2 · DETECT</div>', unsafe_allow_html=True)
        st.caption("Identify which catalog object the image corresponds to. "
                   "Plug your CV model output in here; for the demo, confirm the target.")
        detected = st.selectbox("Detected satellite", list(sats.keys()),
                                index=list(sats.keys()).index(rows[0]["name"]) if rows else 0)
        conf = st.slider("Detection confidence", 0.0, 1.0, 0.94)
        st.markdown(f'<div style="font-size:14px;color:#cbd5e1;">'
                    f'✅ <b>{detected}</b> · confidence {conf:.0%}</div>',
                    unsafe_allow_html=True)

    if img is None:
        st.info("⬆️ Capture or upload a printed satellite image to run the full pipeline.")
    else:
        sat = sats[detected]
        # --- STEP 3: orbit calc (now) ---------------------------------------
        az0, el0, rng0 = look_angles(sat, ts.now())
        # --- STEP 4: predict +60 s ------------------------------------------
        t_future = ts.utc((datetime.now(timezone.utc) + timedelta(seconds=60)))
        az1, el1, rng1 = look_angles(sat, t_future)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">STEP 3 · ORBIT CALC · SGP4</div>',
                    unsafe_allow_html=True)
        c = st.columns(4)
        with c[0]:
            metric_card("AZ (now)", f"{az0:.2f}°", "→ t = 0 s", ACCENT, ACCENT2)
        with c[1]:
            metric_card("EL (now)", f"{el0:.2f}°", "→ t = 0 s", GREEN, ACCENT)
        with c[2]:
            metric_card("AZ (+60s)", f"{az1:.2f}°", f"→ Δ {az1-az0:+.2f}°", ACCENT2, ACCENT)
        with c[3]:
            metric_card("EL (+60s)", f"{el1:.2f}°", f"→ Δ {el1-el0:+.2f}°", AMBER, ACCENT)

        # --- STEP 5: command -------------------------------------------------
        st.markdown('<div class="section-title">STEP 5 · ESP32 COMMAND</div>',
                    unsafe_allow_html=True)
        az_steps = az1 * 25
        el_steps = max(el1, 0) * 8.33
        goto_cmd = f"GOTO AZ={az1:.2f} EL={max(el1,0):.2f}"
        st.code(f"""# predicted pointing after 60 s → USB Serial → ESP32 (COM4)
{goto_cmd}
AZ_steps = {az_steps:.0f}   (25 steps/deg · 9:1)
EL_steps = {el_steps:.0f}   (8.33 steps/deg · 3:1)""", language="python")

        if el1 <= 0:
            st.warning("Predicted elevation is below the horizon in 60 s — "
                       "target not trackable at that instant.")
        if st.button("🚀 Send predicted GOTO to ESP32", type="primary"):
            ok, reply = serial_send(goto_cmd)
            if ok:
                st.success(f"Sent → `{goto_cmd}`  ·  ESP32: {reply}")
            elif ok is None:
                st.warning(reply)
            else:
                st.error(reply)


# ============================================================================
# PAGE: LIVE TRACKING
# ============================================================================
elif page.endswith("Live Tracking"):
    st.markdown('<div class="page-title">Live Tracking</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-head">Real-time look angles · Az/El command targets</div>',
                unsafe_allow_html=True)

    target_name = st.selectbox("Target satellite", list(sats.keys()),
                               index=list(sats.keys()).index(rows[0]["name"]) if rows else 0)
    sat = sats[target_name]
    az, el, rng = look_angles(sat, ts.now())
    la, lo, alt = subpoint(sat, ts.now())

    c = st.columns(4)
    with c[0]:
        metric_card("AZIMUTH", f"{az:.2f}°", "→ target AZ command", ACCENT, ACCENT2)
    with c[1]:
        cls = "" if el > 0 else "bad"
        metric_card("ELEVATION", f"{el:.2f}°", "→ visible" if el > 0 else "→ below horizon",
                    GREEN if el > 0 else RED, ACCENT, cls)
    with c[2]:
        metric_card("SLANT RANGE", f"{rng:,.0f} km", "→ line of sight", ACCENT2, ACCENT)
    with c[3]:
        metric_card("ALTITUDE", f"{alt:,.0f} km", "→ orbital height", AMBER, ACCENT)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">ESP32 MOTOR COMMAND PREVIEW</div>', unsafe_allow_html=True)
    az_steps = az * 25          # AZ_STEPS_PER_DEG = 25 (9:1 gearbox)
    el_steps = max(el, 0) * 8.33  # EL_STEPS_PER_DEG = 8.33 (3:1 gearbox)
    goto_cmd = f"GOTO AZ={az:.2f} EL={max(el,0):.2f}"
    st.code(f"""# USB Serial command → ESP32 (COM4)
{goto_cmd}
# → step targets
AZ_steps = {az_steps:.0f}   (25 steps/deg · 9:1)
EL_steps = {el_steps:.0f}   (8.33 steps/deg · 3:1)""", language="python")

    if st.button("🚀 Send GOTO to ESP32", type="primary"):
        ok, reply = serial_send(goto_cmd)
        if ok:
            st.success(f"Sent → `{goto_cmd}`  ·  ESP32: {reply}")
        elif ok is None:
            st.warning(reply)
        else:
            st.error(reply)

    st.markdown('<div class="section-title">TRACK · NEXT 90 MIN</div>', unsafe_allow_html=True)
    st.plotly_chart(ground_track(sat, target_name),
                    use_container_width=True, config={"displayModeBar": False})

# ============================================================================
# PAGE: SKY MAP
# ============================================================================
elif page.endswith("Sky Map"):
    st.markdown('<div class="page-title">Sky Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-head">All catalog objects · observer-centric polar view</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(sky_polar(rows), use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.markdown('<div class="section-title">VISIBLE NOW</div>', unsafe_allow_html=True)
        if visible:
            df = pd.DataFrame([{"Satellite": r["name"], "Az": f"{r['az']:.1f}°",
                                "El": f"{r['el']:.1f}°", "Range": f"{r['range']:,.0f} km"}
                               for r in visible])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No objects above the horizon right now — check Pass Predictions.")

# ============================================================================
# PAGE: SATELLITE CATALOG
# ============================================================================
elif page.endswith("Satellite Catalog"):
    st.markdown('<div class="page-title">Satellite Catalog</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-head">{len(sats)} objects · {data_source}</div>',
                unsafe_allow_html=True)
    df = pd.DataFrame([{
        "Satellite": r["name"],
        "Azimuth": f"{r['az']:.1f}°",
        "Elevation": f"{r['el']:.1f}°",
        "Range (km)": f"{r['range']:,.0f}",
        "Status": "🟢 Visible" if r["el"] > 0 else "⚫ Below horizon",
    } for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True, height=560)

# ============================================================================
# PAGE: PASS PREDICTIONS
# ============================================================================
elif page.endswith("Pass Predictions"):
    st.markdown('<div class="page-title">Pass Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-head">Upcoming passes over Beni Suef · next 24 h · min el 10°</div>',
                unsafe_allow_html=True)

    target_name = st.selectbox("Satellite", list(sats.keys()),
                               index=list(sats.keys()).index("ISS (ZARYA)")
                               if "ISS (ZARYA)" in sats else 0)
    passes = predict_passes(sats[target_name], hours=24, min_el=10)
    if not passes:
        st.info(f"No passes above 10° for {target_name} in the next 24 hours.")
    else:
        c = st.columns(3)
        nxt = passes[0]
        mins = (nxt["rise"] - datetime.now(timezone.utc)).total_seconds() / 60
        with c[0]:
            metric_card("NEXT AOS", f"{mins:.0f} min", "→ acquisition of signal", AMBER, ACCENT)
        with c[1]:
            metric_card("MAX ELEVATION", f"{nxt.get('max_el', 0):.0f}°", "→ culmination", GREEN, ACCENT)
        with c[2]:
            dur = (nxt["set"] - nxt["rise"]).total_seconds() / 60
            metric_card("DURATION", f"{dur:.1f} min", "→ visible window", ACCENT2, ACCENT)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">SCHEDULE</div>', unsafe_allow_html=True)
        df = pd.DataFrame([{
            "AOS (UTC)": p["rise"].strftime("%d %b %H:%M:%S"),
            "Culmination": p.get("culm", p["rise"]).strftime("%H:%M:%S"),
            "Max El": f"{p.get('max_el', 0):.0f}°",
            "LOS (UTC)": p["set"].strftime("%H:%M:%S"),
            "Duration": f"{(p['set']-p['rise']).total_seconds()/60:.1f} min",
        } for p in passes])
        st.dataframe(df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: GROUND STATION
# ============================================================================
elif page.endswith("Ground Station"):
    st.markdown('<div class="page-title">Ground Station</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-head">Hardware telemetry · ESP32 tracker · alt-az mount</div>',
                unsafe_allow_html=True)

    c = st.columns(4)
    with c[0]:
        metric_card("MCU LINK", "COM4", "→ ESP32 · CH340", GREEN, ACCENT)
    with c[1]:
        metric_card("I2C HEALTH", "99.6%", "→ retry @ 100kHz", GREEN, ACCENT)
    with c[2]:
        metric_card("COMPASS REF", "245.9°", "→ QMC5883L offset", ACCENT2, ACCENT)
    with c[3]:
        metric_card("MICROSTEP", "1600", "→ pulses/rev", AMBER, ACCENT)

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">SUBSYSTEM STATUS</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:14px; color:#cbd5e1; line-height:2.1;">
        <span class="status-dot live"></span> AZ motor — NEMA23 · DM542 · 2.84 A<br>
        <span class="status-dot live"></span> EL motor — NEMA23 · DM542 · 2.84 A<br>
        <span class="status-dot live"></span> MPU6050 IMU — 0x68 · OK<br>
        <span class="status-dot idle"></span> QMC5883L — 0x0D · startup-only read<br>
        <span class="status-dot live"></span> Limit switches — AZ GPIO5 · EL GPIO17<br>
        <span class="status-dot idle"></span> Power rail — 24V buck · brownout on EL start
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title">AXIS CALIBRATION</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:14px; color:#cbd5e1; line-height:2.1;">
        <b>AZ</b> — 25 steps/deg · 9:1 gearbox<br>
        <b>EL</b> — 8.33 steps/deg · 3:1 gearbox · dir −1<br>
        <b>PID AZ</b> — Kp 16 · Ki 64 · Kd 1.56<br>
        <b>PID EL</b> — Kp 9.6 · Ki 38.4 · Kd 0.934<br>
        <b>Phase margin</b> — 74°<br>
        <b>Comms</b> — USB Serial (primary) · WiFi AP (backup)
        </div>""", unsafe_allow_html=True)

    st.caption("Telemetry panel reflects last known calibration. Connect pyserial to COM4 "
               "for live readback when running locally.")
