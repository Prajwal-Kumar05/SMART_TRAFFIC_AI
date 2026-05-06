"""
app.py  —  Smart Traffic AI — Bengaluru
Run:  streamlit run app.py
Tabs: Live Map · Prediction · Signals · Emergency · YOLO Detection · Flood Risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import cv2
from datetime import datetime, timedelta
from PIL import Image
import io

from simulator  import generate_traffic_data, build_adjacency_matrix, NODES, EDGES
from models     import train_models, SignalOptimizer
from features   import compute_green_corridor, HOSPITALS, VEHICLE_TYPES
from flood_predictor import FloodRiskPredictor, find_nearest_safe_route, find_junction_route, NODE_ELEVATION, NODE_FLOOD_HISTORY
from yolo_detector   import YOLODetector, DENSITY_COLORS

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic AI — Bengaluru",
    page_icon="🚦", layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero banner ── */
.hero {
  background: linear-gradient(135deg, #050d18 0%, #0a1f35 50%, #0d2545 100%);
  border-radius: 16px; padding: 1.8rem 2.2rem; margin-bottom: 1.2rem;
  color: #fff; border: 1px solid rgba(0,180,216,.25);
  box-shadow: 0 4px 24px rgba(0,0,0,.4);
}
.hero h1 { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -.5px; }
.hero p  { margin: .35rem 0 0; font-size: .9rem; opacity: .7; }
.badge {
  display: inline-block; background: rgba(255,255,255,.07);
  border: 1px solid rgba(255,255,255,.18); border-radius: 99px;
  padding: .15rem .65rem; font-size: .72rem; margin: .55rem .2rem 0 0;
  letter-spacing: .3px;
}

/* ── KPI row ── */
.krow { display: flex; gap: .8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
.kpi {
  flex: 1; min-width: 105px; background: #0a1525; border-radius: 12px;
  padding: .85rem 1rem; color: #fff; border-left: 3px solid #00b4d8;
  border: 1px solid rgba(255,255,255,.06); border-left: 3px solid #00b4d8;
}
.kpi.r { border-left-color: #ef233c; }
.kpi.g { border-left-color: #52b788; }
.kpi.y { border-left-color: #f4a261; }
.kpi.p { border-left-color: #c77dff; }
.kpi .v { font-size: 1.65rem; font-weight: 800; line-height: 1.1; }
.kpi .l { font-size: .65rem; opacity: .55; margin-top: .2rem; text-transform: uppercase; letter-spacing: .4px; }

/* ── Section header ── */
.sec {
  font-size: 1rem; font-weight: 700; color: #e2e8f0;
  margin: 1.1rem 0 .5rem; padding-bottom: .3rem;
  border-bottom: 2px solid rgba(0,180,216,.2);
}

/* ── Alert boxes ── */
.al  { border-radius: 10px; padding: .8rem 1.1rem; margin: .35rem 0; font-size: .86rem; }
.r2  { background: #ef233c14; border: 1px solid #ef233c55; color: #fca5a5; }
.o2  { background: #f4a26114; border: 1px solid #f4a26155; color: #fdba74; }
.g2  { background: #52b78814; border: 1px solid #52b78855; color: #86efac; }
.b2  { background: #00b4d814; border: 1px solid #00b4d855; color: #7dd3fc; }
.p2  { background: #c77dff14; border: 1px solid #c77dff55; color: #d8b4fe; }
.cr2 { background: #7209b714; border: 1px solid #7209b755; color: #e9d5ff; }

/* ── Junction card ── */
.cnode {
  display: flex; align-items: center; gap: .7rem; background: #0a1525;
  border-radius: 9px; padding: .55rem .85rem; margin: .25rem 0;
  border: 1px solid rgba(255,255,255,.06);
}
.dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }

/* ── YOLO stat card ── */
.ycard {
  background: #0a1525; border-radius: 12px; padding: 1rem 1.1rem;
  border: 1px solid rgba(255,255,255,.07); text-align: center;
}
.ycard .big { font-size: 2rem; font-weight: 800; }
.ycard .sub { font-size: .68rem; opacity: .5; text-transform: uppercase; letter-spacing: .4px; margin-top: .15rem; }

/* ── Flood node card ── */
.fcard {
  background: #0a1525; border-radius: 10px; padding: .6rem .9rem;
  margin: .2rem 0; border-left: 3px solid; border-top: 1px solid rgba(255,255,255,.05);
}

/* ── Route step ── */
.rnode {
  display: flex; align-items: center; gap: .6rem; background: #0a1525;
  border-radius: 8px; padding: .5rem .75rem; margin: .2rem 0;
  border: 1px solid rgba(255,255,255,.05);
}
</style>
""", unsafe_allow_html=True)

# ── Plot layout defaults ───────────────────────────────────────────────────────
DL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,15,24,.85)",
    font=dict(color="#e2e8f0"), margin=dict(l=20, r=20, t=20, b=40)
)


def cc(c):
    if c < .3:  return "#52b788"
    if c < .5:  return "#95d5b2"
    if c < .65: return "#f4a261"
    if c < .8:  return "#e76f51"
    return "#ef233c"


def cl(c):
    if c < .3:  return "🟢 Free Flow"
    if c < .5:  return "🟡 Moderate"
    if c < .65: return "🟠 Heavy"
    if c < .8:  return "🔴 Congested"
    return "🚨 Severe"


# ── Data loaders ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    return generate_traffic_data(days=7), build_adjacency_matrix(15)


@st.cache_resource(show_spinner=False)
def load_models(_df, _adj):
    ph = st.empty(); bar = st.progress(0)
    def cb(ep, tot, loss):
        bar.progress(ep / tot)
        ph.info(f"⚙️ Training… epoch {ep}/{tot} | loss={loss:.5f}")
    m = train_models(_df, n_nodes=15, epochs=20, adj=_adj, progress_callback=cb)
    ph.empty(); bar.empty()
    return m


@st.cache_resource(show_spinner=False)
def load_rl():
    return SignalOptimizer()


@st.cache_resource(show_spinner=False)
def load_flood():
    return FloodRiskPredictor()


@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLODetector()


# ── Predict node ──────────────────────────────────────────────────────────────
def predict_node(models, df, nid):
    nd = df[df["node_id"] == nid].sort_values("timestamp").tail(12).copy()
    if len(nd) < 12:
        return np.zeros(6)
    nd["sn"] = nd["speed"] / 60; nd["dn"] = nd["density"] / 200
    nd["hs"] = np.sin(2 * np.pi * nd["hour"] / 24)
    nd["hc"] = np.cos(2 * np.pi * nd["hour"] / 24)
    nd["ds"] = np.sin(2 * np.pi * nd["day_of_week"] / 7)
    f = nd[["congestion", "sn", "dn", "hs", "hc", "ds"]].values.astype(np.float32)
    x = torch.tensor(f).unsqueeze(0)
    with torch.no_grad():
        lp  = models["lstm"](x)
        ge  = models["gat"](models["snap"], models["adj"])
        out = models["fuse"](ge[nid].unsqueeze(0), lp)
    return out.squeeze().numpy()


# ── Base map builder ──────────────────────────────────────────────────────────
def base_map(df_snap, extras=None, flood_risks=None):
    nm = {n["id"]: n for n in NODES}
    traces = []
    for i, j, _ in EDGES:
        ni, nj = nm[i], nm[j]
        traces.append(go.Scattermapbox(
            lat=[ni["lat"], nj["lat"], None], lon=[ni["lon"], nj["lon"], None],
            mode="lines", line=dict(width=1.5, color="rgba(255,255,255,.08)"),
            hoverinfo="none", showlegend=False))

    if flood_risks:
        colors = [FloodRiskPredictor.risk_color(flood_risks.get(int(r["node_id"]), 0))
                  for _, r in df_snap.iterrows()]
        sizes  = [12 + flood_risks.get(int(r["node_id"]), 0) * 32
                  for _, r in df_snap.iterrows()]
        hover  = [f"<b>{r['node_name']}</b><br>"
                  f"Flood Risk: {flood_risks.get(int(r['node_id']), 0):.0%}<br>"
                  f"{FloodRiskPredictor.risk_label(flood_risks.get(int(r['node_id']), 0))}"
                  for _, r in df_snap.iterrows()]
    else:
        colors = [cc(c) for c in df_snap["congestion"]]
        sizes  = [12 + c * 28 for c in df_snap["congestion"]]
        hover  = [f"<b>{r['node_name']}</b><br>{r['congestion']:.0%} · {r['speed']:.0f}km/h"
                  for _, r in df_snap.iterrows()]

    traces.append(go.Scattermapbox(
        lat=df_snap["lat"], lon=df_snap["lon"], mode="markers+text",
        marker=dict(size=sizes, color=colors, opacity=.9),
        text=df_snap["node_name"], textfont=dict(size=8, color="white"),
        textposition="top right", hovertext=hover,
        hoverinfo="text", showlegend=False))

    if extras:
        traces += extras

    fig = go.Figure(data=traces)
    fig.update_layout(
        mapbox=dict(style="carto-darkmatter", center=dict(lat=12.97, lon=77.62), zoom=10.8),
        margin=dict(l=0, r=0, t=0, b=0), height=440, paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Load resources ────────────────────────────────────────────────────────────
with st.spinner("Loading traffic data…"):
    df, adj = load_data()
with st.spinner("Training AI models…"):
    models = load_models(df, adj)
with st.spinner("Initialising signal optimizer…"):
    rl = load_rl()
with st.spinner("Loading flood predictor…"):
    flood_pred = load_flood()
with st.spinner("Loading YOLO detector…"):
    yolo = load_yolo()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Smart Traffic AI")
    st.caption("Bengaluru Road Network v3.0")
    st.divider()
    sim_off  = st.slider("🕐 Time offset (hrs)", -168, 0, -2, 1)
    sel_node = st.selectbox("📍 Focus junction", range(15),
                             format_func=lambda x: NODES[x]["name"])
    st.divider()
    st.markdown("**AI Stack:** GAT · LSTM · Fusion · RL · LSTM Flood")
    st.markdown("**Features:** 🗺️ Live Map · 📈 Prediction · 🚦 Signals · 🚑 Corridor · 📷 YOLO · 🌊 Flood")
    st.caption("PyTorch · Streamlit · YOLOv8")

# ── Snapshot ──────────────────────────────────────────────────────────────────
sim_now = datetime.now() + timedelta(hours=sim_off)
sim_now = sim_now.replace(second=0, microsecond=0, minute=(sim_now.minute // 5) * 5)
all_t = df["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
ct = all_t.iloc[(all_t - sim_now).abs().argmin()]
snap = df[df["timestamp"] == ct].copy()

avg_c  = snap["congestion"].mean()
avg_s  = snap["speed"].mean()
worst = snap.loc[snap["congestion"].idxmax()]
selected_data = snap[snap["node_id"] == sel_node].iloc[0]
n_sev  = (snap["congestion"] > .75).sum()
kc     = "r" if avg_c > .65 else "y" if avg_c > .4 else "g"

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>🏙️ Smart Traffic AI — Bengaluru</h1>
  <p>GAT + LSTM congestion forecasting · RL signal optimisation · Emergency green corridor · YOLOv8 vehicle detection · LSTM flood risk routing</p>
  <span class="badge">🕸️ Graph Attention Net</span>
  <span class="badge">📈 LSTM Forecast</span>
  <span class="badge">🤖 Q-Learning RL</span>
  <span class="badge">🚑 Green Corridor</span>
  <span class="badge">📷 YOLOv8</span>
  <span class="badge">🌊 Flood Risk</span>
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class="krow">
  <div class="kpi {kc}"><div class="v">{avg_c:.0%}</div><div class="l">City Congestion</div></div>
  <div class="kpi g"><div class="v">{avg_s:.0f}</div><div class="l">Avg Speed km/h</div></div>
  <div class="kpi r"><div class="v">{n_sev}</div><div class="l">Severe Junctions</div></div>

  <div class="kpi y">
    <div class="v">{selected_data['node_name'].split()[0]}</div>
    <div class="l">Selected: {selected_data['congestion']:.0%}</div>
  </div>

  <div class="kpi">
    <div class="v">{ct.strftime('%H:%M')}</div>
    <div class="l">{ct.strftime('%a %d %b')}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
T = st.tabs(["🗺️ Live Map", "📈 Prediction", "🚦 Signals",
              "🚑 Emergency", "📷 YOLO Detection", "🌊 Flood Risk"])

# ════════════════════════════════════════════════════
# TAB 1 — Live Map
# ════════════════════════════════════════════════════
with T[0]:
    ca, cb = st.columns([3, 1])
    with ca:
        st.markdown('<div class="sec">🗺️ Live Road Network — Bengaluru</div>', unsafe_allow_html=True)
        st.plotly_chart(base_map(snap), use_container_width=True, key="m1")
    with cb:
        st.markdown('<div class="sec">📋 Junction Status</div>', unsafe_allow_html=True)
        for _, r in snap.sort_values("congestion", ascending=False).iterrows():
            col = cc(r["congestion"])
            st.markdown(f"""<div style='background:#0a1525;border-left:3px solid {col};
border-radius:8px;padding:.45rem .75rem;margin:.25rem 0;'>
<b style='font-size:.85rem;'>{r['node_name']}</b><br>
<span style='color:{col};font-weight:700;font-size:.82rem;'>{r['congestion']:.0%}</span>
<span style='opacity:.45;font-size:.77rem;'> · {r['speed']:.0f} km/h</span>
</div>""", unsafe_allow_html=True)

    ds = snap.sort_values("congestion")
    fig = go.Figure(go.Bar(
        x=ds["node_name"], y=ds["congestion"],
        marker_color=[cc(c) for c in ds["congestion"]],
        text=[f"{c:.0%}" for c in ds["congestion"]], textposition="outside"))
    fig.update_layout(yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                      height=280, **DL)
    st.plotly_chart(fig, use_container_width=True, key="b1")


# ════════════════════════════════════════════════════
# TAB 2 — Prediction
# ════════════════════════════════════════════════════
with T[1]:
    st.markdown(f'<div class="sec">📈 30-Min Forecast — {NODES[sel_node]["name"]}</div>',
                unsafe_allow_html=True)
    fc   = predict_node(models, df, sel_node)
    hist = df[df["node_id"] == sel_node].sort_values("timestamp").tail(24)
    ht, hv = list(hist["timestamp"]), list(hist["congestion"])
    ft = [ht[-1] + timedelta(minutes=5 * (i + 1)) for i in range(6)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ht, y=hv, mode="lines+markers", name="Historical",
                             line=dict(color="#00b4d8", width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=[ht[-1]] + ft, y=[hv[-1]] + list(fc),
                             mode="lines", showlegend=False,
                             line=dict(color="#f4a261", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=ft, y=list(fc), mode="lines+markers",
                             name="GAT+LSTM Forecast",
                             line=dict(color="#f4a261", width=3),
                             marker=dict(size=9, symbol="diamond")))
    fig.add_hrect(y0=.75, y1=1, fillcolor="rgba(239,35,60,.07)", line_width=0)
    fig.update_layout(yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                      legend=dict(orientation="h", y=1.05), height=330, **DL)
    st.plotly_chart(fig, use_container_width=True, key="pred")

    st.dataframe(pd.DataFrame({
        "Time":       [t.strftime("%H:%M") for t in ft],
        "Congestion": [f"{v:.1%}" for v in fc],
        "Speed est":  [f"{60*(1-v):.0f} km/h" for v in fc],
        "Status":     [cl(v) for v in fc],
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec">🌡️ City-Wide Forecast Heatmap</div>', unsafe_allow_html=True)
    rows = []
    for nid in range(15):
        f2 = predict_node(models, df, nid)
        for s, v in enumerate(f2):
            rows.append({"Node": NODES[nid]["name"], "Step": f"+{(s+1)*5}min", "C": float(v)})
    ph = pd.DataFrame(rows).pivot(index="Node", columns="Step", values="C")
    cols = [f"+{i*5}min" for i in range(1, 7)]
    ph = ph[[c for c in cols if c in ph.columns]]
    fig2 = go.Figure(go.Heatmap(
        z=ph.values, x=ph.columns.tolist(), y=ph.index.tolist(),
        colorscale=[[0, "#52b788"], [.5, "#f4a261"], [1, "#ef233c"]],
        zmin=0, zmax=1,
        text=[[f"{v:.0%}" for v in row] for row in ph.values],
        texttemplate="%{text}",
        colorbar=dict(tickformat=".0%")))
    fig2.update_layout(height=400, **DL)
    st.plotly_chart(fig2, use_container_width=True, key="heat2")


# ════════════════════════════════════════════════════
# TAB 3 — Signals
# ════════════════════════════════════════════════════
with T[2]:
    st.markdown('<div class="sec">🚦 RL Signal Optimizer</div>', unsafe_allow_html=True)
    sa, sb = st.columns(2)
    with sa:
        cns = st.slider("🡹 N-S Congestion", 0.0, 1.0, 0.75, .01)
        cew = st.slider("🡺 E-W Congestion", 0.0, 1.0, 0.35, .01)
        res = rl.optimize(cns, cew)
        st.markdown(f"""<div class="al b2"><b>🤖 {res['action']}</b><br><br>
🟢 N-S Green: <b>{res['green_ns']}s</b> &nbsp;|&nbsp; Red: <b>{60-res['green_ns']}s</b><br>
🟢 E-W Green: <b>{res['green_ew']}s</b> &nbsp;|&nbsp; Red: <b>{60-res['green_ew']}s</b><br><br>
⏱️ Wait: <b>{res['wait_before']:.0f}s</b> → <b>{res['wait_after']:.0f}s</b>
&nbsp; 📉 <b style='color:#52b788;'>{res['reduction']:.1f}% reduction</b></div>""",
                    unsafe_allow_html=True)
    with sb:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=res["reduction"],
            number={"suffix": "%", "font": {"color": "white", "size": 34}},
            title={"text": "Wait Reduction", "font": {"color": "white"}},
            gauge={"axis": {"range": [0, 60]}, "bar": {"color": "#52b788"},
                  "steps": [
    {"range": [0, 20], "color": "rgba(82, 183, 136, 0.2)"},
    {"range": [20, 50], "color": "rgba(244, 162, 97, 0.2)"},
    {"range": [50, 100], "color": "rgba(239, 35, 60, 0.2)"}
]}))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"),
                            height=240, margin=dict(l=30, r=30, t=40, b=10))
        st.plotly_chart(fig_g, use_container_width=True, key="gauge3")

    q  = rl.q_heatmap()
    lb = ["Very Low", "Low", "Med", "High", "Very High"]
    fig_q = go.Figure(go.Heatmap(
        z=q, x=[f"EW:{l}" for l in lb], y=[f"NS:{l}" for l in lb],
        colorscale="Blues",
        text=[[f"{v:.2f}" for v in row] for row in q], texttemplate="%{text}"))
    fig_q.update_layout(height=290, **DL)
    st.plotly_chart(fig_q, use_container_width=True, key="qt3")


# ════════════════════════════════════════════════════
# TAB 4 — Emergency Green Corridor
# ════════════════════════════════════════════════════
with T[3]:
    st.markdown('<div class="sec">🚑 Emergency Green Corridor System</div>', unsafe_allow_html=True)
    st.markdown("Computes the fastest route and pre-clears traffic signals 90 seconds before the emergency vehicle arrives at each junction.")

    e1, e2, e3 = st.columns(3)
    with e1: vt   = st.selectbox("🚨 Vehicle Type", list(VEHICLE_TYPES.keys()))
    with e2: ori  = st.selectbox("📍 Origin Junction", range(15), format_func=lambda x: NODES[x]["name"])
    with e3: hosp = st.selectbox("🏥 Destination Hospital", list(HOSPITALS.keys()))

    if st.button("🚨 ACTIVATE GREEN CORRIDOR", use_container_width=True, type="primary"):
        with st.spinner("Computing route and cascading signals…"):
            cor = compute_green_corridor(ori, hosp, vt, snap)
        if "error" in cor:
            st.error(cor["error"])
        else:
            uc = "r2" if cor["urgency"] == "CRITICAL" else "o2"
            st.markdown(
                f'<div class="al {uc}">⚡ <b>{cor["urgency"]}</b> — {vt} '
                f'from <b>{NODES[ori]["name"]}</b> to <b>{hosp}</b></div>',
                unsafe_allow_html=True)

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Distance",      f"{cor['dist_km']} km")
            k2.metric("Normal ETA",    f"{cor['eta_base_min']} min")
            k3.metric("With Corridor", f"{cor['eta_corridor_min']} min")
            k4.metric("Time Saved",    f"{cor['eta_saved_min']} min",
                      delta=f"-{cor['eta_saved_min']} min")
            k5.metric("Survival Boost", f"+{cor['lives_impact']:.1f}%")

            nm2  = {n["id"]: n for n in NODES}
            lats = [nm2[n]["lat"] for n in cor["path"]]
            lons = [nm2[n]["lon"] for n in cor["path"]]
            hn   = cor["hospital_node"]
            vc   = cor["vehicle"]["color"]

            extras = [
                go.Scattermapbox(lat=lats, lon=lons, mode="lines",
                                 line=dict(width=7, color=vc),
                                 name="Emergency Route", showlegend=True),
                go.Scattermapbox(lat=[hn["lat"]], lon=[hn["lon"]], mode="markers+text",
                                 marker=dict(size=22, color="#ffffff"),
                                 text=[f"🏥 {hosp}"], textfont=dict(color="white", size=11),
                                 textposition="top right", name=hosp, showlegend=True),
            ]
            st.plotly_chart(base_map(snap, extras), use_container_width=True, key="cm4")

            st.markdown('<div class="sec">⏱️ Signal Cascade Timeline</div>', unsafe_allow_html=True)
            for seg in cor["segments"]:
                c2 = "#52b788" if seg["status"] == "🟢 CLEARED" else "#ef233c"
                st.markdown(f"""<div class="cnode">
<div class="dot" style="background:{c2};"></div>
<div style="flex:1;"><b>{seg['node_name']}</b>
<span style="opacity:.45;font-size:.76rem;"> {seg['seg_dist_km']} km</span></div>
<div style="text-align:right;font-size:.78rem;">
<b style="color:{c2};">{seg['status']}</b><br>
<span style="opacity:.45;">+{seg['arrival_min']:.1f} min · pre-clear {seg['pre_clear_min']:.1f} min</span>
</div></div>""", unsafe_allow_html=True)

            fig_eta = go.Figure()
            fig_eta.add_trace(go.Bar(name="Without Corridor", x=["ETA"],
                                     y=[cor["eta_base_min"]], marker_color="#ef233c",
                                     text=[f"{cor['eta_base_min']} min"], textposition="outside"))
            fig_eta.add_trace(go.Bar(name="With Corridor", x=["ETA"],
                                     y=[cor["eta_corridor_min"]], marker_color="#52b788",
                                     text=[f"{cor['eta_corridor_min']} min"], textposition="outside"))
            fig_eta.update_layout(barmode="group", yaxis_title="Minutes",
                                  height=250, **DL)
            st.plotly_chart(fig_eta, use_container_width=True, key="eta4")
    else:
        st.markdown('<div class="al b2">👆 Select vehicle type, origin junction, and hospital — then click <b>ACTIVATE</b></div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 5 — YOLO Detection
# ════════════════════════════════════════════════════
with T[4]:
    st.markdown('<div class="sec">📷 YOLOv8 Vehicle Detection & Traffic Density</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="al b2">
<b>What this does:</b> Upload a traffic camera image and the system will automatically
detect every vehicle in the frame — cars, trucks, buses, and motorcycles — count them,
estimate the current traffic density level, and recommend the optimal signal timing.
Emergency vehicles are flagged instantly for priority signal override.
If the YOLOv8 model weights are not present, a realistic demo mode is used.
</div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a traffic image (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        help="Any road / intersection image works. The detector runs on the uploaded frame.")

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not decode the image. Please try a different file.")
        else:
            with st.spinner("Running vehicle detection…"):
                result = yolo.detect(frame)

            # Status badge
            if result["using_mock"]:
                st.markdown('<div class="al o2">⚠️ <b>Demo Mode</b> — YOLOv8 model weights not found. '
                            'Showing realistic simulated detections. Place <code>yolov8n.pt</code> '
                            'in the <code>models/</code> folder to enable live detection.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="al g2">✅ <b>YOLOv8 Active</b> — Real-time detection running.</div>',
                            unsafe_allow_html=True)

            # Stat cards
            dc = DENSITY_COLORS.get(result["density_label"], "#00b4d8")
            sc = result["signal_recommendation"]
            pri_color = {"none": "#52b788", "low": "#95d5b2",
                         "medium": "#f4a261", "high": "#ef233c", "critical": "#7209b7"}
            pc = pri_color.get(sc["priority"], "#52b788")

            ya, yb, yc, yd = st.columns(4)
            ya.markdown(f'<div class="ycard"><div class="big" style="color:#00b4d8;">{result["vehicle_count"]}</div><div class="sub">Vehicles Detected</div></div>', unsafe_allow_html=True)
            yb.markdown(f'<div class="ycard"><div class="big" style="color:{dc};">{result["density_label"]}</div><div class="sub">Traffic Density</div></div>', unsafe_allow_html=True)
            yc.markdown(f'<div class="ycard"><div class="big" style="color:{"#ef233c" if result["emergency_count"] > 0 else "#52b788"};">{result["emergency_count"]}</div><div class="sub">Emergency Vehicles</div></div>', unsafe_allow_html=True)
            yd.markdown(f'<div class="ycard"><div class="big" style="color:{pc};">+{sc["extend_green"]}s</div><div class="sub">Signal Extension</div></div>', unsafe_allow_html=True)

            # Signal recommendation
            sig_cls = "r2" if sc["priority"] == "critical" else \
                      "o2" if sc["priority"] == "high" else \
                      "y2" if sc["priority"] == "medium" else "g2"
            st.markdown(f'<div class="al {sig_cls}" style="margin-top:.6rem;">🚦 <b>Signal Recommendation:</b> {sc["description"]}</div>',
                        unsafe_allow_html=True)

            # Annotated image + detection table
            img_col, tbl_col = st.columns([1.4, 1])
            with img_col:
                st.markdown('<div class="sec">🖼️ Annotated Detection Frame</div>', unsafe_allow_html=True)
                annotated_rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)

            with tbl_col:
                st.markdown('<div class="sec">📊 Detection Breakdown</div>', unsafe_allow_html=True)
                dets = result["detections"]
                if dets:
                    df_det = pd.DataFrame(dets)
                    counts = df_det["class_name"].value_counts().reset_index()
                    counts.columns = ["Vehicle Type", "Count"]
                    st.dataframe(counts, use_container_width=True, hide_index=True)

                    # Density gauge
                    fig_dens = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=result["traffic_density"] * 100,
                        number={"suffix": "%", "font": {"color": "white", "size": 30}},
                        title={"text": "Density Score", "font": {"color": "white", "size": 13}},
                        gauge={"axis": {"range": [0, 100]},
                               "bar": {"color": dc.replace("#", "#")},
                               "bgcolor": "#1e2a38",
                               "steps": [
    {"range": [0, 20],  "color": "rgba(82, 183, 136, 0.2)"},
    {"range": [20, 50], "color": "rgba(244, 162, 97, 0.2)"},
    {"range": [50, 100],"color": "rgba(239, 35, 60, 0.2)"}
]}))
                    fig_dens.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                           font=dict(color="white"),
                                           height=210, margin=dict(l=20, r=20, t=40, b=10))
                    st.plotly_chart(fig_dens, use_container_width=True, key="yd_gauge")
                else:
                    st.info("No vehicles detected in this frame.")

    else:
        st.markdown('<div class="al b2" style="margin-top:.5rem;">📤 Upload a traffic camera image above to run vehicle detection.</div>',
                    unsafe_allow_html=True)

        # Demo info cards
        d1, d2, d3 = st.columns(3)
        d1.markdown("""<div class="ycard" style="text-align:left;padding:1.1rem;">
<b style="color:#00b4d8;">🚗 Vehicle Types</b><br><br>
<small style="opacity:.75;line-height:1.8;">
Cars · Motorcycles<br>Buses · Trucks<br>Emergency vehicles</small></div>""", unsafe_allow_html=True)
        d2.markdown("""<div class="ycard" style="text-align:left;padding:1.1rem;">
<b style="color:#f4a261;">📊 Density Levels</b><br><br>
<small style="opacity:.75;line-height:1.8;">
🟢 Low (0–10 vehicles)<br>🟡 Medium (10–25)<br>🔴 High (25–50)<br>🚨 Very High (50+)</small></div>""", unsafe_allow_html=True)
        d3.markdown("""<div class="ycard" style="text-align:left;padding:1.1rem;">
<b style="color:#52b788;">🚦 Signal Actions</b><br><br>
<small style="opacity:.75;line-height:1.8;">
Normal cycle<br>+10s green extension<br>+20s green extension<br>🚨 Emergency override</small></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
# TAB 6 — Flood Risk
# ════════════════════════════════════════════════════
with T[5]:
    st.markdown('<div class="sec">🌊 Flood Risk Dashboard — Bengaluru Road Network</div>',
                unsafe_allow_html=True)

    st.markdown("""<div class="al b2">
Enter the current <b>rain intensity</b> below. The LSTM flood model will compute
waterlogging risk for every junction using elevation data, historical flood incidents,
and a rolling rain accumulation window. High-risk roads are excluded from routing —
and the <b>nearest safe route</b> from any flooded area is shown directly on this map.
</div>""", unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    f_col1, f_col2, f_col3 = st.columns([2, 1, 1])
    with f_col1:
        rain_intensity = st.slider(
            "🌧️ Rain Intensity (0 = dry · 100 = extreme monsoon)",
            min_value=0, max_value=100, value=45, step=1,
            help="Represents mm/hr equivalent rainfall intensity across Bengaluru")
    with f_col2:
        origin_flood = st.selectbox(
            "📍 Your Location (find safe route from here)",
            range(15), index=0,
            format_func=lambda x: NODES[x]["name"])
    with f_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_flood = st.button("🔍 Analyse Flood Risk", use_container_width=True, type="primary")

    # ── Run prediction ────────────────────────────────────────────────────────
    if run_flood or rain_intensity > 0:
        with st.spinner("Running LSTM flood prediction…"):
            flood_risks   = flood_pred.predict_all(float(rain_intensity))
            penalties     = flood_pred.get_flood_penalties(flood_risks)
            alerts        = flood_pred.generate_alerts(
                flood_risks, {n["id"]: n["name"] for n in NODES})
            cong_snap     = dict(zip(snap["node_id"], snap["congestion"]))
            safe_route    = find_nearest_safe_route(origin_flood, flood_risks, penalties, cong_snap)

        # ── KPI strip ─────────────────────────────────────────────────────────
        flooded  = sum(1 for r in flood_risks.values() if r >= FloodRiskPredictor.CRITICAL_THRESHOLD)
        high_r   = sum(1 for r in flood_risks.values() if FloodRiskPredictor.ALERT_THRESHOLD <= r < FloodRiskPredictor.CRITICAL_THRESHOLD)
        avg_risk = np.mean(list(flood_risks.values()))
        safe_ct  = sum(1 for r in flood_risks.values() if r < FloodRiskPredictor.FLOOD_THRESHOLD)

        fk1, fk2, fk3, fk4, fk5 = st.columns(5)
        fk1.metric("🌧️ Rain Intensity", f"{rain_intensity}/100")
        fk2.metric("🌊 Flooded Roads",  flooded,    delta=f"{flooded} closed" if flooded else "All clear", delta_color="inverse")
        fk3.metric("🔴 High Risk",      high_r)
        fk4.metric("✅ Safe Roads",     safe_ct)
        fk5.metric("📊 Avg City Risk",  f"{avg_risk:.0%}")

        # ── Map + Safe route ──────────────────────────────────────────────────
        map_col, route_col = st.columns([1.6, 1])

        with map_col:
            st.markdown('<div class="sec">🗺️ Flood Risk Map</div>', unsafe_allow_html=True)

            flood_extras = []
            nm_map = {n["id"]: n for n in NODES}
            # Draw safe route on map if found
            if safe_route and len(safe_route["path"]) > 1:
                r_lats = [nm_map[nid]["lat"] for nid in safe_route["path"]]
                r_lons = [nm_map[nid]["lon"] for nid in safe_route["path"]]
                flood_extras.append(go.Scattermapbox(
                    lat=r_lats, lon=r_lons, mode="lines",
                    line=dict(width=5, color="#00b4d8"),
                    name="Safe Route", showlegend=True))
                # Destination marker
                dest = nm_map[safe_route["target_id"]]
                flood_extras.append(go.Scattermapbox(
                    lat=[dest["lat"]], lon=[dest["lon"]], mode="markers+text",
                    marker=dict(size=20, color="#00b4d8"),
                    text=[f"✅ {dest['name']}"], textfont=dict(color="white", size=10),
                    textposition="top right", name="Safe Destination", showlegend=True))

            st.plotly_chart(base_map(snap, flood_extras, flood_risks),
                            use_container_width=True, key="flood_map")

            # Map legend
            leg_items = [
                ("#52b788", "Safe (< 25%)"),
                ("#f4a261", "Moderate (45–55%)"),
                ("#ef233c", "High Risk (70–85%)"),
                ("#7209b7", "Flooded (> 85%)"),
                ("#00b4d8", "Safe Route"),
            ]
            legend_html = "<div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:.4rem;'>"
            for color, label in leg_items:
                legend_html += (f"<div style='display:flex;align-items:center;gap:.35rem;font-size:.75rem;opacity:.8;'>"
                                f"<div style='width:12px;height:12px;border-radius:50%;background:{color};'></div>"
                                f"{label}</div>")
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

        with route_col:
            # ── Safe Route Panel ──────────────────────────────────────────────
            st.markdown('<div class="sec">🧭 Nearest Safe Route</div>', unsafe_allow_html=True)
            origin_risk = flood_risks.get(origin_flood, 0)
            origin_name = NODES[origin_flood]["name"]

            if origin_risk >= FloodRiskPredictor.CRITICAL_THRESHOLD:
                st.markdown(f'<div class="al r2">🌊 <b>{origin_name}</b> is currently <b>FLOODED</b>. '
                            f'Stay in place — road is closed.</div>', unsafe_allow_html=True)
            elif origin_risk >= FloodRiskPredictor.ALERT_THRESHOLD:
                st.markdown(f'<div class="al o2">🔴 <b>{origin_name}</b> has HIGH flood risk ({origin_risk:.0%}). '
                            f'Finding nearest safe exit…</div>', unsafe_allow_html=True)

            if safe_route:
                dest_risk = flood_risks.get(safe_route["target_id"], 0)
                st.markdown(f"""<div class="al g2">
✅ <b>Safe Destination: {safe_route['target_name']}</b><br>
Risk at destination: {dest_risk:.0%} ({FloodRiskPredictor.risk_label(dest_risk)})<br>
Distance: <b>{safe_route['dist_km']} km</b> · ETA: <b>{safe_route['time_min']:.0f} min</b>
</div>""", unsafe_allow_html=True)

                st.markdown("**Route Steps:**")
                for idx, (nid, nname) in enumerate(zip(safe_route["path"], safe_route["path_names"])):
                    r = flood_risks.get(nid, 0)
                    rc = FloodRiskPredictor.risk_color(r)
                    icon = "🟢" if r < 0.45 else "🟡" if r < 0.55 else "🔴"
                    st.markdown(f"""<div class="rnode">
<span style="font-size:.85rem;opacity:.5;">{idx+1}.</span>
<div class="dot" style="background:{rc};"></div>
<div style="flex:1;font-size:.85rem;"><b>{nname}</b></div>
<span style="font-size:.75rem;color:{rc};">{icon} {r:.0%}</span>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown('<div class="al o2">⚠️ No safe route found — all nearby roads are flooded. '
                            'Await BBMP advisory.</div>', unsafe_allow_html=True)

        # ── Route Between Junctions ───────────────────────────────────────────
        st.markdown('<div class="sec">🛣️ Route Between Junctions — Flood-Aware</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Find the safest, fastest route between any two junctions, "
            "automatically avoiding flooded or closed roads.",
            unsafe_allow_html=True,
        )

        rj_c1, rj_c2, rj_c3 = st.columns([2, 2, 1])
        with rj_c1:
            rj_src = st.selectbox(
                "🟢 From Junction",
                range(15),
                format_func=lambda x: NODES[x]["name"],
                key="rj_src",
            )
        with rj_c2:
            rj_dst = st.selectbox(
                "🔴 To Junction",
                range(15),
                format_func=lambda x: NODES[x]["name"],
                index=5,
                key="rj_dst",
            )
        with rj_c3:
            st.write("")
            rj_find = st.button("🔍 Find Route", use_container_width=True, key="rj_find")

        if rj_find:
            if rj_src == rj_dst:
                st.markdown(
                    '<div class="al o2">⚠️ Source and destination are the same junction.</div>',
                    unsafe_allow_html=True,
                )
            else:
                with st.spinner("Computing flood-aware route…"):
                    rj_result = find_junction_route(
                        rj_src, rj_dst, flood_risks, penalties, cong_snap, k=3
                    )

                src_risk = rj_result.get("source_risk", 0)
                dst_risk = rj_result.get("target_risk", 0)

                # Endpoint risk banners
                ep_c1, ep_c2 = st.columns(2)
                with ep_c1:
                    src_color = FloodRiskPredictor.risk_color(src_risk)
                    src_label = FloodRiskPredictor.risk_label(src_risk)
                    st.markdown(
                        f"<div class='al' style='border-left:4px solid {src_color};padding:.55rem .8rem;'>"
                        f"🟢 <b>{NODES[rj_src]['name']}</b><br>"
                        f"<span style='font-size:.78rem;opacity:.8;'>{src_label} — {src_risk:.0%} flood risk</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with ep_c2:
                    dst_color = FloodRiskPredictor.risk_color(dst_risk)
                    dst_label = FloodRiskPredictor.risk_label(dst_risk)
                    st.markdown(
                        f"<div class='al' style='border-left:4px solid {dst_color};padding:.55rem .8rem;'>"
                        f"🔴 <b>{NODES[rj_dst]['name']}</b><br>"
                        f"<span style='font-size:.78rem;opacity:.8;'>{dst_label} — {dst_risk:.0%} flood risk</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                if rj_result.get("blocked") or not rj_result.get("routes"):
                    st.markdown(
                        '<div class="al r2">🚫 No passable route found — all paths between these '
                        "junctions pass through flooded or closed roads. "
                        "Please wait for flood levels to recede or choose a different destination.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    rj_routes = rj_result["routes"]
                    fastest   = rj_routes[0]

                    # Summary banner for fastest route
                    st.markdown(
                        f'<div class="al g2">✅ <b>Best Route Found</b> — '
                        f'{fastest["route_label"]}<br>'
                        f'<span style="font-size:.8rem;opacity:.85;">'
                        f'🕐 {fastest["time_min"]:.0f} min &nbsp;·&nbsp; '
                        f'📏 {fastest["dist_km"]} km &nbsp;·&nbsp; '
                        f'{fastest["hops"]} hop{"s" if fastest["hops"] != 1 else ""}'
                        f'</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Draw route on map
                    rj_lats = [nm_map[nid]["lat"] for nid in fastest["path"]]
                    rj_lons = [nm_map[nid]["lon"] for nid in fastest["path"]]
                    rj_extras = list(flood_extras) + [
                        go.Scattermapbox(
                            lat=rj_lats, lon=rj_lons,
                            mode="lines+markers",
                            line=dict(width=4, color="#f4a261"),
                            marker=dict(size=10, color="#f4a261"),
                            name="Junction Route",
                            showlegend=True,
                        )
                    ]
                    st.plotly_chart(
                        base_map(snap, rj_extras, flood_risks),
                        use_container_width=True,
                        key="rj_map",
                    )

                    # All route options
                    st.markdown("**Route Options:**")
                    for i, route in enumerate(rj_routes):
                        badge = "🥇 Fastest" if i == 0 else f"Option {i + 1}"
                        st.markdown(
                            f"**{badge}**: {' → '.join(route['path_names'])} "
                            f"| {route['time_min']:.0f} min · {route['dist_km']} km · {route['hops']} hop{'s' if route['hops'] != 1 else ''}"
                        )

                    # Step-by-step breakdown for fastest route
                    st.markdown("**Step-by-step (fastest route):**")
                    for idx, (nid, nname) in enumerate(
                        zip(fastest["path"], fastest["path_names"])
                    ):
                        r      = flood_risks.get(nid, 0)
                        rc     = FloodRiskPredictor.risk_color(r)
                        icon   = "🟢" if r < 0.45 else "🟡" if r < 0.55 else "🔴"
                        closed = penalties.get(nid, 0) == float("inf")
                        closed_badge = "&nbsp;🚫 CLOSED" if closed else ""
                        st.markdown(
                            f"<div class='rnode'>"
                            f"<span style='font-size:.85rem;opacity:.5;'>{idx + 1}.</span>"
                            f"<div class='dot' style='background:{rc};'></div>"
                            f"<div style='flex:1;font-size:.85rem;'><b>{nname}</b>{closed_badge}</div>"
                            f"<span style='font-size:.75rem;color:{rc};'>{icon} {r:.0%}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        # ── Junction flood details ─────────────────────────────────────────────
        st.markdown('<div class="sec">📋 All Junctions — Flood Risk Details</div>',
                    unsafe_allow_html=True)
        sorted_nodes = sorted(flood_risks.items(), key=lambda x: x[1], reverse=True)

        grid_cols = st.columns(3)
        for idx, (nid, risk) in enumerate(sorted_nodes):
            node = NODES[nid]
            rc = FloodRiskPredictor.risk_color(risk)
            rl = FloodRiskPredictor.risk_label(risk)
            elev = NODE_ELEVATION[nid]
            hist = NODE_FLOOD_HISTORY[nid]
            closed = penalties.get(nid, 0) == float('inf')
            with grid_cols[idx % 3]:
                st.markdown(f"""<div class="fcard" style="border-left-color:{rc};">
<div style="display:flex;justify-content:space-between;align-items:center;">
  <b style="font-size:.88rem;">{node['name']}</b>
  <span style="color:{rc};font-weight:700;font-size:.82rem;">{risk:.0%}</span>
</div>
<div style="margin-top:.3rem;font-size:.75rem;opacity:.65;">
  {rl} {"&nbsp;🚫 CLOSED" if closed else ""}<br>
  ⛰️ Elev: {elev}m &nbsp;·&nbsp; 📅 Hist: {hist} incidents/season
</div></div>""", unsafe_allow_html=True)

        # ── Alerts ────────────────────────────────────────────────────────────
        if alerts:
            st.markdown('<div class="sec">🚨 Active Flood Alerts</div>', unsafe_allow_html=True)
            for alert in alerts:
                cls = "cr2" if alert["risk"] >= FloodRiskPredictor.CRITICAL_THRESHOLD else "r2"
                st.markdown(f"""<div class="al {cls}">
<b>{alert['level']} — {alert['node_name']}</b>
<span style='float:right;opacity:.7;font-size:.8rem;'>Risk: {alert['risk']:.0%}</span><br>
<span style='font-size:.82rem;opacity:.85;'>{alert['action']}</span>
</div>""", unsafe_allow_html=True)

        # ── Risk bar chart ────────────────────────────────────────────────────
        st.markdown('<div class="sec">📊 Flood Risk by Junction</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame([
            {"Junction": NODES[nid]["name"], "Risk": risk,
             "Color": FloodRiskPredictor.risk_color(risk)}
            for nid, risk in sorted(flood_risks.items(), key=lambda x: x[1], reverse=True)
        ])
        fig_flood = go.Figure(go.Bar(
            x=chart_df["Junction"], y=chart_df["Risk"],
            marker_color=chart_df["Color"],
            text=[f"{v:.0%}" for v in chart_df["Risk"]], textposition="outside"))
        fig_flood.add_hline(y=FloodRiskPredictor.CRITICAL_THRESHOLD,
                            line_dash="dash", line_color="#7209b7",
                            annotation_text="Critical threshold", annotation_font_color="#7209b7")
        fig_flood.add_hline(y=FloodRiskPredictor.ALERT_THRESHOLD,
                            line_dash="dash", line_color="#ef233c",
                            annotation_text="Alert threshold", annotation_font_color="#ef233c")
        fig_flood.update_layout(yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                                height=310, **DL)
        st.plotly_chart(fig_flood, use_container_width=True, key="flood_bar")

    else:
        st.markdown('<div class="al b2" style="margin-top:.5rem;">👆 '
                    'Move the rain intensity slider and click <b>Analyse Flood Risk</b> to begin.</div>',
                    unsafe_allow_html=True)
        # Info cards
        fi1, fi2, fi3 = st.columns(3)
        fi1.markdown("""<div class="ycard" style="text-align:left;padding:1.1rem;">
<b style="color:#00b4d8;">🧠 LSTM Flood Model</b><br><br>
<small style="opacity:.75;line-height:1.8;">
Trained on synthetic rain sequences.<br>
Uses node elevation + historical<br>
flood incident data (2015–2023).</small></div>""", unsafe_allow_html=True)
        fi2.markdown("""<div class="ycard" style="text-align:left;padding:1.1rem;">
<b style="color:#ef233c;">🔴 Risk Thresholds</b><br><br>
<small style="opacity:.75;line-height:1.8;">
55% → Routing penalty applied<br>
70% → Citizen alert issued<br>
85% → Road closed in graph</small></div>""", unsafe_allow_html=True)
        fi3.markdown("""<div class="ycard" style="text-align:left;padding:1.1rem;">
<b style="color:#52b788;">🧭 Safe Routing</b><br><br>
<small style="opacity:.75;line-height:1.8;">
Dijkstra shortest-path<br>skips all flooded roads.<br>
Nearest safe junction shown.</small></div>""", unsafe_allow_html=True)
