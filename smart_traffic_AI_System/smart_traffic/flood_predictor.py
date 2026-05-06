"""
flood_predictor.py
──────────────────
Flood & Waterlogging Risk Prediction — Bengaluru Road Network
LSTM-based predictor using rain intensity, node elevation, and historical flood data.

Output:
  - flood_risk score [0.0 – 1.0] per node
  - flood_penalty for route graph
  - citizen alerts for high-risk nodes
"""

import numpy as np
import torch
import torch.nn as nn
import heapq
from datetime import datetime
from typing import Dict, List

from simulator import NODES, EDGES

# ── Elevation Data (approx. metres above sea level, BBMP DEM surveys) ─────────
NODE_ELEVATION = {
    0: 887.0, 1: 920.0, 2: 900.0, 3: 918.0, 4: 895.0,
    5: 910.0, 6: 905.0, 7: 915.0, 8: 870.0, 9: 908.0,
    10: 912.0, 11: 883.0, 12: 890.0, 13: 905.0, 14: 918.0,
}

# Historical flood incidents per monsoon season (BBMP data 2015–2023)
NODE_FLOOD_HISTORY = {
    0: 9, 1: 2, 2: 7, 3: 2, 4: 8,
    5: 3, 6: 6, 7: 2, 8: 8, 9: 3,
    10: 2, 11: 5, 12: 6, 13: 2, 14: 2,
}

MAX_FLOOD_HISTORY = max(NODE_FLOOD_HISTORY.values())
MAX_ELEVATION = max(NODE_ELEVATION.values())
MIN_ELEVATION = min(NODE_ELEVATION.values())


# ── LSTM Flood Risk Model ──────────────────────────────────────────────────────
class FloodLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, num_layers=2,
                 static_size=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.static_fc = nn.Linear(static_size, hidden_size)
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.head = nn.Sequential(
            nn.ReLU(), nn.Linear(hidden_size, 16),
            nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x_seq, x_static):
        _, (h_n, _) = self.lstm(x_seq)
        lstm_out = h_n[-1]
        static_out = torch.relu(self.static_fc(x_static))
        fused = torch.relu(self.fusion(torch.cat([lstm_out, static_out], dim=1)))
        return self.head(fused).squeeze(-1)


# ── Predictor ─────────────────────────────────────────────────────────────────
class FloodRiskPredictor:
    FLOOD_THRESHOLD    = 0.55
    ALERT_THRESHOLD    = 0.70
    CRITICAL_THRESHOLD = 0.85

    def __init__(self, seq_len: int = 12):
        self.seq_len = seq_len
        self.model = FloodLSTM()
        self._train_synthetic()
        self.model.eval()
        self._rain_history: Dict[int, List[float]] = {
            nid: [0.0] * seq_len for nid in range(15)
        }

    def _train_synthetic(self, n_samples: int = 2000, epochs: int = 30):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()
        self.model.train()
        np.random.seed(42)
        for _ in range(epochs):
            x_seq_list, x_stat_list, y_list = [], [], []
            for _ in range(n_samples):
                nid = np.random.randint(0, 15)
                rain_seq = np.random.exponential(10, self.seq_len).clip(0, 100)
                cumulative_rain = rain_seq.sum() / 100.0
                peak_rain = rain_seq.max() / 100.0
                trend = (rain_seq[-3:].mean() - rain_seq[:3].mean()) / 100.0
                elev_norm = 1.0 - (NODE_ELEVATION[nid] - MIN_ELEVATION) / max(MAX_ELEVATION - MIN_ELEVATION, 1)
                hist_norm = NODE_FLOOD_HISTORY[nid] / MAX_FLOOD_HISTORY
                base_risk = (0.40 * cumulative_rain + 0.25 * peak_rain +
                             0.20 * elev_norm + 0.15 * hist_norm)
                base_risk += 0.10 * max(0, trend)
                base_risk = float(np.clip(base_risk + np.random.normal(0, 0.03), 0, 1))
                cum_window = np.cumsum(rain_seq) / (100 * self.seq_len)
                trend_window = np.gradient(rain_seq) / 100.0
                seq_feat = np.stack([rain_seq / 100.0, cum_window, trend_window], axis=1)
                x_seq_list.append(seq_feat)
                x_stat_list.append([elev_norm, hist_norm])
                y_list.append(base_risk)
            X_seq = torch.tensor(np.array(x_seq_list), dtype=torch.float32)
            X_stat = torch.tensor(np.array(x_stat_list), dtype=torch.float32)
            Y = torch.tensor(y_list, dtype=torch.float32)
            optimizer.zero_grad()
            pred = self.model(X_seq, X_stat)
            loss = loss_fn(pred, Y)
            loss.backward()
            optimizer.step()

    def update_rain(self, rain_intensity: float):
        for nid in range(15):
            elev_norm = 1.0 - (NODE_ELEVATION[nid] - MIN_ELEVATION) / max(MAX_ELEVATION - MIN_ELEVATION, 1)
            spatial_mult = 0.7 + 0.6 * elev_norm
            local_rain = float(np.clip(rain_intensity * spatial_mult + np.random.normal(0, 2), 0, 100))
            history = self._rain_history[nid]
            history.append(local_rain)
            self._rain_history[nid] = history[-self.seq_len:]

    @torch.no_grad()
    def predict_all(self, rain_intensity: float) -> Dict[int, float]:
        self.update_rain(rain_intensity)
        x_seq_list, x_stat_list = [], []
        for nid in range(15):
            rain_seq = np.array(self._rain_history[nid], dtype=np.float32)
            cum_window = np.cumsum(rain_seq) / (100 * self.seq_len)
            trend_window = np.gradient(rain_seq) / 100.0
            seq_feat = np.stack([rain_seq / 100.0, cum_window, trend_window], axis=1)
            elev_norm = 1.0 - (NODE_ELEVATION[nid] - MIN_ELEVATION) / max(MAX_ELEVATION - MIN_ELEVATION, 1)
            hist_norm = NODE_FLOOD_HISTORY[nid] / MAX_FLOOD_HISTORY
            x_seq_list.append(seq_feat)
            x_stat_list.append([elev_norm, hist_norm])
        X_seq = torch.tensor(np.array(x_seq_list), dtype=torch.float32)
        X_stat = torch.tensor(np.array(x_stat_list), dtype=torch.float32)
        preds = self.model(X_seq, X_stat).numpy()
        return {nid: float(preds[nid]) for nid in range(15)}

    def get_flood_penalties(self, flood_risks: Dict[int, float]) -> Dict[int, float]:
        penalties = {}
        for nid, risk in flood_risks.items():
            if risk >= self.CRITICAL_THRESHOLD:
                penalties[nid] = float('inf')
            elif risk >= self.FLOOD_THRESHOLD:
                scale = (risk - self.FLOOD_THRESHOLD) / (self.CRITICAL_THRESHOLD - self.FLOOD_THRESHOLD)
                penalties[nid] = scale * 3600.0
            else:
                penalties[nid] = 0.0
        return penalties

    def generate_alerts(self, flood_risks: Dict[int, float],
                        node_names: Dict[int, str]) -> List[Dict]:
        alerts = []
        ts = datetime.now().strftime("%d %b %Y %H:%M")
        for nid, risk in flood_risks.items():
            if risk >= self.ALERT_THRESHOLD:
                level = "🔴 CRITICAL" if risk >= self.CRITICAL_THRESHOLD else "🟠 HIGH"
                action = ("ROAD CLOSED — Do NOT enter. Use alternate route."
                          if risk >= self.CRITICAL_THRESHOLD
                          else "Waterlogging likely. Avoid if possible. Drive slowly.")
                alerts.append({
                    "node_id": nid,
                    "node_name": node_names.get(nid, f"Node {nid}"),
                    "risk": round(risk, 4),
                    "level": level,
                    "message": (f"[BBMP FLOOD ALERT | {ts}] {level} at "
                                f"{node_names.get(nid, '')}. Risk: {risk:.0%}. {action}"),
                    "action": action,
                    "elevation": NODE_ELEVATION[nid],
                    "history_incidents": NODE_FLOOD_HISTORY[nid],
                })
        alerts.sort(key=lambda a: a["risk"], reverse=True)
        return alerts

    @staticmethod
    def risk_label(risk: float) -> str:
        if risk < 0.25: return "✅ Safe"
        if risk < 0.45: return "🟡 Low Risk"
        if risk < 0.55: return "🟠 Moderate"
        if risk < 0.70: return "🔴 High Risk"
        if risk < 0.85: return "🚨 Very High"
        return "🌊 FLOODED"

    @staticmethod
    def risk_color(risk: float) -> str:
        if risk < 0.25: return "#52b788"
        if risk < 0.45: return "#95d5b2"
        if risk < 0.55: return "#f4a261"
        if risk < 0.70: return "#e76f51"
        if risk < 0.85: return "#ef233c"
        return "#7209b7"


# ── Flood-aware Dijkstra routing ───────────────────────────────────────────────
def build_flood_graph(flood_penalties: Dict[int, float],
                      congestion_snap: dict = None) -> dict:
    """Build graph excluding flooded roads; remaining weighted by travel time."""
    graph = {n["id"]: [] for n in NODES}
    for (i, j, dist_km) in EDGES:
        pi = flood_penalties.get(i, 0.0)
        pj = flood_penalties.get(j, 0.0)
        if pi == float('inf') or pj == float('inf'):
            continue  # road closed
        ci = (congestion_snap or {}).get(i, 0.5)
        cj = (congestion_snap or {}).get(j, 0.5)
        avg_cong = (ci + cj) / 2
        speed_kmh = max(5.0, 60.0 * (1.0 - avg_cong))
        travel_min = (dist_km / speed_kmh) * 60
        flood_pen_min = (pi + pj) / 2 / 60
        total = travel_min + flood_pen_min
        graph[i].append((j, dist_km, total))
        graph[j].append((i, dist_km, total))
    return graph


def find_nearest_safe_route(origin_id: int, flood_risks: Dict[int, float],
                             flood_penalties: Dict[int, float],
                             congestion_snap: dict = None):
    """
    From origin, find the nearest safe junction (risk < FLOOD_THRESHOLD)
    using Dijkstra, respecting flood-closed roads.
    Returns: {target_id, path, dist_km, time_min, route_label}
    """
    graph = build_flood_graph(flood_penalties, congestion_snap)
    predictor = FloodRiskPredictor.__new__(FloodRiskPredictor)  # no re-train

    dist = {n["id"]: float("inf") for n in NODES}
    prev = {n["id"]: None for n in NODES}
    dist_km_map = {n["id"]: 0.0 for n in NODES}
    dist[origin_id] = 0.0
    pq = [(0.0, origin_id)]

    safe_threshold = FloodRiskPredictor.FLOOD_THRESHOLD

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, seg_km, seg_time in graph[u]:
            nd = dist[u] + seg_time
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                dist_km_map[v] = dist_km_map[u] + seg_km
                heapq.heappush(pq, (nd, v))

    # Find nearest safe node (not the origin itself)
    safe_nodes = [
        (nid, dist[nid])
        for nid in dist
        if nid != origin_id and flood_risks.get(nid, 0) < safe_threshold and dist[nid] < float("inf")
    ]
    if not safe_nodes:
        return None

    target_id = min(safe_nodes, key=lambda x: x[1])[0]

    # Reconstruct path
    path, cur = [], target_id
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    node_map = {n["id"]: n["name"] for n in NODES}
    return {
        "target_id": target_id,
        "target_name": node_map.get(target_id, f"Node {target_id}"),
        "path": path,
        "path_names": [node_map.get(n, f"Node {n}") for n in path],
        "dist_km": round(dist_km_map[target_id], 2),
        "time_min": round(dist[target_id], 1),
        "route_label": " → ".join(node_map.get(n, str(n)) for n in path),
    }


# ── Junction-to-Junction Flood-Aware Routing ──────────────────────────────────
def _k_shortest_flood_paths(graph: dict, source: int, target: int,
                             k: int = 3, n_nodes: int = 15):
    """
    Yen's K-Shortest Paths on an adjacency-list graph
    {node: [(neighbour, dist_km, time_min), ...]}.
    Returns list of (path, total_time_min, total_km).
    """
    def _dijkstra(g, src, tgt):
        dist = {i: float("inf") for i in range(n_nodes)}
        km_d = {i: 0.0         for i in range(n_nodes)}
        prev = {i: None        for i in range(n_nodes)}
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == tgt:
                break
            for v, seg_km, seg_t in g.get(u, []):
                nd = dist[u] + seg_t
                if nd < dist[v]:
                    dist[v] = nd
                    km_d[v] = km_d[u] + seg_km
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        if dist[tgt] == float("inf"):
            return [], float("inf"), 0.0
        path, cur = [], tgt
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return (path, dist[tgt], km_d[tgt]) if path[0] == src else ([], float("inf"), 0.0)

    p0, d0, km0 = _dijkstra(graph, source, target)
    if not p0:
        return []

    paths = [(p0, d0, km0)]
    candidates = []

    for _ in range(k - 1):
        last_path, _, _ = paths[-1]
        for spur_idx in range(len(last_path) - 1):
            spur_node = last_path[spur_idx]
            root_path = last_path[:spur_idx + 1]

            # Build a mutable copy of the graph, removing spur edges & root nodes
            g_copy = {n: list(edges) for n, edges in graph.items()}

            for p, _, _ in paths:
                if p[:spur_idx + 1] == root_path and spur_idx + 1 < len(p):
                    nxt = p[spur_idx + 1]
                    g_copy[spur_node] = [(v, sk, st) for v, sk, st in g_copy.get(spur_node, []) if v != nxt]
                    g_copy[nxt]       = [(v, sk, st) for v, sk, st in g_copy.get(nxt,       []) if v != spur_node]

            for rn in root_path[:-1]:
                g_copy[rn] = []
                for n in g_copy:
                    g_copy[n] = [(v, sk, st) for v, sk, st in g_copy[n] if v != rn]

            sp, sd, skm = _dijkstra(g_copy, spur_node, target)
            if sp:
                root_time = sum(
                    next((st for v, _, st in graph.get(root_path[i], [])
                          if v == root_path[i + 1]), 0.0)
                    for i in range(len(root_path) - 1)
                )
                root_km = sum(
                    next((sk for v, sk, _ in graph.get(root_path[i], [])
                          if v == root_path[i + 1]), 0.0)
                    for i in range(len(root_path) - 1)
                )
                full_path = root_path + sp[1:]
                full_time = root_time + sd
                full_km   = root_km   + skm
                entry = (full_path, full_time, full_km)
                if entry not in candidates:
                    candidates.append(entry)

        if not candidates:
            break
        candidates.sort(key=lambda x: x[1])
        best = candidates.pop(0)
        if best not in paths:
            paths.append(best)

    return paths


def find_junction_route(source: int, target: int,
                        flood_risks: Dict[int, float],
                        flood_penalties: Dict[int, float],
                        congestion_snap: dict = None,
                        k: int = 3) -> Dict:
    """
    Find the top-k flood-aware routes between any two junctions.

    Edges through critically flooded nodes are blocked entirely.
    Remaining edges are weighted by travel time + proportional flood penalty.

    Returns a dict with:
      routes     – list of {path, path_names, dist_km, time_min, route_label}
      source_risk / target_risk – flood risk at endpoints
      blocked    – True if no route exists
    """
    if source == target:
        return {"error": "Source and target junctions are the same."}

    graph = build_flood_graph(flood_penalties, congestion_snap)

    raw = _k_shortest_flood_paths(graph, source, target, k=k)
    if not raw:
        return {"routes": [], "blocked": True,
                "source_risk": flood_risks.get(source, 0),
                "target_risk": flood_risks.get(target, 0)}

    node_map = {n["id"]: n["name"] for n in NODES}
    routes = []
    for path, time_min, dist_km in raw:
        routes.append({
            "path":        path,
            "path_names":  [node_map.get(n, f"Node {n}") for n in path],
            "dist_km":     round(dist_km, 2),
            "time_min":    round(time_min, 1),
            "route_label": " → ".join(node_map.get(n, str(n)) for n in path),
            "hops":        len(path) - 1,
        })

    return {
        "routes":       routes,
        "blocked":      False,
        "source_risk":  flood_risks.get(source, 0),
        "target_risk":  flood_risks.get(target, 0),
    }
