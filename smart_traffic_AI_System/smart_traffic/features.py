"""
features.py
───────────
Feature: Emergency Vehicle Green Corridor
"""

import numpy as np
import heapq
from simulator import NODES, EDGES


def build_graph():
    g = {n["id"]: [] for n in NODES}
    for (i, j, d) in EDGES:
        g[i].append((j, d))
        g[j].append((i, d))
    return g


def dijkstra(graph, src, dst):
    dist = {n: float("inf") for n in graph}
    prev = {n: None for n in graph}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dst:
            break
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[dst], path


# ══════════════════════════════════════════════════════════
#  Emergency Vehicle Green Corridor
# ══════════════════════════════════════════════════════════
HOSPITALS = {
    "NIMHANS":           {"node": 9,  "lat": 12.9402, "lon": 77.5960},
    "Manipal Hospital":  {"node": 10, "lat": 12.9766, "lon": 77.6483},
    "St. John's":        {"node": 11, "lat": 12.9255, "lon": 77.6218},
    "Victoria Hospital": {"node": 2,  "lat": 12.9579, "lon": 77.5823},
    "Narayana Health":   {"node": 8,  "lat": 12.8605, "lon": 77.6732},
}

VEHICLE_TYPES = {
    "🚑 Ambulance":  {"speed_kmh": 45, "urgency": "CRITICAL", "color": "#ef233c"},
    "💊 Blood Van":  {"speed_kmh": 50, "urgency": "CRITICAL", "color": "#c77dff"},
}


def compute_green_corridor(origin_node, hospital_name, vehicle_type, df_snap):
    graph = build_graph()
    dest_node = HOSPITALS[hospital_name]["node"]
    veh = VEHICLE_TYPES[vehicle_type]
    dist_km, path = dijkstra(graph, origin_node, dest_node)
    if dist_km == float("inf") or len(path) < 2:
        return {"error": "No route found"}

    speed = veh["speed_kmh"]
    eta_base = (dist_km / speed) * 60
    node_map = {n["id"]: n for n in NODES}
    segments = []
    cumulative = 0.0

    for i, nid in enumerate(path):
        node = node_map[nid]
        cong_vals = df_snap[df_snap["node_id"] == nid]["congestion"].values
        cong = float(cong_vals[0]) if len(cong_vals) > 0 else 0.5
        seg_dist = 0.0
        if i > 0:
            pnid = path[i - 1]
            for (a, b, d) in EDGES:
                if (a == pnid and b == nid) or (b == pnid and a == nid):
                    seg_dist = d
                    break
            cumulative += (seg_dist / speed) * 60 + cong * 2.5
        segments.append({
            "node_id": nid, "node_name": node["name"],
            "lat": node["lat"], "lon": node["lon"],
            "congestion": round(cong, 3), "seg_dist_km": round(seg_dist, 2),
            "arrival_min": round(cumulative, 2),
            "pre_clear_min": round(max(0.0, cumulative - 1.5), 2),
            "status": "🟢 CLEARED" if cong < 0.4 else "🔴 CLEARING",
        })

    eta_corridor = cumulative
    eta_saved = max(0.0, eta_base - eta_corridor)
    return {
        "path": path, "segments": segments,
        "dist_km": round(dist_km, 2),
        "eta_base_min": round(eta_base, 1),
        "eta_corridor_min": round(eta_corridor, 1),
        "eta_saved_min": round(eta_saved, 1),
        "lives_impact": round(eta_saved * 2.1, 1),
        "vehicle": veh, "vehicle_type": vehicle_type,
        "hospital": hospital_name,
        "hospital_node": HOSPITALS[hospital_name],
        "urgency": veh["urgency"], "n_junctions": len(path),
    }
