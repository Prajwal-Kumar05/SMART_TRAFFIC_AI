import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── Bengaluru Road Network ────────────────────────────────────────────────────
NODES = [
    {"id": 0,  "name": "Silk Board",     "lat": 12.9176, "lon": 77.6228},
    {"id": 1,  "name": "MG Road",        "lat": 12.9757, "lon": 77.6011},
    {"id": 2,  "name": "Koramangala",    "lat": 12.9352, "lon": 77.6245},
    {"id": 3,  "name": "Brigade Road",   "lat": 12.9722, "lon": 77.6052},
    {"id": 4,  "name": "Marathahalli",   "lat": 12.9591, "lon": 77.7012},
    {"id": 5,  "name": "Whitefield",     "lat": 12.9698, "lon": 77.7499},
    {"id": 6,  "name": "Hebbal",         "lat": 13.0350, "lon": 77.5970},
    {"id": 7,  "name": "Yeshwantpur",   "lat": 13.0194, "lon": 77.5354},
    {"id": 8,  "name": "Electronic City","lat": 12.8399, "lon": 77.6770},
    {"id": 9,  "name": "Jayanagar",      "lat": 12.9305, "lon": 77.5830},
    {"id": 10, "name": "Indiranagar",    "lat": 12.9784, "lon": 77.6408},
    {"id": 11, "name": "HSR Layout",     "lat": 12.9116, "lon": 77.6389},
    {"id": 12, "name": "BTM Layout",     "lat": 12.9166, "lon": 77.6101},
    {"id": 13, "name": "Banashankari",   "lat": 12.9256, "lon": 77.5462},
    {"id": 14, "name": "Rajajinagar",    "lat": 12.9906, "lon": 77.5553},
]

# (from_id, to_id, distance_km)
EDGES = [
    (0, 2, 3.2), (0, 8, 12.1), (0, 11, 2.8), (0, 12, 3.1),
    (1, 3, 0.8), (1, 10, 2.1), (1, 6, 8.5),
    (2, 3, 2.0), (2, 11, 4.1), (2, 12, 2.3),
    (3, 10, 1.9),
    (4, 5, 8.2), (4, 10, 7.3), (4, 11, 5.8),
    (6, 7, 4.2), (6, 14, 3.8),
    (7, 13, 6.5), (7, 14, 2.1),
    (8, 11, 9.3), (8, 12, 7.8),
    (9, 12, 2.4), (9, 13, 3.6),
    (10, 4, 7.3), (10, 1, 2.1),
    (11, 12, 3.2), (11, 0, 2.8),
    (12, 9, 2.4), (12, 0, 3.1),
    (13, 9, 3.6), (13, 7, 6.5),
    (14, 6, 3.8), (14, 7, 2.1),
]

# Node-specific congestion multipliers (Silk Board always worst!)
NODE_MULTIPLIERS = {
    0: 1.35, 1: 1.10, 2: 1.05, 3: 0.95, 4: 1.10,
    5: 0.70, 6: 0.80, 7: 0.80, 8: 0.90, 9: 0.70,
    10: 0.90, 11: 0.80, 12: 0.75, 13: 0.65, 14: 0.75,
}

def get_congestion_factor(hour: int, dow: int) -> float:
    """Realistic Bengaluru time-of-day traffic factor."""
    if dow < 5:  # Weekday
        if 8 <= hour <= 10:    return 0.85   # Morning rush
        elif 17 <= hour <= 20: return 0.90   # Evening rush
        elif 12 <= hour <= 14: return 0.55   # Lunch
        elif 0 <= hour <= 5:   return 0.10   # Night
        else:                  return 0.40
    else:                                    # Weekend
        if 11 <= hour <= 20:   return 0.50
        elif 0 <= hour <= 6:   return 0.10
        else:                  return 0.35

def generate_traffic_data(days: int = 7, interval_min: int = 5) -> pd.DataFrame:
    """Generate 7-day synthetic traffic dataset for 15 Bengaluru nodes."""
    np.random.seed(42)
    start = datetime.now() - timedelta(days=days)
    timestamps = [start + timedelta(minutes=i * interval_min)
                  for i in range(int(days * 24 * 60 / interval_min))]

    records = []
    for ts in timestamps:
        hour, dow = ts.hour, ts.weekday()
        base = get_congestion_factor(hour, dow)
        for node in NODES:
            nid = node["id"]
            noise = np.random.normal(0, 0.07)
            cong = float(np.clip(base * NODE_MULTIPLIERS[nid] + noise, 0, 1))
            speed = float(np.clip(60 * (1 - cong) + np.random.normal(0, 2), 5, 60))
            density = float(np.clip(200 * cong + np.random.normal(0, 8), 0, 200))
            records.append({
                "timestamp": ts,
                "node_id": nid,
                "node_name": node["name"],
                "lat": node["lat"],
                "lon": node["lon"],
                "hour": hour,
                "day_of_week": dow,
                "congestion": round(cong, 4),
                "speed": round(speed, 1),
                "density": round(density, 1),
            })
    return pd.DataFrame(records)

def build_adjacency_matrix(n_nodes: int = 15) -> np.ndarray:
    """Build symmetric adjacency matrix for the road graph."""
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (i, j, _) in EDGES:
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    # Self-loops
    np.fill_diagonal(adj, 1.0)
    # Degree normalization
    deg = adj.sum(axis=1, keepdims=True)
    return adj / np.maximum(deg, 1e-6)
