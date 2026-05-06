"""
models.py
─────────
Graph Attention Network (GAT) + LSTM Temporal Model + Fusion Layer
Q-Learning Signal Optimizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ──────────────────────────────────────────────────────────
# 1.  Graph Attention Network (pure PyTorch, no extra libs)
# ──────────────────────────────────────────────────────────
class GATLayer(nn.Module):
    """Single multi-head GAT layer."""
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.W     = nn.Linear(in_dim, out_dim, bias=False)
        self.attn  = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.drop  = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.W(x).view(N, self.n_heads, self.head_dim)          # [N, H, D]

        h_i = h.unsqueeze(1).expand(N, N, self.n_heads, self.head_dim)
        h_j = h.unsqueeze(0).expand(N, N, self.n_heads, self.head_dim)
        e = F.leaky_relu(
            self.attn(torch.cat([h_i, h_j], dim=-1)).squeeze(-1),   # [N, N, H]
            negative_slope=0.2
        )
        # Mask non-edges
        mask = (adj == 0).unsqueeze(-1).expand_as(e)
        e = e.masked_fill(mask, -1e9)
        alpha = self.drop(F.softmax(e, dim=1))                       # [N, N, H]

        out = torch.einsum("ijh,jhd->ihd", alpha, h).reshape(N, -1) # [N, H*D]
        return F.elu(out)


class GATModel(nn.Module):
    """2-layer GAT for spatial congestion modelling."""
    def __init__(self, in_dim: int = 6, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.layer1 = GATLayer(in_dim, hidden, n_heads=4)
        self.layer2 = GATLayer(hidden, out_dim, n_heads=4)
        self.bn1    = nn.BatchNorm1d(hidden)
        self.bn2    = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.layer1(x, adj))
        x = self.bn2(self.layer2(x, adj))
        return x                                                       # [N, out_dim]


# ──────────────────────────────────────────────────────────
# 2.  LSTM Temporal Model
# ──────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """Predicts next 6 congestion steps (30 min) from last 12 steps (1 hr)."""
    def __init__(self, input_size: int = 6, hidden: int = 64,
                 num_layers: int = 2, pred_steps: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, pred_steps),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)              # [B, T, H]
        return self.head(out[:, -1, :])    # [B, pred_steps]


# ──────────────────────────────────────────────────────────
# 3.  Fusion Model
# ──────────────────────────────────────────────────────────
class FusionModel(nn.Module):
    """Combines GAT spatial embedding + LSTM temporal predictions."""
    def __init__(self, gat_dim: int = 16, lstm_out: int = 6, pred_steps: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gat_dim + lstm_out, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, pred_steps),
            nn.Sigmoid(),
        )

    def forward(self, gat_emb: torch.Tensor, lstm_pred: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([gat_emb, lstm_pred], dim=-1))


# ──────────────────────────────────────────────────────────
# 4.  Training Helper
# ──────────────────────────────────────────────────────────
def prepare_lstm_sequences(df, node_id: int, seq_len: int = 12, pred_steps: int = 6):
    """Extract (X, y) sequences for a single node."""
    node_df = df[df["node_id"] == node_id].sort_values("timestamp").reset_index(drop=True)

    # Features: congestion, speed_norm, density_norm, hour_sin, hour_cos, day_sin, day_cos
    node_df["speed_n"]   = node_df["speed"] / 60.0
    node_df["density_n"] = node_df["density"] / 200.0
    node_df["hr_sin"]    = np.sin(2 * np.pi * node_df["hour"] / 24)
    node_df["hr_cos"]    = np.cos(2 * np.pi * node_df["hour"] / 24)
    node_df["dw_sin"]    = np.sin(2 * np.pi * node_df["day_of_week"] / 7)
    node_df["dw_cos"]    = np.cos(2 * np.pi * node_df["day_of_week"] / 7)

    feats = node_df[["congestion","speed_n","density_n","hr_sin","hr_cos","dw_sin","dw_cos"]].values
    # Trim to 6 features (drop day_cos for in_dim=6)
    feats = feats[:, :6]

    X, y = [], []
    for i in range(len(feats) - seq_len - pred_steps):
        X.append(feats[i:i + seq_len])
        y.append(feats[i + seq_len:i + seq_len + pred_steps, 0])  # congestion only
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_models(df, n_nodes: int = 15, epochs: int = 15,
                 adj: np.ndarray = None, progress_callback=None):
    """Train LSTM + GAT + Fusion. Returns trained model dict."""
    device = torch.device("cpu")

    lstm  = LSTMModel(input_size=6, hidden=64, num_layers=2, pred_steps=6).to(device)
    gat   = GATModel(in_dim=6, hidden=32, out_dim=16).to(device)
    fuse  = FusionModel(gat_dim=16, lstm_out=6, pred_steps=6).to(device)

    optimizer = torch.optim.Adam(
        list(lstm.parameters()) + list(gat.parameters()) + list(fuse.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    adj_t = torch.tensor(adj, dtype=torch.float32).to(device)
    
    # Build dataset from all nodes together
    all_X, all_y, all_nid = [], [], []
    for nid in range(n_nodes):
        X, y = prepare_lstm_sequences(df, nid)
        if len(X) > 0:
            all_X.append(X); all_y.append(y)
            all_nid += [nid] * len(X)
    all_X = np.concatenate(all_X); all_y = np.concatenate(all_y)
    all_nid = np.array(all_nid)

    # Get snapshot of current features for GAT (use last-known step per node)
    def get_node_features_snapshot(df):
        latest = df.sort_values("timestamp").groupby("node_id").last().reset_index()
        latest = latest.sort_values("node_id")
        speed_n   = latest["speed"].values   / 60.0
        density_n = latest["density"].values / 200.0
        hr_sin    = np.sin(2 * np.pi * latest["hour"].values / 24)
        hr_cos    = np.cos(2 * np.pi * latest["hour"].values / 24)
        dw_sin    = np.sin(2 * np.pi * latest["day_of_week"].values / 7)
        dw_cos    = np.cos(2 * np.pi * latest["day_of_week"].values / 7)
        return np.stack([latest["congestion"].values, speed_n, density_n,
                         hr_sin, hr_cos, dw_sin], axis=1).astype(np.float32)

    snap  = get_node_features_snapshot(df)
    snap_t = torch.tensor(snap).to(device)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(all_X), torch.tensor(all_y), torch.tensor(all_nid)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    losses = []
    for ep in range(epochs):
        lstm.train(); gat.train(); fuse.train()
        ep_loss = 0.0
        for xb, yb, nids in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            lstm_pred = lstm(xb)                    # [B, 6]
            gat_emb   = gat(snap_t, adj_t)          # [N, 16]
            gat_sel   = gat_emb[nids]               # [B, 16]
            fused     = fuse(gat_sel, lstm_pred)    # [B, 6]

           # Regression loss (value prediction)
            reg_loss = mse_loss(fused, yb)

# Classification labels (>= 0.5 = congested)
            yb_cls = (yb >= 0.5).float()

# Classification loss
            cls_loss = bce_loss(fused, yb_cls)

# Final combined loss
            loss = reg_loss + 0.7 * cls_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(lstm.parameters()) + list(gat.parameters()) + list(fuse.parameters()), 1.0
            )
            optimizer.step()
            ep_loss += loss.item()

        avg = ep_loss / len(loader)
        losses.append(avg)
        if progress_callback:
            progress_callback(ep + 1, epochs, avg)

    return {"lstm": lstm, "gat": gat, "fuse": fuse,
            "adj": adj_t, "snap": snap_t, "losses": losses}


# ──────────────────────────────────────────────────────────
# 5.  Q-Learning Signal Optimizer
# ──────────────────────────────────────────────────────────
class SignalOptimizer:
    """Tabular Q-learning for traffic signal timing."""

    ACTIONS = [
        {"label": "Balanced",       "ns": 30, "ew": 30},
        {"label": "Extend N-S 🡹",  "ns": 50, "ew": 20},
        {"label": "Extend E-W 🡺",  "ns": 20, "ew": 50},
        {"label": "Heavy N-S 🡹🡹", "ns": 60, "ew": 10},
        {"label": "Heavy E-W 🡺🡺", "ns": 10, "ew": 60},
    ]

    def __init__(self, n_states: int = 5, lr: float = 0.15,
                 gamma: float = 0.90, eps: float = 0.15):
        self.n = n_states
        self.n_actions = len(self.ACTIONS)
        self.lr = lr; self.gamma = gamma; self.eps = eps
        self.Q = np.zeros((n_states, n_states, self.n_actions))
        self._pretrain(episodes=3000)

    def _disc(self, c: float) -> int:
        return min(int(c * self.n), self.n - 1)

    def _wait(self, action_idx: int, c_ns: float, c_ew: float) -> float:
        a = self.ACTIONS[action_idx]
        cycle = a["ns"] + a["ew"]
        w_ns = (cycle / max(a["ns"], 1)) * c_ns * 90
        w_ew = (cycle / max(a["ew"], 1)) * c_ew * 90
        return (w_ns + w_ew) / 2

    def _pretrain(self, episodes: int):
        for _ in range(episodes):
            c_ns = np.random.random(); c_ew = np.random.random()
            s_ns = self._disc(c_ns);   s_ew = self._disc(c_ew)
            a = (np.random.randint(self.n_actions) if np.random.random() < self.eps
                 else int(np.argmax(self.Q[s_ns, s_ew])))
            reward = -self._wait(a, c_ns, c_ew) / 90.0
            self.Q[s_ns, s_ew, a] += self.lr * (
                reward + self.gamma * np.max(self.Q[s_ns, s_ew]) - self.Q[s_ns, s_ew, a]
            )

    def optimize(self, c_ns: float, c_ew: float) -> dict:
        s_ns = self._disc(c_ns); s_ew = self._disc(c_ew)
        a    = int(np.argmax(self.Q[s_ns, s_ew]))
        act  = self.ACTIONS[a]

        baseline_wait = self._wait(1, c_ns, c_ew)   # action=Balanced as baseline
        opt_wait      = self._wait(a, c_ns, c_ew)
        reduction     = max(0.0, (baseline_wait - opt_wait) / max(baseline_wait, 1) * 100)

        return {
            "action":     act["label"],
            "green_ns":   act["ns"],
            "green_ew":   act["ew"],
            "wait_before": round(baseline_wait, 1),
            "wait_after":  round(opt_wait, 1),
            "reduction":   round(reduction, 1),
        }

    def q_heatmap(self):
        """Return Q-table slice (best action value) for visualization."""
        return self.Q.max(axis=2)   # [n_states, n_states]
