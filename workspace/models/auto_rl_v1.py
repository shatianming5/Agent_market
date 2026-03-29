from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from agent_market.freqai.model.base import BaseModelAdapter, TrainResult


def _as_2d_float_array(X: Any) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"X must be 2D array-like, got shape={arr.shape}")
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _as_1d_float_array(y: Any) -> np.ndarray:
    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return arr.astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    if a.size != b.size:
        raise ValueError("corrcoef inputs must have same length")
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if not np.isfinite(a_std) or not np.isfinite(b_std) or a_std == 0.0 or b_std == 0.0:
        return 0.0
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return 0.0
    return c


def _ensure_strictly_increasing(edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return edges
    e = np.asarray(edges, dtype=np.float32)
    out = [float(e[0])]
    for v in e[1:]:
        vv = float(v)
        if vv > out[-1]:
            out.append(vv)
    return np.asarray(out, dtype=np.float32)


class AutoRL_v1(BaseModelAdapter):
    registry_name = "auto_rl_v1"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.q_table_: Optional[Dict[tuple, np.ndarray]] = None
        self.sa_counts_: Optional[Dict[tuple, int]] = None
        self.median_: Optional[np.ndarray] = None
        self.bin_edges_: Optional[list[np.ndarray]] = None
        self.n_features_in_: Optional[int] = None
        self.n_bins_: int = int(config.get("n_bins", 8))
        self.model_path_: Optional[Path] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
    ) -> TrainResult:
        X_tr = _as_2d_float_array(X_train)
        y_tr = _as_1d_float_array(y_train)

        if X_tr.shape[0] != y_tr.shape[0]:
            raise ValueError(f"X_train and y_train length mismatch: {X_tr.shape[0]} vs {y_tr.shape[0]}")
        if X_tr.size == 0:
            raise ValueError("X_train is empty")

        self.n_features_in_ = int(X_tr.shape[1])

        model_dir = self.config.get("model_dir")
        if model_dir is None:
            raise ValueError('config["model_dir"] is required')
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        seed = int(self.config.get("seed", self.config.get("random_state", 42)))
        rng = np.random.RandomState(seed)

        n_bins = int(self.config.get("n_bins", self.n_bins_))
        n_bins = max(2, n_bins)
        self.n_bins_ = int(n_bins)

        # Preprocess: replace non-finites with per-feature median
        X64 = X_tr.astype(np.float64, copy=False)
        finite_mask = np.isfinite(X64)
        med = np.zeros((self.n_features_in_,), dtype=np.float32)
        X_filled = X64.copy()
        for j in range(self.n_features_in_):
            col = X64[:, j]
            m = finite_mask[:, j]
            if np.any(m):
                mj = float(np.median(col[m]))
            else:
                mj = 0.0
            if not np.isfinite(mj):
                mj = 0.0
            med[j] = np.float32(mj)
            bad = ~m
            if np.any(bad):
                X_filled[bad, j] = mj
        self.median_ = med

        # Quantile bin edges for each feature (n_bins -> n_bins-1 cut points)
        qs = np.linspace(0.0, 100.0, n_bins + 1, dtype=np.float64)[1:-1]
        edges_list: list[np.ndarray] = []
        for j in range(self.n_features_in_):
            col = X_filled[:, j]
            if col.size == 0:
                edges = np.asarray([], dtype=np.float32)
            else:
                try:
                    edges = np.percentile(col, qs).astype(np.float32, copy=False)
                except Exception:
                    edges = np.asarray([], dtype=np.float32)
            edges = _ensure_strictly_increasing(edges)
            if edges.size == 0:
                # Fallback: create at least one edge if possible
                cmin = float(np.min(col)) if col.size else 0.0
                cmax = float(np.max(col)) if col.size else 0.0
                if np.isfinite(cmin) and np.isfinite(cmax) and cmax > cmin:
                    edges = np.linspace(cmin, cmax, n_bins + 1, dtype=np.float64)[1:-1].astype(np.float32)
                    edges = _ensure_strictly_increasing(edges)
            edges_list.append(edges)
        self.bin_edges_ = edges_list

        # RL hyperparams
        eps_start = float(self.config.get("eps_start", 0.25))
        eps_min = float(self.config.get("eps_min", 0.02))
        eps_decay = float(self.config.get("eps_decay", 0.90))
        gamma = float(self.config.get("gamma", 0.0))
        base_alpha = float(self.config.get("alpha", 0.25))
        alpha_decay_pow = float(self.config.get("alpha_decay_pow", 0.60))
        n_epochs = int(self.config.get("epochs", 20))
        trade_cost = float(self.config.get("trade_cost", 0.0002))

        # Action mapping: 0 flat, 1 long, 2 short
        a_to_pos = np.asarray([0.0, 1.0, -1.0], dtype=np.float32)
        pos_to_idx = {-1.0: 0, 0.0: 1, 1.0: 2}

        q_table: Dict[tuple, np.ndarray] = {}
        sa_counts: Dict[tuple, int] = {}

        X_bins = self._discretize_all(X_filled.astype(np.float32, copy=False))

        # Sequential Q-learning over the dataset; include prev-position in state to keep Markov w/ trade cost.
        for epoch in range(max(1, n_epochs)):
            eps = max(eps_min, eps_start * (eps_decay ** epoch))
            prev_pos = 0.0

            for t in range(X_bins.shape[0]):
                prev_idx = int(pos_to_idx.get(float(prev_pos), 1))
                s = self._state_key(X_bins[t], prev_idx)

                if rng.rand() < eps:
                    a = int(rng.randint(0, 3))
                else:
                    qvals = q_table.get(s)
                    if qvals is None:
                        a = 0
                    else:
                        a = int(np.argmax(qvals))

                pos = float(a_to_pos[a])
                y = float(y_tr[t])

                # Reward: position * future return minus trade cost on position change
                r = pos * y - trade_cost * abs(pos - float(prev_pos))

                if t + 1 < X_bins.shape[0]:
                    next_prev_idx = int(pos_to_idx.get(float(pos), 1))
                    s2 = self._state_key(X_bins[t + 1], next_prev_idx)
                    q_next = q_table.get(s2)
                    max_next = float(np.max(q_next)) if q_next is not None else 0.0
                    target = r + gamma * max_next
                else:
                    target = r

                q = q_table.get(s)
                if q is None:
                    q = np.zeros((3,), dtype=np.float32)
                    q_table[s] = q

                sa_key = (s, a)
                cnt = int(sa_counts.get(sa_key, 0)) + 1
                sa_counts[sa_key] = cnt

                alpha_t = base_alpha / ((1.0 + float(cnt)) ** alpha_decay_pow)
                q[a] = np.float32(float(q[a]) + alpha_t * (target - float(q[a])))

                prev_pos = pos

        self.q_table_ = q_table
        self.sa_counts_ = sa_counts

        metrics: Dict[str, float] = {}

        yhat_tr = self.predict(X_tr).astype(np.float64, copy=False)
        ytr64 = y_tr.astype(np.float64, copy=False)
        resid_tr = yhat_tr - ytr64
        metrics["train_mse"] = float(np.mean(resid_tr ** 2)) if ytr64.size else 0.0
        metrics["train_mae"] = float(np.mean(np.abs(resid_tr))) if ytr64.size else 0.0
        metrics["train_corr"] = _safe_corrcoef(yhat_tr, ytr64)
        metrics["train_dir_acc"] = (
            float(np.mean((np.sign(yhat_tr) == np.sign(ytr64)).astype(np.float64))) if ytr64.size else 0.0
        )

        if X_valid is not None and y_valid is not None:
            X_va = _as_2d_float_array(X_valid)
            y_va = _as_1d_float_array(y_valid)
            if X_va.size:
                if self.n_features_in_ is not None and X_va.shape[1] != self.n_features_in_:
                    raise ValueError(f"X_valid has {X_va.shape[1]} features, expected {self.n_features_in_}")
                if X_va.shape[0] != y_va.shape[0]:
                    raise ValueError(f"X_valid and y_valid length mismatch: {X_va.shape[0]} vs {y_va.shape[0]}")
                yhat_va = self.predict(X_va).astype(np.float64, copy=False)
                yva64 = y_va.astype(np.float64, copy=False)
                resid_va = yhat_va - yva64
                metrics["valid_mse"] = float(np.mean(resid_va ** 2)) if yva64.size else 0.0
                metrics["valid_mae"] = float(np.mean(np.abs(resid_va))) if yva64.size else 0.0
                metrics["valid_corr"] = _safe_corrcoef(yhat_va, yva64)
                metrics["valid_dir_acc"] = (
                    float(np.mean((np.sign(yhat_va) == np.sign(yva64)).astype(np.float64))) if yva64.size else 0.0
                )

        model_path = model_dir / "auto_rl_v1.pkl"
        self.save(model_path)
        self.model_path_ = model_path

        return TrainResult(model_path=model_path, metrics=metrics)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.q_table_ is None or self.median_ is None or self.bin_edges_ is None:
            raise RuntimeError("Model is not loaded/fitted yet")

        X_in = _as_2d_float_array(X)
        if X_in.size == 0:
            return np.asarray([], dtype=np.float32)

        if self.n_features_in_ is not None and X_in.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_in.shape[1]} features, expected {self.n_features_in_}")

        X64 = X_in.astype(np.float64, copy=False)
        X_filled = X64.copy()
        for j in range(X_filled.shape[1]):
            col = X_filled[:, j]
            m = np.isfinite(col)
            if not np.all(m):
                X_filled[~m, j] = float(self.median_[j])

        X_bins = self._discretize_all(X_filled.astype(np.float32, copy=False))

        # Return estimate of E[y | state] using symmetric long/short Q values at prev_pos=flat.
        prev_pos_idx = 1  # flat
        out = np.zeros((X_bins.shape[0],), dtype=np.float32)
        for i in range(X_bins.shape[0]):
            s = self._state_key(X_bins[i], prev_pos_idx)
            q = self.q_table_.get(s)
            if q is None:
                out[i] = np.float32(0.0)
            else:
                q_long = float(q[1])
                q_short = float(q[2])
                out[i] = np.float32(0.5 * (q_long - q_short))
        if out.shape[0] != X_in.shape[0]:
            raise RuntimeError("predict() returned wrong length")
        return out

    def save(self, path: Path) -> None:
        if self.q_table_ is None or self.median_ is None or self.bin_edges_ is None:
            raise RuntimeError("Nothing to save: model is not fitted/loaded")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "registry_name": self.registry_name,
            "n_features_in": self.n_features_in_,
            "n_bins": int(self.n_bins_),
            "median": np.asarray(self.median_, dtype=np.float32),
            "bin_edges": [np.asarray(e, dtype=np.float32) for e in self.bin_edges_],
            "q_table": self.q_table_,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: Path) -> None:
        path = Path(path)
        with path.open("rb") as f:
            payload = pickle.load(f)

        self.n_features_in_ = payload.get("n_features_in", None)
        self.n_bins_ = int(payload.get("n_bins", 8))
        self.median_ = np.asarray(payload.get("median"), dtype=np.float32)
        edges = payload.get("bin_edges")
        if not isinstance(edges, list):
            raise ValueError("Invalid checkpoint: bin_edges missing")
        self.bin_edges_ = [np.asarray(e, dtype=np.float32) for e in edges]

        q_table = payload.get("q_table")
        if not isinstance(q_table, dict):
            raise ValueError("Invalid checkpoint: q_table missing")
        self.q_table_ = q_table

        self.model_path_ = path

    def _discretize_all(self, X_filled: np.ndarray) -> np.ndarray:
        if self.bin_edges_ is None:
            raise RuntimeError("bin_edges not initialized")
        X = np.asarray(X_filled, dtype=np.float32)
        n, f = X.shape
        out = np.zeros((n, f), dtype=np.int16)
        for j in range(f):
            edges = self.bin_edges_[j]
            if edges.size == 0:
                out[:, j] = 0
            else:
                out[:, j] = np.searchsorted(edges, X[:, j], side="right").astype(np.int16, copy=False)
        return out

    def _state_key(self, bins_row: np.ndarray, prev_pos_idx: int) -> tuple:
        row = np.asarray(bins_row).reshape(-1)
        return tuple(int(v) for v in row) + (int(prev_pos_idx),)