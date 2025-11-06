#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


def _load_pair(data_dir: Path, pair: str, timeframe: str) -> pd.DataFrame:
    file = data_dir / f"{pair.replace('/', '_')}-{timeframe}.feather"
    if not file.exists():
        raise FileNotFoundError(f"data file missing: {file}")
    df = pd.read_feather(file)[["date", "close"]]
    df["date"] = pd.to_datetime(df["date"])
    tag = pair.replace("/", "_")
    return df.rename(columns={"close": f"close_{tag}"})


def _merge_pairs(data_dir: Path, pairs: List[str], timeframe: str) -> pd.DataFrame:
    merged = None
    for pair in pairs:
        df = _load_pair(data_dir, pair, timeframe)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="date", how="inner")
    if merged is None or merged.empty:
        raise ValueError("no overlapping data across pairs")
    merged = merged.sort_values("date").reset_index(drop=True)
    return merged


def _compute_returns(merged: pd.DataFrame, pairs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    return_cols = []
    for pair in pairs:
        tag = pair.replace("/", "_")
        close = merged[f"close_{tag}"].astype(float).values
        log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
        merged[f"ret_{tag}"] = log_ret
        return_cols.append(f"ret_{tag}")
    returns = merged[return_cols].values.astype(np.float32)
    dates = merged["date"].values
    return returns, dates


def _build_sequences(
    returns: np.ndarray,
    window: int,
    horizon: int,
    target_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seqs: List[np.ndarray] = []
    targets: List[float] = []
    anchors: List[int] = []
    limit = len(returns) - horizon
    for idx in range(window, limit):
        window_slice = returns[idx - window : idx]
        target_value = returns[idx : idx + horizon, target_idx].sum()
        seqs.append(window_slice)
        targets.append(target_value)
        anchors.append(idx + horizon - 1)
    features = np.stack(seqs).astype(np.float32)
    labels = np.asarray(targets, dtype=np.float32)
    anchor_idx = np.asarray(anchors, dtype=np.int64)
    return features, labels, anchor_idx


class ReturnSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(features)  # (N, window, num_features)
        self.targets = torch.from_numpy(targets).unsqueeze(-1)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.lstm(x)
        last_hidden = seq_out[:, -1, :]
        return self.head(last_hidden)


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        encoded = self.encoder(x)
        last = encoded[:, -1, :]
        return self.head(last)


@dataclass
class TrainResult:
    final_loss: float
    mse_test: float
    direction_accuracy: float
    cumulative_return: float
    device: str


def train_model(
    dataset: ReturnSequenceDataset,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    train_split: float = 0.8,
    input_size: int = 1,
) -> Tuple[TrainResult, nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_len = int(len(dataset) * train_split)
    test_len = len(dataset) - train_len
    if train_len == 0 or test_len == 0:
        raise ValueError("dataset too small for the requested split")

    train_set, test_set = random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    model = LSTMRegressor(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()

    model.train()
    last_loss = float("nan")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_set)
        scheduler.step(epoch_loss)
        last_loss = epoch_loss

    model.eval()
    mse_total = 0.0
    total = 0
    preds_all: List[float] = []
    targets_all: List[float] = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            mse_total += criterion(preds, yb).item() * xb.size(0)
            total += xb.size(0)
            preds_all.extend(preds.squeeze(-1).cpu().tolist())
            targets_all.extend(yb.squeeze(-1).cpu().tolist())
    mse_test = mse_total / total if total else float("nan")

    preds_np = np.array(preds_all)
    targets_np = np.array(targets_all)
    direction_accuracy = float(np.mean(np.sign(preds_np) == np.sign(targets_np)))
    cumulative_return = float(np.sum(np.sign(preds_np) * targets_np))

    summary = TrainResult(
        final_loss=float(last_loss),
        mse_test=float(mse_test),
        direction_accuracy=direction_accuracy,
        cumulative_return=cumulative_return,
        device=str(device),
    )
    return summary, model, device


def generate_predictions(
    model: nn.Module,
    device: torch.device,
    features: np.ndarray,
    targets: np.ndarray,
    anchor_idx: np.ndarray,
    dates: np.ndarray,
    horizon: int,
    batch_size: int = 512,
) -> pd.DataFrame:
    model.eval()
    tensor = torch.from_numpy(features)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            xb = tensor[start : start + batch_size].to(device)
            batch_pred = model(xb).squeeze(-1).cpu().numpy()
            preds.append(batch_pred)
    predictions = np.concatenate(preds)

    aligned_dates = dates[anchor_idx]
    result = pd.DataFrame(
        {
            "date": aligned_dates,
            "prediction": predictions,
            "actual": targets,
            "horizon": horizon,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GPU LSTM on multi-asset log returns.")
    parser.add_argument("--data-dir", default="resources/user_data/data/binanceus", help="Directory containing feather OHLCV files.")
    parser.add_argument("--pairs", default="BTC/USDT,ETH/USDT,SOL/USDT", help="Comma-separated list of pairs.")
    parser.add_argument("--target-pair", default=None, help="Which pair's future return to predict (defaults to first pair).")
    parser.add_argument("--timeframe", default="4h", help="Feather timeframe suffix.")
    parser.add_argument("--window", type=int, default=64, help="Sequence window length.")
    parser.add_argument("--horizon", type=int, default=6, help="Number of future steps aggregated into the target.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--output", default="user_data/reports/deep_learning/lstm_summary.json")
    parser.add_argument("--prediction-output", default="user_data/reports/deep_learning/lstm_predictions.csv")
    parser.add_argument("--prediction-dir", default=None, help="Optional directory to store predictions as <pair>-<timeframe>.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    if not pairs:
        raise ValueError("no trading pairs provided")
    target_pair = args.target_pair or pairs[0]
    if target_pair not in pairs:
        raise ValueError(f"target pair {target_pair} not in pairs list {pairs}")

    merged = _merge_pairs(data_dir, pairs, args.timeframe)
    returns_matrix, dates = _compute_returns(merged, pairs)
    target_idx = pairs.index(target_pair)
    features, labels, anchors = _build_sequences(returns_matrix, args.window, args.horizon, target_idx)

    mask = np.isfinite(features).all(axis=(1, 2)) & np.isfinite(labels)
    features = features[mask]
    labels = labels[mask]
    anchors = anchors[mask]
    if len(features) < 100:
        raise ValueError("dataset too small after filtering; adjust window/horizon or data range.")

    dataset = ReturnSequenceDataset(features, labels)
    train_result, model, device = train_model(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_split=args.train_split,
        input_size=features.shape[-1],
    )

    pred_df = generate_predictions(model, device, features, labels, anchors, dates, args.horizon, batch_size=args.batch_size)
    if args.prediction_dir:
        prediction_output = Path(args.prediction_dir) / f"{target_pair.replace('/', '_')}-{args.timeframe}.csv"
    else:
        prediction_output = Path(args.prediction_output)
    prediction_output.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(prediction_output, index=False)

    summary = {
        "pairs": pairs,
        "target_pair": target_pair,
        "timeframe": args.timeframe,
        "window": args.window,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_split": args.train_split,
        "device": train_result.device,
        "final_loss": train_result.final_loss,
        "mse_test": train_result.mse_test,
        "direction_accuracy": train_result.direction_accuracy,
        "cumulative_return": train_result.cumulative_return,
        "samples": len(dataset),
        "prediction_file": str(prediction_output),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[deep-seq] summary written -> {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
