"""
Train and evaluate deep learning baselines (1D CNN and LSTM) on wearable BVP data.

This script trains CNN and LSTM models using PyTorch Lightning on pooled data 
from participants 3–10 of the CogWear pilot dataset, and then measures inference 
time and accuracy on unseen participants (0–2).

The goal is to compare the inference efficiency of deep learning models 
against the rule-based, interpretable cognitive load detection pipeline.
"""

import os, time, json
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from utils import load_signals_from_csv, create_windows
from sklearn.metrics import accuracy_score

# -------------------------------
# 1. Data utilities
# -------------------------------
def load_participant_windows(pid, w=512, s=64):
    base = load_signals_from_csv(f"data/pilot/{pid}/baseline/empatica_bvp.csv")
    load = load_signals_from_csv(f"data/pilot/{pid}/cognitive_load/empatica_bvp.csv")
    fb = create_windows(base, w, s)
    fc = create_windows(load, w, s)
    n = min(len(fb), len(fc))
    X = np.vstack([fb[:n], fc[:n]])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
    y = torch.tensor(y, dtype=torch.float32)
    return X, y

def load_training_data(pids=range(3, 11), w=512, s=64):
    X_all, y_all = [], []
    for pid in pids:
        X, y = load_participant_windows(pid, w, s)
        X_all.append(X)
        y_all.append(y)
    return torch.cat(X_all), torch.cat(y_all)

# -------------------------------
# 2. Models
# -------------------------------
class CNN1D(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x).squeeze()

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LSTMNet(pl.LightningModule):
    def __init__(self, input_size=1, hidden_size=64, lr=1e-3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 1), nn.Sigmoid())
        self.loss_fn = nn.BCELoss()
        self.lr = lr

    def forward(self, x):
        x = x.transpose(1, 2)  # (N, L, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1]).squeeze()

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# -------------------------------
# 3. Inference timing
# -------------------------------
def time_inference(model, X):
    model.eval()
    start = time.time()
    with torch.no_grad():
        preds = model(X).cpu().numpy()
    duration = time.time() - start
    preds_bin = (preds > 0.5).astype(int)
    return preds_bin, duration

# -------------------------------
# 4. Training + evaluation
# -------------------------------
def run_experiment():
    os.makedirs("results", exist_ok=True)
    w, s = 512, 64

    # --- Train data ---
    X_train, y_train = load_training_data(range(3, 11), w, s)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    # --- Train CNN ---
    cnn = CNN1D()
    pl.Trainer(max_epochs=3, enable_checkpointing=False, logger=False).fit(cnn, loader)

    # --- Train LSTM ---
    lstm = LSTMNet()
    pl.Trainer(max_epochs=3, enable_checkpointing=False, logger=False).fit(lstm, loader)

    # --- Inference on participants 0–2 ---
    results = []
    for pid in [0, 1, 2]:
        print(f"⚙️ Inference for participant {pid}")
        X_test, y_test = load_participant_windows(pid, w, s)

        preds_cnn, t_cnn = time_inference(cnn, X_test)
        preds_lstm, t_lstm = time_inference(lstm, X_test)

        acc_cnn = accuracy_score(y_test, preds_cnn)
        acc_lstm = accuracy_score(y_test, preds_lstm)

        print(f"PID {pid}: CNN={t_cnn:.4f}s ({acc_cnn:.3f}) | LSTM={t_lstm:.4f}s ({acc_lstm:.3f})")

        results.append({
            "participant": pid,
            "cnn_time": t_cnn,
            "lstm_time": t_lstm,
            "cnn_acc": acc_cnn,
            "lstm_acc": acc_lstm
        })

    df = pd.DataFrame(results)
    df.to_csv("results/inference_times_lightning.csv", index=False)
    print("\n✅ Inference timings saved → results/inference_times_lightning.csv")

# -------------------------------
if __name__ == "__main__":
    run_experiment()
