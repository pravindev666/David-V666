"""
DAVID PROPHETIC ORACLE — Transformer Attention Model
======================================================
Uses PyTorch TransformerEncoder to identify non-linear patterns.
3rd Pillar of the V6.6.6+ Meta-Ensemble.
"""
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
import math

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, UP, DOWN, C

BINARY_MAP = {0: UP, 1: DOWN}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerNet(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super(TransformerNet, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 2)
        
    def forward(self, x):
        # x: (batch, seq, features)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Global average pooling over the sequence dimension
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

class TransformerModel:
    def __init__(self, seq_length=20):
        self.seq_length = seq_length
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accuracy = 0.0
        
    def _create_sequences(self, X, y=None):
        xs, ys = [], []
        for i in range(len(X) - self.seq_length + 1):
            xs.append(X[i:(i + self.seq_length)])
            if y is not None:
                ys.append(y[i + self.seq_length - 1])
        if y is not None:
            return np.array(xs), np.array(ys)
        return np.array(xs)
        
    def train(self, df, feature_cols, verbose=True, epochs=40, batch_size=64):
        self.feature_cols = feature_cols
        train_df = df[df["target_binary"] != 2].copy()
        if len(train_df) < self.seq_length * 2:
            return
            
        X_raw = train_df[feature_cols].values
        y_raw = train_df["target_binary"].values.astype(int)
        
        if verbose:
            print(f"\n{C.header('TRAINING TRANSFORMER ATTENTION MODEL')}")
            
        X_scaled = self.scaler.fit_transform(X_raw)
        X_seq, y_seq = self._create_sequences(X_scaled, y_raw)
        
        split_idx = int(len(X_seq) * 0.8)
        X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
        X_val, y_val = X_seq[split_idx:], y_seq[split_idx:]
        
        train_tensor = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False)
        
        # d_model should be divisible by nhead (4)
        self.model = TransformerNet(input_size=len(feature_cols)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0005, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            self.model.eval()
            with torch.no_grad():
                val_X, val_y = torch.FloatTensor(X_val).to(self.device), torch.LongTensor(y_val).to(self.device)
                val_outputs = self.model(val_X)
                val_loss = criterion(val_outputs, val_y).item()
                acc = accuracy_score(y_val, np.argmax(val_outputs.cpu().numpy(), axis=1))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    self.accuracy = acc
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.is_trained = True
            if verbose:
                print(f"  {C.GREEN}[OK] Transformer trained. Best Accuracy: {self.accuracy:.1%}{C.RESET}")

    def predict(self, df):
        if not self.is_trained: return {"prob_up": 0.5, "prob_down": 0.5}
        df_seq = df.iloc[-self.seq_length:]
        X_raw = df_seq[self.feature_cols].values
        X_scaled = self.scaler.transform(X_raw)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        return {"prob_up": float(probs[0]), "prob_down": float(probs[1])}

    def save(self, path=None):
        if path is None: path = os.path.join(MODEL_DIR, "transformer_model.pkl")
        state = {"model_state": self.model.state_dict(), "scaler": self.scaler, "feature_cols": self.feature_cols, "seq_length": self.seq_length}
        with open(path, "wb") as f: pickle.dump(state, f)

    def load(self, path=None):
        if path is None: path = os.path.join(MODEL_DIR, "transformer_model.pkl")
        if not os.path.exists(path): return False
        with open(path, "rb") as f: state = pickle.load(f)
        self.seq_length = state["seq_length"]
        self.scaler = state["scaler"]
        self.feature_cols = state["feature_cols"]
        self.model = TransformerNet(input_size=len(self.feature_cols)).to(self.device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        self.is_trained = True
        return True
