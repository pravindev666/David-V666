"""
DAVID PROPHETIC ORACLE — LSTM Sequence Model
================================================
Uses PyTorch to train an LSTM on a 10-day rolling window of features.
Captures temporal patterns missing in tree-based ensembles.
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

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, UP, DOWN, C

BINARY_MAP = {0: UP, 1: DOWN}

class SequenceNet(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(SequenceNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, (hc, cn) = self.lstm(x)
        # We want the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class SequenceModel:
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
        
    def train(self, df, feature_cols, verbose=True, epochs=30, batch_size=64):
        self.feature_cols = feature_cols
        
        # Filter noise
        train_df = df[df["target_binary"] != 2].copy()
        if len(train_df) < self.seq_length * 2:
            raise ValueError("Not enough data to train SequenceModel after noise filtering.")
            
        X_raw = train_df[feature_cols].values
        y_raw = train_df["target_binary"].values.astype(int)
        
        if verbose:
            print(f"\n{C.header('TRAINING LSTM SEQUENCE MODEL')}")
            print(f"  Device: {self.device} | Sequence Length: {self.seq_length} days")
            
        # Scale -> we scale before creating sequences
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_raw)
        
        if verbose:
            print(f"  Samples: {len(X_seq)} | Features: {len(feature_cols)}")
        
        # Split 80/20 chronological
        split_idx = int(len(X_seq) * 0.8)
        X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
        X_val, y_val = X_seq[split_idx:], y_seq[split_idx:]
        
        train_tensor = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_tensor = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=False)
        
        self.model = SequenceNet(input_size=len(feature_cols)).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            self.model.eval()
            with torch.no_grad():
                val_X, val_y = torch.FloatTensor(X_val).to(self.device), torch.LongTensor(y_val).to(self.device)
                val_outputs = self.model(val_X)
                val_loss = criterion(val_outputs, val_y).item()
                
                probs = torch.softmax(val_outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                acc = accuracy_score(y_val, preds)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    self.accuracy = acc
                    
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:2d}/{epochs} | t_loss: {train_loss/len(train_loader):.4f} | v_loss: {val_loss:.4f} | v_acc: {acc:.2%}")
                
        self.model.load_state_dict(best_model_state)
        self.is_trained = True
        
        if verbose:
            print(f"  {C.GREEN}[OK] LSTM trained. Best Val Accuracy: {self.accuracy:.1%}{C.RESET}")
            
    def predict(self, df):
        if not self.is_trained:
            raise RuntimeError("SequenceModel not trained!")
            
        if len(df) < self.seq_length:
            return {"direction": BINARY_MAP[0], "confidence": 0.5, "prob_up": 0.5, "prob_down": 0.5}
            
        df_seq = df.iloc[-self.seq_length:]
        X_raw = df_seq[self.feature_cols].values
        X_scaled = self.scaler.transform(X_raw)
        
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        pred_class = np.argmax(probs)
        return {
            "direction": BINARY_MAP[pred_class],
            "confidence": float(probs[pred_class]),
            "prob_up": float(probs[0]),
            "prob_down": float(probs[1])
        }

    def save(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, "sequence_model.pkl")
            
        state = {
            "model_state": self.model.state_dict() if self.model else None,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "is_trained": self.is_trained,
            "seq_length": self.seq_length,
            "accuracy": self.accuracy
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  {C.GREEN}[SAVED] Sequence Model → {path}{C.RESET}")
        
    def load(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, "sequence_model.pkl")
        if not os.path.exists(path):
            return False
            
        with open(path, "rb") as f:
            state = pickle.load(f)
            
        self.seq_length = state["seq_length"]
        self.scaler = state["scaler"]
        self.feature_cols = state["feature_cols"]
        self.is_trained = state["is_trained"]
        self.accuracy = state.get("accuracy", 0.0)
        
        if self.is_trained and state["model_state"] is not None:
            self.model = SequenceNet(input_size=len(self.feature_cols)).to(self.device)
            self.model.load_state_dict(state["model_state"])
            self.model.eval()
            
        print(f"  {C.GREEN}[LOADED] Sequence Model from {path}{C.RESET}")
        return True

