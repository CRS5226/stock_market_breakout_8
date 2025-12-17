#!/usr/bin/env python3
# quant_model_buy_only_csv.py — Simplified Quant Model (XGB + RF + LR)
# Predicts only BUY or NOACTION (SELL removed)
# Reads candle+indicator data from CSV and trains with 80:20 split

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =====================================================
# CONFIG
# =====================================================
TIMEFRAMES = ["5min", "15min", "30min", "45min", "1hour", "4hour"]
LOOKBACK = 60
DATA_BASE_PATH = "history_data"
PRICE_MOVE_THRESHOLD = 0.002  # 0.2% up move = BUY

FEATURE_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RSI14",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "MA_Fast",
    "MA_Slow",
    "ATR14",
    "ADX",
    "+DI",
    "-DI",
    "BB_Upper",
    "BB_Lower",
]


# =====================================================
# FEATURE ENGINEERING
# =====================================================
def build_feature_matrix(df: pd.DataFrame):
    """Create rolling-window features; classify only BUY vs NOACTION."""
    df = df.copy().reset_index(drop=True)

    # Compute returns and label only BUY (> threshold)
    df["ret"] = (df["Close"].shift(-1) - df["Close"]) / df["Close"]
    df["target"] = np.where(df["ret"] > PRICE_MOVE_THRESHOLD, 1, 0)

    X, y = [], []
    for i in range(len(df) - LOOKBACK):
        window = df.iloc[i : i + LOOKBACK]
        feats = []
        for col in FEATURE_COLS:
            if col not in window.columns:
                feats += [0, 0, 0, 0]
                continue
            arr = window[col].values
            feats += [
                np.mean(arr),
                np.std(arr),
                arr[-1] - arr[0],
                (arr[-1] / (arr[0] + 1e-6)) - 1,
            ]
        X.append(feats)
        y.append(df["target"].iloc[i + LOOKBACK - 1])

    if not X:
        return pd.DataFrame(), np.array([])

    return pd.DataFrame(X), np.array(y)


# =====================================================
# MODEL TRAINING + PREDICTION
# =====================================================
def train_models(X_train, y_train):
    """Train 3 base models (XGB, RF, LR)."""
    models = {
        "xgb": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        ),
        "rf": RandomForestClassifier(
            n_estimators=150, max_depth=6, n_jobs=-1, random_state=42
        ),
        "lr": LogisticRegression(solver="liblinear", C=1.0),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models, X_test, y_test):
    """Evaluate models individually and ensemble-averaged."""
    probs = np.mean(
        [model.predict_proba(X_test)[:, 1] for model in models.values()], axis=0
    )
    y_pred = (probs > 0.5).astype(int)

    return {
        "train_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "train_precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "train_recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "train_f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }


# =====================================================
# LIVE PREDICTION
# =====================================================
def predict_latest(df, models):
    """Predict latest candle as BUY or NOACTION."""
    recent = df.tail(LOOKBACK)
    feats = []
    for col in FEATURE_COLS:
        arr = recent[col].values if col in recent.columns else np.zeros(LOOKBACK)
        feats += [
            np.mean(arr),
            np.std(arr),
            arr[-1] - arr[0],
            (arr[-1] / (arr[0] + 1e-6)) - 1,
        ]

    X_live = pd.DataFrame([feats])
    probs = np.mean(
        [float(model.predict_proba(X_live)[:, 1].item()) for model in models.values()]
    )
    signal = "BUY" if probs > 0.6 else "NOACTION"
    return signal, round(probs, 4)


# =====================================================
# LOAD CSV
# =====================================================
def load_csv_for_tf(stock_code: str, tf: str):
    """Load CSV for given timeframe (latest file)."""
    folder = os.path.join(DATA_BASE_PATH, f"history_data_{stock_code}")
    if not os.path.exists(folder):
        print(f"[⚠️] No folder found for {stock_code}")
        return pd.DataFrame()

    candidates = [f for f in os.listdir(folder) if tf in f and f.endswith(".csv")]
    if not candidates:
        print(f"[⚠️] No CSV found for {stock_code} {tf}")
        return pd.DataFrame()

    latest_file = sorted(candidates)[-1]
    path = os.path.join(folder, latest_file)
    print(f"📂 Loading {path}")
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").dropna().reset_index(drop=True)
    return df


# =====================================================
# MAIN LOGIC
# =====================================================
def run_quant(stock_code: str):
    all_forecasts = []

    for tf in TIMEFRAMES:
        df = load_csv_for_tf(stock_code, tf)
        if df.empty or len(df) <= LOOKBACK:
            continue

        X, y = build_feature_matrix(df)
        if X.empty or len(y) < 100:
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        start = time.perf_counter()
        models = train_models(X_train, y_train)
        metrics = evaluate_models(models, X_test, y_test)
        signal, conf = predict_latest(df, models)
        elapsed = round(time.perf_counter() - start, 3)

        forecast = {
            "timeframe": tf,
            "quant_signal": signal,
            "quant_confidence": conf,
            "train_time_sec": elapsed,
            **metrics,
            "last_updated": datetime.now().isoformat(),
        }
        print(f"📊 {stock_code} {tf} → {json.dumps(forecast, indent=2)}")
        all_forecasts.append(forecast)

    if not all_forecasts:
        print(f"[⚠️] No forecasts generated for {stock_code}")
        return None

    combined = {
        "stock_code": stock_code,
        "generated_at": datetime.now().isoformat(),
        "forecasts": all_forecasts,
    }

    out_dir = os.path.join("result", stock_code.upper())
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stock_code.lower()}_quant_result.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved forecast JSON → {out_path}")

    return combined


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    STOCKS = ["FORCEMOT", "GOKEX"]
    for s in STOCKS:
        try:
            run_quant(s)
        except Exception as e:
            print(f"[❌ Error running quant for {s}]: {e}")
