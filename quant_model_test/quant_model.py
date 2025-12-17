#!/usr/bin/env python3
# quant_model_multi_tf.py — Ensemble Quantitative Model (XGB + RF + LR) for multi-timeframe prediction + metrics + JSON save

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from redis_util import get_redis, get_recent_indicators_tf, save_forecast_to_redis

# =====================================================
# CONFIG
# =====================================================
TIMEFRAMES = ["5min", "15min", "30min", "45min", "1hour", "4hour"]
LOOKBACK = 60
MIN_SAMPLES = 80
N_SPLITS = 3  # time-series folds for stacking

FEATURE_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RSI14",
    "ADX",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "MA_Fast",
    "MA_Slow",
    "ATR14",
    "vwap_diff_bps",
    "ema20_slope_bps",
    "ema50_slope_bps",
    "dist_hh20_bps",
    "bb_width_bps",
]


# =====================================================
# FEATURE ENGINEERING
# =====================================================
def build_feature_matrix(df: pd.DataFrame):
    """Create sliding-window features + labels (next candle direction)."""
    df = df.copy().reset_index(drop=True)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna(subset=["target"])

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
                arr[-1] / (arr[0] + 1e-6),
            ]
        X.append(feats)
        y.append(df["target"].iloc[i + LOOKBACK - 1])

    if not X:
        return pd.DataFrame(), np.array([])

    return pd.DataFrame(X), np.array(y)


# =====================================================
# STACKING ENSEMBLE
# =====================================================
def train_stacked_ensemble(X, y):
    """Train stacking ensemble (XGB + RF + LR meta) using small time-series folds."""
    if len(X) < MIN_SAMPLES:
        print(f"[⚠️] Too few samples ({len(X)}) for ensemble training")
        return None, None, None

    base_models = {
        "xgb": XGBClassifier(
            n_estimators=40,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        ),
        "rf": RandomForestClassifier(
            n_estimators=80, max_depth=4, n_jobs=-1, random_state=42
        ),
        "lr": LogisticRegression(solver="liblinear", C=1.0),
    }

    oof_preds = {m: np.zeros(len(X)) for m in base_models}
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        for name, model in base_models.items():
            model.fit(X_train, y_train)
            oof_preds[name][val_idx] = model.predict_proba(X_val)[:, 1]

    meta_X = pd.DataFrame({k: oof_preds[k] for k in base_models})
    meta_y = y

    meta_model = LogisticRegression(solver="liblinear")
    meta_model.fit(meta_X, meta_y)

    # retrain base models on full data
    for m in base_models.values():
        m.fit(X, y)

    return base_models, meta_model, meta_X.columns.tolist()


# =====================================================
# EVALUATION + LIVE PREDICTION
# =====================================================
def eval_metrics(y_true, y_pred):
    return {
        "train_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "train_precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "train_recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "train_f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def ensemble_predict(base_models, meta_model, X_row):
    """Compute base probs + meta stacked probability."""
    base_probs = {
        name: float(model.predict_proba(X_row)[:, 1])
        for name, model in base_models.items()
    }
    meta_input = pd.DataFrame([base_probs])
    final_prob = float(meta_model.predict_proba(meta_input)[:, 1])
    return final_prob, base_probs


# =====================================================
# PER-TIMEFRAME PROCESS
# =====================================================
def train_and_predict_ensemble(df: pd.DataFrame, timeframe: str):
    if len(df) < LOOKBACK + 1:
        print(f"[⚠️] Not enough data for {timeframe}")
        return None

    X, y = build_feature_matrix(df)
    if X.empty or len(y) < MIN_SAMPLES:
        print(f"[⚠️] Insufficient samples for {timeframe}")
        return None

    start = time.perf_counter()

    # split for eval
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    base_models, meta_model, used_models = train_stacked_ensemble(X_train, y_train)
    if base_models is None:
        return None

    # Evaluate
    y_pred_test = []
    for i in range(len(X_test)):
        p, _ = ensemble_predict(base_models, meta_model, X_test.iloc[[i]])
        y_pred_test.append(1 if p > 0.5 else 0)

    metrics = eval_metrics(y_test, np.array(y_pred_test))
    elapsed = round(time.perf_counter() - start, 3)

    # Live prediction
    recent = df.tail(LOOKBACK)
    feats = []
    for col in FEATURE_COLS:
        if col not in recent.columns:
            feats += [0, 0, 0, 0]
            continue
        arr = recent[col].values
        feats += [
            np.mean(arr),
            np.std(arr),
            arr[-1] - arr[0],
            arr[-1] / (arr[0] + 1e-6),
        ]

    X_live = pd.DataFrame([feats])
    prob, per_model_probs = ensemble_predict(base_models, meta_model, X_live)

    signal = "BUY" if prob > 0.6 else "SELL" if prob < 0.4 else "NOACTION"

    forecast = {
        "timeframe": timeframe,
        "quant_signal": signal,
        "quant_confidence": round(prob if signal == "BUY" else 1 - prob, 4),
        "train_time_sec": elapsed,
        "used_models": used_models,
        **metrics,
        "per_model_probs": per_model_probs,
        "last_updated": datetime.now().isoformat(),
    }
    return forecast


# =====================================================
# MULTI-TF WRAPPER
# =====================================================
def run_quant_multi(stock_code: str):
    r = get_redis()
    all_forecasts = []

    for tf in TIMEFRAMES:
        try:
            rows = get_recent_indicators_tf(r, stock_code, tf, n=400)
            if not rows:
                print(f"[⚠️] No data found for {stock_code} {tf}")
                continue

            df = pd.DataFrame(rows).sort_values("Timestamp")
            for c in df.columns:
                if c != "Timestamp":
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            forecast = train_and_predict_ensemble(df, tf)
            if forecast:
                all_forecasts.append(forecast)
                save_forecast_to_redis(r, stock_code, forecast)
                print(f"📊 {stock_code} {tf} → {json.dumps(forecast, indent=2)}")

        except Exception as e:
            print(f"[❌ Error] {stock_code} {tf}: {e}")

    if not all_forecasts:
        print(f"[⚠️] No forecasts generated for {stock_code}")
        return None

    combined = {
        "stock_code": stock_code,
        "generated_at": datetime.now().isoformat(),
        "forecasts": all_forecasts,
    }

    print(f"\n✅ Final multi-timeframe ensemble forecast for {stock_code}:")
    print(json.dumps(combined, indent=2))

    output_dir = os.path.join("result", stock_code.upper())
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{stock_code.lower()}_quant_result.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved forecast JSON → {file_path}")
    save_forecast_to_redis(r, stock_code, combined)
    return combined


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    STOCKS = ["FORCEMOT", "GOKEX"]
    for s in STOCKS:
        try:
            run_quant_multi(s)
        except Exception as e:
            print(f"[❌ Error running quant for {s}]: {e}")
