# -*- coding: utf-8 -*-
"""
test_real_dataset.py
--------------------
Evaluates the trained Fraud Detection Model (XGBoost / Random Forest) on an
external/real-world credit card dataset (CSV or Parquet).

Compatible with standard benchmark datasets such as:
1. Kaggle Credit Card Transactions Dataset (fraudTest.csv / fraudTrain.csv)
2. Any custom financial CSV with transaction amount, time/category, and fraud labels.

Usage:
  python test_real_dataset.py --file path/to/dataset.csv
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)

DEFAULT_MODEL_PATH = "fraud_xgb_pipeline.joblib"

def map_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes column names and extracts the required features:
    ['amount', 'hour_of_day', 'is_night_int', 'geo_risk_score', 'lat', 'lon', 'merchant_category', 'amount_bucket']
    """
    print("[1/4] Mapping columns and engineering features...")
    cols_lower = {col.lower(): col for col in df.columns}
    
    # 1. Amount mapping
    amt_col = cols_lower.get("amt") or cols_lower.get("amount") or cols_lower.get("transaction_amount")
    if not amt_col:
        raise ValueError(f"Could not find amount column. Available columns: {list(df.columns)}")
    df["amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

    # 2. Timestamp -> hour_of_day & is_night_int
    time_col = (
        cols_lower.get("trans_date_trans_time") or
        cols_lower.get("timestamp") or
        cols_lower.get("time") or
        cols_lower.get("date")
    )
    if time_col:
        try:
            dt_series = pd.to_datetime(df[time_col], errors="coerce")
            df["hour_of_day"] = dt_series.dt.hour.fillna(12).astype(int)
        except Exception:
            df["hour_of_day"] = 12
    else:
        df["hour_of_day"] = 12
    
    df["is_night_int"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)

    # 3. Location (lat & lon) -> geo_risk_score
    lat_col = cols_lower.get("lat") or cols_lower.get("merch_lat") or cols_lower.get("latitude")
    lon_col = cols_lower.get("long") or cols_lower.get("lon") or cols_lower.get("merch_long") or cols_lower.get("longitude")
    
    if lat_col:
        df["lat"] = pd.to_numeric(df[lat_col], errors="coerce").fillna(0.0)
    else:
        df["lat"] = 0.0
        
    if lon_col:
        df["lon"] = pd.to_numeric(df[lon_col], errors="coerce").fillna(0.0)
    else:
        df["lon"] = 0.0

    def get_geo_risk(lat):
        abs_lat = abs(lat)
        if abs_lat > 60: return 2.0
        if abs_lat > 40: return 1.0
        return 0.5
    df["geo_risk_score"] = df["lat"].apply(get_geo_risk)

    # 4. Merchant Category
    cat_col = cols_lower.get("category") or cols_lower.get("merchant_category")
    if cat_col:
        # Standardize category name
        df["merchant_category"] = (
            df[cat_col].astype(str)
            .str.lower()
            .str.replace("misc_pos", "retail")
            .str.replace("misc_net", "retail")
            .str.replace("shopping_pos", "retail")
            .str.replace("shopping_net", "retail")
            .str.replace("gas_transport", "travel")
            .str.replace("grocery_pos", "food")
            .str.replace("grocery_net", "food")
        )
    else:
        df["merchant_category"] = "retail"

    # 5. Amount bucket
    def bucket_amount(a):
        if a < 100: return "Low"
        if a < 500: return "Medium"
        if a < 2000: return "High"
        return "Very High"
    df["amount_bucket"] = df["amount"].apply(bucket_amount)

    # 6. Target Label (is_fraud)
    label_col = (
        cols_lower.get("is_fraud") or
        cols_lower.get("isfraud") or
        cols_lower.get("class") or
        cols_lower.get("label") or
        cols_lower.get("fraud")
    )
    if label_col:
        df["target_label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    else:
        df["target_label"] = None

    return df

def main():
    parser = argparse.ArgumentParser(description="Test trained fraud model on a real dataset")
    parser.add_argument("--file", "-f", type=str, default="training_data_clean.parquet", help="Path to real dataset (.csv or .parquet)")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained pipeline (.joblib)")
    parser.add_argument("--sample", "-s", type=int, default=None, help="Limit sample size (e.g. 50000) for faster evaluation")
    args = parser.parse_args()

    print("=" * 70)
    print("  REAL-WORLD DATASET FRAUD DETECTION BENCHMARK")
    print("=" * 70)

    if not os.path.exists(args.file):
        print(f"[ERROR] Dataset file not found: {args.file}")
        print("Please provide a valid file path using: python test_real_dataset.py --file <path_to_file.csv>")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("Please ensure fraud_xgb_pipeline.joblib exists in the directory.")
        sys.exit(1)

    print(f"\n[Step 1] Loading model: {args.model} ...")
    model = joblib.load(args.model)
    print("  [OK] Model pipeline loaded.")

    print(f"\n[Step 2] Loading dataset: {args.file} ...")
    t0 = time.time()
    if args.file.endswith(".csv"):
        if args.sample:
            df = pd.read_csv(args.file, nrows=args.sample)
        else:
            df = pd.read_csv(args.file)
    elif args.file.endswith(".parquet"):
        df = pd.read_parquet(args.file)
        if args.sample and len(df) > args.sample:
            df = df.sample(n=args.sample, random_state=42)
    else:
        print("[ERROR] Unsupported file format. Please provide a .csv or .parquet file.")
        sys.exit(1)

    print(f"  [OK] Loaded {len(df):,} transactions in {time.time()-t0:.2f}s")

    # Feature transformation
    df = map_and_engineer_features(df)
    feature_cols = ['amount', 'hour_of_day', 'is_night_int', 'geo_risk_score', 'lat', 'lon', 'merchant_category', 'amount_bucket']
    X = df[feature_cols]

    print(f"\n[Step 3] Running Inference on {len(X):,} transactions ...")
    t1 = time.time()
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    inference_time = time.time() - t1
    latency_per_sample_ms = (inference_time / len(X)) * 1000

    print(f"  [OK] Inference finished in {inference_time:.2f}s ({latency_per_sample_ms:.4f} ms per transaction)")
    flagged_fraud_count = int(np.sum(preds == 1))
    print(f"  Total Flagged as Fraud: {flagged_fraud_count:,} ({flagged_fraud_count / len(X) * 100:.2f}%)")

    # If ground truth labels exist, calculate evaluation metrics
    if df["target_label"] is not None and df["target_label"].nunique() > 1:
        y_true = df["target_label"]
        print(f"\n[Step 4] Ground Truth Evaluation Metrics:")
        actual_fraud = int(y_true.sum())
        actual_legit = len(y_true) - actual_fraud
        print(f"  Actual Legit : {actual_legit:,} | Actual Fraud : {actual_fraud:,}")

        auc_roc = roc_auc_score(y_true, probas)
        auc_pr = average_precision_score(y_true, probas)
        f1 = f1_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        acc = accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()

        print("\n" + "=" * 70)
        print("  MODEL EVALUATION RESULTS ON REAL DATASET")
        print("=" * 70)
        print(f"  Accuracy                : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  AUC-ROC                 : {auc_roc:.4f}")
        print(f"  AUC-PR                  : {auc_pr:.4f}")
        print(f"  F1-Score                : {f1:.4f}")
        print(f"  Precision               : {prec:.4f}")
        print(f"  Recall (Catch Rate)     : {rec*100:.2f}% ({tp} of {actual_fraud} caught)")
        print(f"  False Alarm Rate        : {(fp / actual_legit)*100:.4f}% ({fp} false positives)")
        print(f"  Inference Latency       : {latency_per_sample_ms:.4f} ms/txn")
        print("\n  Confusion Matrix:")
        print(f"                    Predicted: Legit    Predicted: Fraud")
        print(f"  Actual: Legit     {tn:<19} {fp:<16}")
        print(f"  Actual: Fraud     {fn:<19} {tp:<16}")
        print("=" * 70)
    else:
        print("\n[NOTE] No ground-truth fraud label column found in dataset.")
        print("Model generated predictions for all records successfully.")

    # Show top 5 flagged transactions
    df["predicted_fraud"] = preds
    df["risk_score"] = probas
    sample_frauds = df[df["predicted_fraud"] == 1][["amount", "merchant_category", "hour_of_day", "risk_score"]].head(5)
    if not sample_frauds.empty:
        print("\nSample Fraudulent Transactions Detected:")
        print(sample_frauds.to_string(index=False))

if __name__ == "__main__":
    main()
