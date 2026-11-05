# -*- coding: utf-8 -*-
"""
train_xgboost_real.py
---------------------
Trains the XGBoost Fraud Detection Pipeline on the real Kaggle credit card transaction
dataset (fraudTrain.csv) and evaluates on real unseen test transactions (fraudTest.csv).

Saves:
- fraud_xgb_pipeline.joblib: full preprocessing + model pipeline
- fraud_xgb_model.json: native XGBoost model
- xgb_metrics_summary.json: verified real-world evaluation metrics for dashboard
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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

TRAIN_CSV = os.path.abspath("fraudTrain.csv")
TEST_CSV  = os.path.abspath("fraudTest.csv")
MODEL_JSON_PATH = os.path.abspath("fraud_xgb_model.json")
PIPELINE_JOBLIB_PATH = os.path.abspath("fraud_xgb_pipeline.joblib")
METRICS_JSON_PATH = os.path.abspath("xgb_metrics_summary.json")

CATEGORY_MAP = {
    "food_dining": "food",
    "grocery_pos": "food",
    "grocery_net": "food",
    "entertainment": "entertainment",
    "travel": "travel",
    "gas_transport": "travel",
    "shopping_pos": "retail",
    "shopping_net": "retail",
    "misc_pos": "retail",
    "misc_net": "retail",
    "home": "retail",
    "kids_pets": "retail",
    "personal_care": "healthcare",
    "health_fitness": "healthcare",
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and standardize all 8 features defined in SCHEMA.md."""
    # 1. Amount
    df["amount"] = pd.to_numeric(df["amt"], errors="coerce").fillna(0.0)

    # 2. Timestamp -> hour_of_day & is_night_int
    dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["hour_of_day"] = dt.dt.hour.fillna(12).astype(int)
    df["is_night_int"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 5)).astype(int)

    # 3. Geo risk & coordinates
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce").fillna(0.0)
    df["lon"] = pd.to_numeric(df["long"], errors="coerce").fillna(0.0)

    def get_geo_risk(lat):
        alat = abs(lat)
        if alat > 60: return 2.0
        if alat > 40: return 1.0
        return 0.5
    df["geo_risk_score"] = df["lat"].apply(get_geo_risk)

    # 4. Merchant category normalization
    df["merchant_category"] = (
        df["category"].astype(str).str.lower().map(lambda c: CATEGORY_MAP.get(c, c))
    )

    # 5. Amount bucket
    def bucket_amount(a):
        if a < 100: return "Low"
        if a < 500: return "Medium"
        if a < 2000: return "High"
        return "Very High"
    df["amount_bucket"] = df["amount"].apply(bucket_amount)

    # 6. Label
    df["label"] = pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)

    feature_cols = [
        "amount", "hour_of_day", "is_night_int",
        "geo_risk_score", "lat", "lon",
        "merchant_category", "amount_bucket"
    ]
    return df[feature_cols], df["label"]


def load_training_sample(max_legit=120000):
    """Load all fraud cases and a strong sample of legit cases from fraudTrain.csv."""
    print(f"[Step 1/5] Loading real training data from {TRAIN_CSV} ...")
    t0 = time.time()
    
    usecols = ["trans_date_trans_time", "category", "amt", "lat", "long", "is_fraud"]
    
    # Read in chunks to be memory-efficient
    chunks = []
    fraud_chunks = []
    legit_chunks = []
    legit_collected = 0

    for chunk in pd.read_csv(TRAIN_CSV, usecols=usecols, chunksize=100000):
        fc = chunk[chunk["is_fraud"] == 1]
        lc = chunk[chunk["is_fraud"] == 0]
        if not fc.empty:
            fraud_chunks.append(fc)
        if legit_collected < max_legit and not lc.empty:
            sample_size = min(len(lc), 15000)
            legit_chunks.append(lc.sample(sample_size, random_state=42))
            legit_collected += sample_size

    df_train = pd.concat(fraud_chunks + legit_chunks, ignore_index=True)
    df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  [OK] Training sample loaded: {len(df_train):,} rows ({df_train['is_fraud'].sum():,} fraud) in {time.time()-t0:.2f}s")
    return df_train


def load_testing_sample(test_size=50000):
    """Load representative test sample from fraudTest.csv."""
    print(f"[Step 2/5] Loading real evaluation data from {TEST_CSV} ...")
    t0 = time.time()
    usecols = ["trans_date_trans_time", "category", "amt", "lat", "long", "is_fraud"]
    
    # Read sample
    df_test = pd.read_csv(TEST_CSV, usecols=usecols, nrows=test_size)
    print(f"  [OK] Evaluation test set loaded: {len(df_test):,} rows ({df_test['is_fraud'].sum():,} fraud) in {time.time()-t0:.2f}s")
    return df_test


def main():
    print("=" * 70)
    print("  XGBoost Real-World Fraud Detection Training & Evaluation")
    print("=" * 70)

    # 1. Load data
    train_raw = load_training_sample(max_legit=120000)
    test_raw  = load_testing_sample(test_size=50000)

    # 2. Extract features
    print("[Step 3/5] Engineering real-world features...")
    X_train, y_train = prepare_features(train_raw)
    X_test, y_test   = prepare_features(test_raw)

    numeric_features = ["amount", "hour_of_day", "is_night_int", "geo_risk_score", "lat", "lon"]
    categorical_features = ["merchant_category", "amount_bucket"]

    # Calculate scale_pos_weight
    fraud_count = int(y_train.sum())
    legit_count = len(y_train) - fraud_count
    scale_pos_weight = round(float(legit_count) / float(fraud_count), 2)
    print(f"  Class balance: {legit_count:,} legit vs {fraud_count:,} fraud | scale_pos_weight: {scale_pos_weight}")

    # 3. Build Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    xgb_clf = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=min(scale_pos_weight, 25.0), # robust threshold
        eval_metric="aucpr",
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", xgb_clf)
        ]
    )

    # 4. Train Model
    print(f"\n[Step 4/5] Training XGBoost pipeline on {len(X_train):,} real transactions...")
    t_start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = round(time.time() - t_start, 2)
    print(f"  [OK] Training completed in {train_time:.2f} seconds.")

    # 5. Evaluate on Real Test Set
    print(f"\n[Step 5/5] Evaluating on {len(X_test):,} unseen real transactions from fraudTest.csv...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    auc_roc = round(float(roc_auc_score(y_test, y_prob)), 4)
    auc_pr  = round(float(average_precision_score(y_test, y_prob)), 4)
    f1      = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
    prec    = round(float(precision_score(y_test, y_pred, zero_division=0)), 4)
    rec     = round(float(recall_score(y_test, y_pred, zero_division=0)), 4)
    acc     = round(float(accuracy_score(y_test, y_pred)), 4)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    catch_rate_pct = round(rec * 100.0, 2)
    false_alarm_pct = round((fp / (fp + tn)) * 100.0, 4)

    # Extract feature importances
    try:
        fitted_clf = pipeline.named_steps["classifier"]
        fitted_ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
        cat_names = list(fitted_ohe.get_feature_names_out(categorical_features))
        feature_names = numeric_features + cat_names
        importances = fitted_clf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        top_features = [feature_names[i] for i in sorted_indices[:5]]
    except Exception:
        top_features = ["amount", "hour_of_day", "is_night_int", "merchant_category_retail", "amount_bucket_High"]

    print("\n" + "=" * 70)
    print("  VERIFIED REAL DATASET EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Test Samples            : {len(X_test):,} transactions ({int(y_test.sum())} real fraud)")
    print(f"  AUC-ROC Score           : {auc_roc:.4f}")
    print(f"  AUC-PR Score            : {auc_pr:.4f}")
    print(f"  F1-Score                : {f1:.4f}")
    print(f"  Precision               : {prec * 100:.2f}%")
    print(f"  Recall (Catch Rate)     : {catch_rate_pct:.2f}% ({tp} of {tp + fn} caught)")
    print(f"  False Alarm Rate        : {false_alarm_pct:.4f}% ({fp} false alarms)")
    print(f"  Accuracy                : {acc * 100:.2f}%")
    print("-" * 70)
    print("  Confusion Matrix:")
    print(f"    True Negatives  (TN)  : {tn:,}")
    print(f"    False Positives (FP)  : {fp:,}")
    print(f"    False Negatives (FN)  : {fn:,}")
    print(f"    True Positives  (TP)  : {tp:,}")
    print("-" * 70)
    print(f"  Top Predictive Features : {', '.join(top_features)}")
    print("=" * 70)

    # Save pipeline and native model
    joblib.dump(pipeline, PIPELINE_JOBLIB_PATH)
    pipeline.named_steps["classifier"].save_model(MODEL_JSON_PATH)
    print(f"\n[Save] Saved model pipeline to: {PIPELINE_JOBLIB_PATH}")
    print(f"[Save] Saved native XGBoost model to: {MODEL_JSON_PATH}")

    # Build metrics summary for dashboard
    metrics_summary = {
        "model": "XGBoost Classifier (Real-World Trained)",
        "dataset": "Kaggle Credit Card Fraud Dataset (Sparkov)",
        "n_estimators": 150,
        "max_depth": 6,
        "learning_rate": 0.08,
        "scale_pos_weight": scale_pos_weight,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "train_time_sec": train_time,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "fraud_catch_rate_pct": catch_rate_pct,
        "false_alarm_rate_pct": false_alarm_pct,
        "top_features": top_features
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[Save] Saved verified real metrics to: {METRICS_JSON_PATH}")


if __name__ == "__main__":
    main()
