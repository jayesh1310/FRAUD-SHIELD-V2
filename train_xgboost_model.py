# -*- coding: utf-8 -*-
"""
train_xgboost_model.py
----------------------
Trains an extreme Gradient Boosting (XGBoost) classifier on the fraud detection
streaming dataset using the exact features defined in SCHEMA.md Section 4.

Features used:
  - amount (float)
  - hour_of_day (int)
  - is_night_transaction (binary 0/1)
  - geo_risk_score (float)
  - lat, lon (float, extracted from location)
  - merchant_category (one-hot encoded)
  - amount_bucket (encoded)

Handles severe class imbalance via scale_pos_weight.
Saves model as:
  - fraud_xgb_model.json (native XGBoost format)
  - fraud_xgb_pipeline.joblib (full preprocessing + model pipeline)
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from sklearn.model_selection import train_test_split
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

DATA_PATH = os.path.abspath("training_data_clean.parquet")
MODEL_JSON_PATH = os.path.abspath("fraud_xgb_model.json")
PIPELINE_JOBLIB_PATH = os.path.abspath("fraud_xgb_pipeline.joblib")

def load_data():
    if not os.path.exists(DATA_PATH):
        alt_path = os.path.abspath("training_data")
        if os.path.exists(alt_path):
            import glob
            files = glob.glob(os.path.join(alt_path, "*.parquet"))
            if files:
                df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            else:
                raise FileNotFoundError(f"No parquet files in {alt_path}")
        else:
            raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    else:
        df = pd.read_parquet(DATA_PATH)
    
    # Extract lat and lon if nested in location dict
    if "lat" not in df.columns or "lon" not in df.columns:
        def get_coord(loc, key):
            if isinstance(loc, dict):
                return float(loc.get(key, 0.0))
            if hasattr(loc, "__getitem__"):
                try:
                    return float(loc[key])
                except Exception:
                    pass
            return getattr(loc, key, 0.0)
        
        df["lat"] = df["location"].apply(lambda x: get_coord(x, "lat"))
        df["lon"] = df["location"].apply(lambda x: get_coord(x, "lon"))

    # Convert boolean columns to int
    df["is_night_int"] = df["is_night_transaction"].astype(int)
    df["label"] = df["is_fraud"].astype(int)
    
    return df

def main():
    print("=" * 65)
    print("  TASK: XGBoost Fraud Detection Model Training & Evaluation")
    print("=" * 65)

    print(f"\n[Step 1] Loading dataset from {DATA_PATH} ...")
    t0 = time.time()
    df = load_data()
    total_rows = len(df)
    fraud_count = int(df["label"].sum())
    legit_count = total_rows - fraud_count

    print(f"  [OK] Total records : {total_rows:,} (loaded in {time.time()-t0:.2f}s)")
    print(f"       Legit (0)     : {legit_count:,} ({legit_count / total_rows * 100:.2f}%)")
    print(f"       Fraud (1)     : {fraud_count:,} ({fraud_count / total_rows * 100:.2f}%)")

    if fraud_count == 0:
        print("\n[ERROR] No fraud instances found in dataset. Cannot train classifier.")
        sys.exit(1)

    # Features definition
    numeric_features = ["amount", "hour_of_day", "is_night_int", "geo_risk_score", "lat", "lon"]
    categorical_features = ["merchant_category", "amount_bucket"]
    all_feature_cols = numeric_features + categorical_features

    X = df[all_feature_cols]
    y = df["label"]

    # Compute scale_pos_weight to counteract class imbalance
    scale_pos_weight = float(legit_count) / float(fraud_count)
    print(f"\n[Step 2] Class Imbalance Configuration:")
    print(f"  scale_pos_weight : {scale_pos_weight:.2f} (penalizes missed fraud proportionally)")

    # Train / Test split (80/20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train set        : {len(X_train):,} samples ({int(y_train.sum())} fraud)")
    print(f"  Test set         : {len(X_test):,} samples ({int(y_test.sum())} fraud)")

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    # XGBoost Classifier
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42
    )

    full_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", xgb_clf)
        ]
    )

    print("\n[Step 3] Fitting XGBoost Pipeline ...")
    t1 = time.time()
    full_pipeline.fit(X_train, y_train)
    train_time = time.time() - t1
    print(f"  [OK] Training completed in {train_time:.2f} seconds.")

    print("\n[Step 4] Evaluating Model on Test Set ...")
    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fraud_catch_rate = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    false_alarm_rate = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0

    print("\n" + "=" * 65)
    print("  XGBOOST MODEL EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Accuracy          : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  AUC-ROC           : {auc_roc:.4f}  <-- Primary discrimination metric")
    print(f"  AUC-PR            : {auc_pr:.4f}  <-- Precision-Recall curve area")
    print(f"  F1-Score          : {f1:.4f}  <-- Harmonic mean of precision & recall")
    print(f"  Precision         : {prec:.4f}  (Of all flagged frauds, % real)")
    print(f"  Recall            : {rec:.4f}  (Of all actual frauds, % caught)")

    print("\n  Confusion Matrix:")
    print(f"                    Predicted: Legit    Predicted: Fraud")
    print(f"  Actual: Legit     {tn:<19} {fp:<16}  ({fp} false alarms)")
    print(f"  Actual: Fraud     {fn:<19} {tp:<16}  ({fn} missed frauds)")
    print(f"\n  Fraud Catch Rate  : {fraud_catch_rate:.2f}% ({tp} of {tp + fn} caught)")
    print(f"  False Alarm Rate  : {false_alarm_rate:.4f}% ({fp} legit transactions flagged)")

    # Extract feature names & importances
    encoder = full_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    encoded_cat_names = encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + encoded_cat_names

    importances = xgb_clf.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("\n[Step 5] Top Feature Importances (XGBoost Gain):")
    print(f"  {'Rank':<5} {'Feature':<30} {'Importance':<12} Relative Bar")
    print("  " + "-" * 60)
    for rank, (feat, imp) in enumerate(ranked[:8], 1):
        bar = "|" * max(1, int(imp * 50))
        print(f"  {rank:<5} {feat:<30} {imp:<12.4f} {bar}")

    print(f"\n[Step 6] Saving XGBoost Model Artifacts ...")
    # Save standalone XGBoost model
    xgb_clf.save_model(MODEL_JSON_PATH)
    print(f"  [OK] Saved model JSON to: {MODEL_JSON_PATH}")

    # Save complete pipeline (includes scaler + encoder + XGBoost)
    joblib.dump(full_pipeline, PIPELINE_JOBLIB_PATH)
    print(f"  [OK] Saved full pipeline to: {PIPELINE_JOBLIB_PATH}")

    # Save metadata summary
    summary_path = os.path.abspath("xgb_metrics_summary.json")
    summary = {
        "model": "XGBoost Classifier",
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "scale_pos_weight": round(scale_pos_weight, 2),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_time_sec": round(train_time, 2),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "f1_score": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "accuracy": round(acc, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "fraud_catch_rate_pct": round(fraud_catch_rate, 2),
        "false_alarm_rate_pct": round(false_alarm_rate, 4),
        "top_features": [f for f, _ in ranked[:5]]
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Saved metrics summary to: {summary_path}")

    print("\n" + "=" * 65)
    print("  XGBoost Training and Evaluation Complete!")
    print("=" * 65)

if __name__ == "__main__":
    main()
