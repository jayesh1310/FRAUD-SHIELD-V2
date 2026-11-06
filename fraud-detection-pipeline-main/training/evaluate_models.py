"""
evaluate_models.py
------------------
Phase 4: Evaluate all three trained models on the held-out fraudTest.csv.

Rules:
  - fraudTest.csv is ONLY used here — never during training or threshold tuning.
  - Natural class distribution is preserved (no resampling).
  - Metrics computed: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC,
    Confusion Matrix, FPR, FNR, Fraud Detection Rate, Training Time, Inference Time.

Usage:
  cd fraud-detection-pipeline-main
  python training/evaluate_models.py

Outputs:
  results/model_comparison.json
"""

import json
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# ── Ensure HADOOP_HOME and winutils.exe are configured on Windows ────────────
if os.name == "nt":
    if os.path.exists(r"C:\hadoop"):
        os.environ["HADOOP_HOME"] = r"C:\hadoop"
        hadoop_bin = r"C:\hadoop\bin"
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

# ── Set Python executable for PySpark workers BEFORE importing pyspark ────────
_py_exe = sys.executable
if _py_exe and os.path.isfile(_py_exe):
    os.environ["PYSPARK_PYTHON"]        = _py_exe
    os.environ["PYSPARK_DRIVER_PYTHON"] = _py_exe

ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = Path(os.getenv("DATA_DIR",  str(ROOT_DIR.parent)))
MODELS_DIR  = Path(os.getenv("MODELS_DIR", str(ROOT_DIR / "models")))
RESULTS_DIR = ROOT_DIR / "results"
TEST_CSV    = DATA_DIR / "fraudTest.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.functions import vector_to_array

sys.path.insert(0, str(ROOT_DIR))
from features.feature_engineering import (
    enrich_transaction, LABEL_COL, FEATURES_COL, SCALED_FEATURES_COL,
)


def create_spark():
    builder = (
        SparkSession.builder
        .appName("FraudShield-Evaluation")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.driver.maxResultSize", "2g")
    )
    if os.name == "nt" and os.path.exists(r"C:\hadoop"):
        builder = (
            builder
            .config("spark.driver.extraJavaOptions", "-Dhadoop.home.dir=C:/hadoop")
            .config("spark.executor.extraJavaOptions", "-Dhadoop.home.dir=C:/hadoop")
        )
    return builder.getOrCreate()


def compute_metrics_from_counts(tp: int, fp: int, tn: int, fn: int, roc_auc: float, pr_auc: float, label: str) -> dict:
    """Compute comprehensive metrics directly from confusion matrix counts."""
    total = max(tp + fp + tn + fn, 1)
    acc   = (tp + tn) / total
    prec  = tp / max(tp + fp, 1) if (tp + fp) > 0 else 0.0
    rec   = tp / max(tp + fn, 1) if (tp + fn) > 0 else 0.0
    f1    = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    fpr   = fp / max(fp + tn, 1) if (fp + tn) > 0 else 0.0
    fnr   = fn / max(fn + tp, 1) if (fn + tp) > 0 else 0.0
    fdr   = rec

    print(f"\n  [{label}] Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  [{label}] ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"  [{label}] TP={tp}  FP={fp}  TN={tn}  FN={fn}  FPR={fpr:.4f}  FNR={fnr:.4f}")

    return {
        "accuracy":             round(float(acc),     4),
        "precision":            round(float(prec),    4),
        "recall":               round(float(rec),     4),
        "f1_score":             round(float(f1),      4),
        "roc_auc":              round(float(roc_auc), 4),
        "pr_auc":               round(float(pr_auc),  4),
        "fraud_detection_rate": round(float(fdr),     4),
        "false_positive_rate":  round(float(fpr),     4),
        "false_negative_rate":  round(float(fnr),     4),
        "confusion_matrix": {
            "true_positive":  tp,
            "false_positive": fp,
            "true_negative":  tn,
            "false_negative": fn,
        },
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, label: str) -> dict:
    """Compute comprehensive metrics for binary classification."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except Exception:
        roc_auc = 0.0
    try:
        pr_auc = average_precision_score(y_true, y_score)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]),
                      int(cm[1, 0]), int(cm[1, 1]))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fdr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # fraud detection rate = recall

    print(f"\n  [{label}] Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")
    print(f"  [{label}] ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
    print(f"  [{label}] TP={tp}  FP={fp}  TN={tn}  FN={fn}  FPR={fpr:.4f}  FNR={fnr:.4f}")

    return {
        "accuracy":             round(float(acc),     4),
        "precision":            round(float(prec),    4),
        "recall":               round(float(rec),     4),
        "f1_score":             round(float(f1),      4),
        "roc_auc":              round(float(roc_auc), 4),
        "pr_auc":               round(float(pr_auc),  4),
        "fraud_detection_rate": round(float(fdr),     4),
        "false_positive_rate":  round(float(fpr),     4),
        "false_negative_rate":  round(float(fnr),     4),
        "confusion_matrix": {
            "true_positive":  tp,
            "false_positive": fp,
            "true_negative":  tn,
            "false_negative": fn,
        },
    }


def main():
    # ── Verify sklearn is installed ──────────────────────────────────────────
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("[ERROR] scikit-learn not installed. Run: pip install scikit-learn")

    # ── Load model metadata (thresholds) ────────────────────────────────────
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        sys.exit("[ERROR] model_metadata.json not found. Run train_models.py first.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    lr_threshold  = float(meta.get("lr_threshold",  0.5))
    xgb_threshold = float(meta.get("xgb_threshold", 0.5))

    print("=" * 70)
    print("  FraudShield 2.0 — Model Evaluation on fraudTest.csv")
    print(f"  LR threshold : {lr_threshold}   XGB threshold: {xgb_threshold}")
    print("=" * 70)

    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Load test CSV ─────────────────────────────────────────────────────
    print("\n[1/5] Loading fraudTest.csv …")
    test_raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(TEST_CSV))
    )
    test_rows  = test_raw.count()
    test_fraud = test_raw.filter(col("is_fraud") == 1).count()
    print(f"    Rows: {test_rows:,}  |  Fraud: {test_fraud:,} ({test_fraud/test_rows*100:.2f}%)")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("\n[2/5] Feature engineering …")
    test_enriched = enrich_transaction(test_raw)
    test_enriched = test_enriched.withColumn(LABEL_COL, col(LABEL_COL).cast("double"))

    # ── 3. Apply shared preprocessing pipeline ───────────────────────────────
    print("\n[3/5] Applying preprocessing pipeline …")
    prep_path_str = str(MODELS_DIR / "preprocessor").replace("\\", "/")
    j_prep = spark._jvm.org.apache.spark.ml.PipelineModel.load(prep_path_str)
    prep_model = PipelineModel._from_java(j_prep)
    test_preprocessed = prep_model.transform(test_enriched)
    test_preprocessed.cache()
    _ = test_preprocessed.count()

    # ── 4. Load models and score ─────────────────────────────────────────────
    print("\n[4/5] Scoring test set with all 3 models ...")
    results = {}

    from pyspark.sql.functions import count as spark_count, when
    from pyspark.ml.evaluation import BinaryClassificationEvaluator

    # --- Model 1: Linear Regression ---
    print("\n  [LR] Loading and scoring ...")
    lr_path_str = str(MODELS_DIR / "linear_regression").replace("\\", "/")
    j_lr = spark._jvm.org.apache.spark.ml.regression.LinearRegressionModel.load(lr_path_str)
    lr_model = LinearRegressionModel._from_java(j_lr)
    t_inf = time.time()
    lr_scored = lr_model.transform(test_preprocessed)
    lr_scored = lr_scored.withColumn("lr_pred", when(col("prediction") >= lr_threshold, 1.0).otherwise(0.0))
    lr_scored.cache()
    _ = lr_scored.count()
    lr_inf_time = time.time() - t_inf

    cm_row = lr_scored.groupBy().agg(
        spark_count(when((col("lr_pred") == 1.0) & (col(LABEL_COL) == 1.0), 1)).alias("tp"),
        spark_count(when((col("lr_pred") == 1.0) & (col(LABEL_COL) == 0.0), 1)).alias("fp"),
        spark_count(when((col("lr_pred") == 0.0) & (col(LABEL_COL) == 0.0), 1)).alias("tn"),
        spark_count(when((col("lr_pred") == 0.0) & (col(LABEL_COL) == 1.0), 1)).alias("fn"),
    ).collect()[0]

    tp, fp, tn, fn = int(cm_row["tp"]), int(cm_row["fp"]), int(cm_row["tn"]), int(cm_row["fn"])
    roc_eval = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol=LABEL_COL, metricName="areaUnderROC")
    pr_eval  = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol=LABEL_COL, metricName="areaUnderPR")
    lr_roc   = roc_eval.evaluate(lr_scored)
    lr_pr    = pr_eval.evaluate(lr_scored)

    lr_metrics = compute_metrics_from_counts(tp, fp, tn, fn, lr_roc, lr_pr, "LR")
    lr_metrics["training_time"]   = round(float(meta.get("lr_training_time",  0.0)), 2)
    lr_metrics["inference_time"]  = round(float(lr_inf_time), 2)
    lr_metrics["threshold"]       = lr_threshold
    lr_metrics["model_note"]      = ("LinearRegression used as a regression-based fraud scorer. "
                                     "Continuous scores are thresholded for binary classification. "
                                     "This is a baseline approach, not the standard classification method.")
    results["linear_regression"] = lr_metrics

    # --- Model 2: Random Forest ---
    print("\n  [RF] Loading and scoring ...")
    rf_path_str = str(MODELS_DIR / "random_forest").replace("\\", "/")
    j_rf = spark._jvm.org.apache.spark.ml.classification.RandomForestClassificationModel.load(rf_path_str)
    rf_model = RandomForestClassificationModel._from_java(j_rf)
    t_inf = time.time()
    rf_scored = rf_model.transform(test_preprocessed)
    rf_scored.cache()
    _ = rf_scored.count()
    rf_inf_time = time.time() - t_inf

    rf_cm = rf_scored.groupBy().agg(
        spark_count(when((col("prediction") == 1.0) & (col(LABEL_COL) == 1.0), 1)).alias("tp"),
        spark_count(when((col("prediction") == 1.0) & (col(LABEL_COL) == 0.0), 1)).alias("fp"),
        spark_count(when((col("prediction") == 0.0) & (col(LABEL_COL) == 0.0), 1)).alias("tn"),
        spark_count(when((col("prediction") == 0.0) & (col(LABEL_COL) == 1.0), 1)).alias("fn"),
    ).collect()[0]

    tp_rf, fp_rf, tn_rf, fn_rf = int(rf_cm["tp"]), int(rf_cm["fp"]), int(rf_cm["tn"]), int(rf_cm["fn"])
    rf_roc_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=LABEL_COL, metricName="areaUnderROC")
    rf_pr_eval  = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol=LABEL_COL, metricName="areaUnderPR")
    rf_roc      = rf_roc_eval.evaluate(rf_scored)
    rf_pr       = rf_pr_eval.evaluate(rf_scored)

    rf_metrics = compute_metrics_from_counts(tp_rf, fp_rf, tn_rf, fn_rf, rf_roc, rf_pr, "RF")
    rf_metrics["training_time"]  = round(float(meta.get("rf_training_time", 0.0)), 2)
    rf_metrics["inference_time"] = round(float(rf_inf_time), 2)
    results["random_forest"] = rf_metrics

    # --- Model 3: XGBoost ---
    print("\n  [XGB] Loading and scoring ...")
    try:
        import xgboost as xgb
        import pandas as pd
        import shutil
    except ImportError:
        sys.exit("[ERROR] xgboost or pandas not installed.")
    xgb_booster = xgb.Booster()
    xgb_booster.load_model(str(MODELS_DIR / "xgboost" / "xgb_model.json"))

    temp_xgb_parquet = RESULTS_DIR / "temp_xgb_test_data"
    if temp_xgb_parquet.exists():
        shutil.rmtree(temp_xgb_parquet, ignore_errors=True)

    print("    Exporting test feature vectors to optimized Parquet ...")
    (
        test_preprocessed
        .withColumn("features_arr", vector_to_array(FEATURES_COL))
        .select("features_arr", LABEL_COL)
        .write
        .mode("overwrite")
        .parquet(str(temp_xgb_parquet))
    )

    t_inf = time.time()
    print("    Reading Parquet and predicting with XGBoost Booster ...")
    xgb_df = pd.read_parquet(str(temp_xgb_parquet))
    X_test = np.vstack(xgb_df["features_arr"].values)
    y_test = xgb_df[LABEL_COL].values.astype(int)
    xgb_probs = xgb_booster.predict(xgb.DMatrix(X_test))
    xgb_preds = (xgb_probs >= xgb_threshold).astype(int)
    xgb_inf_time = time.time() - t_inf

    # Clean up temporary Parquet
    shutil.rmtree(temp_xgb_parquet, ignore_errors=True)

    xgb_metrics = compute_metrics(y_test, xgb_preds, xgb_probs, "XGB")
    xgb_metrics["training_time"]  = round(float(meta.get("xgb_training_time", 0.0)), 2)
    xgb_metrics["inference_time"] = round(float(xgb_inf_time), 2)
    xgb_metrics["threshold"]      = xgb_threshold
    results["xgboost"] = xgb_metrics

    # ── 5. Save comparison JSON ──────────────────────────────────────────────
    comparison = {
        "metadata": {
            "dataset":          "Sparkov-generated simulated credit-card transaction data",
            "source":           "https://www.kaggle.com/datasets/kartik2112/fraud-detection",
            "note":             ("This dataset was generated by the Sparkov Data Generation tool. "
                                 "It simulates realistic credit-card transaction patterns but is NOT "
                                 "actual bank-collected transaction data."),
            "train_file":       "fraudTrain.csv",
            "test_file":        "fraudTest.csv",
            "test_rows":        test_rows,
            "test_fraud_rows":  int(test_fraud),
            "test_fraud_pct":   round(test_fraud / test_rows * 100, 4),
        },
        "linear_regression": results["linear_regression"],
        "random_forest":     results["random_forest"],
        "xgboost":           results["xgboost"],
    }

    out_path = RESULTS_DIR / "model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[5/5] Results saved -> {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print(f"  {'Metric':<22} {'Linear Reg':>14} {'Random Forest':>14} {'XGBoost':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")
    for metric in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
        lr_v  = results["linear_regression"].get(metric, 0)
        rf_v  = results["random_forest"].get(metric, 0)
        xgb_v = results["xgboost"].get(metric, 0)
        print(f"  {metric:<22} {lr_v:>14.4f} {rf_v:>14.4f} {xgb_v:>14.4f}")
    print("=" * 70)
    print("\nNote: For highly imbalanced data, Recall, F1, and PR-AUC are more")
    print("meaningful than Accuracy. Do not select a model based on Accuracy alone.")
    print("\nDashboard will load results from: results/model_comparison.json")

    spark.stop()


if __name__ == "__main__":
    main()
