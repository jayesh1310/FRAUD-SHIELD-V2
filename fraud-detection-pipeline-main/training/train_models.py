"""
train_models.py
---------------
Phase 3: Train three ML models on fraudTrain.csv (Sparkov-generated simulation data).

Models trained:
  1. LinearRegression   (MLlib, regression-based fraud scorer with tuned threshold)
  2. RandomForestClassifier (MLlib, 100 trees, class-weight-balanced)
  3. XGBoost            (xgboost Python library, Booster saved as JSON)

All models share one fitted preprocessing pipeline (saved separately).
Thresholds are tuned on a 20% validation split derived from training data.
Test data (fraudTest.csv) is NEVER touched here.

Usage:
  cd fraud-detection-pipeline-main
  python training/train_models.py

Outputs:
  models/preprocessor/      - fitted PySpark PipelineModel
  models/linear_regression/ - fitted LinearRegressionModel
  models/random_forest/     - fitted RandomForestClassificationModel
  models/xgboost/xgb_model.json
  models/model_metadata.json  (thresholds, feature count, class counts)
"""

import json
import os
import sys
import time
import shutil
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = Path(os.getenv("DATA_DIR",  str(ROOT_DIR.parent)))   # where CSVs live
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(ROOT_DIR / "models")))
TRAIN_CSV  = DATA_DIR / "fraudTrain.csv"

# ── PySpark / MLlib imports ────────────────────────────────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.functions import vector_to_array

# Feature engineering (shared module)
sys.path.insert(0, str(ROOT_DIR))
from features.feature_engineering import (
    enrich_transaction,
    build_preprocessing_pipeline,
    LABEL_COL, FEATURES_COL, SCALED_FEATURES_COL,
    NUMERIC_FEATURE_COLS, CATEGORICAL_COLS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def create_spark_session() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("FraudShield-Training")
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


def cleanup_model_dir(path: Path):
    """Remove existing model directory so save() doesn't fail."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def tune_lr_threshold(scores, labels):
    """
    Find the classification threshold that maximises F1 on a validation set.
    scores, labels: 1-D numpy arrays.
    """
    import numpy as np
    best_f1, best_thresh = 0.0, 0.5
    for t in [i / 100 for i in range(5, 65, 5)]:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"[LR]  Best validation F1={best_f1:.4f} at threshold={best_thresh:.2f}")
    return float(best_thresh)


def tune_xgb_threshold(probs, labels):
    """Same F1-maximising threshold search for XGBoost scores."""
    import numpy as np
    best_f1, best_thresh = 0.0, 0.5
    for t in [i / 100 for i in range(5, 65, 5)]:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"[XGB] Best validation F1={best_f1:.4f} at threshold={best_thresh:.2f}")
    return float(best_thresh)


# ── Main training routine ─────────────────────────────────────────────────────
def main():
    import numpy as np

    print("=" * 70)
    print("  FraudShield 2.0 — Model Training")
    print(f"  Train CSV : {TRAIN_CSV}")
    print(f"  Models dir: {MODELS_DIR}")
    print("=" * 70)

    if not TRAIN_CSV.exists():
        sys.exit(f"[ERROR] Training file not found: {TRAIN_CSV}\n"
                 "Place fraudTrain.csv in the project root (parent of fraud-detection-pipeline-main).")

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # ── 1. Load training CSV ─────────────────────────────────────────────────
    print("\n[1/6] Loading fraudTrain.csv …")
    t0 = time.time()
    train_raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(TRAIN_CSV))
    )
    # Rename ambiguous column to avoid conflicts
    if "long" not in train_raw.columns and "long" not in train_raw.columns:
        pass
    train_raw = train_raw.withColumnRenamed("long", "long") if "long" in train_raw.columns else train_raw
    
    total_rows  = train_raw.count()
    fraud_rows  = train_raw.filter(col("is_fraud") == 1).count()
    legit_rows  = total_rows - fraud_rows
    print(f"    Rows: {total_rows:,}  |  Fraud: {fraud_rows:,} ({fraud_rows/total_rows*100:.2f}%)  |  Legit: {legit_rows:,}")
    print(f"    Load time: {time.time()-t0:.1f}s")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    print("\n[2/6] Feature engineering …")
    t0 = time.time()
    train_enriched = enrich_transaction(train_raw)
    # Cast label to double for MLlib
    train_enriched = train_enriched.withColumn(LABEL_COL, col(LABEL_COL).cast("double"))
    train_enriched.cache()
    _ = train_enriched.count()  # materialise cache
    print(f"    Feature engineering done in {time.time()-t0:.1f}s")

    # 80/20 train/validation split (for threshold tuning)
    train_df, val_df = train_enriched.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    val_df.cache()
    _ = train_df.count()
    _ = val_df.count()
    print(f"    Train split: {train_df.count():,} rows | Val split: {val_df.count():,} rows")

    # ── 3. Fit shared preprocessing pipeline ─────────────────────────────────
    print("\n[3/6] Fitting preprocessing pipeline (StringIndexer + OHE + VectorAssembler + StandardScaler) …")
    t0 = time.time()
    prep_pipeline = build_preprocessing_pipeline()
    prep_model    = prep_pipeline.fit(train_df)

    prep_dir = MODELS_DIR / "preprocessor"
    cleanup_model_dir(prep_dir)
    prep_model.write().overwrite().save(str(prep_dir))
    print(f"    Preprocessing pipeline saved -> {prep_dir}  ({time.time()-t0:.1f}s)")

    train_preprocessed = prep_model.transform(train_df)
    val_preprocessed   = prep_model.transform(val_df)
    train_preprocessed.cache()
    val_preprocessed.cache()
    _ = train_preprocessed.count()
    _ = val_preprocessed.count()

    # ── 4. Model 1: Linear Regression ────────────────────────────────────────
    print("\n[4a/6] Training Model 1: Linear Regression …")
    print("       (Regression-based fraud scorer — scores are continuous, not probabilities)")
    t0_lr = time.time()

    lr = LinearRegression(
        featuresCol=SCALED_FEATURES_COL,
        labelCol=LABEL_COL,
        maxIter=20,
        regParam=0.001,
        elasticNetParam=0.0,
    )
    lr_model = lr.fit(train_preprocessed)
    lr_time  = time.time() - t0_lr

    # Save LR model
    lr_dir = MODELS_DIR / "linear_regression"
    cleanup_model_dir(lr_dir)
    lr_model.write().overwrite().save(str(lr_dir))
    print(f"    LR saved -> {lr_dir}  (training: {lr_time:.1f}s)")

    # Threshold calibration on validation set
    print("    Calibrating LR threshold on validation set ...")
    val_lr_scored = lr_model.transform(val_preprocessed)
    val_lr_pd     = val_lr_scored.select("prediction", LABEL_COL).toPandas()
    lr_scores     = val_lr_pd["prediction"].values
    lr_labels     = val_lr_pd[LABEL_COL].values.astype(int)
    lr_threshold  = tune_lr_threshold(lr_scores, lr_labels)

    # ── 5. Model 2: Random Forest ─────────────────────────────────────────────
    print("\n[4b/6] Training Model 2: Random Forest Classifier …")
    t0_rf = time.time()

    # Class weights to handle imbalance
    n_total  = float(total_rows)
    w_fraud  = n_total / (2.0 * max(fraud_rows, 1))
    w_legit  = n_total / (2.0 * max(legit_rows, 1))
    train_weighted = train_preprocessed.withColumn(
        "weight",
        when(col(LABEL_COL) == 1.0, w_fraud).otherwise(w_legit),
    )

    rf = RandomForestClassifier(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        weightCol="weight",
        numTrees=100,
        maxDepth=10,
        subsamplingRate=0.8,
        seed=42,
    )
    rf_model = rf.fit(train_weighted)
    rf_time  = time.time() - t0_rf

    rf_dir = MODELS_DIR / "random_forest"
    cleanup_model_dir(rf_dir)
    rf_model.write().overwrite().save(str(rf_dir))
    print(f"    RF saved -> {rf_dir}  (training: {rf_time:.1f}s)")

    # ── 6. Model 3: XGBoost ───────────────────────────────────────────────────
    print("\n[4c/6] Training Model 3: XGBoost ...")
    print("       Extracting feature vectors to pandas for XGBoost training ...")
    t0_xgb = time.time()

    try:
        import xgboost as xgb
    except ImportError:
        sys.exit("[ERROR] xgboost not installed. Run: pip install xgboost")

    # Extract the "features" vector column → numpy array
    train_arr = (
        train_preprocessed
        .withColumn("features_arr", vector_to_array(FEATURES_COL))
        .select("features_arr", LABEL_COL)
        .toPandas()
    )
    val_arr = (
        val_preprocessed
        .withColumn("features_arr", vector_to_array(FEATURES_COL))
        .select("features_arr", LABEL_COL)
        .toPandas()
    )

    X_train = np.vstack(train_arr["features_arr"].values)
    y_train = train_arr[LABEL_COL].values.astype(int)
    X_val   = np.vstack(val_arr["features_arr"].values)
    y_val   = val_arr[LABEL_COL].values.astype(int)

    scale_pos_weight = float(legit_rows) / max(fraud_rows, 1)
    print(f"    scale_pos_weight = {scale_pos_weight:.1f}  (legit/fraud ratio)")

    xgb_clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        use_label_encoder=False,
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    xgb_time = time.time() - t0_xgb

    xgb_dir = MODELS_DIR / "xgboost"
    xgb_dir.mkdir(parents=True, exist_ok=True)
    xgb_path = xgb_dir / "xgb_model.json"
    xgb_clf.get_booster().save_model(str(xgb_path))
    print(f"    XGB saved -> {xgb_path}  (training: {xgb_time:.1f}s)")

    # Threshold calibration for XGBoost
    xgb_probs     = xgb_clf.predict_proba(X_val)[:, 1]
    xgb_threshold = tune_xgb_threshold(xgb_probs, y_val)

    # ── 7. Save model metadata ────────────────────────────────────────────────
    # Number of features in the assembled vector
    feature_count = int(X_train.shape[1])

    metadata = {
        "lr_threshold":      lr_threshold,
        "xgb_threshold":     xgb_threshold,
        "feature_count":     feature_count,
        "train_rows":        total_rows,
        "train_fraud_rows":  fraud_rows,
        "train_legit_rows":  legit_rows,
        "scale_pos_weight":  scale_pos_weight,
        "lr_training_time":  round(lr_time, 2),
        "rf_training_time":  round(rf_time, 2),
        "xgb_training_time": round(xgb_time, 2),
        "numeric_features":  NUMERIC_FEATURE_COLS,
        "categorical_features": CATEGORICAL_COLS,
    }
    meta_path = MODELS_DIR / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[6/6] Model metadata saved -> {meta_path}")

    print("\n" + "=" * 70)
    print("  Training complete!")
    print(f"  LR  training time : {lr_time:.1f}s  |  threshold: {lr_threshold:.2f}")
    print(f"  RF  training time : {rf_time:.1f}s")
    print(f"  XGB training time : {xgb_time:.1f}s  |  threshold: {xgb_threshold:.2f}")
    print(f"  Feature vector dim: {feature_count}")
    print("=" * 70)
    print("\nNext step: python training/evaluate_models.py")

    spark.stop()


if __name__ == "__main__":
    main()
