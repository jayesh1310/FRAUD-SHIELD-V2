"""
spark_consumer_ml.py
--------------------
FraudShield 2.0 — Three-Model Spark Structured Streaming Consumer.

For every Kafka micro-batch:
  1. Deserialize JSON using the flat schema (raw Kaggle fields).
  2. Apply feature engineering (shared features/feature_engineering.py).
  3. Apply shared preprocessing pipeline (StringIndexer+OHE+VectorAssembler+StandardScaler).
  4. Score with Linear Regression model  -> lr_score, lr_prediction.
  5. Score with Random Forest model      -> rf_score, rf_prediction.
  6. Score with XGBoost model            -> xgb_score, xgb_prediction.
  7. Record actual_is_fraud (ground-truth label — NOT used as ML feature).
  8. Write combined predictions to Parquet (Snappy compression).

Output schema per transaction:
  transaction_id, trans_date_trans_time, amt, merchant, category,
  city, state, actual_is_fraud,
  lr_score, lr_prediction, lr_correct,
  rf_score, rf_prediction, rf_correct,
  xgb_score, xgb_prediction, xgb_correct,
  prediction_timestamp
"""

import json
import os
import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, when
from pyspark.sql.types import (
    DoubleType, IntegerType, LongType, StringType, StructField, StructType,
)
from pyspark.ml import PipelineModel
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.functions import vector_to_array

# ── Configuration ─────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "transactions")
SPARK_MASTER            = os.getenv("SPARK_MASTER", "local[2]")

# Mounted volume paths (see docker-compose.yml)
MODELS_DIR        = os.getenv("MODELS_DIR",    "/app/models")
OUTPUT_PATH       = os.getenv("OUTPUT_PATH",   "/data/storage/predictions/combined_predictions")
CHECKPOINT_PATH   = os.getenv("CHECKPOINT_PATH", "/data/storage/checkpoints/consumer_v2")
PYTHONPATH_ENV    = os.getenv("PYTHONPATH",    "/app")

# ── Incoming Kafka message schema (matches producer) ──────────────────────────
TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id",        StringType()),
    StructField("cc_num",               StringType()),
    StructField("trans_num",            StringType()),
    StructField("trans_date_trans_time", StringType()),
    StructField("unix_time",            LongType()),
    StructField("merchant",             StringType()),
    StructField("category",             StringType()),
    StructField("amt",                  DoubleType()),
    StructField("first",                StringType()),
    StructField("last",                 StringType()),
    StructField("gender",               StringType()),
    StructField("dob",                  StringType()),
    StructField("city",                 StringType()),
    StructField("state",                StringType()),
    StructField("zip",                  StringType()),
    StructField("city_pop",             LongType()),
    StructField("lat",                  DoubleType()),
    StructField("long",                 DoubleType()),
    StructField("merch_lat",            DoubleType()),
    StructField("merch_long",           DoubleType()),
    # Ground-truth label — stored for evaluation AFTER prediction, never as ML input
    StructField("actual_is_fraud",      IntegerType()),
])


# ── SparkSession ──────────────────────────────────────────────────────────────
def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("FraudShield-3ModelConsumer")
        .config("spark.executorEnv.PYTHONPATH", PYTHONPATH_ENV)
        .config("spark.yarn.appMasterEnv.PYTHONPATH", PYTHONPATH_ENV)
        .config("spark.sql.parquet.compression.codec", "snappy")
        .getOrCreate()
    )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_PATH,     exist_ok=True)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # ── Load model metadata (thresholds) ────────────────────────────────────
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    lr_threshold  = 0.5
    xgb_threshold = 0.5
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        lr_threshold  = float(meta.get("lr_threshold",  0.5))
        xgb_threshold = float(meta.get("xgb_threshold", 0.5))
    print(f"[Consumer] LR threshold={lr_threshold}  XGB threshold={xgb_threshold}", flush=True)

    # ── Load models ──────────────────────────────────────────────────────────
    print("[Consumer] Loading preprocessing pipeline …", flush=True)
    prep_path_str = os.path.join(MODELS_DIR, "preprocessor").replace("\\", "/")
    j_prep = spark._jvm.org.apache.spark.ml.PipelineModel.load(prep_path_str)
    prep_model = PipelineModel._from_java(j_prep)

    print("[Consumer] Loading Linear Regression model …", flush=True)
    lr_path_str = os.path.join(MODELS_DIR, "linear_regression").replace("\\", "/")
    j_lr = spark._jvm.org.apache.spark.ml.regression.LinearRegressionModel.load(lr_path_str)
    lr_model = LinearRegressionModel._from_java(j_lr)

    print("[Consumer] Loading Random Forest model …", flush=True)
    rf_path_str = os.path.join(MODELS_DIR, "random_forest").replace("\\", "/")
    j_rf = spark._jvm.org.apache.spark.ml.classification.RandomForestClassificationModel.load(rf_path_str)
    rf_model = RandomForestClassificationModel._from_java(j_rf)

    xgb_model_path = os.path.join(MODELS_DIR, "xgboost", "xgb_model.json")
    print(f"[Consumer] Loading XGBoost model: {xgb_model_path}", flush=True)
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(xgb_model_path)

    sys.path.insert(0, "/app")
    from features.feature_engineering import enrich_transaction, FEATURES_COL

    # ── 1. Read from Kafka ───────────────────────────────────────────────────
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # ── 2. Parse JSON ────────────────────────────────────────────────────────
    parsed = (
        raw_stream
        .select(from_json(col("value").cast("string"), TRANSACTION_SCHEMA).alias("d"))
        .select("d.*")
        .filter(col("transaction_id").isNotNull())
    )

    # ── 3. Micro-batch processor ─────────────────────────────────────────────
    def process_batch(batch_df, batch_id):
        if not batch_df.take(1):
            return

        try:
            # 1. Feature engineering
            enriched = enrich_transaction(batch_df)

            # 2. Shared preprocessing
            preprocessed = prep_model.transform(enriched)

            # 3. LR scoring
            lr_scored = lr_model.transform(preprocessed)
            scored = (
                lr_scored
                .withColumnRenamed("prediction", "lr_raw_score")
                .withColumn("lr_score",      col("lr_raw_score").cast(DoubleType()))
                .withColumn("lr_prediction", when(col("lr_raw_score") >= lr_threshold, 1).otherwise(0).cast(IntegerType()))
            )

            # 4. RF scoring
            rf_scored = rf_model.transform(scored)
            scored = (
                rf_scored
                .withColumn("rf_score",      vector_to_array(col("probability"))[1].cast(DoubleType()))
                .withColumnRenamed("prediction", "rf_prediction")
                .withColumn("rf_prediction", col("rf_prediction").cast(IntegerType()))
            )

            # 5. Extract for XGBoost & Final Parquet
            scored = scored.withColumn("features_array", vector_to_array(FEATURES_COL))
            pdf = scored.select(
                "transaction_id", "trans_date_trans_time", "amt", "merchant", "category",
                "city", "state",
                "actual_is_fraud",
                "lr_score", "lr_prediction",
                "rf_score", "rf_prediction",
                "features_array",
            ).toPandas()

            if len(pdf) == 0:
                return

            import numpy as np
            import pandas as pd

            X = np.vstack(pdf["features_array"].values)
            xgb_probs = booster.predict(xgb.DMatrix(X))

            pdf["xgb_score"]      = [float(p) for p in np.round(xgb_probs, 4)]
            pdf["xgb_prediction"] = (xgb_probs >= xgb_threshold).astype(int)
            pdf["lr_correct"]     = (pdf["lr_prediction"]  == pdf["actual_is_fraud"]).astype(int)
            pdf["rf_correct"]     = (pdf["rf_prediction"]  == pdf["actual_is_fraud"]).astype(int)
            pdf["xgb_correct"]    = (pdf["xgb_prediction"] == pdf["actual_is_fraud"]).astype(int)
            pdf["prediction_timestamp"] = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
            pdf.drop(columns=["features_array"], inplace=True)

            out_spark_df = spark.createDataFrame(pdf)
            (
                out_spark_df.write
                .mode("append")
                .format("parquet")
                .option("compression", "snappy")
                .save(OUTPUT_PATH)
            )

            flagged = int(((pdf["lr_prediction"] == 1) | (pdf["rf_prediction"] == 1) | (pdf["xgb_prediction"] == 1)).sum())
            print(f"[Consumer] Batch #{batch_id}: {len(pdf)} txs processed, {flagged} flagged as fraud.", flush=True)

        except Exception as err:
            print(f"[Consumer] Error processing batch #{batch_id}: {err}", file=sys.stderr, flush=True)

    # ── 4. Start Streaming Query ─────────────────────────────────────────────
    query = (
        parsed.writeStream
        .foreachBatch(process_batch)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="5 seconds")
        .start()
    )

    print(f"[Consumer] 3-model streaming query started (id={query.id})", flush=True)
    print(f"[Consumer] Output -> {OUTPUT_PATH}", flush=True)
    print(f"[Consumer] Checkpoint -> {CHECKPOINT_PATH}", flush=True)
    print("[Consumer] Awaiting termination …", flush=True)

    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("[Consumer] Interrupted — stopping.", flush=True)
        query.stop()
    except Exception as exc:
        print(f"[Consumer] Query failed: {exc}", file=sys.stderr, flush=True)
        query.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
