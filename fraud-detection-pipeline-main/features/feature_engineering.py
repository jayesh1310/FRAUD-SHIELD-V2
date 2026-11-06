"""
feature_engineering.py
-----------------------
Shared PySpark feature-engineering pipeline for FraudShield 2.0.

Used by:
  - training/train_models.py        (fit on fraudTrain.csv)
  - training/evaluate_models.py     (transform fraudTest.csv)
  - consumer/spark_consumer_ml.py   (transform live Kafka stream)

Dataset:  Sparkov-generated simulated credit-card transaction data.
          Source: https://www.kaggle.com/datasets/kartik2112/fraud-detection

Features engineered:
  Temporal  : transaction_hour, transaction_day, transaction_month,
              transaction_day_of_week, is_night_transaction, is_weekend
  Transaction: transaction_amount, log_transaction_amount
  Geographic : cardholder_latitude/longitude, merchant_latitude/longitude,
               home_to_merchant_distance
  Customer   : customer_age
  Population : log_city_pop
  Categorical: category, gender, state  (kept raw, encoded in preprocessing pipeline)
"""

import math
import os

# ── Ensure HADOOP_HOME and PySpark Python on Windows ─────────────────────────
if os.name == "nt":
    if os.path.exists(r"C:\hadoop"):
        os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
        hadoop_bin = r"C:\hadoop\bin"
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

import sys as _sys
_py = _sys.executable
if _py and os.path.isfile(_py):
    os.environ.setdefault("PYSPARK_PYTHON",        _py)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON",  _py)

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, hour, dayofmonth, month, dayofweek, log1p, when, sqrt, pow as spark_pow,
    to_timestamp, datediff, to_date, lit, coalesce, radians, sin, cos, atan2,
)
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler,
)

# ─── Feature column registry ──────────────────────────────────────────────────
NUMERIC_FEATURE_COLS = [
    "transaction_amount",        # raw amount
    "log_transaction_amount",    # log(1 + amount) — reduces right skew
    "transaction_hour",          # 0-23
    "transaction_day",           # 1-31
    "transaction_month",         # 1-12
    "transaction_day_of_week",   # 1=Sun, 7=Sat (Spark convention)
    "is_night_transaction",      # 1 if 22:00–05:00 else 0
    "is_weekend",                # 1 if Sat/Sun else 0
    "cardholder_latitude",
    "cardholder_longitude",
    "merchant_latitude",
    "merchant_longitude",
    "home_to_merchant_distance", # haversine km
    "customer_age",              # years at transaction time
    "log_city_pop",              # log(1 + city_pop)
]

CATEGORICAL_COLS      = ["category", "gender", "state"]
CATEGORICAL_INDEX_COLS = [f"{c}_idx" for c in CATEGORICAL_COLS]
CATEGORICAL_OHE_COLS  = [f"{c}_ohe" for c in CATEGORICAL_COLS]

ASSEMBLER_INPUT_COLS  = NUMERIC_FEATURE_COLS + CATEGORICAL_OHE_COLS
LABEL_COL             = "is_fraud"
FEATURES_COL          = "features"
SCALED_FEATURES_COL   = "scaled_features"


# ─── Pure Spark SQL haversine (no Python UDF / no Python worker required) ──────
def haversine_col(lat1_c, lon1_c, lat2_c, lon2_c):
    """
    Haversine distance in km as a Spark Column expression.
    Uses only built-in Spark functions — no Python UDF, no Python worker process.
    """
    R = 6371.0
    dlat = radians(lat2_c - lat1_c)
    dlon = radians(lon2_c - lon1_c)
    a = (spark_pow(sin(dlat / 2), 2)
         + cos(radians(lat1_c)) * cos(radians(lat2_c)) * spark_pow(sin(dlon / 2), 2))
    # atan2(sqrt(a), sqrt(1-a)) * 2 * R
    dist = lit(2.0 * R) * atan2(sqrt(a), sqrt(lit(1.0) - a))
    return dist


# ─── Feature computation functions ───────────────────────────────────────────
def add_temporal_features(df: DataFrame, ts_col: str = "trans_date_trans_time") -> DataFrame:
    """Extract temporal features from the transaction timestamp string."""
    ts = to_timestamp(col(ts_col), "yyyy-MM-dd HH:mm:ss")
    return (
        df
        .withColumn("_ts", ts)
        .withColumn("transaction_hour",        hour("_ts").cast(DoubleType()))
        .withColumn("transaction_day",         dayofmonth("_ts").cast(DoubleType()))
        .withColumn("transaction_month",       month("_ts").cast(DoubleType()))
        .withColumn("transaction_day_of_week", dayofweek("_ts").cast(DoubleType()))
        .withColumn("is_night_transaction",
                    when((hour("_ts") >= 22) | (hour("_ts") <= 5), 1.0)
                    .otherwise(0.0))
        .withColumn("is_weekend",
                    when(dayofweek("_ts").isin([1, 7]), 1.0).otherwise(0.0))
        .drop("_ts")
    )


def add_transaction_features(df: DataFrame) -> DataFrame:
    """Amount-based features."""
    return (
        df
        .withColumn("transaction_amount",     col("amt").cast(DoubleType()))
        .withColumn("log_transaction_amount", log1p(col("amt").cast(DoubleType())))
    )


def add_geographic_features(df: DataFrame) -> DataFrame:
    """Coordinates and haversine distance between cardholder home and merchant."""
    return (
        df
        .withColumn("cardholder_latitude",       col("lat").cast(DoubleType()))
        .withColumn("cardholder_longitude",      col("long").cast(DoubleType()))
        .withColumn("merchant_latitude",         col("merch_lat").cast(DoubleType()))
        .withColumn("merchant_longitude",        col("merch_long").cast(DoubleType()))
        .withColumn("home_to_merchant_distance",
                    haversine_col(
                        col("lat").cast(DoubleType()),
                        col("long").cast(DoubleType()),
                        col("merch_lat").cast(DoubleType()),
                        col("merch_long").cast(DoubleType()),
                    ).cast(DoubleType()))
    )


def add_customer_features(df: DataFrame) -> DataFrame:
    """Customer age at time of transaction (from dob and trans_date_trans_time)."""
    return (
        df
        .withColumn("_trans_date", to_date(col("trans_date_trans_time"), "yyyy-MM-dd HH:mm:ss"))
        .withColumn("_dob_date",   to_date(col("dob"), "yyyy-MM-dd"))
        .withColumn("customer_age",
                    (datediff(col("_trans_date"), col("_dob_date")) / 365.25)
                    .cast(DoubleType()))
        .withColumn("customer_age", coalesce(col("customer_age"), lit(35.0)))
        .drop("_trans_date", "_dob_date")
    )


def add_city_pop_log(df: DataFrame) -> DataFrame:
    """Log-scaled city population."""
    return df.withColumn("log_city_pop", log1p(col("city_pop").cast(DoubleType())))


def add_categorical_defaults(df: DataFrame) -> DataFrame:
    """Fill nulls in categorical columns so StringIndexer never sees None."""
    return df.fillna({"category": "misc_net", "gender": "M", "state": "XX"})


def enrich_transaction(df: DataFrame, ts_col: str = "trans_date_trans_time") -> DataFrame:
    """
    Apply the full feature-engineering pipeline to a transaction DataFrame.
    Works identically for training data (CSV) and streaming data (Kafka JSON).

    Steps:
      1. Amount features
      2. Temporal features
      3. Geographic features (lat/lon + distance)
      4. Customer age
      5. Log city population
      6. Categorical null-filling

    Args:
        df: Spark DataFrame with raw Kaggle columns.
        ts_col: Name of the timestamp column.

    Returns:
        Enriched DataFrame with additional feature columns.
    """
    df = add_transaction_features(df)
    df = add_temporal_features(df, ts_col=ts_col)
    df = add_geographic_features(df)
    df = add_customer_features(df)
    df = add_city_pop_log(df)
    df = add_categorical_defaults(df)
    return df


# ─── Preprocessing ML pipeline ────────────────────────────────────────────────
def build_preprocessing_pipeline() -> Pipeline:
    """
    Build an UNFITTED PySpark ML preprocessing pipeline.

    Stages:
      StringIndexer (category, gender, state)
      → OneHotEncoder
      → VectorAssembler  (numeric + OHE cols  → "features")
      → StandardScaler   ("features" → "scaled_features")

    The pipeline must be FIT on training data only, then APPLIED to both
    training and test/streaming data to prevent data leakage.

    Returns:
        Unfitted Pipeline instance.
    """
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL_COLS
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe", handleInvalid="keep")
        for c in CATEGORICAL_COLS
    ]
    assembler = VectorAssembler(
        inputCols=ASSEMBLER_INPUT_COLS,
        outputCol=FEATURES_COL,
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol=FEATURES_COL,
        outputCol=SCALED_FEATURES_COL,
        withMean=True,
        withStd=True,
    )
    return Pipeline(stages=indexers + encoders + [assembler, scaler])
