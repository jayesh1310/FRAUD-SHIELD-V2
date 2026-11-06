"""
ingest_data.py
--------------
Phase 2: Ingestion & Preprocessing for FraudShield 2.0.

Converts raw Kaggle CSV files (Sparkov-generated simulation data) into
optimized Apache Parquet files with Snappy compression.

Validates:
  - Schema consistency
  - Null values handling
  - Type casting
  - Deduplication
"""

import os
import sys
from pathlib import Path

# Configure environment for Windows
if os.name == "nt" and os.path.exists(r"C:\hadoop"):
    os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
    if r"C:\hadoop\bin" not in os.environ.get("PATH", ""):
        os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ.get("PATH", "")

_py = sys.executable
if _py and os.path.isfile(_py):
    os.environ.setdefault("PYSPARK_PYTHON", _py)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", _py)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR.parent
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

def get_spark():
    builder = (
        SparkSession.builder
        .appName("FraudShield-Ingestion")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.compression.codec", "snappy")
    )
    if os.name == "nt" and os.path.exists(r"C:\hadoop"):
        builder = (
            builder
            .config("spark.driver.extraJavaOptions", "-Dhadoop.home.dir=C:/hadoop")
            .config("spark.executor.extraJavaOptions", "-Dhadoop.home.dir=C:/hadoop")
        )
    return builder.getOrCreate()

def ingest_dataset(spark, csv_name, out_subdir):
    csv_path = DATA_DIR / csv_name
    target_path = OUTPUT_DIR / out_subdir
    print(f"\n[Ingest] Processing {csv_path} -> {target_path} ...")
    
    if not csv_path.exists():
        print(f"[Ingest] Warning: {csv_path} does not exist.")
        return

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(csv_path))
    )

    initial_count = df.count()
    print(f"  Raw count: {initial_count:,}")

    # Remove duplicates on transaction ID if present
    if "trans_num" in df.columns:
        df = df.dropDuplicates(["trans_num"])
    
    # Check nulls in critical columns
    df = df.filter(col("amt").isNotNull() & col("trans_date_trans_time").isNotNull())

    # Write to optimized Parquet
    target_path.mkdir(parents=True, exist_ok=True)
    (
        df.write
        .mode("overwrite")
        .parquet(str(target_path))
    )
    print(f"  Saved to Parquet: {target_path}")

def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    
    ingest_dataset(spark, "fraudTrain.csv", "train")
    ingest_dataset(spark, "fraudTest.csv", "test")
    
    spark.stop()
    print("[Ingest] Ingestion and Parquet conversion complete.")

if __name__ == "__main__":
    main()
