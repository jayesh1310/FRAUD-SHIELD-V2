# -*- coding: utf-8 -*-
"""
stream_consumer_xgb.py
----------------------
Real-Time Fraud Detection Streaming Consumer using XGBoost.

Consumes live financial transaction events from Apache Kafka (localhost:9092),
performs feature engineering on the fly (hour_of_day, night flag, geo-risk, amount tiers),
and predicts fraud probability using the trained XGBoost Pipeline (fraud_xgb_pipeline.joblib).

Flagged fraudulent transactions are saved to ./temp_flagged_xgb/ in Parquet format
for audit and dashboard visualization. Also writes live scoring logs to consumer_stream.log.
"""

import json
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from datetime import datetime, timezone
import pandas as pd
import joblib
from kafka import KafkaConsumer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "transactions")
MODEL_PATH              = os.getenv("MODEL_PATH", "fraud_xgb_pipeline.joblib")
OUTPUT_DIR              = os.getenv("OUTPUT_DIR", "temp_flagged_xgb")
BATCH_FLUSH_INTERVAL    = 5  # seconds between saving flagged fraud to Parquet

def enrich_transaction(raw_txn: dict) -> dict:
    """Apply the exact feature engineering used during model training."""
    # Parse timestamp
    ts_str = raw_txn.get("timestamp", datetime.now(timezone.utc).isoformat())
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)
    
    hour_of_day = dt.hour
    is_night_int = 1 if (hour_of_day >= 22 or hour_of_day <= 5) else 0

    # Amount tier
    amount = float(raw_txn.get("amount", 0.0))
    if amount < 100:
        amount_bucket = "Low"
    elif amount < 500:
        amount_bucket = "Medium"
    elif amount < 2000:
        amount_bucket = "High"
    else:
        amount_bucket = "Very High"

    # Geo risk heuristic
    loc = raw_txn.get("location", {})
    lat = float(loc.get("lat", 0.0)) if isinstance(loc, dict) else 0.0
    lon = float(loc.get("lon", 0.0)) if isinstance(loc, dict) else 0.0

    abs_lat = abs(lat)
    if abs_lat > 60:
        geo_risk_score = 2.0
    elif abs_lat > 40:
        geo_risk_score = 1.0
    else:
        geo_risk_score = 0.5

    return {
        "amount": amount,
        "hour_of_day": int(hour_of_day),
        "is_night_int": int(is_night_int),
        "geo_risk_score": float(geo_risk_score),
        "lat": lat,
        "lon": lon,
        "merchant_category": str(raw_txn.get("merchant_category", "retail")),
        "amount_bucket": amount_bucket
    }

def main():
    print("=" * 70)
    print("  XGBoost Real-Time Streaming Fraud Consumer")
    print("=" * 70)

    # 1. Load model
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: {MODEL_PATH}")
        print("Please run `python train_xgboost_model.py` first to train and generate the model.")
        sys.exit(1)

    print(f"[Step 1] Loading XGBoost Pipeline from: {MODEL_PATH} ...")
    model = joblib.load(MODEL_PATH)
    print("  [OK] Model loaded successfully.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Open local consumer log file for dashboard integration
    log_file_path = os.path.abspath("consumer_stream.log")
    c_log_fp = open(log_file_path, "a", encoding="utf-8")
    c_log_fp.write(f"{datetime.now(timezone.utc).isoformat()} [XGBoost] Consumer initialized. Ready for transactions.\n")
    c_log_fp.flush()

    # 2. Connect to Kafka
    print(f"\n[Step 2] Connecting to Kafka broker at '{KAFKA_BOOTSTRAP_SERVERS}' ...")
    print(f"         Subscribed to topic: '{KAFKA_TOPIC}'")
    
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=1000  # unblock periodically to flush data
        )
    except Exception as e:
        print(f"[ERROR] Failed to connect to Kafka: {e}")
        print("Make sure Docker pipeline is running: `docker compose up -d` in fraud-detection-pipeline-main")
        c_log_fp.close()
        sys.exit(1)

    print("\n[Step 3] Stream Active! Monitoring live transactions in real-time...", flush=True)
    print("         (Press Ctrl+C to terminate)\n", flush=True)
    print(f"{'STATUS':<10} {'TXN ID':<38} {'AMOUNT':<12} {'MERCHANT':<15} {'SCORE':<8}", flush=True)
    print("-" * 88, flush=True)

    flagged_buffer = []
    last_flush_time = time.time()
    total_processed = 0
    total_flagged = 0

    try:
        while True:
            # Poll Kafka messages
            records = consumer.poll(timeout_ms=1000)
            
            for tp, messages in records.items():
                for msg in messages:
                    raw_txn = msg.value
                    total_processed += 1

                    # Feature engineering
                    features = enrich_transaction(raw_txn)
                    features_df = pd.DataFrame([features])

                    # XGBoost prediction
                    pred = int(model.predict(features_df)[0])
                    prob = float(model.predict_proba(features_df)[0, 1])

                    is_fraud_flagged = (pred == 1)
                    tid = raw_txn.get("transaction_id", "N/A")
                    amount = raw_txn.get("amount", 0.0)
                    category = raw_txn.get("merchant_category", "N/A")

                    if is_fraud_flagged:
                        total_flagged += 1
                        status = "[FRAUD]"
                        print(f"{status:<10} {tid:<38} ${amount:<11.2f} {category:<15} {prob:<8.4f}", flush=True)

                        # Append to output buffer
                        record_to_save = dict(raw_txn)
                        record_to_save.update(features)
                        record_to_save["fraud_probability"] = prob
                        record_to_save["flagged_fraud"] = True
                        record_to_save["scored_by"] = "XGBoost"
                        flagged_buffer.append(record_to_save)
                    else:
                        status = "[LEGIT]"
                        print(f"{status:<10} {tid:<38} ${amount:<11.2f} {category:<15} {prob:<8.4f}", flush=True)

                    # Write to consumer_stream.log for real-time dashboard display
                    log_entry = f"[XGBoost] {status} {tid} \u2014 ${amount:.2f} [{category}] Prob: {prob:.4f}"
                    c_log_fp.write(f"{datetime.now(timezone.utc).isoformat()} {log_entry}\n")
                    if total_processed % 5 == 0:
                        c_log_fp.flush()

            # Periodic Parquet flush
            if (time.time() - last_flush_time >= BATCH_FLUSH_INTERVAL) and flagged_buffer:
                flush_df = pd.DataFrame(flagged_buffer)
                part_filename = f"part-xgb-{int(time.time() * 1000)}.parquet"
                flush_path = os.path.join(OUTPUT_DIR, part_filename)
                flush_df.to_parquet(flush_path, index=False)
                flagged_buffer.clear()
                last_flush_time = time.time()
                c_log_fp.flush()

    except KeyboardInterrupt:
        print("\n[Consumer] Interrupted by user. Shutting down...")
    finally:
        # Flush remaining
        if flagged_buffer:
            flush_df = pd.DataFrame(flagged_buffer)
            part_filename = f"part-xgb-{int(time.time() * 1000)}.parquet"
            flush_path = os.path.join(OUTPUT_DIR, part_filename)
            flush_df.to_parquet(flush_path, index=False)
            flagged_buffer.clear()

        c_log_fp.flush()
        c_log_fp.close()
        consumer.close()
        print("=" * 70)
        print("  Consumer Session Summary")
        print(f"  Total Processed : {total_processed:,}")
        print(f"  Flagged Fraud   : {total_flagged:,}")
        print("=" * 70)

if __name__ == "__main__":
    main()
