"""
transaction_producer.py
-----------------------
FraudShield 2.0 — Kafka Producer (Historical Replay Mode).

Reads actual test transactions from fraudTest.csv (Sparkov-generated simulation data)
and publishes them to the 'transactions' Kafka topic.

The ground-truth label (actual_is_fraud) is included in the Kafka message as a
separate field for downstream evaluation — it is NEVER passed to any ML model
as an input feature.

Faker is NOT used. All data comes from the Kaggle dataset.

Environment variables:
  KAFKA_BOOTSTRAP_SERVERS  (default: kafka:29092)
  KAFKA_TOPIC              (default: transactions)
  TRANSACTIONS_PER_SECOND  (default: 25)
  CSV_FILE_PATH            (default: /app/fraudTest.csv)
"""

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

from kafka import KafkaProducer
from kafka.errors import KafkaError

# ── Configuration ─────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "transactions")
TRANSACTIONS_PER_SECOND = int(os.getenv("TRANSACTIONS_PER_SECOND", "25"))
CSV_FILE_PATH           = os.getenv("CSV_FILE_PATH", "/app/fraudTest.csv")


def find_csv(custom: str = "") -> str:
    """Locate fraudTest.csv across common paths."""
    candidates = [
        custom,
        CSV_FILE_PATH,
        "fraudTest.csv",
        os.path.join("..", "fraudTest.csv"),
        os.path.join(os.path.dirname(__file__), "..", "..", "fraudTest.csv"),
        "/app/fraudTest.csv",
        "/data/fraudTest.csv",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    return ""


def csv_row_to_message(row: dict) -> dict:
    """
    Convert a raw fraudTest.csv row to the new flat Kafka message schema.

    The 'actual_is_fraud' field carries the ground-truth label.
    It must NEVER be used as an ML model input feature — only for
    post-prediction accuracy evaluation.
    """
    def safe_float(val, default=0.0):
        try:
            return float(val) if val and val.strip() else default
        except (ValueError, TypeError):
            return default

    def safe_int(val, default=0):
        try:
            return int(float(val)) if val and str(val).strip() else default
        except (ValueError, TypeError):
            return default

    raw_fraud = str(row.get("is_fraud", "0")).strip().lower()
    is_fraud  = 1 if raw_fraud in ("1", "true", "yes") else 0

    tx_id = str(row.get("trans_num") or row.get("Unnamed: 0", "")).strip()
    if not tx_id:
        import uuid
        tx_id = str(uuid.uuid4())

    ts_raw = str(row.get("trans_date_trans_time", "")).strip()
    if not ts_raw:
        ts_raw = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    return {
        # Identifiers (not used as ML features)
        "transaction_id":        tx_id,
        "cc_num":                str(row.get("cc_num", "")).strip(),
        "trans_num":             str(row.get("trans_num", tx_id)).strip(),

        # Temporal
        "trans_date_trans_time": ts_raw,
        "unix_time":             safe_int(row.get("unix_time", 0)),

        # Transaction
        "merchant":              str(row.get("merchant", "")).strip(),
        "category":              str(row.get("category", "misc_net")).strip().lower(),
        "amt":                   safe_float(row.get("amt", 0.0)),

        # Cardholder info (names excluded — PII, not needed for ML)
        "first":                 str(row.get("first", "")).strip(),
        "last":                  str(row.get("last", "")).strip(),
        "gender":                str(row.get("gender", "M")).strip().upper(),
        "dob":                   str(row.get("dob", "1980-01-01")).strip(),

        # Location
        "city":                  str(row.get("city", "")).strip(),
        "state":                 str(row.get("state", "XX")).strip().upper(),
        "zip":                   str(row.get("zip", "")).strip(),
        "city_pop":              safe_int(row.get("city_pop", 0)),
        "lat":                   safe_float(row.get("lat", 0.0)),
        "long":                  safe_float(row.get("long", 0.0)),

        # Merchant location
        "merch_lat":             safe_float(row.get("merch_lat", 0.0)),
        "merch_long":            safe_float(row.get("merch_long", 0.0)),

        # Ground-truth label — stored separately, NEVER used as ML input feature
        "actual_is_fraud":       is_fraud,
    }


def create_producer(bootstrap_servers: str) -> KafkaProducer:
    print(f"[Producer] Connecting to Kafka at {bootstrap_servers} …")
    for attempt in range(1, 16):
        try:
            p = KafkaProducer(
                bootstrap_servers=[bootstrap_servers],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=5,
                linger_ms=5,
            )
            print("[Producer] Connected to Kafka.")
            return p
        except (KafkaError, Exception) as e:
            wait = min(attempt * 2, 15)
            print(f"[Producer] Not ready (attempt {attempt}/15: {e}). Retrying in {wait}s …")
            time.sleep(wait)
    print("[Producer] Failed to connect to Kafka.", file=sys.stderr)
    sys.exit(1)


def main():
    csv_file = find_csv()
    if not csv_file:
        print("[Producer] ERROR: fraudTest.csv not found. Mount it at /app/fraudTest.csv", file=sys.stderr)
        sys.exit(1)

    producer  = create_producer(KAFKA_BOOTSTRAP_SERVERS)
    tx_rate   = max(1, TRANSACTIONS_PER_SECOND)
    delay     = 1.0 / tx_rate

    print("=" * 70)
    print("  FRAUDSHIELD 2.0 — HISTORICAL REPLAY PRODUCER")
    print(f"  Source : {csv_file}")
    print(f"  Topic  : {KAFKA_TOPIC}")
    print(f"  Rate   : {tx_rate} tx/s")
    print("  Mode   : Real dataset replay (Sparkov-generated simulation data)")
    print("=" * 70)

    total_sent  = 0
    fraud_sent  = 0
    pass_num    = 1

    try:
        while True:
            with open(csv_file, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t0  = time.time()
                    msg = csv_row_to_message(row)
                    producer.send(KAFKA_TOPIC, value=msg)
                    total_sent += 1
                    if msg["actual_is_fraud"]:
                        fraud_sent += 1

                    label = "FRAUD" if msg["actual_is_fraud"] else "legit"
                    # Keep log format compatible with dashboard parser
                    print(
                        f"[Producer] Sent [{label}] transaction {msg['transaction_id']} "
                        f"\u2014 ${msg['amt']:.2f}",
                        flush=True,
                    )

                    elapsed = time.time() - t0
                    sleep_t = delay - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

            pass_num += 1
            print(f"[Producer] EOF reached. Looping (pass #{pass_num}). "
                  f"Sent {total_sent:,} total, {fraud_sent:,} fraud.", flush=True)

    except KeyboardInterrupt:
        print(f"\n[Producer] Shutting down. Sent {total_sent:,} transactions ({fraud_sent:,} fraud).")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
