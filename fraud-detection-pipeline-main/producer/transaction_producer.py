"""
transaction_producer.py
-----------------------
Streams credit-card transactions to an Apache Kafka topic in real time.

Supports TWO modes:
1. Real Dataset Mode:
   If `fraudTest.csv` (or path specified by CSV_FILE_PATH or --file) exists,
   reads transactions line-by-line from the CSV, maps to standard schema,
   and publishes to Kafka at a configurable rate (e.g. 10 to 50 tx/s).
   Loops continuously when reaching EOF.

2. Synthetic Mode (Fallback):
   Generates synthetic credit-card transaction events using Faker.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone

try:
    from faker import Faker
    fake = Faker()
except ImportError:
    fake = None

from kafka import KafkaProducer
from kafka.errors import KafkaError

# ---------------------------------------------------------------------------
# Configuration (overridden via CLI or environment variables)
# ---------------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
KAFKA_TOPIC             = os.getenv("KAFKA_TOPIC", "transactions")
TRANSACTIONS_PER_SECOND = int(os.getenv("TRANSACTIONS_PER_SECOND", "20"))
FRAUD_PROBABILITY       = float(os.getenv("FRAUD_PROBABILITY", "0.02"))
CSV_FILE_PATH           = os.getenv("CSV_FILE_PATH", "")

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


def find_csv_file(custom_path: str = "") -> str:
    """Locate real CSV dataset across common project directories."""
    candidates = [
        custom_path,
        CSV_FILE_PATH,
        "fraudTest.csv",
        os.path.join("..", "fraudTest.csv"),
        os.path.join("..", "..", "fraudTest.csv"),
        os.path.join(os.path.dirname(__file__), "..", "..", "fraudTest.csv"),
        "/app/fraudTest.csv",
        "/data/fraudTest.csv",
    ]
    for c in candidates:
        if c and os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
    return ""


def csv_row_to_transaction(row: dict) -> dict:
    """Transform a CSV row from fraudTest.csv into the exact TRANSACTION_SCHEMA."""
    raw_cat = str(row.get("category", "retail")).strip().lower()
    norm_cat = CATEGORY_MAP.get(raw_cat, raw_cat)

    raw_ts = str(row.get("trans_date_trans_time", "")).strip()
    if raw_ts:
        ts = raw_ts.replace(" ", "T")
    else:
        ts = datetime.now(timezone.utc).isoformat()

    try:
        amount = round(float(row.get("amt", 0.0)), 2)
    except (ValueError, TypeError):
        amount = 0.0

    try:
        lat = float(row.get("lat", 0.0))
    except (ValueError, TypeError):
        lat = 0.0
    try:
        lon = float(row.get("long", 0.0))
    except (ValueError, TypeError):
        lon = 0.0

    raw_fraud = str(row.get("is_fraud", "0")).strip().lower()
    is_fraud = raw_fraud in ("1", "true", "yes")

    tx_id = str(row.get("trans_num") or row.get("transaction_id") or str(uuid.uuid4()))
    cc_num = str(row.get("cc_num", "")).strip()
    first = str(row.get("first", "")).strip()
    last = str(row.get("last", "")).strip()
    user_id = f"user_{cc_num}" if cc_num else f"{first}_{last}"

    return {
        "transaction_id": tx_id,
        "timestamp": ts,
        "user_id": user_id,
        "card_number": cc_num,
        "merchant": str(row.get("merchant", "")).strip(),
        "merchant_category": norm_cat,
        "amount": amount,
        "currency": "USD",
        "location": {
            "city": str(row.get("city", "")).strip(),
            "country": "US",
            "lat": lat,
            "lon": lon,
        },
        "is_fraud": is_fraud,
    }


def generate_synthetic_transaction(is_fraud: bool = False) -> dict:
    """Return a single synthetic transaction record (fallback mode)."""
    if fake is None:
        raise RuntimeError("Faker is not installed. Please install faker or provide a CSV dataset.")
    amount = round(random.uniform(5_000, 25_000), 2) if is_fraud else round(random.uniform(1, 2_000), 2)
    return {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": fake.uuid4(),
        "card_number": fake.credit_card_number(),
        "merchant": fake.company(),
        "merchant_category": random.choice(
            ["retail", "food", "travel", "entertainment", "electronics", "healthcare"]
        ),
        "amount": amount,
        "currency": "USD",
        "location": {
            "city": fake.city(),
            "country": fake.country_code(),
            "lat": float(fake.latitude()),
            "lon": float(fake.longitude()),
        },
        "is_fraud": is_fraud,
    }


def create_producer(bootstrap_servers: str) -> KafkaProducer:
    """Create and return a configured KafkaProducer with connection retry."""
    print(f"[Producer] Connecting to Kafka at {bootstrap_servers}...")
    for attempt in range(1, 11):
        try:
            p = KafkaProducer(
                bootstrap_servers=[bootstrap_servers],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=5,
                linger_ms=5,
            )
            print("[Producer] Connected to Kafka successfully.")
            return p
        except (KafkaError, Exception) as e:
            print(f"[Producer] Kafka not ready yet (attempt {attempt}/10: {e}). Retrying in 2s...")
            time.sleep(2)

    print(f"[Producer] Failed to connect to Kafka at {bootstrap_servers}.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Credit Card Transaction Kafka Producer")
    parser.add_argument("--file", "-f", default="", help="Path to real CSV dataset")
    parser.add_argument("--rate", "-r", type=int, default=TRANSACTIONS_PER_SECOND, help="Transactions per second")
    parser.add_argument("--servers", "-b", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--topic", "-t", default=KAFKA_TOPIC, help="Kafka topic name")
    args, _ = parser.parse_known_args()

    producer = create_producer(args.servers)
    tx_rate = max(1, args.rate)
    delay = 1.0 / tx_rate

    # Check for real CSV dataset
    csv_file = find_csv_file(args.file)

    if csv_file:
        print("=" * 70)
        print("  STREAMING MODE: REAL DATASET")
        print(f"  Source file : {csv_file}")
        print(f"  Target topic: {args.topic}")
        print(f"  Rate        : {tx_rate} tx/s")
        print("=" * 70)

        total_sent = 0
        fraud_sent = 0
        pass_num = 1

        try:
            while True:
                with open(csv_file, mode="r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        t0 = time.time()
                        txn = csv_row_to_transaction(row)
                        producer.send(args.topic, value=txn)
                        total_sent += 1
                        if txn["is_fraud"]:
                            fraud_sent += 1

                        label = "FRAUD" if txn["is_fraud"] else "legit"
                        print(f"[Producer] Sent [{label}] transaction {txn['transaction_id']} \u2014 ${txn['amount']:.2f}", flush=True)

                        elapsed = time.time() - t0
                        remaining = delay - elapsed
                        if remaining > 0:
                            time.sleep(remaining)

                pass_num += 1
                print(f"[Producer] Reached EOF. Looping to beginning (pass #{pass_num})...", flush=True)
        except KeyboardInterrupt:
            print("\n[Producer] Shutting down...")
        finally:
            producer.flush()
            producer.close()
    else:
        print("=" * 70)
        print("  STREAMING MODE: SYNTHETIC (FAKER)")
        print(f"  Target topic: {args.topic}")
        print(f"  Rate        : {tx_rate} tx/s")
        print(f"  Fraud prob  : {FRAUD_PROBABILITY * 100:.1f}%")
        print("=" * 70)

        try:
            while True:
                is_fraud = random.random() < FRAUD_PROBABILITY
                txn = generate_synthetic_transaction(is_fraud=is_fraud)
                producer.send(args.topic, value=txn)
                label = "FRAUD" if is_fraud else "legit"
                print(f"[Producer] Sent [{label}] transaction {txn['transaction_id']} \u2014 ${txn['amount']:.2f}", flush=True)
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\n[Producer] Shutting down...")
        finally:
            producer.flush()
            producer.close()


if __name__ == "__main__":
    main()
