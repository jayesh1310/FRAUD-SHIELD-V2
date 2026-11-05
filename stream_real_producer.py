# -*- coding: utf-8 -*-
"""
stream_real_producer.py
-----------------------
Streams real-world credit card transactions from Kaggle/Sparkov CSV datasets
(e.g., fraudTest.csv or fraudTrain.csv) line-by-line into Apache Kafka topic 'transactions'.

Features:
- Configurable streaming rate (10 to 50+ tx/sec)
- Converts CSV rows into the exact pipeline TRANSACTION_SCHEMA JSON
- Standardizes merchant categories to match model features
- Continuous loop option for uninterrupted testing
- Real-time terminal stats with fraud alerts
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from kafka import KafkaProducer
from kafka.errors import KafkaError

# ---------------------------------------------------------------------------
# Category normalization mapping
# ---------------------------------------------------------------------------
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream real CSV transactions into Kafka topic."
    )
    parser.add_argument(
        "--file", "-f",
        default="fraudTest.csv",
        help="Path to CSV dataset (default: fraudTest.csv)"
    )
    parser.add_argument(
        "--rate", "-r",
        type=float,
        default=25.0,
        help="Transactions per second to publish (default: 25.0, range: 1-100)"
    )
    parser.add_argument(
        "--bootstrap-servers", "-b",
        default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092"),
        help="Kafka bootstrap server address (default: 127.0.0.1:9092)"
    )
    parser.add_argument(
        "--topic", "-t",
        default=os.getenv("KAFKA_TOPIC", "transactions"),
        help="Kafka topic name (default: transactions)"
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Loop back to beginning of CSV upon reaching end (default: True)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after sending N transactions (0 = unlimited, default: 0)"
    )
    parser.add_argument(
        "--use-current-time",
        action="store_true",
        help="Replace historical CSV timestamps with current UTC wall-clock time"
    )
    return parser.parse_args()


def row_to_transaction_json(row: dict, use_current_time: bool = False) -> dict:
    """Map a raw CSV row to the exact schema expected by the streaming consumers."""
    raw_cat = str(row.get("category", "retail")).strip().lower()
    norm_cat = CATEGORY_MAP.get(raw_cat, raw_cat)

    # Timestamp handling
    if use_current_time:
        ts = datetime.now(timezone.utc).isoformat()
    else:
        raw_ts = str(row.get("trans_date_trans_time", "")).strip()
        if raw_ts:
            ts = raw_ts.replace(" ", "T")
        else:
            ts = datetime.now(timezone.utc).isoformat()

    # Amount
    try:
        amount = round(float(row.get("amt", 0.0)), 2)
    except (ValueError, TypeError):
        amount = 0.0

    # Coordinates
    try:
        lat = float(row.get("lat", 0.0))
    except (ValueError, TypeError):
        lat = 0.0
    try:
        lon = float(row.get("long", 0.0))
    except (ValueError, TypeError):
        lon = 0.0

    # Fraud label
    raw_fraud = str(row.get("is_fraud", "0")).strip().lower()
    is_fraud = raw_fraud in ("1", "true", "yes")

    # Transaction ID (trans_num)
    tx_id = str(row.get("trans_num") or row.get("transaction_id") or f"tx_{int(time.time()*1000)}")

    # Card / User
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


def create_producer(bootstrap_servers: str) -> KafkaProducer:
    """Create Kafka producer with retry logic."""
    print(f"[Producer] Connecting to Kafka broker at {bootstrap_servers}...")
    for attempt in range(1, 11):
        try:
            p = KafkaProducer(
                bootstrap_servers=[bootstrap_servers],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks=1,
                retries=3,
                linger_ms=5,
            )
            print("  [OK] Connected to Kafka successfully.")
            return p
        except (KafkaError, Exception) as e:
            print(f"  [Wait] Kafka not available yet ({e}). Retrying in 2s (attempt {attempt}/10)...")
            time.sleep(2)

    print(f"[ERROR] Could not connect to Kafka at {bootstrap_servers}.")
    print("Please make sure Kafka container is running: `docker compose up -d kafka zookeeper`")
    sys.exit(1)


def main():
    args = parse_args()

    # Verify CSV file existence
    csv_path = os.path.abspath(args.file)
    if not os.path.exists(csv_path):
        alt = os.path.join(os.getcwd(), args.file)
        if os.path.exists(alt):
            csv_path = alt
        else:
            print(f"[ERROR] CSV dataset file not found at: {csv_path}")
            sys.exit(1)

    producer = create_producer(args.bootstrap_servers)

    print("=" * 75)
    print("  Real Dataset Kafka Stream Producer")
    print("=" * 75)
    print(f"  Source file : {csv_path}")
    print(f"  Kafka topic : {args.topic}")
    print(f"  Rate target : {args.rate:.1f} transactions / second")
    print(f"  Loop mode   : {'Enabled' if args.loop else 'Disabled'}")
    print(f"  Timestamp   : {'Current UTC' if args.use_current_time else 'Original CSV timestamp'}")
    print("=" * 75)
    print("Streaming started. Press Ctrl+C to stop.\n", flush=True)

    delay = 1.0 / max(0.1, args.rate)
    total_sent = 0
    fraud_sent = 0
    start_time = time.time()
    last_stat_time = start_time
    pass_number = 1

    # Optional local log file for dashboard integration
    log_file_path = os.path.join(os.path.dirname(csv_path), "producer_stream.log")
    log_fp = open(log_file_path, "a", encoding="utf-8")

    try:
        while True:
            with open(csv_path, mode="r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_start = time.time()
                    txn = row_to_transaction_json(row, use_current_time=args.use_current_time)

                    # Send to Kafka
                    producer.send(args.topic, value=txn)
                    total_sent += 1

                    label = "FRAUD" if txn["is_fraud"] else "legit"
                    if txn["is_fraud"]:
                        fraud_sent += 1

                    # Log matching dashboard regex:
                    # [Producer] Sent [FRAUD] transaction <tid> — $<amt>
                    log_line = f"[Producer] Sent [{label}] transaction {txn['transaction_id']} \u2014 ${txn['amount']:.2f}"
                    print(log_line, flush=True)
                    log_fp.write(f"{datetime.now(timezone.utc).isoformat()} {log_line}\n")
                    if total_sent % 20 == 0:
                        log_fp.flush()

                    if args.limit > 0 and total_sent >= args.limit:
                        print(f"\n[Done] Limit of {args.limit} transactions reached.")
                        return

                    # Throttle rate
                    elapsed = time.time() - t_start
                    sleep_time = delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    # Periodic stats summary every 10 seconds
                    now = time.time()
                    if now - last_stat_time >= 10.0:
                        dur = now - start_time
                        actual_rate = total_sent / dur if dur > 0 else 0
                        fraud_pct = (fraud_sent / total_sent * 100) if total_sent > 0 else 0
                        print(f"\n---> [STATS] Sent: {total_sent:,} | Frauds: {fraud_sent:,} ({fraud_pct:.2f}%) | Speed: {actual_rate:.1f} tx/s\n", flush=True)
                        last_stat_time = now

            if not args.loop:
                print("\n[Done] Reached end of CSV file.")
                break

            pass_number += 1
            print(f"\n[Loop] Reached EOF. Starting pass #{pass_number}...\n", flush=True)

    except KeyboardInterrupt:
        print("\n[Producer] Interrupted by user. Shutting down cleanly...")
    finally:
        log_fp.close()
        producer.flush()
        producer.close()
        dur = time.time() - start_time
        actual_rate = total_sent / dur if dur > 0 else 0
        fraud_pct = (fraud_sent / total_sent * 100) if total_sent > 0 else 0
        print("=" * 75)
        print("  Stream Producer Summary")
        print("=" * 75)
        print(f"  Total sent   : {total_sent:,}")
        print(f"  Fraud sent   : {fraud_sent:,} ({fraud_pct:.2f}%)")
        print(f"  Duration     : {dur:.1f} seconds")
        print(f"  Average rate : {actual_rate:.1f} tx/sec")
        print("=" * 75)


if __name__ == "__main__":
    main()
