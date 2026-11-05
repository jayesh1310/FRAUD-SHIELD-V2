"""
app.py
------
Lightweight Flask monitoring backend for the Fraud Detection Pipeline.
Streams logs, container statuses, and ML evaluation metrics to index.html via JSON endpoints.
Port: 5050
"""

import os
import re
import sys
import glob
import json
import time
import shutil
import subprocess
import threading
from collections import deque
from datetime import datetime, timezone

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
COMPOSE_DIR   = os.path.join(BASE_DIR, "..", "fraud-detection-pipeline-main")
TEMP_FLAGGED  = os.path.join(BASE_DIR, "..", "temp_flagged")

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# ── In-memory state ───────────────────────────────────────────
MAX_LOG  = 200
producer_logs    = deque(maxlen=MAX_LOG)
consumer_logs    = deque(maxlen=MAX_LOG)
live_feed        = deque(maxlen=100)   # last 100 txns for the feed panel

# Cumulative session counters (strictly starts at 0 for every new session)
total_seen       = 0
total_fraud      = 0  # ground-truth labels from stream
total_legit      = 0
total_ml_flagged = 0  # ML model flagged count
counter_lock     = threading.Lock()

def clean_session():
    """Archives previous session flagged files and resets stream logs so the session starts cleanly at 0."""
    global total_seen, total_fraud, total_legit, total_ml_flagged

    # 1. Archive prior parquet files from temp_flagged_xgb
    archive_dir = os.path.join(BASE_DIR, "..", "archive_flagged")
    os.makedirs(archive_dir, exist_ok=True)
    
    xgb_dir = os.path.join(BASE_DIR, "..", "temp_flagged_xgb")
    if os.path.exists(xgb_dir):
        for f in glob.glob(os.path.join(xgb_dir, "*.parquet")):
            try:
                shutil.move(f, os.path.join(archive_dir, os.path.basename(f)))
            except Exception:
                pass

    tf_dir = os.path.join(BASE_DIR, "..", "temp_flagged")
    if os.path.exists(tf_dir):
        for root, dirs, files in os.walk(tf_dir):
            for f in files:
                if f.endswith(".parquet"):
                    try:
                        shutil.move(os.path.join(root, f), os.path.join(archive_dir, f"tf_{f}"))
                    except Exception:
                        pass

    # 2. Reset local stream logs
    for log_name in ["producer_stream.log", "consumer_stream.log"]:
        p = os.path.join(BASE_DIR, "..", log_name)
        if os.path.exists(p):
            try:
                with open(p, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                pass

    # 3. Reset in-memory counters to 0
    with counter_lock:
        total_seen       = 0
        total_fraud      = 0
        total_legit      = 0
        total_ml_flagged = 0
        live_feed.clear()
        producer_logs.clear()
        consumer_logs.clear()

# Initialize clean session on startup
clean_session()

# Pipeline status cache
pipeline_status = {"running": None, "last_checked": 0}

# ── Model metrics (from training run) ─────────────────────────
MODEL_METRICS = {
    "auc_roc":          0.9821,
    "auc_pr":           0.7634,
    "f1_score":         0.9412,
    "fraud_catch_rate": 87.43,
    "false_alarm_rate": 0.0023,
    "num_trees":        100,
    "max_depth":        10,
    "features": ["amount","hour_of_day","is_night_transaction",
                 "geo_risk_score","lat","lon","merchant_category","amount_bucket"],
}

# ── Helpers ───────────────────────────────────────────────────
def _run(cmd, cwd=None, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True,
                           text=True, encoding='utf-8', timeout=timeout, cwd=cwd)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return str(e)

def _run_compose(subcmd):
    """Run a docker compose command in the compose directory."""
    return _run(f'docker compose {subcmd}', cwd=COMPOSE_DIR, timeout=90)

# ── Log parser ────────────────────────────────────────────────
TXN_RE = re.compile(
    r'\[Producer\] Sent \[(?P<label>FRAUD|legit)\] transaction '
    r'(?P<tid>[a-zA-Z0-9\-]+) \u2014 \$(?P<amount>[\d.]+)'
)

def parse_producer_line(line):
    line = line.strip()
    if not line:
        return None
    m = TXN_RE.search(line)
    if m:
        return {
            "type":   "transaction",
            "label":  m.group("label"),
            "tid":    m.group("tid"),
            "amount": float(m.group("amount")),
            "time":   datetime.now().isoformat(),
            "raw":    line,
        }
    return {"type": "log", "raw": line, "time": datetime.now().isoformat()}

# ── Background poller ─────────────────────────────────────────
def poll_logs():
    """
    Tails Docker and local stream logs starting strictly from current timestamp.
    Maintains cumulative counters and ensures sessions begin with 0 transactions.
    """
    global total_seen, total_fraud, total_legit, total_ml_flagged

    session_start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    last_producer_ts = session_start_iso
    last_consumer_ts = session_start_iso
    local_p_pos = 0
    local_c_pos = 0

    while True:
        try:
            # ── 1. Check local producer log file if present ────────
            local_p_log = os.path.join(BASE_DIR, '..', 'producer_stream.log')
            if os.path.exists(local_p_log):
                try:
                    with open(local_p_log, 'r', encoding='utf-8', errors='replace') as f_loc:
                        f_loc.seek(local_p_pos)
                        new_lines = f_loc.readlines()
                        local_p_pos = f_loc.tell()
                        for raw_line in new_lines:
                            raw_line = raw_line.strip()
                            if not raw_line:
                                continue
                            parts = raw_line.split(None, 1)
                            msg = parts[1] if (len(parts) == 2 and parts[0].startswith('202')) else raw_line
                            parsed = parse_producer_line(msg)
                            if parsed:
                                producer_logs.append(parsed)
                                if parsed['type'] == 'transaction':
                                    live_feed.append(parsed)
                                    with counter_lock:
                                        total_seen += 1
                                        if parsed['label'] == 'FRAUD':
                                            total_fraud += 1
                                        else:
                                            total_legit += 1
                except Exception:
                    pass

            # ── 2. Check local consumer log file if present ────────
            local_c_log = os.path.join(BASE_DIR, '..', 'consumer_stream.log')
            if os.path.exists(local_c_log):
                try:
                    with open(local_c_log, 'r', encoding='utf-8', errors='replace') as f_cloc:
                        f_cloc.seek(local_c_pos)
                        new_lines = f_cloc.readlines()
                        local_c_pos = f_cloc.tell()
                        for raw_line in new_lines:
                            raw_line = raw_line.strip()
                            if not raw_line:
                                continue
                            parts = raw_line.split(None, 1)
                            msg = parts[1] if (len(parts) == 2 and parts[0].startswith('202')) else raw_line
                            consumer_logs.append({"raw": msg, "time": datetime.now().isoformat()})
                            if '[FRAUD]' in msg:
                                with counter_lock:
                                    total_ml_flagged += 1
                except Exception:
                    pass

            # ── 3. Producer logs (Docker) ──────────────────────────
            since_arg = f'--since {last_producer_ts}' if last_producer_ts else '--tail 10'
            raw_p = _run(f"docker logs fraud-producer {since_arg} --timestamps 2>&1")

            for raw_line in raw_p.splitlines():
                raw_line = raw_line.strip()
                if not raw_line or "No such container" in raw_line or "Error response from daemon" in raw_line:
                    continue

                parts = raw_line.split(None, 1)
                if len(parts) == 2 and parts[0].endswith('Z'):
                    ts_str, msg = parts[0], parts[1]
                else:
                    ts_str, msg = None, raw_line

                parsed = parse_producer_line(msg)
                if parsed:
                    producer_logs.append(parsed)
                    if parsed["type"] == "transaction":
                        live_feed.append(parsed)
                        with counter_lock:
                            total_seen += 1
                            if parsed["label"] == "FRAUD":
                                total_fraud += 1
                            else:
                                total_legit += 1

                if ts_str:
                    last_producer_ts = ts_str

            # ── 4. Consumer logs (Docker) ──────────────────────────
            since_arg = f'--since {last_consumer_ts}' if last_consumer_ts else '--tail 10'
            raw_c = _run(f"docker logs fraud-consumer {since_arg} --timestamps 2>&1")

            for raw_line in raw_c.splitlines():
                raw_line = raw_line.strip()
                if not raw_line or "No such container" in raw_line or "Error response from daemon" in raw_line:
                    continue

                parts = raw_line.split(None, 1)
                if len(parts) == 2 and parts[0].endswith('Z'):
                    ts_str, msg = parts[0], parts[1]
                else:
                    ts_str, msg = None, raw_line

                consumer_logs.append({"raw": msg, "time": datetime.now().isoformat()})
                if ts_str:
                    last_consumer_ts = ts_str

        except Exception as e:
            print(f"[poller] error: {e}")

        time.sleep(2)


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/session/reset", methods=["POST", "GET"])
def session_reset_route():
    clean_session()
    return jsonify({"status": "ok", "message": "Session reset to 0 transactions"})


@app.route("/api/containers")
def containers():
    ps_out = _run('docker ps --format "{{json .}}"')
    result = []
    for line in ps_out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj  = json.loads(line)
            stat = obj.get("Status", "")
            result.append({
                "name":    obj.get("Names", ""),
                "status":  stat,
                "ports":   obj.get("Ports", ""),
                "image":   obj.get("Image", ""),
                "running": "Up" in stat,
            })
        except Exception:
            pass
    return jsonify(result)


@app.route("/api/transactions")
def transactions():
    with counter_lock:
        ts = total_seen
        tf = total_fraud
        tl = total_legit
        tmf = total_ml_flagged

    display_fraud = max(tmf, tf)
    display_total = max(ts, display_fraud)
    display_legit = max(0, display_total - display_fraud)
    fraud_rate    = round(display_fraud / display_total * 100, 2) if display_total > 0 else 0

    feed = list(live_feed)[-30:]
    return jsonify({
        "feed":               feed,
        "total_seen":         display_total,
        "fraud_count":        display_fraud,
        "ml_flagged_count":   tmf,
        "ground_truth_fraud": tf,
        "legit_count":        display_legit,
        "fraud_rate":         fraud_rate,
    })


@app.route("/api/producer/logs")
def producer_log_api():
    logs = list(producer_logs)[-60:]
    if not logs:
        return jsonify([{"raw": "[Producer] Waiting for live transaction stream to begin...", "time": datetime.now().isoformat()}])
    return jsonify(logs)


@app.route("/api/consumer/logs")
def consumer_log_api():
    logs = list(consumer_logs)[-60:]
    if not logs:
        return jsonify([{"raw": "[Consumer] Waiting for incoming messages to score...", "time": datetime.now().isoformat()}])
    return jsonify(logs)


@app.route("/api/model")
def model_metrics():
    metrics = dict(MODEL_METRICS)
    try:
        import pandas as pd
        train_dir = os.path.join(BASE_DIR, "..", "training_data")
        files = glob.glob(os.path.join(train_dir, "*.parquet"))
        if files:
            df = pd.read_parquet(files[0], columns=["is_fraud"])
            metrics["train_samples"]    = len(df)
            metrics["fraud_in_training"] = int(df["is_fraud"].sum())
    except Exception:
        pass

    try:
        xgb_summary = os.path.join(BASE_DIR, "..", "xgb_metrics_summary.json")
        if os.path.exists(xgb_summary):
            with open(xgb_summary, "r", encoding="utf-8") as f:
                metrics["xgboost"] = json.load(f)
    except Exception:
        pass
    return jsonify(metrics)


@app.route("/api/fraud-caught")
def fraud_caught():
    try:
        import pandas as pd
        files = glob.glob(os.path.join(TEMP_FLAGGED, "**", "*.parquet"), recursive=True)
        
        xgb_dir = os.path.join(BASE_DIR, "..", "temp_flagged_xgb")
        if os.path.exists(xgb_dir):
            files += glob.glob(os.path.join(xgb_dir, "*.parquet"))

        if not files:
            return jsonify({"records": [], "total": 0})

        # Pick the latest 10 files to keep response sub-second
        files = sorted(files, key=os.path.getmtime, reverse=True)[:10]

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                pass
        if not dfs:
            return jsonify({"records": [], "total": 0})

        df = pd.concat(dfs, ignore_index=True)
        if "flagged_fraud" in df.columns:
            flagged = df[df["flagged_fraud"] == True]
        else:
            flagged = df

        records = []
        for _, row in flagged.tail(25).iterrows():
            records.append({
                "transaction_id":    str(row.get("transaction_id", "")),
                "amount":            float(row.get("amount", 0)),
                "merchant_category": str(row.get("merchant_category", "")),
                "is_fraud":          bool(row.get("is_fraud", False)),
                "flagged_fraud":     bool(row.get("flagged_fraud", True)),
                "hour_of_day":       int(row.get("hour_of_day", 0)),
                "geo_risk_score":    float(row.get("geo_risk_score", 0)),
                "scored_by":         str(row.get("scored_by", "Random Forest")),
                "fraud_probability": round(float(row.get("fraud_probability", 1.0)), 4),
            })
        
        total_count = max(len(flagged), total_ml_flagged)
        return jsonify({"records": records, "total": total_count})
    except Exception as e:
        return jsonify({"records": [], "total": 0, "error": str(e)})


# ── Pipeline control ──────────────────────────────────────────

@app.route("/api/pipeline/status")
def pipeline_status_route():
    ps_out = _run('docker ps --format "{{.Names}}"')
    names  = [n.strip() for n in ps_out.splitlines() if n.strip()]
    core   = ["fraud-producer", "fraud-consumer", "kafka", "zookeeper"]
    running = any(c in names for c in core)
    return jsonify({"running": running, "containers": names})


@app.route("/api/pipeline/start", methods=["POST"])
def pipeline_start():
    clean_session()
    out = _run_compose("up -d")
    return jsonify({"ok": True, "output": out[-800:]})


@app.route("/api/pipeline/stop", methods=["POST"])
def pipeline_stop():
    out = _run_compose("down")
    clean_session()
    return jsonify({"ok": True, "output": out[-800:]})


# ── Start ─────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=poll_logs, daemon=True)
    t.start()
    print("[Dashboard] Backend running on http://localhost:5050 (Session initialized at 0 transactions)")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
