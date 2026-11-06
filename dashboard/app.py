"""
app.py
------
FraudShield 2.0 — Flask monitoring backend.
Serves the dashboard and REST API for 3-model fraud detection pipeline.
Port: 5050

New endpoints:
  GET  /api/models/comparison         — model_comparison.json
  GET  /api/models/linear-regression  — LR metrics
  GET  /api/models/random-forest      — RF metrics
  GET  /api/models/xgboost           — XGBoost metrics
  GET  /api/model-agreement           — prediction agreement analysis
  POST /api/pipeline/pause
  POST /api/pipeline/resume
  POST /api/pipeline/replay
"""

import glob
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR    = os.path.join(BASE_DIR, "..", "fraud-detection-pipeline-main")
COMPOSE_DIR     = PIPELINE_DIR
MODELS_DIR      = os.path.join(PIPELINE_DIR, "models")
RESULTS_FILE    = os.path.join(PIPELINE_DIR, "results", "model_comparison.json")
PREDICTIONS_DIR = os.path.join(PIPELINE_DIR, "storage", "predictions", "combined_predictions")
# Legacy paths (backward compat)
TEMP_FLAGGED    = os.path.join(BASE_DIR, "..", "temp_flagged")

app = Flask(__name__, static_folder=BASE_DIR)
CORS(app)

# ── In-memory state ───────────────────────────────────────────────────────────
MAX_LOG  = 300
producer_logs = deque(maxlen=MAX_LOG)
consumer_logs = deque(maxlen=MAX_LOG)
live_feed     = deque(maxlen=200)   # last 200 transactions

total_seen       = 0
total_fraud      = 0
total_legit      = 0
total_ml_flagged = 0
counter_lock     = threading.Lock()

# 3-model agreement counters
agree_fraud   = 0
agree_legit   = 0
one_fraud     = 0
two_fraud     = 0
disagree      = 0
agreement_lock = threading.Lock()


def clean_session():
    """Archive parquet files and reset all in-memory counters."""
    global total_seen, total_fraud, total_legit, total_ml_flagged
    global agree_fraud, agree_legit, one_fraud, two_fraud, disagree

    archive_dir = os.path.join(BASE_DIR, "..", "archive_flagged")
    os.makedirs(archive_dir, exist_ok=True)

    for d in [TEMP_FLAGGED, os.path.join(BASE_DIR, "..", "temp_flagged_xgb")]:
        if os.path.isdir(d):
            for f in glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True):
                try:
                    shutil.move(f, os.path.join(archive_dir, os.path.basename(f)))
                except Exception:
                    pass

    with counter_lock:
        total_seen = total_fraud = total_legit = total_ml_flagged = 0
        live_feed.clear()
        producer_logs.clear()
        consumer_logs.clear()
    with agreement_lock:
        agree_fraud = agree_legit = one_fraud = two_fraud = disagree = 0


clean_session()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _run(cmd, cwd=None, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           encoding="utf-8", timeout=timeout, cwd=cwd)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return str(e)


def _run_compose(subcmd):
    return _run(f"docker compose {subcmd}", cwd=COMPOSE_DIR, timeout=120)


def _load_comparison() -> dict:
    """Load results/model_comparison.json if available."""
    if os.path.isfile(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ── Log parsing ───────────────────────────────────────────────────────────────
TXN_RE = re.compile(
    r"\[Producer\] Sent \[(?P<label>FRAUD|legit)\] transaction "
    r"(?P<tid>[a-zA-Z0-9\-]+) \u2014 \$(?P<amount>[\d.]+)"
)


def parse_producer_line(line: str) -> dict | None:
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


# ── Background poller ─────────────────────────────────────────────────────────
def poll_logs():
    global total_seen, total_fraud, total_legit, total_ml_flagged

    session_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    last_producer_ts = session_start
    last_consumer_ts = session_start

    while True:
        try:
            # Docker producer logs
            since = f"--since {last_producer_ts}"
            raw_p = _run(f"docker logs fraud-producer {since} --timestamps 2>&1")
            for line in raw_p.splitlines():
                line = line.strip()
                if not line or "No such container" in line or "Error response" in line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0].endswith("Z"):
                    ts_str, msg = parts[0], parts[1]
                    last_producer_ts = ts_str
                else:
                    msg = line
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

            # Docker consumer logs
            since_c = f"--since {last_consumer_ts}"
            raw_c = _run(f"docker logs fraud-consumer {since_c} --timestamps 2>&1")
            for line in raw_c.splitlines():
                line = line.strip()
                if not line or "No such container" in line or "Error response" in line:
                    continue
                parts = line.split(None, 1)
                if len(parts) == 2 and parts[0].endswith("Z"):
                    ts_str, msg = parts[0], parts[1]
                    last_consumer_ts = ts_str
                else:
                    msg = line
                consumer_logs.append({"raw": msg, "time": datetime.now().isoformat()})

            # Read recent Parquet predictions for agreement stats
            _update_agreement_stats()

        except Exception as e:
            print(f"[poller] error: {e}")

        time.sleep(3)


def _update_agreement_stats():
    """Update agreement counters from latest Parquet predictions."""
    global agree_fraud, agree_legit, one_fraud, two_fraud, disagree
    try:
        import pandas as pd
        files = sorted(
            glob.glob(os.path.join(PREDICTIONS_DIR, "**", "*.parquet"), recursive=True),
            key=os.path.getmtime, reverse=True,
        )[:5]
        if not files:
            return
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f, columns=[
                    "lr_prediction", "rf_prediction", "xgb_prediction", "actual_is_fraud"
                ]))
            except Exception:
                pass
        if not dfs:
            return
        df = pd.concat(dfs, ignore_index=True)
        preds_sum = df["lr_prediction"] + df["rf_prediction"] + df["xgb_prediction"]
        with agreement_lock:
            agree_fraud = int((preds_sum == 3).sum())
            agree_legit = int((preds_sum == 0).sum())
            one_fraud   = int((preds_sum == 1).sum())
            two_fraud   = int((preds_sum == 2).sum())
            disagree    = int((preds_sum.isin([1, 2])).sum())
    except Exception:
        pass


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat(), "version": "2.0"})


@app.route("/api/session/reset", methods=["POST", "GET"])
def session_reset():
    clean_session()
    return jsonify({"status": "ok", "message": "Session reset"})


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

    fraud_rate = round(tf / ts * 100, 2) if ts > 0 else 0.0
    feed = list(live_feed)[-50:]
    return jsonify({
        "feed":               feed,
        "total_seen":         ts,
        "fraud_count":        tf,
        "legit_count":        tl,
        "ml_flagged_count":   tmf,
        "ground_truth_fraud": tf,
        "fraud_rate":         fraud_rate,
    })


@app.route("/api/producer/logs")
def producer_log_api():
    logs = list(producer_logs)[-80:]
    if not logs:
        return jsonify([{"raw": "[Producer] Waiting for stream …", "time": datetime.now().isoformat()}])
    return jsonify(logs)


@app.route("/api/consumer/logs")
def consumer_log_api():
    logs = list(consumer_logs)[-80:]
    if not logs:
        return jsonify([{"raw": "[Consumer] Waiting for messages …", "time": datetime.now().isoformat()}])
    return jsonify(logs)


# ── Model comparison (reads from results/model_comparison.json) ───────────────
@app.route("/api/models/comparison")
def models_comparison():
    data = _load_comparison()
    if not data:
        return jsonify({"error": "model_comparison.json not found. Run training/evaluate_models.py first."}), 404
    return jsonify(data)


@app.route("/api/models/linear-regression")
def model_lr():
    data = _load_comparison()
    lr = data.get("linear_regression", {})
    if not lr:
        return jsonify({"error": "LR metrics not available"}), 404
    return jsonify(lr)


@app.route("/api/models/random-forest")
def model_rf():
    data = _load_comparison()
    rf = data.get("random_forest", {})
    if not rf:
        return jsonify({"error": "RF metrics not available"}), 404
    return jsonify(rf)


@app.route("/api/models/xgboost")
def model_xgb():
    data = _load_comparison()
    xgb = data.get("xgboost", {})
    if not xgb:
        return jsonify({"error": "XGBoost metrics not available"}), 404
    return jsonify(xgb)


@app.route("/api/model")
def model_metrics_legacy():
    """Legacy endpoint — returns XGBoost metrics for backward compatibility."""
    data = _load_comparison()
    xgb = data.get("xgboost", {})
    # Remap to legacy keys
    return jsonify({
        "auc_roc":          xgb.get("roc_auc", 0),
        "auc_pr":           xgb.get("pr_auc", 0),
        "f1_score":         xgb.get("f1_score", 0),
        "fraud_catch_rate": round(xgb.get("fraud_detection_rate", 0) * 100, 2),
        "false_alarm_rate": xgb.get("false_positive_rate", 0),
        "comparison":       data,
    })


# ── Live prediction data from Parquet ─────────────────────────────────────────
@app.route("/api/fraud-caught")
def fraud_caught():
    try:
        import pandas as pd
        files = sorted(
            glob.glob(os.path.join(PREDICTIONS_DIR, "**", "*.parquet"), recursive=True),
            key=os.path.getmtime, reverse=True,
        )[:10]
        if not files:
            return jsonify({"records": [], "total": 0})

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                pass
        if not dfs:
            return jsonify({"records": [], "total": 0})

        df = pd.concat(dfs, ignore_index=True)
        # Show rows where at least one model predicted fraud
        fraud_rows = df[(df.get("lr_prediction", 0) == 1) |
                        (df.get("rf_prediction", 0) == 1) |
                        (df.get("xgb_prediction", 0) == 1)]

        records = []
        for _, row in fraud_rows.tail(50).iterrows():
            records.append({
                "transaction_id":    str(row.get("transaction_id", "")),
                "amount":            float(row.get("amt", 0)),
                "merchant":          str(row.get("merchant", "")),
                "merchant_category": str(row.get("category", "")),
                "actual_is_fraud":   int(row.get("actual_is_fraud", 0)),
                "lr_prediction":     int(row.get("lr_prediction", 0)),
                "rf_prediction":     int(row.get("rf_prediction", 0)),
                "xgb_prediction":    int(row.get("xgb_prediction", 0)),
                "lr_score":          round(float(row.get("lr_score", 0)), 4),
                "rf_score":          round(float(row.get("rf_score", 0)), 4),
                "xgb_score":         round(float(row.get("xgb_score", 0)), 4),
            })
        return jsonify({"records": records, "total": len(fraud_rows)})
    except Exception as e:
        return jsonify({"records": [], "total": 0, "error": str(e)})


@app.route("/api/model-agreement")
def model_agreement():
    with agreement_lock:
        return jsonify({
            "all_agree_fraud":  agree_fraud,
            "all_agree_legit":  agree_legit,
            "one_model_fraud":  one_fraud,
            "two_model_fraud":  two_fraud,
            "disagreement":     disagree,
        })


# ── Live prediction feed from Parquet ─────────────────────────────────────────
@app.route("/api/predictions/live")
def predictions_live():
    try:
        import pandas as pd
        files = sorted(
            glob.glob(os.path.join(PREDICTIONS_DIR, "**", "*.parquet"), recursive=True),
            key=os.path.getmtime, reverse=True,
        )[:5]
        if not files:
            return jsonify({"records": [], "total_processed": 0})

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                pass
        if not dfs:
            return jsonify({"records": [], "total_processed": 0})

        df = pd.concat(dfs, ignore_index=True)
        records = []
        for _, row in df.tail(100).iterrows():
            lr_p  = int(row.get("lr_prediction", 0))
            rf_p  = int(row.get("rf_prediction", 0))
            xgb_p = int(row.get("xgb_prediction", 0))
            preds_sum = lr_p + rf_p + xgb_p
            if preds_sum == 0:
                agreement = "all_legit"
            elif preds_sum == 3:
                agreement = "all_fraud"
            elif preds_sum == 1:
                agreement = "one_fraud"
            else:
                agreement = "two_fraud"

            records.append({
                "transaction_id":   str(row.get("transaction_id", "")),
                "timestamp":        str(row.get("trans_date_trans_time", "")),
                "amount":           float(row.get("amt", 0)),
                "merchant":         str(row.get("merchant", "")),
                "category":         str(row.get("category", "")),
                "actual_is_fraud":  int(row.get("actual_is_fraud", 0)),
                "lr_prediction":    lr_p,
                "rf_prediction":    rf_p,
                "xgb_prediction":   xgb_p,
                "lr_score":         round(float(row.get("lr_score", 0)), 4),
                "rf_score":         round(float(row.get("rf_score", 0)), 4),
                "xgb_score":        round(float(row.get("xgb_score", 0)), 4),
                "lr_correct":       int(row.get("lr_correct", 0)),
                "rf_correct":       int(row.get("rf_correct", 0)),
                "xgb_correct":      int(row.get("xgb_correct", 0)),
                "agreement":        agreement,
            })
        return jsonify({"records": records, "total_processed": len(df)})
    except Exception as e:
        return jsonify({"records": [], "total_processed": 0, "error": str(e)})


# ── Pipeline control ──────────────────────────────────────────────────────────
@app.route("/api/pipeline/status")
def pipeline_status():
    ps_out = _run('docker ps --format "{{.Names}}"')
    names  = [n.strip() for n in ps_out.splitlines() if n.strip()]
    core   = ["fraud-producer", "fraud-consumer", "kafka", "zookeeper"]
    return jsonify({"running": any(c in names for c in core), "containers": names})


@app.route("/api/pipeline/start", methods=["POST"])
def pipeline_start():
    clean_session()
    out = _run_compose("up -d")
    return jsonify({"ok": True, "output": out[-1000:]})


@app.route("/api/pipeline/stop", methods=["POST"])
def pipeline_stop():
    out = _run_compose("down")
    clean_session()
    return jsonify({"ok": True, "output": out[-1000:]})


@app.route("/api/pipeline/pause", methods=["POST"])
def pipeline_pause():
    out = _run("docker pause fraud-producer", timeout=15)
    return jsonify({"ok": True, "output": out})


@app.route("/api/pipeline/resume", methods=["POST"])
def pipeline_resume():
    out = _run("docker unpause fraud-producer", timeout=15)
    return jsonify({"ok": True, "output": out})


@app.route("/api/pipeline/replay", methods=["POST"])
def pipeline_replay():
    """Restart producer to replay from beginning."""
    clean_session()
    _run("docker restart fraud-producer", timeout=30)
    return jsonify({"ok": True, "message": "Producer restarted — replaying fraudTest.csv from beginning"})


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=poll_logs, daemon=True)
    t.start()
    print("[FraudShield 2.0] Dashboard backend starting on http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
