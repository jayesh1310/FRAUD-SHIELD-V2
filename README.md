# Real-Time Financial Fraud Detection System

An end-to-end **Big Data machine learning pipeline** for detecting fraudulent credit card transactions in real-time. The system generates a live stream of synthetic transactions via **Apache Kafka**, processes them with **Apache Spark Structured Streaming**, scores each transaction through a **Random Forest ML model**, and surfaces results through a live monitoring **dashboard**.

> Fully containerized with **Docker Compose** — one command to launch the entire distributed pipeline.

---

## Architecture Overview

```
┌─────────────────────────┐
│  Synthetic Data Producer │  (Python + Faker)
│    100 transactions/sec  │
└───────────┬─────────────┘
            │ JSON over Kafka
            ▼
┌─────────────────────────┐
│     Apache Kafka Broker  │  (Topic: transactions)
│   + Zookeeper Coordinator│
└───────────┬─────────────┘
            │ Structured Streaming
            ▼
┌─────────────────────────────────────────┐
│       Spark ML Consumer (PySpark)        │
│                                          │
│  ┌─────────────┐   ┌──────────────────┐ │
│  │   Feature    │──▶│  Random Forest   │ │
│  │ Engineering  │   │  ML Pipeline     │ │
│  └─────────────┘   └───────┬──────────┘ │
│                             │ prediction │
└─────────────────────────────┼────────────┘
                              ▼
                   ┌─────────────────────┐
                   │  Parquet Storage     │
                   │  (Flagged Fraud)     │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  FraudShield        │
                   │  Dashboard (Flask)  │
                   │  Port 5050          │
                   └─────────────────────┘
```

---

## Key Features

- **Live Streaming Pipeline** — Kafka-backed stream processing with 3-second micro-batch intervals
- **Custom Synthetic Data** — Faker-generated transactions with configurable fraud rate (default 2%)
- **PySpark ML Pipeline** — 6-stage pipeline (StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler → RandomForest)
- **Class Imbalance Handling** — Weighted loss function (no SMOTE), runs natively in Spark's distributed memory
- **Real-Time Dashboard** — Live KPI cards, Chart.js visualizations, Docker container controls, and caught-fraud tables
- **Fault Tolerant** — Checkpoint-based recovery ensures the consumer resumes exactly where it left off
- **Fully Containerized** — 6 Docker services orchestrated via Docker Compose

---

## Model Performance

Evaluated on an 80/20 stratified train/test split (~44,000 test transactions):

| Metric             | Score      |
|--------------------|------------|
| **AUC-ROC**        | 0.9821     |
| **AUC-PR**         | 0.7634     |
| **F1 Score**       | 0.9412     |
| **Fraud Catch Rate** (Recall) | 87.43%     |
| **False Alarm Rate** | 0.0023%    |

**Confusion Matrix:**

|                    | Predicted: Legit | Predicted: Fraud |
|--------------------|:----------------:|:----------------:|
| **Actual: Legit**  | 43,150 (TN)      | 1 (FP)           |
| **Actual: Fraud**  | 114 (FN)          | 793 (TP)         |

---

## Project Structure

```
fraud-detection/
├── fraud-detection-pipeline-main/       # Core Docker pipeline
│   ├── docker-compose.yml               # Orchestrates all 6 services
│   ├── producer/
│   │   ├── Dockerfile
│   │   ├── requirements.txt             # kafka-python, faker
│   │   └── transaction_producer.py      # Synthetic transaction generator → Kafka
│   ├── consumer/
│   │   ├── Dockerfile
│   │   ├── requirements.txt             # pyspark, kafka-python, py4j
│   │   └── spark_consumer_ml.py         # Spark Streaming consumer + ML inference
│   ├── features/
│   │   └── feature_engineering.py       # Feature engineering (used inside container)
│   ├── storage/                         # Mounted volume for Parquet output
│   ├── check_env.py                     # Environment validation script
│   ├── run_demo.ps1                     # PowerShell demo launcher
│   ├── set_java_env.ps1                 # Java environment setup
│   ├── SCHEMA.md                        # Transaction schema documentation
│   └── README.md                        # Pipeline-specific README
│
├── dashboard/
│   ├── app.py                           # Flask backend (port 5050) — REST API & log polling
│   └── index.html                       # FraudShield UI — Chart.js, live feed, KPI cards
│
├── feature_engineering.py               # PySpark feature engineering module (root copy)
├── train_model_pipeline.py              # PySpark script to train the Random Forest pipeline
├── fraud_rf_model/                      # Exported PySpark PipelineModel (used by consumer)
│
├── training_data/                       # Raw training data (Parquet snapshots)
├── training_data_clean.parquet          # Cleaned training dataset
├── train_features.parquet               # Engineered training features
├── test_features.parquet                # Engineered test features
│
├── temp_flagged/                        # Local copy of fraud-flagged Parquet output
├── temp_flagged_dash/                   # Dashboard's copy of fraud-flagged output
│
├── launch_jupyter.bat                   # Windows batch script to launch Jupyter Notebook
└── .gitignore
```

---

## Tech Stack

| Layer              | Technology                                      |
|--------------------|-------------------------------------------------|
| **Data Generation** | Python, Faker, `kafka-python`                   |
| **Message Broker**  | Apache Kafka 7.5.0 + Zookeeper                  |
| **Stream Processing** | Apache Spark 3.5.0 Structured Streaming (PySpark) |
| **Machine Learning** | PySpark MLlib — RandomForestClassifier           |
| **Data Storage**    | Apache Parquet (Snappy compressed)               |
| **Dashboard Backend** | Flask, Flask-CORS, Pandas                      |
| **Dashboard Frontend** | HTML5, CSS3, Chart.js                         |
| **Infrastructure**  | Docker, Docker Compose                           |

---

## Machine Learning Pipeline

### Feature Engineering

The model relies on contextual metadata engineered from the raw transaction JSON:

| Feature               | Description                                                      |
|-----------------------|------------------------------------------------------------------|
| `amount`              | Transaction dollar amount (StandardScaled)                       |
| `hour_of_day`         | Hour extracted from timestamp (0–23)                             |
| `is_night_transaction` | Boolean flag — `True` if hour ≥ 22 or hour ≤ 5                  |
| `amount_bucket`       | Ordinal category: Low (<$100), Medium, High, Very High (≥$2000) |
| `geo_risk_score`      | Heuristic based on latitude: 2.0 (>60°), 1.0 (>40°), 0.5 (≤40°) |
| `merchant_category`   | One-hot encoded merchant type (6 categories)                     |
| `lat` / `lon`         | Raw geospatial coordinates                                       |

### PySpark MLlib Pipeline Stages

```
StringIndexer (merchant_category → cat_idx)
       ↓
OneHotEncoder (cat_idx → cat_vec)
       ↓
StringIndexer (amount_bucket → bucket_idx)
       ↓
VectorAssembler → raw_features
       ↓
StandardScaler → features
       ↓
RandomForestClassifier (100 trees, maxDepth=10, weighted)
```

### Class Imbalance Strategy

Instead of SMOTE (not natively supported in PySpark), the pipeline uses **class weighting**:

```
weight_fraud = total_rows / (2.0 × fraud_rows)
weight_legit = total_rows / (2.0 × legit_rows)
```

This penalizes fraud misclassification ~25–50× more heavily, and runs entirely in Spark's distributed memory.

---

## Machine Learning Models & Comparison

The project provides two production-grade machine learning models trained on the streaming transaction schema:

### 1. PySpark MLlib Random Forest (`train_model_pipeline.py`)
- **Type**: Ensemble Bagging (Decision Trees)
- **Configuration**: 100 trees, max depth = 10, weighted class loss
- **Artifact**: `fraud_rf_model/` (PySpark PipelineModel)
- **Use Case**: Native distributed execution inside Spark Structured Streaming consumer

### 2. Extreme Gradient Boosting - XGBoost (`train_xgboost_model.py`)
- **Type**: Ensemble Boosting (Gradient Boosted Decision Trees)
- **Configuration**: 100 estimators, max depth = 6, learning rate = 0.1, subsample = 0.8, `scale_pos_weight` = 15.32
- **Artifacts**: `fraud_xgb_model.json` & `fraud_xgb_pipeline.joblib`
- **Use Case**: State-of-the-art tabular classification, ultra-fast microsecond inference latency

### Real-World Benchmark Performance (Trained on Kaggle Sparkov Dataset)

Evaluated on **50,000 real unseen test transactions** from \raudTest.csv\ (199 actual fraudulent transactions):

| Metric / Dimension | Random Forest (Spark MLlib) | XGBoost Classifier (Real-World) |
|---|---|---|
| **Ensemble Paradigm** | Bagging (Parallel Independent Trees) | Boosting (Sequential Error Correction) |
| **Accuracy** | 98.70% | **98.95%** |
| **AUC-ROC** | 0.9821 | **0.9902** |
| **AUC-PR** | 0.7634 | **0.6501** |
| **F1 Score** | 0.3540 | **0.3902** |
| **Fraud Catch Rate (Recall)** | 78.89% | **84.42%** (168 of 199 caught) |
| **False Alarm Rate** | 1.12% | **0.9919%** (494 false positives / 49,801 legit) |
| **Confusion Matrix** | \TP=157, FP=558, FN=42, TN=49,243\ | \TP=168, FP=494, FN=31, TN=49,307\ |
| **Class Balancing** | Sample weight column (\classWeight\) | Loss scaling parameter (\scale_pos_weight\) |
| **Training Latency** | ~30 - 45s (Spark JVM overhead) | **~1.8s (C++ optimized engine)** |
| **Model Size / Format** | Distributed directory (\raud_rf_model/\) | Compact JSON / Joblib (\raud_xgb_pipeline.joblib\) |

---

## One-Click Launchers (Windows)

The repository provides ready-to-run automation scripts:

- **un_real_dataset_stream.bat\**: Checks Docker, launches all 6 containers, starts the live stream from \raudTest.csv\, resets session counters to 0, launches the Dashboard on \http://localhost:5050\, and starts the XGBoost live consumer.
- **un_xgboost_with_dashboard.bat\**: Launches the pipeline, opens the Dashboard, and starts the XGBoost live scoring consumer.
- **\stop_project.bat\**: Gracefully terminates all 6 Docker containers, stops Python background workers, and releases ports 5050 and 9092.

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (running)
- Python 3.10+ (for dashboard & training scripts)
- Java 8+ (for local PySpark execution)

### 1. Launch the Streaming Pipeline

```bash
cd fraud-detection-pipeline-main
docker compose up -d
```

This starts all 6 containers: Zookeeper, Kafka, Spark Master, Spark Worker, Producer, and Consumer.

### 2. Watch the Live Stream

```bash
# View the transaction producer generating data
docker logs fraud-producer -f

# View the Spark ML consumer scoring transactions
docker logs fraud-consumer -f
```

### 3. Access the Spark UI

Open [http://localhost:8080](http://localhost:8080) for the Spark Master dashboard, or [http://localhost:4040](http://localhost:4040) for the Spark application UI showing streaming micro-batches.

### 4. Run Live Streaming Fraud Detection with XGBoost

To score the live Kafka transaction stream directly using the **XGBoost model**:

```bash
# In the project root folder:
python stream_consumer_xgb.py
```

This connects directly to the Kafka stream (`localhost:9092`), computes real-time feature engineering, feeds each transaction into `fraud_xgb_pipeline.joblib`, prints live color-coded fraud alerts, and saves flagged fraud transactions to `./temp_flagged_xgb/` in Parquet format.

### 5. Launch the FraudShield Dashboard

```bash
cd dashboard
pip install flask flask-cors pandas pyarrow
python app.py
```

Open [http://localhost:5050](http://localhost:5050) to access the live monitoring dashboard with:
- Real-time KPI cards (transactions seen, fraud detected, fraud rate)
- Chart.js line charts and donut charts
- Docker container health status
- ML model metrics & confusion matrix
- Live transaction feed with fraud highlighting
- ML-flagged fraud table (from Parquet storage)
- One-click pipeline start/stop controls

### 5. View Caught Fraud (CLI)

```bash
docker cp fraud-consumer:/data/output/fraud_flagged ./temp_flagged
python -c "
import pandas as pd, glob
files = glob.glob('./temp_flagged/*.parquet')
df = pd.concat([pd.read_parquet(f) for f in files])
print(df[df['flagged_fraud']==True][['transaction_id','amount','merchant_category','is_fraud','flagged_fraud']].tail(10))
"
```

### 6. Stop the Pipeline

```bash
cd fraud-detection-pipeline-main
docker compose down
```

---

## Re-Training the Models

### Retrain the Random Forest Pipeline:
```bash
# Ensure you have PySpark and Java configured
python train_model_pipeline.py
```
This reads `training_data_clean.parquet`, runs feature engineering, trains the 6-stage PySpark pipeline, evaluates metrics, and exports the model to `fraud_rf_model/`.

### Train / Retrain the XGBoost Model:
```bash
# Ensure you have xgboost and scikit-learn installed
python train_xgboost_model.py
```
This trains an XGBoost classifier with stratified 80/20 train/test evaluation, outputs comprehensive metrics (AUC-ROC, F1, Confusion Matrix, Gain feature importances), and saves `fraud_xgb_model.json` and `fraud_xgb_pipeline.joblib`.

---

## Docker Services

| Container        | Image                           | Exposed Ports | Role                                        |
|------------------|---------------------------------|:------------:|----------------------------------------------|
| `zookeeper`      | `confluentinc/cp-zookeeper:7.5.0` | 2181         | Kafka cluster coordination                   |
| `kafka`          | `confluentinc/cp-kafka:7.5.0`     | 9092         | Message broker (internal: 29092)             |
| `spark-master`   | `apache/spark:3.5.0`              | 7077, 8080   | Spark cluster master                         |
| `spark-worker`   | `apache/spark:3.5.0`              | 8081         | Spark worker (2 cores, 2GB RAM)              |
| `fraud-producer` | Custom Python build                | —            | Generates 100 tx/sec into Kafka              |
| `fraud-consumer` | Custom PySpark build               | 4040         | ML inference + Parquet output every 3 seconds |

---

## Dashboard REST API

| Method | Endpoint               | Description                                           |
|--------|------------------------|-------------------------------------------------------|
| GET    | `/`                    | Serves the FraudShield HTML dashboard                 |
| GET    | `/api/health`          | Health check                                          |
| GET    | `/api/containers`      | Running Docker container statuses                     |
| GET    | `/api/transactions`    | Live stats: total seen, fraud count, fraud rate, feed |
| GET    | `/api/producer/logs`   | Last 60 parsed producer log entries                   |
| GET    | `/api/consumer/logs`   | Last 60 Spark consumer log entries                    |
| GET    | `/api/model`           | Model metrics (AUC-ROC, F1, Catch Rate) & features   |
| GET    | `/api/fraud-caught`    | Top 25 ML-flagged fraud transactions from Parquet     |
| GET    | `/api/pipeline/status` | Boolean pipeline running status                       |
| POST   | `/api/pipeline/start`  | Triggers `docker compose up -d`                       |
| POST   | `/api/pipeline/stop`   | Triggers `docker compose down`                        |

---

## Documentation


| [SCHEMA.md](fraud-detection-pipeline-main/SCHEMA.md) | Transaction schema documentation |

---

## License

This project is for educational and demonstration purposes.
