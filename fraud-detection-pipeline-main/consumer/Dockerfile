FROM apache/spark:3.5.0

# Build context is PROJECT ROOT (see docker-compose.yml)
USER root

# Install Python 3, pip, build tools
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip wget build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (pyspark, kafka-python, xgboost, sklearn, etc.)
COPY consumer/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Pre-download Kafka connector JARs for Spark 3.5.0 / Scala 2.12
# ---------------------------------------------------------------------------
ENV MAVEN_REPO=https://repo1.maven.org/maven2
ENV SPARK_JARS_DIR=/opt/spark/jars

RUN rm -f "${SPARK_JARS_DIR}"/commons-pool2-*.jar \
          "${SPARK_JARS_DIR}"/kafka-clients-*.jar \
          "${SPARK_JARS_DIR}"/kafka-*.jar \
          "${SPARK_JARS_DIR}"/spark-sql-kafka-*.jar \
          "${SPARK_JARS_DIR}"/spark-token-provider-kafka-*.jar \
    && wget -q -P "${SPARK_JARS_DIR}" \
        "${MAVEN_REPO}/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar" \
    && wget -q -P "${SPARK_JARS_DIR}" \
        "${MAVEN_REPO}/org/apache/kafka/kafka-clients/3.4.1/kafka-clients-3.4.1.jar" \
    && wget -q -P "${SPARK_JARS_DIR}" \
        "${MAVEN_REPO}/org/apache/spark/spark-sql-kafka-0-10_2.12/3.5.0/spark-sql-kafka-0-10_2.12-3.5.0.jar" \
    && wget -q -P "${SPARK_JARS_DIR}" \
        "${MAVEN_REPO}/org/apache/spark/spark-token-provider-kafka-0-10_2.12/3.5.0/spark-token-provider-kafka-0-10_2.12-3.5.0.jar" \
    && echo "--- Kafka JARs installed ---" \
    && ls -lh "${SPARK_JARS_DIR}"/*.jar | grep -E "kafka|commons-pool2"

WORKDIR /app

# Copy shared features package
COPY features/ /app/features/

# Copy consumer application
COPY consumer/ /app/

# Note: models/ are mounted via volume at runtime (not baked into the image)
# This allows updating models without rebuilding the Docker image.

ENV PYTHONPATH=/app

USER spark

CMD ["spark-submit", \
     "--master", "local[2]", \
     "spark_consumer_ml.py"]
