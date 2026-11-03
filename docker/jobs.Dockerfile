# PySpark + Delta Lake + MLflow + training/rollout tooling.
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/jobs.txt /tmp/jobs.txt
RUN pip install --no-cache-dir -r /tmp/jobs.txt

WORKDIR /workspace
ENV PYTHONPATH=/workspace \
    PYTHONUNBUFFERED=1
