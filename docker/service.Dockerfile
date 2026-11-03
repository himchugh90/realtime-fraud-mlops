# Slim image shared by the model servers and the gateway.
# Code is mounted at /app by docker-compose, so only dependencies live here.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/service.txt /tmp/service.txt
RUN pip install --no-cache-dir -r /tmp/service.txt

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1
