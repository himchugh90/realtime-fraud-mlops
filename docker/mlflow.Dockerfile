FROM python:3.11-slim

# RUN pip install --no-cache-dir mlflow==2.17.2 psycopg2-binary==2.9.10
RUN pip install --no-cache-dir mlflow==2.17.2 psycopg2-binary==2.9.10 "sqlalchemy>=2.0,<2.1"
