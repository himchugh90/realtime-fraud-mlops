"""Raw PaySim CSV -> bronze Delta table.

Usage: python -m pipelines.ingest --source data/raw/paysim
"""
import argparse
import os

from pyspark.sql import functions as F

from pipelines.spark import BRONZE_PATH, get_spark, latest_delta_version

RAW_SCHEMA = (
    "step INT, type STRING, amount DOUBLE, nameOrig STRING, oldbalanceOrg DOUBLE, "
    "newbalanceOrig DOUBLE, nameDest STRING, oldbalanceDest DOUBLE, newbalanceDest DOUBLE, "
    "isFraud INT, isFlaggedFraud INT"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/paysim", help="Folder containing PaySim CSV file(s)")
    args = parser.parse_args()

    spark = get_spark("ingest")
    raw = spark.read.csv(os.path.join(os.path.abspath(args.source), "*.csv"), header=True, schema=RAW_SCHEMA)

    # Deterministic id so predictions can later be joined back to (delayed) labels.
    bronze = raw.withColumn(
        "txn_id",
        F.xxhash64("step", "type", "amount", "nameOrig", "nameDest", "oldbalanceOrg").cast("string"),
    )
    bronze.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(BRONZE_PATH)

    stats = spark.read.format("delta").load(BRONZE_PATH).agg(F.count("*").alias("rows"), F.sum("isFraud").alias("frauds"))
    row = stats.collect()[0]
    print(f"bronze: {row.rows:,} rows, {row.frauds:,} fraud, delta version {latest_delta_version(spark, BRONZE_PATH)}")
    spark.stop()


if __name__ == "__main__":
    main()
