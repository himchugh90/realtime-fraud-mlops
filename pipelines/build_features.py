"""Bronze Delta -> (1) dest activity parquet for Feast, (2) training features Delta.

Usage: python -m pipelines.build_features
"""
from common.features import DEST_ENTITY_KEY, spark_feature_columns
from pipelines.dest_features import build_dest_activity, point_in_time_join
from pipelines.spark import BRONZE_PATH, DEST_ACTIVITY_PATH, FEATURES_PATH, get_spark, latest_delta_version


def main():
    spark = get_spark("build_features")
    bronze = spark.read.format("delta").load(BRONZE_PATH)

    # Feast's offline source: what the online store will be loaded from.
    dest_activity = build_dest_activity(bronze)
    dest_activity.write.mode("overwrite").parquet(DEST_ACTIVITY_PATH)
    dest_activity = spark.read.parquet(DEST_ACTIVITY_PATH)

    txns = bronze.select("txn_id", "step", "nameOrig", DEST_ENTITY_KEY, "isFraud", *spark_feature_columns())
    features = point_in_time_join(txns, dest_activity)
    features.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(FEATURES_PATH)

    print(f"dest activity: {DEST_ACTIVITY_PATH}")
    print(f"features: delta version {latest_delta_version(spark, FEATURES_PATH)} at {FEATURES_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
