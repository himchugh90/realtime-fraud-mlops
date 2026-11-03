"""Local Spark session with Delta Lake, sized for a 16 GB laptop."""
import os

from delta import configure_spark_with_delta_pip
from delta.tables import DeltaTable
from pyspark.sql import SparkSession

BRONZE_PATH = os.path.abspath("data/delta/bronze_transactions")
FEATURES_PATH = os.path.abspath("data/delta/features")
DEST_ACTIVITY_PATH = os.path.abspath("data/feast/dest_activity")


def get_spark(app_name: str) -> SparkSession:
    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "4g"))
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # UTC + standard parquet timestamps, so Feast/pandas read event times correctly.
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def latest_delta_version(spark: SparkSession, path: str) -> int:
    """Delta table version, logged with every model so training data is reproducible."""
    return DeltaTable.forPath(spark, path).history(1).select("version").collect()[0][0]
