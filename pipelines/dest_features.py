"""History features for the receiving account (nameDest), and the point-in-time join.

PaySim senders almost never repeat, but receivers do (mule accounts), so history
is keyed on nameDest.

Timing contract, shared by training and serving:
  A feature row for (nameDest, step s) summarises activity up to and including step s.
  It becomes available at the start of step s + 1 (its event_timestamp).
  A transaction at step t sees the latest row available at or before t, i.e. s <= t - 1.
  So a transaction never sees its own hour, offline or online.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from common.features import DEST_ENTITY_KEY, DEST_FEATURES, SIM_START

_SIM_START_EPOCH = int(SIM_START.timestamp())


def build_dest_activity(bronze: DataFrame) -> DataFrame:
    """One row per (nameDest, step) with rolling inbound activity, ready for Feast."""
    hourly = bronze.groupBy(DEST_ENTITY_KEY, "step").agg(
        F.count("*").alias("n"), F.sum("amount").alias("amt")
    )

    def last_hours(hours: int):
        return Window.partitionBy(DEST_ENTITY_KEY).orderBy("step").rangeBetween(-(hours - 1), 0)

    return hourly.select(
        DEST_ENTITY_KEY,
        "step",
        F.timestamp_seconds(F.lit(_SIM_START_EPOCH) + (F.col("step") + 1) * 3600).alias("event_timestamp"),
        F.col("n").cast("long").alias("dest_in_count_1h"),
        F.sum("n").over(last_hours(24)).cast("long").alias("dest_in_count_24h"),
        F.sum("amt").over(last_hours(24)).cast("double").alias("dest_in_amount_24h"),
        F.sum("n").over(last_hours(168)).cast("long").alias("dest_in_count_7d"),
        F.sum("amt").over(last_hours(168)).cast("double").alias("dest_in_amount_7d"),
    )


def point_in_time_join(txns: DataFrame, dest_activity: DataFrame) -> DataFrame:
    """As-of join: attach to each transaction the latest dest row available before it.

    Done in Spark because it scales; tests check it matches what Feast serves online.
    """
    feats = dest_activity.select(
        DEST_ENTITY_KEY,
        (F.col("step") + 1).alias("_available_at"),
        F.lit(0).alias("_is_txn"),
        *DEST_FEATURES,
    )
    tx = txns.withColumn("_available_at", F.col("step")).withColumn("_is_txn", F.lit(1))

    # Feature rows sort before transactions at the same step, so a row available at
    # step t (built from step t - 1) is visible to transactions at step t.
    w = (
        Window.partitionBy(DEST_ENTITY_KEY)
        .orderBy("_available_at", "_is_txn")
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    unioned = tx.unionByName(feats, allowMissingColumns=True)
    other_cols = [c for c in unioned.columns if c not in DEST_FEATURES]
    joined = unioned.select(*other_cols, *[F.last(c, ignorenulls=True).over(w).alias(c) for c in DEST_FEATURES])

    # No history yet -> 0, exactly what the gateway does for a missing entity.
    return (
        joined.filter(F.col("_is_txn") == 1)
        .drop("_available_at", "_is_txn")
        .fillna(0, subset=DEST_FEATURES)
    )
