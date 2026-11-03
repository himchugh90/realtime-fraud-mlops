"""Single source of truth for feature logic.

Two kinds of features:
  REQUEST_FEATURES  computed from the incoming transaction alone (online: compute_features,
                    offline: spark_feature_columns). tests/test_feature_parity.py checks they agree.
  DEST_FEATURES     history of the receiving account. Built in Spark (pipelines/dest_features.py),
                    served online from Feast/Redis, keyed by nameDest.
"""
import math
from datetime import datetime, timedelta, timezone

TXN_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# PaySim steps are hours; map them onto a calendar so Feast can do time-based joins.
SIM_START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def step_to_timestamp(step: int) -> datetime:
    return SIM_START + timedelta(hours=step)


DEST_FEATURE_VIEW = "dest_activity"
DEST_ENTITY_KEY = "nameDest"
DEST_FEATURES = [
    "dest_in_count_1h",
    "dest_in_count_24h",
    "dest_in_amount_24h",
    "dest_in_count_7d",
    "dest_in_amount_7d",
]

REQUEST_FEATURES = [
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_error",
    "dest_balance_error",
    "orig_drained",
    "amount_to_orig_balance",
    "hour_of_day",
    *[f"type_{t}" for t in TXN_TYPES],
]

# Model input order. Changing this list means retraining and a new model version.
FEATURE_COLUMNS = REQUEST_FEATURES + DEST_FEATURES


def compute_features(txn: dict) -> dict:
    """Online path: one raw transaction -> request feature dict (all floats)."""
    amount = float(txn["amount"])
    old_orig = float(txn["oldbalanceOrg"])
    new_orig = float(txn["newbalanceOrig"])
    old_dest = float(txn["oldbalanceDest"])
    new_dest = float(txn["newbalanceDest"])

    features = {
        "amount": amount,
        "log_amount": math.log1p(amount),
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "orig_balance_error": new_orig + amount - old_orig,
        "dest_balance_error": old_dest + amount - new_dest,
        "orig_drained": 1.0 if old_orig > 0 and new_orig == 0 else 0.0,
        "amount_to_orig_balance": amount / (old_orig + 1.0),
        "hour_of_day": float(int(txn["step"]) % 24),
    }
    for t in TXN_TYPES:
        features[f"type_{t}"] = 1.0 if txn["type"] == t else 0.0
    return features


def spark_feature_columns():
    """Offline path: the same features as Spark column expressions."""
    from pyspark.sql import functions as F

    amount = F.col("amount").cast("double")
    old_orig = F.col("oldbalanceOrg").cast("double")
    new_orig = F.col("newbalanceOrig").cast("double")
    old_dest = F.col("oldbalanceDest").cast("double")
    new_dest = F.col("newbalanceDest").cast("double")

    return [
        amount.alias("amount"),
        F.log1p(amount).alias("log_amount"),
        old_orig.alias("oldbalanceOrg"),
        new_orig.alias("newbalanceOrig"),
        old_dest.alias("oldbalanceDest"),
        new_dest.alias("newbalanceDest"),
        (new_orig + amount - old_orig).alias("orig_balance_error"),
        (old_dest + amount - new_dest).alias("dest_balance_error"),
        F.when((old_orig > 0) & (new_orig == 0), 1.0).otherwise(0.0).alias("orig_drained"),
        (amount / (old_orig + 1.0)).alias("amount_to_orig_balance"),
        (F.col("step") % 24).cast("double").alias("hour_of_day"),
        *[F.when(F.col("type") == t, 1.0).otherwise(0.0).alias(f"type_{t}") for t in TXN_TYPES],
    ]
