"""Train/serve skew guard: Spark features must equal online features row for row.

Run: docker compose run --rm jobs pytest -q tests
"""
import math

import pytest

from common.features import REQUEST_FEATURES, compute_features, spark_feature_columns

SAMPLE_TXNS = [
    dict(step=1, type="TRANSFER", amount=181.0, nameOrig="C1", oldbalanceOrg=181.0, newbalanceOrig=0.0,
         oldbalanceDest=0.0, newbalanceDest=0.0),
    dict(step=25, type="PAYMENT", amount=9839.64, nameOrig="C2", oldbalanceOrg=170136.0, newbalanceOrig=160296.36,
         oldbalanceDest=0.0, newbalanceDest=0.0),
    dict(step=743, type="CASH_IN", amount=0.0, nameOrig="C3", oldbalanceOrg=0.0, newbalanceOrig=0.0,
         oldbalanceDest=1e7, newbalanceDest=1e7),
]


def test_offline_matches_online(spark):
    df = spark.createDataFrame(SAMPLE_TXNS)
    offline = [row.asDict() for row in df.select(*spark_feature_columns()).collect()]
    for txn, off in zip(SAMPLE_TXNS, offline):
        online = compute_features(txn)
        assert list(online) == REQUEST_FEATURES
        for col in REQUEST_FEATURES:
            assert math.isclose(online[col], off[col], rel_tol=1e-9, abs_tol=1e-9), (col, online[col], off[col])
