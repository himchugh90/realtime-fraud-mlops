"""Online/offline skew check: what Feast serves from Redis must equal what the model trained on.

Integration test: needs the stack up, features built, and
`python -m pipelines.materialize --until-step N` run *before* any replay.
For transactions at step N + 1, the offline point-in-time features and the
online store must agree exactly.

Run: docker compose run --rm -e MATERIALIZED_UNTIL_STEP=400 jobs pytest -q -m integration
"""
import math
import os
import socket

import pytest
from pyspark.sql import functions as F

from common.features import DEST_ENTITY_KEY, DEST_FEATURE_VIEW, DEST_FEATURES
from pipelines.spark import FEATURES_PATH

pytestmark = pytest.mark.integration
UNTIL_STEP = int(os.getenv("MATERIALIZED_UNTIL_STEP", "400"))
SAMPLE = 200


def redis_up() -> bool:
    try:
        socket.create_connection(("redis", 6379), timeout=1).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not redis_up(), reason="redis not reachable; start the stack")
def test_online_features_match_offline(spark):
    from feast import FeatureStore

    offline = (
        spark.read.format("delta")
        .load(FEATURES_PATH)
        .filter(F.col("step") == UNTIL_STEP + 1)
        .filter(F.col("dest_in_count_7d") > 0)  # accounts with history: the interesting case
        .select(DEST_ENTITY_KEY, *DEST_FEATURES)
        .dropDuplicates([DEST_ENTITY_KEY])
        .limit(SAMPLE)
        .toPandas()
    )
    assert len(offline), f"no transactions with history at step {UNTIL_STEP + 1}"

    online = (
        FeatureStore(repo_path="feature_repo")
        .get_online_features(
            features=[f"{DEST_FEATURE_VIEW}:{f}" for f in DEST_FEATURES],
            entity_rows=[{DEST_ENTITY_KEY: d} for d in offline[DEST_ENTITY_KEY]],
        )
        .to_df()
    )

    merged = offline.merge(online, on=DEST_ENTITY_KEY, suffixes=("_offline", "_online"))
    mismatches = [
        (row[DEST_ENTITY_KEY], f, row[f"{f}_offline"], row[f"{f}_online"])
        for _, row in merged.iterrows()
        for f in DEST_FEATURES
        if row[f"{f}_online"] is None
        or not math.isclose(float(row[f"{f}_offline"]), float(row[f"{f}_online"]), rel_tol=1e-9)
    ]
    assert not mismatches, f"{len(mismatches)} skewed values, e.g. {mismatches[:5]}"
