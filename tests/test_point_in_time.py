"""Leakage guard: a transaction must only see dest activity from earlier hours.

Run: docker compose run --rm jobs pytest -q tests/test_point_in_time.py
"""
from pipelines.dest_features import build_dest_activity, point_in_time_join

# Account D receives: 2 transfers at step 1, 1 at step 3, 1 at step 30.
BRONZE = [
    dict(txn_id="a", step=1, nameDest="D", amount=100.0),
    dict(txn_id="b", step=1, nameDest="D", amount=50.0),
    dict(txn_id="c", step=3, nameDest="D", amount=10.0),
    dict(txn_id="d", step=30, nameDest="D", amount=5.0),
    dict(txn_id="e", step=2, nameDest="OTHER", amount=1.0),
]


def joined(spark) -> dict:
    bronze = spark.createDataFrame(BRONZE)
    dest = build_dest_activity(bronze)
    txns = bronze.select("txn_id", "step", "nameDest")
    return {r.txn_id: r.asDict() for r in point_in_time_join(txns, dest).collect()}


def test_first_hour_sees_no_history(spark):
    rows = joined(spark)
    # Same-hour transactions must not see each other.
    for txn_id in ("a", "b"):
        assert rows[txn_id]["dest_in_count_24h"] == 0
        assert rows[txn_id]["dest_in_amount_24h"] == 0


def test_later_hour_sees_only_earlier_hours(spark):
    c = joined(spark)["c"]  # step 3: sees step 1 only, not itself
    assert c["dest_in_count_24h"] == 2
    assert c["dest_in_amount_24h"] == 150.0
    assert c["dest_in_count_1h"] == 2  # "last active hour" = step 1


def test_windows_are_as_of_last_activity(spark):
    d = joined(spark)["d"]  # step 30: latest row is step 3, whose 24h window covers steps 1-3
    assert d["dest_in_count_24h"] == 3
    assert d["dest_in_amount_7d"] == 160.0


def test_new_account_defaults_to_zero(spark):
    e = joined(spark)["e"]
    assert e["dest_in_count_7d"] == 0
