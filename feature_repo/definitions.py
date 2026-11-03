"""Feast definitions. Register with: feast -c feature_repo apply"""
import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float64, Int64

from common.features import DEST_ENTITY_KEY, DEST_FEATURE_VIEW

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "feast"))

dest_account = Entity(
    name="dest_account",
    join_keys=[DEST_ENTITY_KEY],
    value_type=ValueType.STRING,
    description="Receiving account; fraud funnels money into a small set of mule accounts.",
)

dest_activity_source = FileSource(
    name="dest_activity_source",
    path=os.path.join(DATA_DIR, "dest_activity"),  # written by pipelines.build_features
    timestamp_field="event_timestamp",
)

dest_activity = FeatureView(
    name=DEST_FEATURE_VIEW,
    entities=[dest_account],
    # 0 = no expiry. Training (Spark as-of join) also has no expiry; keep the two in step.
    ttl=timedelta(0),
    schema=[
        Field(name="dest_in_count_1h", dtype=Int64),
        Field(name="dest_in_count_24h", dtype=Int64),
        Field(name="dest_in_amount_24h", dtype=Float64),
        Field(name="dest_in_count_7d", dtype=Int64),
        Field(name="dest_in_amount_7d", dtype=Float64),
    ],
    source=dest_activity_source,
    online=True,
)
