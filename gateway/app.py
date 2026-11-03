"""Prediction gateway: builds the feature vector once, then routes between champion and challenger.

Feature vector = request features (computed from the transaction) + dest-account history
(looked up in the Feast online store). Building it here means champion and challenger
always score the exact same input, which keeps shadow comparisons fair.

Modes (set via POST /admin/config):
  champion_only  all traffic to champion
  shadow         champion answers; challenger gets a copy, its answer is logged and discarded
  canary         canary_pct% of users go to challenger (sticky per user); falls back to champion on error
"""
import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from common.features import DEST_ENTITY_KEY, DEST_FEATURE_VIEW, DEST_FEATURES, compute_features

UPSTREAMS = {
    "champion": os.getenv("CHAMPION_URL", "http://champion:8080"),
    "challenger": os.getenv("CHALLENGER_URL", "http://challenger:8080"),
}
PREDICTION_LOG = os.getenv("PREDICTION_LOG", "/logs/predictions.jsonl")
UPSTREAM_TIMEOUT_S = float(os.getenv("UPSTREAM_TIMEOUT_S", "0.5"))
FEAST_REPO = os.getenv("FEAST_REPO", "/app/feature_repo")
ONLINE_FEATURE_REFS = [f"{DEST_FEATURE_VIEW}:{f}" for f in DEST_FEATURES]

REQUESTS = Counter("gateway_upstream_requests_total", "Upstream calls", ["variant", "outcome"])
LATENCY = Histogram(
    "gateway_upstream_latency_seconds",
    "Gateway -> model server round trip",
    ["variant"],
    buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5),
)
SHADOW_COMPARISONS = Counter("gateway_shadow_comparisons_total", "Shadow decisions compared", ["agree"])
CANARY_PERCENT = Gauge("gateway_canary_percent", "Share of users routed to challenger")
ONLINE_FEATURE_LATENCY = Histogram(
    "gateway_online_feature_latency_seconds",
    "Feast online store lookup",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
ONLINE_FEATURE_DEFAULTS = Counter(
    "gateway_online_feature_defaults_total", "Requests scored with default (0) history features", ["reason"]
)


class RollingStats:
    """In-process stats over the most recent requests; the rollout controller reads these."""

    def __init__(self, window: int = 5000):
        self.latencies_ms = deque(maxlen=window)
        self.requests = self.errors = self.flagged = 0

    def record(self, latency_ms: float | None = None, flagged: bool = False, error: bool = False):
        self.requests += 1
        if error:
            self.errors += 1
            return
        self.latencies_ms.append(latency_ms)
        self.flagged += int(flagged)

    def snapshot(self) -> dict:
        lat = sorted(self.latencies_ms)
        ok = self.requests - self.errors

        def pct(p):
            return round(lat[min(int(p * len(lat)), len(lat) - 1)], 2) if lat else None

        return {
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": self.errors / self.requests if self.requests else 0.0,
            "flag_rate": self.flagged / ok if ok else 0.0,
            "p50_ms": pct(0.50),
            "p99_ms": pct(0.99),
        }


class ShadowStats:
    def __init__(self):
        self.compared = self.agreements = 0
        self.abs_score_diff = 0.0

    def snapshot(self) -> dict:
        return {
            "compared": self.compared,
            "agreement_rate": self.agreements / self.compared if self.compared else None,
            "mean_abs_score_diff": self.abs_score_diff / self.compared if self.compared else None,
        }


class UpstreamError(Exception):
    pass


class Transaction(BaseModel):
    txn_id: str | None = None
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str | None = None
    oldbalanceDest: float
    newbalanceDest: float


class RolloutConfig(BaseModel):
    mode: Literal["champion_only", "shadow", "canary"]
    canary_pct: int = Field(0, ge=0, le=100)


config = RolloutConfig(mode="champion_only")
stats = {"champion": RollingStats(), "challenger": RollingStats()}
shadow_stats = ShadowStats()
_background_tasks: set[asyncio.Task] = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        timeout=UPSTREAM_TIMEOUT_S, limits=httpx.Limits(max_connections=200, max_keepalive_connections=100)
    )
    os.makedirs(os.path.dirname(PREDICTION_LOG), exist_ok=True)
    app.state.log = open(PREDICTION_LOG, "a", buffering=1)
    yield
    await app.state.http.aclose()
    app.state.log.close()


app = FastAPI(title="fraud gateway", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


_feature_store: FeatureStore | None = None


def _read_online_features(dest: str) -> dict:
    global _feature_store
    if _feature_store is None:
        _feature_store = FeatureStore(repo_path=FEAST_REPO)
    resp = _feature_store.get_online_features(
        features=ONLINE_FEATURE_REFS, entity_rows=[{DEST_ENTITY_KEY: dest}]
    ).to_dict()
    return {f: resp[f][0] for f in DEST_FEATURES}


async def online_features(dest: str | None) -> tuple[dict, str | None]:
    """Dest history from Feast, plus the reason defaults were used (None if served from the store).

    Defaults (0) match the offline join for accounts with no history. On a store error we
    also degrade to defaults rather than fail the payment; the counter makes that visible.
    """
    defaults = {f: 0.0 for f in DEST_FEATURES}
    if not dest:
        ONLINE_FEATURE_DEFAULTS.labels("no_entity").inc()
        return defaults, "no_entity"
    start = time.perf_counter()
    try:
        values = await asyncio.to_thread(_read_online_features, dest)
    except Exception:
        ONLINE_FEATURE_DEFAULTS.labels("store_error").inc()
        return defaults, "store_error"
    ONLINE_FEATURE_LATENCY.observe(time.perf_counter() - start)
    if all(v is None for v in values.values()):
        ONLINE_FEATURE_DEFAULTS.labels("new_entity").inc()
        return defaults, "new_entity"
    return {f: float(v or 0.0) for f, v in values.items()}, None


def canary_bucket(key: str) -> int:
    # Stable across processes and restarts (Python's hash() is salted per process),
    # so a user stays on the same variant for the whole canary.
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % 100


async def call_model(variant: str, features: dict) -> dict:
    start = time.perf_counter()
    try:
        resp = await app.state.http.post(f"{UPSTREAMS[variant]}/invocations", json={"features": features})
        resp.raise_for_status()
        body = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        stats[variant].record(error=True)
        REQUESTS.labels(variant, "error").inc()
        raise UpstreamError(variant) from e
    elapsed = time.perf_counter() - start
    stats[variant].record(elapsed * 1000, body["is_fraud"])
    REQUESTS.labels(variant, "ok").inc()
    LATENCY.labels(variant).observe(elapsed)
    return body


def log_prediction(
    txn: Transaction,
    request_id: str,
    result: dict,
    served_by: str,
    shadow: bool,
    fallback: bool,
    feature_defaults: str | None,
):
    record = {
        "ts": time.time(),
        "request_id": request_id,
        "txn_id": txn.txn_id,
        "step": txn.step,
        "mode": config.mode,
        "served_by": served_by,
        "model_version": result["model_version"],
        "score": result["score"],
        "is_fraud": result["is_fraud"],
        "shadow": shadow,
        "fallback": fallback,
        "feature_defaults": feature_defaults,
    }
    app.state.log.write(json.dumps(record) + "\n")


async def shadow_compare(
    txn: Transaction, request_id: str, features: dict, champion_result: dict, feature_defaults: str | None
):
    try:
        result = await call_model("challenger", features)
    except UpstreamError:
        return  # already counted in stats["challenger"]
    agree = result["is_fraud"] == champion_result["is_fraud"]
    shadow_stats.compared += 1
    shadow_stats.agreements += int(agree)
    shadow_stats.abs_score_diff += abs(result["score"] - champion_result["score"])
    SHADOW_COMPARISONS.labels(str(agree)).inc()
    log_prediction(txn, request_id, result, "challenger", shadow=True, fallback=False, feature_defaults=feature_defaults)


@app.post("/predict")
async def predict(txn: Transaction):
    features = compute_features(txn.model_dump())
    dest_features, feature_defaults = await online_features(txn.nameDest)
    features.update(dest_features)
    request_id = uuid.uuid4().hex
    mode, pct = config.mode, config.canary_pct

    variant = "challenger" if mode == "canary" and canary_bucket(txn.nameOrig) < pct else "champion"
    fallback = False
    try:
        result = await call_model(variant, features)
    except UpstreamError:
        if variant == "champion":
            raise HTTPException(503, "champion unavailable")
        variant, fallback = "champion", True
        try:
            result = await call_model("champion", features)
        except UpstreamError:
            raise HTTPException(503, "no model available")

    if mode == "shadow":
        # Fire and forget: shadow latency must never add to the user's latency.
        task = asyncio.create_task(shadow_compare(txn, request_id, features, result, feature_defaults))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    log_prediction(txn, request_id, result, variant, shadow=False, fallback=fallback, feature_defaults=feature_defaults)
    return {
        "request_id": request_id,
        "score": result["score"],
        "is_fraud": result["is_fraud"],
        "served_by": variant,
        "model_version": result["model_version"],
        "fallback": fallback,
    }


@app.get("/admin/config")
def get_config():
    return config


@app.post("/admin/config")
def set_config(new: RolloutConfig):
    global config
    config = new
    CANARY_PERCENT.set(new.canary_pct if new.mode == "canary" else 0)
    return config


@app.get("/admin/stats")
def get_stats():
    return {
        "config": config,
        "champion": stats["champion"].snapshot(),
        "challenger": stats["challenger"].snapshot(),
        "shadow": shadow_stats.snapshot(),
    }


@app.post("/admin/stats/reset")
def reset_stats():
    global shadow_stats
    for variant in stats:
        stats[variant] = RollingStats()
    shadow_stats = ShadowStats()
    return get_stats()


@app.get("/health")
def health():
    return {"status": "ok", "config": config}
