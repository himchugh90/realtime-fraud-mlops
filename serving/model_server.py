"""Model server bound to one registry alias (@champion or @challenger).

It polls the registry and hot-swaps when the alias moves: the new version is
loaded and warmed up in the background while the old one keeps serving, then
swapped in one assignment. That is what makes promotion and rollback zero-downtime.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow import MlflowClient
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel

from common.features import FEATURE_COLUMNS
from common.registry import MODEL_NAME, version_for_alias

MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "30"))
WARMUP_CALLS = int(os.getenv("WARMUP_CALLS", "50"))

log = logging.getLogger("model_server")
logging.basicConfig(level=logging.INFO)

PREDICT_LATENCY = Histogram(
    "model_predict_latency_seconds",
    "Time spent inside model.predict",
    ["alias"],
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["alias", "version", "flagged"])
LOADED_VERSION = Gauge("model_loaded_version", "Registry version currently loaded (0 = none)", ["alias"])


@dataclass(frozen=True)
class LoadedModel:
    version: str
    model: Any
    threshold: float


current: LoadedModel | None = None
_reload_lock = asyncio.Lock()


def _load(version: str) -> LoadedModel:
    mv = MlflowClient().get_model_version(MODEL_NAME, version)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{version}")
    # Warm up before taking traffic: first calls are slow (lazy init, caches).
    row = pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS)
    for _ in range(WARMUP_CALLS):
        model.predict(row)
    return LoadedModel(version=version, model=model, threshold=float(mv.tags.get("threshold", 0.5)))


async def refresh() -> None:
    global current
    async with _reload_lock:
        mv = await asyncio.to_thread(version_for_alias, MlflowClient(), MODEL_ALIAS)
        if mv is None:
            if current is not None:
                log.warning("@%s was removed; unloading v%s", MODEL_ALIAS, current.version)
            current = None
            LOADED_VERSION.labels(MODEL_ALIAS).set(0)
            return
        if current is not None and current.version == mv.version:
            return
        log.info("loading %s v%s for @%s", MODEL_NAME, mv.version, MODEL_ALIAS)
        loaded = await asyncio.to_thread(_load, mv.version)
        current = loaded
        LOADED_VERSION.labels(MODEL_ALIAS).set(int(loaded.version))
        log.info("now serving v%s (threshold %.4f)", loaded.version, loaded.threshold)


async def _poll_registry() -> None:
    while True:
        try:
            await refresh()
        except Exception:
            log.exception("registry refresh failed; keeping current model")
        await asyncio.sleep(POLL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_poll_registry())
    yield
    task.cancel()


app = FastAPI(title=f"fraud model server (@{MODEL_ALIAS})", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class InvocationRequest(BaseModel):
    features: dict[str, float]


@app.post("/invocations")
def invocations(req: InvocationRequest):
    # Plain `def` so FastAPI runs it in a thread pool and predict() doesn't block the event loop.
    model = current  # snapshot: a concurrent hot-swap can't change the model mid-request
    if model is None:
        raise HTTPException(503, f"no model behind @{MODEL_ALIAS}")
    try:
        row = pd.DataFrame([[req.features[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    except KeyError as e:
        raise HTTPException(422, f"missing feature {e}") from e

    start = time.perf_counter()
    score = float(model.model.predict(row)[0])
    PREDICT_LATENCY.labels(MODEL_ALIAS).observe(time.perf_counter() - start)

    flagged = score >= model.threshold
    PREDICTIONS.labels(MODEL_ALIAS, model.version, str(flagged)).inc()
    return {"score": score, "is_fraud": flagged, "model_version": model.version, "alias": MODEL_ALIAS}


@app.post("/admin/reload")
async def reload():
    await refresh()
    return health()


@app.get("/health")
def health():
    model = current
    return {
        "alias": MODEL_ALIAS,
        "status": "ok" if model else "no_model",
        "model_version": model.version if model else None,
    }
