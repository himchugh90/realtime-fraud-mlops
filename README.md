# Real-time Fraud Scoring: Safe Model Promotion

A local, fully open-source MLOps platform for a real-time fraud model. The focus is
**model serving and promotion**: MLflow registry, offline gates, shadow deployment,
canary rollout, automated rollback, and delayed-label evaluation (A/B testing).

Stack: PySpark + Delta Lake · Feast + Redis · MLflow (Postgres backend) · LightGBM · FastAPI · Prometheus + Grafana · Docker Compose.
Everything runs on a 16 GB laptop; no cloud account is needed.

---

## Problem statement

A digital payments platform must score **every transaction for fraud before it is authorised**,
within a **P99 latency budget of 50 ms**. Fraud patterns drift, so the model is retrained
regularly. Each new model is a risk: a worse model either lets fraud through (direct losses)
or blocks good customers (lost revenue and complaints).

**Goal:** ship new model versions **often and safely**:

1. Every candidate is compared with the live model on the **same unseen data** before it gets any traffic.
2. It then runs in **shadow** (sees real traffic, affects nothing), then on a **canary** of real users.
3. Any guardrail breach **rolls back automatically**; a bad promotion can be undone **in under a minute**.
4. Every model is **reproducible**: code, parameters, and the exact Delta table version it was trained on.

**Success metrics**

| Type | Metric |
|---|---|
| Model | Recall at 1% false-positive rate, PR-AUC |
| Serving | P99 latency, error rate, zero-downtime model swaps |
| Process | Time from trained model to 100% traffic, rollback time, number of bad models caught before full rollout |

**The hard part (and the interview story):** fraud labels arrive weeks late (chargebacks), so
you cannot prove a new model is better on live traffic right away. The pipeline uses offline
gates and proxy guardrails (flag rate, agreement with the champion) at rollout time, and
evaluates true precision and recall later with `label_join` once labels arrive.

---

## Data

**PaySim** is a synthetic mobile-money dataset (about 6.36M transactions over 30 days) generated
from real transaction logs of an African mobile-money service.

- Download: Kaggle dataset `ealaxi/paysim1` (CC BY-SA 4.0, needs a free Kaggle account).
- Put the CSV (about 470 MB) in `data/raw/paysim/`.
- Columns: `step` (hour 1–743), `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`,
  `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud`.
- About 0.13% of transactions are fraud, and fraud occurs only in `TRANSFER` and `CASH_OUT`.

**No download yet?** `python -m pipelines.generate_synthetic` writes a 300k-row PaySim-shaped
CSV so you can run the whole stack first.

**Time-based splits** (no random shuffling, as in production):

| Window | Steps | Used for |
|---|---|---|
| Champion v1 training | 1–200 | Bootstrap model |
| Champion v1 holdout | 201–300 | Its own threshold and metrics |
| Challenger v2 training | 1–300 | Retrained candidate |
| Gate window | 301–400 | v1 vs v2 on the same unseen data |
| "Live" traffic | 401–743 | Replayed through the gateway |

**Known data caveats (worth discussing in interviews):**
- In PaySim, balance-error features are extremely predictive. That is partly a simulator artifact.
- `newbalanceOrig` may not be known before authorisation in a real system: a possible leakage question.
- Transaction volume drops sharply after step ~400 while fraud continues, so the live window has drift.

---

## Architecture

```
data/raw (CSV) ─► ingest (PySpark) ─► Delta: bronze ─► build_features ─► Delta: features
                                                                              │
                                     train (LightGBM) ◄───────────────────────┘
                                          │ logs params, metrics, Delta version
                                          ▼
                       MLflow registry: fraud-detector  @champion  @challenger  @previous_champion
                                          │ (servers poll aliases and hot-swap)
                    ┌─────────────────────┴─────────────────────┐
              champion server (:8081)                   challenger server (:8082)
                    ▲                                           ▲
                    └──────────── gateway (:8000) ──────────────┘
                     champion_only │ shadow (copy, discard) │ canary (x% of users)
                                   │
                     predictions.jsonl ──► label_join (delayed labels → A/B metrics)
                     /metrics ─────────► Prometheus (:9090) ─► Grafana (:3000)
```

## Features and the feature store (Feast + Redis)

| Kind | Examples | Offline (training) | Online (serving) |
|---|---|---|---|
| Request | amount, balance errors, type, hour | `spark_feature_columns()` | `compute_features(txn)` in the gateway |
| Dest-account history | inbound count/amount in last 1h, 24h, 7d | Spark as-of join (`pipelines/dest_features.py`) | Feast online store (Redis), keyed by `nameDest` |

History is keyed on the **receiving** account because PaySim senders almost never repeat,
while receivers (mule accounts) do.

**Timing contract** (shared by training and serving): a feature row for `(nameDest, step s)`
summarises activity up to step `s` and becomes available at step `s + 1`. A transaction at step `t`
sees the latest row with `s ≤ t − 1`, so it never sees its own hour.

```
pipelines.build_features ─► data/feast/dest_activity (parquet) ─► feast apply / materialize ─► Redis
            └─► Spark as-of join ─► Delta features (training)                                     │
gateway: compute_features(txn) + Feast.get_online_features(nameDest) ◄───────────────────────────┘
rollout.replay: pushes each finished hour to Redis (stands in for a streaming feature job)
```

Guards against train/serve skew:
- `tests/test_feature_parity.py`: Spark request features = online request features.
- `tests/test_point_in_time.py`: no transaction sees its own or future hours (leakage).
- `tests/test_online_offline_skew.py` (integration): Redis values = training values for real accounts.
- Gate check `features_servable_online`: every model input is computable from the request or registered in Feast.
- Gateway metric `gateway_online_feature_defaults_total{reason}`: requests scored with default history (new account, or the store being down).

---

## Setup (Windows, one time)

1. Install WSL2: open PowerShell **as Administrator**, run `wsl --install`, then restart.
2. Install **Docker Desktop** and enable "Use the WSL 2 based engine".
3. Limit WSL memory so Windows stays responsive. Create `%UserProfile%\.wslconfig` containing:
   ```
   [wsl2]
   memory=10GB
   ```
4. Keep the project **inside the WSL filesystem** (e.g. `~/realtime-fraud-mlops`), not under `/mnt/c`.
   It is much faster, and Spark/Delta file operations behave correctly there.

## Quickstart

```bash
docker compose build
docker compose up -d                    # postgres, redis, mlflow, champion, challenger, gateway, prometheus, grafana

# Data → Delta + Feast source
docker compose run --rm jobs python -m pipelines.generate_synthetic           # skip if you have PaySim
docker compose run --rm jobs python -m pipelines.ingest --source data/raw/synthetic   # or data/raw/paysim
docker compose run --rm jobs python -m pipelines.build_features
docker compose run --rm jobs pytest -q                                        # parity + point-in-time tests

# Feature store: register definitions, load Redis with history up to step 400
docker compose run --rm jobs feast -c feature_repo apply
docker compose run --rm jobs python -m pipelines.materialize --until-step 400
docker compose run --rm -e MATERIALIZED_UNTIL_STEP=400 jobs pytest -q -m integration   # online = offline

# Model v1 → champion
docker compose run --rm jobs python -m pipelines.train --train-max-step 200 --make-champion

# Model v2 → candidate → offline gate (sets @challenger if it passes)
docker compose run --rm jobs python -m pipelines.train --train-max-step 300
docker compose run --rm jobs python -m pipelines.gate --version 2
```

Then, in a **second terminal**, keep live traffic flowing:

```bash
docker compose run --rm jobs python -m rollout.replay --from-step 401 --rps 100 --loop
```

The first pass pushes each finished hour's features into Redis as it goes. Later `--loop`
passes re-send old transactions against newer history, which is fine for rollout load but
leaks "future" features, so `label_join` keeps only each transaction's first score. Re-run
`materialize --until-step 400` before running the skew test again.

and run the rollout:

```bash
docker compose run --rm jobs python -m rollout.controller shadow
docker compose run --rm jobs python -m rollout.controller canary --steps 5,25,50,100 --promote
docker compose run --rm jobs python -m rollout.controller status
docker compose run --rm jobs python -m rollout.controller rollback      # practise undoing it
```

To skip the live checks and promote directly after the gate: `python -m rollout.controller promote`.

After some replay, evaluate on delayed labels (the A/B view):

```bash
docker compose run --rm jobs python -m pipelines.label_join --label-delay-steps 24
```

UIs: MLflow http://localhost:5000 · Gateway docs http://localhost:8000/docs · Prometheus http://localhost:9090 · Grafana http://localhost:3000

Useful Grafana / PromQL queries:
- P99 latency per variant: `histogram_quantile(0.99, sum by (le, variant) (rate(gateway_upstream_latency_seconds_bucket[1m])))`
- Error rate: `sum by (variant) (rate(gateway_upstream_requests_total{outcome="error"}[1m]))`
- Shadow agreement: `sum(rate(gateway_shadow_comparisons_total{agree="True"}[5m])) / sum(rate(gateway_shadow_comparisons_total[5m]))`
- Canary share: `gateway_canary_percent`

---

## Promotion lifecycle

| Stage | Who | Checks | On failure |
|---|---|---|---|
| Register | `train` | Signature, params, metrics, Delta version and threshold logged | n/a |
| Offline gate | `gate` | PR-AUC and recall@1%FPR vs champion on the same window, schema = serving features, single-row P99 | Tagged `gate_status=failed`, never gets traffic |
| Shadow | `controller shadow` | Challenger error rate, P99, decision agreement, flag-rate ratio | Tagged `shadow_status=failed` |
| Canary | `controller canary` | Same guardrails at 5 → 25 → 50 → 100% of users | Automatic return to `champion_only` |
| Promote | `controller promote` | Champion server confirms new version before traffic moves | Previous champion untouched |
| Rollback | `controller rollback` | `@champion` ← `@previous_champion`, hot-swapped | n/a |
| Online eval | `label_join` | True precision/recall per version once labels arrive | Feeds the next retrain |

Every decision is recorded as model-version tags in MLflow, giving you an audit trail.

---

## Experiments to run (each one is a write-up)

1. **Break the gate:** train with `--num-rounds 5` and watch the gate reject it.
2. **Break the canary:** stop the challenger mid-canary (`docker compose stop challenger`) and watch the automatic rollback.
3. **Train/serve skew:** change one feature only in `spark_feature_columns`, then watch the parity test catch it.
4. **Latency:** profile `/invocations`. How much of the P99 is pyfunc schema enforcement? Call the native LightGBM booster directly, or export to ONNX, and measure the difference.
5. **Cold start:** set `WARMUP_CALLS=0` and compare P99 right after a model swap.
6. **Delayed labels:** run `label_join` with delays of 0, 24 and 168 steps. How long until you can trust the A/B result?
7. **Load:** increase `--rps` until P99 breaks the 50 ms budget. Where is the bottleneck: gateway, network hop, or model?

## Next steps

- Streaming features: Redpanda + Spark Structured Streaming for per-user velocity features (Redis online store).
- CI: run `pytest` and `gate` on every training PR (GitHub Actions).
- Kubernetes: the same pattern with KServe or Argo Rollouts (k3d locally).

## Resume bullet (fill in your measured numbers)

> Built a real-time fraud-scoring platform with an MLflow-based promotion pipeline (offline gate → shadow → canary 5–100% → automated rollback) serving LightGBM at __ RPS with __ ms P99; caught __ regressed models before full rollout and cut rollback time to __ s.
