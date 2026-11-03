"""Offline promotion gate: candidate vs current champion on the same unseen window.

Passing sets the @challenger alias. Exits non-zero on failure, so it can run in CI.

Usage: python -m pipelines.gate --version 2
"""
import argparse
import json
import sys
import time

import mlflow
import mlflow.pyfunc
import numpy as np
from feast import FeatureStore
from mlflow import MlflowClient
from pyspark.sql import functions as F

from common.features import DEST_FEATURE_VIEW, FEATURE_COLUMNS, REQUEST_FEATURES
from common.registry import CHALLENGER, CHAMPION, MODEL_NAME, version_for_alias
from pipelines.metrics import evaluate
from pipelines.spark import FEATURES_PATH, get_spark, latest_delta_version

EXPERIMENT = "fraud-detector-gate"


def load_window(lo: int, hi: int):
    spark = get_spark("gate")
    delta_version = latest_delta_version(spark, FEATURES_PATH)
    pdf = (
        spark.read.format("delta")
        .option("versionAsOf", delta_version)
        .load(FEATURES_PATH)
        .filter((F.col("step") > lo) & (F.col("step") <= hi))
        .select(*FEATURE_COLUMNS, "isFraud")
        .toPandas()
    )
    spark.stop()
    return pdf, delta_version


def single_row_p99_ms(model, X, n: int = 300) -> float:
    """Online traffic is one row at a time, so measure that, not batch throughput."""
    rows = X.sample(n=min(n, len(X)), random_state=0)
    timings = []
    for i in range(len(rows)):
        row = rows.iloc[[i]]
        start = time.perf_counter()
        model.predict(row)
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.percentile(timings, 99))


def missing_online_features(input_names: list[str]) -> list[str]:
    """Model inputs that are neither computed from the request nor registered in Feast.

    A model that needs a feature the online store can't serve must not be promoted.
    """
    registered = {f.name for f in FeatureStore(repo_path="feature_repo").get_feature_view(DEST_FEATURE_VIEW).features}
    return [c for c in input_names if c not in REQUEST_FEATURES and c not in registered]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, help="Candidate version (default: latest registered)")
    parser.add_argument("--max-pr-auc-drop", type=float, default=0.01)
    parser.add_argument("--max-recall-drop", type=float, default=0.02)
    parser.add_argument("--max-p99-ms", type=float, default=25.0)
    args = parser.parse_args()

    client = MlflowClient()
    version = args.version or max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    candidate = client.get_model_version(MODEL_NAME, str(version))
    champion = version_for_alias(client, CHAMPION)
    if champion is None:
        sys.exit("No @champion yet: train the first model with --make-champion.")
    if champion.version == candidate.version:
        sys.exit(f"v{version} is already @champion.")

    lo = int(candidate.tags["train_max_step"])
    hi = lo + int(candidate.tags["holdout_steps"])
    pdf, delta_version = load_window(lo, hi)
    X, y = pdf[FEATURE_COLUMNS], pdf["isFraud"].to_numpy()

    cand_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{candidate.version}")
    champ_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{champion.version}")
    cand_m = evaluate(y, np.asarray(cand_model.predict(X)))
    champ_m = evaluate(y, np.asarray(champ_model.predict(X)))
    input_names = cand_model.metadata.get_input_schema().input_names()
    p99 = single_row_p99_ms(cand_model, X)
    missing = missing_online_features(input_names)

    checks = [
        ("schema_matches_serving_features", input_names == FEATURE_COLUMNS, f"{len(input_names)} inputs"),
        ("features_servable_online", not missing, f"missing: {missing}" if missing else "all registered in Feast"),
        (
            "pr_auc_not_worse",
            cand_m["pr_auc"] >= champ_m["pr_auc"] - args.max_pr_auc_drop,
            f"candidate {cand_m['pr_auc']:.4f} vs champion {champ_m['pr_auc']:.4f}",
        ),
        (
            "recall_at_1pct_fpr_not_worse",
            cand_m["recall_at_1pct_fpr"] >= champ_m["recall_at_1pct_fpr"] - args.max_recall_drop,
            f"candidate {cand_m['recall_at_1pct_fpr']:.4f} vs champion {champ_m['recall_at_1pct_fpr']:.4f}",
        ),
        ("single_row_p99_latency", p99 <= args.max_p99_ms, f"{p99:.2f} ms (limit {args.max_p99_ms})"),
    ]
    passed = all(ok for _, ok, _ in checks)

    print(f"\nGate: v{candidate.version} vs @champion v{champion.version}, steps ({lo}, {hi}], {len(pdf):,} rows")
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"gate-v{candidate.version}"):
        mlflow.log_metrics({f"candidate_{k}": v for k, v in cand_m.items()})
        mlflow.log_metrics({f"champion_{k}": v for k, v in champ_m.items()})
        mlflow.log_metric("candidate_single_row_p99_ms", p99)
        mlflow.set_tags(
            {
                "candidate_version": candidate.version,
                "champion_version": champion.version,
                "delta_version": delta_version,
                "passed": passed,
            }
        )

    report = json.dumps({name: {"passed": ok, "detail": detail} for name, ok, detail in checks})
    client.set_model_version_tag(MODEL_NAME, candidate.version, "gate_status", "passed" if passed else "failed")
    client.set_model_version_tag(MODEL_NAME, candidate.version, "gate_report", report)

    if not passed:
        sys.exit(1)
    client.set_registered_model_alias(MODEL_NAME, CHALLENGER, candidate.version)
    print(f"\nv{candidate.version} is now @{CHALLENGER}. Next: python -m rollout.controller shadow")


if __name__ == "__main__":
    main()
