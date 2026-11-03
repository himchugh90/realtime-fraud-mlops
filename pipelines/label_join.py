"""Join logged online predictions with labels that arrive late (chargebacks).

This is the A/B view: real precision/recall per model version on live traffic,
using only labels that would be known by now.

Usage: python -m pipelines.label_join --label-delay-steps 24
"""
import argparse

import mlflow
import pandas as pd
from pyspark.sql import functions as F

from pipelines.spark import BRONZE_PATH, get_spark

EXPERIMENT = "fraud-detector-online-eval"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="data/logs/predictions.jsonl")
    parser.add_argument("--label-delay-steps", type=int, default=24, help="Labels arrive this many steps (hours) late")
    args = parser.parse_args()

    # preds = pd.read_json(args.log, lines=True)
    # txn_ids are 19-digit strings; letting pandas parse them as numbers loses digits.
    preds = pd.read_json(args.log, lines=True, dtype={"txn_id": "string"})
    preds = preds[preds.txn_id.notna()]
    # replay --loop re-sends transactions; keep each one's first (point-in-time correct) score.
    preds = preds.drop_duplicates(["txn_id", "model_version", "shadow"], keep="first")
    labels_known_until = preds.step.max() - args.label_delay_steps

    spark = get_spark("label_join")
    labels = (
        spark.read.format("delta")
        .load(BRONZE_PATH)
        .filter((F.col("step") >= int(preds.step.min())) & (F.col("step") <= int(labels_known_until)))
        .select("txn_id", "isFraud")
        .toPandas()
    )
    spark.stop()

    # joined = preds.merge(labels, on="txn_id", how="inner")
        # The log stores txn_id as a string; Delta may not. Join on one type.
    labels["txn_id"] = labels.txn_id.astype("string")
    joined = preds.merge(labels, on="txn_id", how="inner")
    if joined.empty:
        print("No labelled predictions yet: replay more traffic or lower --label-delay-steps.")
        return

    rows = []
    for (variant, version, shadow), g in joined.groupby(["served_by", "model_version", "shadow"]):
        tp = int((g.is_fraud & (g.isFraud == 1)).sum())
        rows.append(
            {
                "variant": variant,
                "model_version": version,
                "shadow": shadow,
                "predictions": len(g),
                "flag_rate": g.is_fraud.mean(),
                "precision": tp / max(int(g.is_fraud.sum()), 1),
                "recall": tp / max(int(g.isFraud.sum()), 1),
            }
        )
    report = pd.DataFrame(rows)
    print(f"Labels known up to step {labels_known_until}; {len(joined):,} of {len(preds):,} predictions labelled\n")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run():
        for r in rows:
            prefix = f"v{r['model_version']}_{'shadow' if r['shadow'] else 'live'}"
            mlflow.log_metrics({f"{prefix}_{k}": float(r[k]) for k in ("predictions", "flag_rate", "precision", "recall")})
        mlflow.log_param("label_delay_steps", args.label_delay_steps)


if __name__ == "__main__":
    main()
