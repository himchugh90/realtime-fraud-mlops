"""Train a LightGBM fraud model on a time window, log it to MLflow and register a new version.

Usage:
  python -m pipelines.train --train-max-step 200 --make-champion   # first model
  python -m pipelines.train --train-max-step 300                   # retrained candidate
"""
import argparse

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import functions as F

from common.features import FEATURE_COLUMNS
from common.registry import CHAMPION, MODEL_NAME
from pipelines.metrics import evaluate
from pipelines.spark import FEATURES_PATH, get_spark, latest_delta_version

EXPERIMENT = "fraud-detector"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-max-step", type=int, default=200, help="Train on steps <= this (time-based split)")
    parser.add_argument("--holdout-steps", type=int, default=100, help="Evaluate on the next N steps")
    parser.add_argument("--neg-sample", type=float, default=0.1, help="Fraction of non-fraud rows kept for training")
    parser.add_argument("--num-rounds", type=int, default=300)
    parser.add_argument("--make-champion", action="store_true", help="Point @champion at this version (bootstrap only)")
    args = parser.parse_args()

    spark = get_spark("train")
    delta_version = latest_delta_version(spark, FEATURES_PATH)
    df = spark.read.format("delta").option("versionAsOf", delta_version).load(FEATURES_PATH)

    hold_lo, hold_hi = args.train_max_step, args.train_max_step + args.holdout_steps
    train_pdf = (
        df.filter(F.col("step") <= args.train_max_step)
        .sampleBy("isFraud", fractions={0: args.neg_sample, 1: 1.0}, seed=42)
        .select(*FEATURE_COLUMNS, "isFraud")
        .toPandas()
    )
    hold_pdf = (
        df.filter((F.col("step") > hold_lo) & (F.col("step") <= hold_hi))
        .select(*FEATURE_COLUMNS, "isFraud")
        .toPandas()
    )
    spark.stop()
    print(f"train rows {len(train_pdf):,} ({train_pdf.isFraud.sum():,} fraud), holdout rows {len(hold_pdf):,}")

    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.9,
        "seed": 42,
        "verbose": -1,
    }
    booster = lgb.train(
        params,
        lgb.Dataset(train_pdf[FEATURE_COLUMNS], label=train_pdf["isFraud"]),
        num_boost_round=args.num_rounds,
    )
    metrics = evaluate(hold_pdf["isFraud"].to_numpy(), booster.predict(hold_pdf[FEATURE_COLUMNS]))
    print({k: round(v, 4) for k, v in metrics.items()})

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                **params,
                "num_rounds": args.num_rounds,
                "neg_sample": args.neg_sample,
                "train_max_step": args.train_max_step,
                "holdout_steps": args.holdout_steps,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"delta_version": delta_version, "features_path": FEATURES_PATH})
        sample = hold_pdf[FEATURE_COLUMNS].head(100).astype("float64")
        mlflow.lightgbm.log_model(booster, artifact_path="model", signature=infer_signature(sample, booster.predict(sample)))

    version = mlflow.register_model(f"runs:/{run.info.run_id}/model", MODEL_NAME).version
    client = MlflowClient()
    for key, value in {
        "threshold": metrics["threshold_at_1pct_fpr"],
        "train_max_step": args.train_max_step,
        "holdout_steps": args.holdout_steps,
        "delta_version": delta_version,
    }.items():
        client.set_model_version_tag(MODEL_NAME, version, key, str(value))

    if args.make_champion:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
        print(f"registered {MODEL_NAME} v{version} as @{CHAMPION}")
    else:
        print(f"registered {MODEL_NAME} v{version}; next: python -m pipelines.gate --version {version}")


if __name__ == "__main__":
    main()
