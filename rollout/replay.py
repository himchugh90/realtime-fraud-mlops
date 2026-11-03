"""Replay unseen transactions against the gateway as 'live' traffic (labels are not sent).

It also plays the role of a streaming feature job: before sending the transactions of
step t, it pushes dest-account rows for step t - 1 into the Feast online store, so online
features stay as fresh as the offline point-in-time join assumes.

Run `python -m pipelines.materialize --until-step <from-step - 1>` first.

Usage: python -m rollout.replay --from-step 401 --rps 100 --loop
"""
import argparse
import asyncio
import os
import time

import httpx
import pandas as pd
from feast import FeatureStore
from pyspark.sql import functions as F

from common.features import DEST_FEATURE_VIEW
from pipelines.spark import BRONZE_PATH, DEST_ACTIVITY_PATH, get_spark

RAW_FIELDS = [
    "txn_id", "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
]


def load_rows(from_step: int, to_step: int, limit: int) -> list[dict]:
    spark = get_spark("replay")
    df = (
        spark.read.format("delta")
        .load(BRONZE_PATH)
        .filter((F.col("step") >= from_step) & (F.col("step") <= to_step))
        .orderBy("step")
        .select(*RAW_FIELDS)
    )
    rows = df.limit(limit).toPandas().to_dict("records")
    spark.stop()
    return rows


def load_feature_updates(from_step: int, to_step: int) -> dict[int, pd.DataFrame]:
    updates = pd.read_parquet(DEST_ACTIVITY_PATH, filters=[("step", ">=", from_step), ("step", "<=", to_step)])
    return {step: g.drop(columns="step") for step, g in updates.groupby("step")}


async def replay(
    rows: list[dict],
    url: str,
    rps: float,
    concurrency: int,
    store: FeatureStore | None,
    updates: dict[int, pd.DataFrame],
    pushed_until: int,
):
    sem = asyncio.Semaphore(concurrency)
    counts = {"ok": 0, "error": 0}

    async with httpx.AsyncClient(timeout=5) as client:

        async def send(row):
            async with sem:
                try:
                    resp = await client.post(url, json=row)
                    counts["ok" if resp.status_code == 200 else "error"] += 1
                except httpx.HTTPError:
                    counts["error"] += 1

        start = time.perf_counter()
        tasks = []
        for i, row in enumerate(rows):
            # Hour boundary: publish the previous hour's aggregates before scoring this hour.
            # (Requests still in flight from the previous hour may see them: a real race in
            # streaming systems, small here.)
            if store is not None and row["step"] - 1 > pushed_until:
                for step in range(pushed_until + 1, row["step"]):
                    if step in updates:
                        await asyncio.to_thread(store.write_to_online_store, DEST_FEATURE_VIEW, updates[step])
                pushed_until = row["step"] - 1

            delay = start + i / rps - time.perf_counter()  # open-loop pacing at a fixed rate
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send(row)))
            if (i + 1) % 1000 == 0:
                print(f"  sent {i + 1:,} | ok {counts['ok']:,} | errors {counts['error']:,} | features up to step {pushed_until}")
        await asyncio.gather(*tasks)
    print(f"done: ok {counts['ok']:,}, errors {counts['error']:,} in {time.perf_counter() - start:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-step", type=int, default=401)
    parser.add_argument("--to-step", type=int, default=743)
    parser.add_argument("--limit", type=int, default=20_000)
    parser.add_argument("--rps", type=float, default=100)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--loop", action="store_true", help="Repeat forever (keeps traffic up during rollouts)")
    parser.add_argument("--no-feature-push", action="store_true", help="Leave the online store frozen")
    parser.add_argument("--gateway", default=os.getenv("GATEWAY_URL", "http://gateway:8000"))
    args = parser.parse_args()

    rows = load_rows(args.from_step, args.to_step, args.limit)
    store = None if args.no_feature_push else FeatureStore(repo_path="feature_repo")
    updates = load_feature_updates(args.from_step, args.to_step) if store else {}
    print(f"replaying {len(rows):,} transactions at {args.rps} rps")

    asyncio.run(replay(rows, f"{args.gateway}/predict", args.rps, args.concurrency, store, updates, args.from_step - 1))
    # Later passes re-send old transactions while the store holds newer history, so their
    # features include "future" activity. Fine for load during rollouts; use only the first
    # pass for label_join.
    while args.loop:
        asyncio.run(replay(rows, f"{args.gateway}/predict", args.rps, args.concurrency, None, {}, 0))


if __name__ == "__main__":
    main()
