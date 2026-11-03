"""Load the Feast online store (Redis) with dest features as of a given step.

Usage: python -m pipelines.materialize --until-step 400
Live replay starts at --until-step + 1, and rollout.replay keeps the store current from there.
"""
import argparse
from datetime import timedelta

from feast import FeatureStore

from common.features import DEST_FEATURE_VIEW, SIM_START, step_to_timestamp

FEAST_REPO = "feature_repo"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--until-step", type=int, default=400, help="Include activity up to and including this step")
    args = parser.parse_args()

    # Rows for step s are stamped at the start of s + 1; +1s makes the end bound inclusive.
    end = step_to_timestamp(args.until_step + 1) + timedelta(seconds=1)
    store = FeatureStore(repo_path=FEAST_REPO)
    store.materialize(start_date=SIM_START, end_date=end, feature_views=[DEST_FEATURE_VIEW])
    print(f"online store now reflects activity up to step {args.until_step}")


if __name__ == "__main__":
    main()
