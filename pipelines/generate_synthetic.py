"""Generate a small PaySim-shaped CSV so the stack runs before you download real data.

Usage: python -m pipelines.generate_synthetic --rows 300000
"""
import argparse
import os

import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_PROBS = [0.22, 0.35, 0.01, 0.34, 0.08]
MAX_STEP = 743  # PaySim: one step = one hour, 30 days


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Like PaySim, volume is concentrated in early steps.
    step = np.minimum(rng.exponential(180, n).astype(int) + 1, MAX_STEP)
    ttype = rng.choice(TYPES, n, p=TYPE_PROBS)
    amount = np.round(rng.lognormal(11, 1.2, n), 2)
    old_orig = np.round(rng.lognormal(10, 2, n) * (rng.random(n) > 0.3), 2)
    old_dest = np.round(rng.lognormal(12, 2, n) * (rng.random(n) > 0.4), 2)

    outflow = ttype != "CASH_IN"
    new_orig = np.where(outflow, np.maximum(old_orig - amount, 0), old_orig + amount)
    new_dest = np.where(
        ttype == "PAYMENT", 0.0, np.where(outflow, old_dest + amount, np.maximum(old_dest - amount, 0))
    )

    # Fraud only happens on TRANSFER / CASH_OUT and usually drains the account.
    eligible = np.isin(ttype, ["TRANSFER", "CASH_OUT"])
    is_fraud = eligible & (rng.random(n) < 0.012) & (old_orig > 0)
    drains = is_fraud & (rng.random(n) < 0.7)
    amount = np.where(drains, old_orig, amount)
    new_orig = np.where(is_fraud, np.maximum(old_orig - amount, 0), new_orig)
    # PaySim quirk: destination balances are often not updated for fraud.
    new_dest = np.where(is_fraud & (rng.random(n) < 0.6), old_dest, new_dest)

    # Some legitimate transfers also empty the account, so the problem isn't trivial.
    legit_drain = eligible & ~is_fraud & (rng.random(n) < 0.01)
    amount = np.where(legit_drain & (old_orig > 0), old_orig, amount)
    new_orig = np.where(legit_drain, 0.0, new_orig)

    df = pd.DataFrame(
        {
            "step": step,
            "type": ttype,
            "amount": amount,
            "nameOrig": [f"C{x}" for x in rng.integers(1e8, 1e9, n)],
            "oldbalanceOrg": old_orig,
            "newbalanceOrig": np.round(new_orig, 2),
            "nameDest": [f"M{x}" if t == "PAYMENT" else f"C{x}" for t, x in zip(ttype, rng.integers(1e8, 1e9, n))],
            "oldbalanceDest": old_dest,
            "newbalanceDest": np.round(new_dest, 2),
            "isFraud": is_fraud.astype(int),
            "isFlaggedFraud": (is_fraud & (ttype == "TRANSFER") & (amount > 200_000)).astype(int),
        }
    )
    return df.sort_values("step").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="data/raw/synthetic")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "synthetic_paysim.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows ({df.isFraud.sum():,} fraud) to {out}")


if __name__ == "__main__":
    main()
