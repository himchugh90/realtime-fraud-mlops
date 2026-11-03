"""Rollout controller: shadow -> canary -> promote, with automatic rollback.

Keep traffic flowing while it runs (python -m rollout.replay --loop in another terminal).

Usage:
  python -m rollout.controller status
  python -m rollout.controller shadow
  python -m rollout.controller canary --steps 5,25,50,100 --promote
  python -m rollout.controller promote      # direct promotion after the gate (no live checks)
  python -m rollout.controller rollback     # undo the last promotion
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import httpx
from mlflow import MlflowClient

from common.registry import CHALLENGER, CHAMPION, MODEL_NAME, PREVIOUS_CHAMPION, version_for_alias

GATEWAY = os.getenv("GATEWAY_URL", "http://gateway:8000")
SERVERS = {
    CHAMPION: os.getenv("CHAMPION_URL", "http://champion:8080"),
    CHALLENGER: os.getenv("CHALLENGER_URL", "http://challenger:8080"),
}

http = httpx.Client(timeout=30)
client = MlflowClient()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require_alias(alias: str):
    mv = version_for_alias(client, alias)
    if mv is None:
        sys.exit(f"No @{alias} set.")
    return mv


def set_gateway(mode: str, canary_pct: int = 0):
    http.post(f"{GATEWAY}/admin/config", json={"mode": mode, "canary_pct": canary_pct}).raise_for_status()


def gateway_stats() -> dict:
    return http.get(f"{GATEWAY}/admin/stats").json()


def reset_stats():
    http.post(f"{GATEWAY}/admin/stats/reset").raise_for_status()


def ensure_serving(alias: str, version: str, timeout_s: int = 180):
    """Ask the alias's server to reload and wait until it reports the expected version."""
    url = SERVERS[alias]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if http.post(f"{url}/admin/reload").json().get("model_version") == version:
                print(f"  @{alias} server is serving v{version}")
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    sys.exit(f"@{alias} server did not load v{version} within {timeout_s}s")


def wait_for(condition, what: str, timeout_s: int):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        s = gateway_stats()
        if condition(s):
            return s
        print(f"  waiting for {what}... champion={s['champion']['requests']} challenger={s['challenger']['requests']}")
        time.sleep(5)
    set_gateway("champion_only")
    sys.exit(f"Timed out waiting for {what}. Is the replay running? Traffic reset to champion_only.")


def flag_rate_ratio(s: dict) -> float:
    eps = 1e-3  # flag rates are tiny; avoid dividing by zero on small samples
    return (s["challenger"]["flag_rate"] + eps) / (s["champion"]["flag_rate"] + eps)


def guardrails(s: dict, args, compare_flag_rate: bool) -> list[tuple[str, bool, str]]:
    ch = s["challenger"]
    checks = [
        ("challenger_error_rate", ch["error_rate"] <= args.max_error_rate, f"{ch['error_rate']:.4f}"),
        ("challenger_p99_ms", (ch["p99_ms"] or 0) <= args.max_p99_ms, f"{ch['p99_ms']} ms"),
    ]
    if compare_flag_rate:
        ratio = flag_rate_ratio(s)
        checks.append(
            ("flag_rate_ratio", 1 / args.max_flag_ratio <= ratio <= args.max_flag_ratio, f"{ratio:.2f}x champion")
        )
    return checks


def report(checks) -> bool:
    for name, ok, detail in checks:
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return all(ok for _, ok, _ in checks)


def cmd_status(args):
    for alias in (CHAMPION, CHALLENGER, PREVIOUS_CHAMPION):
        mv = version_for_alias(client, alias)
        print(f"@{alias}: {'v' + mv.version if mv else '-'}")
    print(json.dumps(gateway_stats(), indent=2))


def cmd_shadow(args):
    challenger = require_alias(CHALLENGER)
    print(f"Shadow test for v{challenger.version}")
    ensure_serving(CHALLENGER, challenger.version)
    set_gateway("shadow")
    reset_stats()
    s = wait_for(lambda s: s["shadow"]["compared"] >= args.min_requests, "shadow comparisons", args.timeout)
    set_gateway("champion_only")

    agreement = s["shadow"]["agreement_rate"]
    checks = guardrails(s, args, compare_flag_rate=True)
    checks.append(("decision_agreement", agreement >= args.min_agreement, f"{agreement:.4f}"))
    passed = report(checks)
    client.set_model_version_tag(MODEL_NAME, challenger.version, "shadow_status", "passed" if passed else "failed")
    client.set_model_version_tag(MODEL_NAME, challenger.version, "shadow_report", json.dumps(s))
    if not passed:
        sys.exit(1)
    print("Shadow passed. Next: python -m rollout.controller canary --promote")


def cmd_canary(args):
    challenger = require_alias(CHALLENGER)
    print(f"Canary for v{challenger.version}")
    ensure_serving(CHALLENGER, challenger.version)

    for pct in [int(p) for p in args.steps.split(",")]:
        print(f"  -> {pct}% of users on challenger")
        set_gateway("canary", pct)
        reset_stats()
        s = wait_for(
            lambda s: s["challenger"]["requests"] >= args.min_requests, f"{args.min_requests} canary requests", args.timeout
        )
        # At 100% the champion gets no traffic, so there is nothing to compare flag rates against.
        if not report(guardrails(s, args, compare_flag_rate=s["champion"]["requests"] >= 100)):
            set_gateway("champion_only")
            client.set_model_version_tag(MODEL_NAME, challenger.version, "canary_status", f"rolled_back_at_{pct}pct")
            sys.exit(f"Guardrail failed at {pct}%: rolled back to champion_only.")

    client.set_model_version_tag(MODEL_NAME, challenger.version, "canary_status", "passed")
    print("Canary passed.")
    if args.promote:
        promote()
    else:
        set_gateway("champion_only")


def promote():
    challenger = require_alias(CHALLENGER)
    old = version_for_alias(client, CHAMPION)
    print(f"Promoting v{challenger.version} (replacing {'v' + old.version if old else 'nothing'})")

    # Move aliases first; the champion server keeps serving the old model until the new one is warm.
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, challenger.version)
    if old:
        client.set_registered_model_alias(MODEL_NAME, PREVIOUS_CHAMPION, old.version)
    ensure_serving(CHAMPION, challenger.version)

    # Only after the champion server has the new model do we stop routing to the challenger server.
    set_gateway("champion_only")
    client.delete_registered_model_alias(MODEL_NAME, CHALLENGER)
    http.post(f"{SERVERS[CHALLENGER]}/admin/reload")
    client.set_model_version_tag(MODEL_NAME, challenger.version, "promoted_at", now())
    print(f"v{challenger.version} is @{CHAMPION}. Rollback target: @{PREVIOUS_CHAMPION}")


def cmd_promote(args):
    promote()


def cmd_rollback(args):
    previous = require_alias(PREVIOUS_CHAMPION)
    bad = require_alias(CHAMPION)
    print(f"Rolling back v{bad.version} -> v{previous.version}")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, previous.version)
    ensure_serving(CHAMPION, previous.version)
    set_gateway("champion_only")
    client.set_model_version_tag(MODEL_NAME, bad.version, "rolled_back_at", now())
    print("Rollback complete.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("promote").set_defaults(func=cmd_promote)
    sub.add_parser("rollback").set_defaults(func=cmd_rollback)

    for name, func in (("shadow", cmd_shadow), ("canary", cmd_canary)):
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument("--min-requests", type=int, default=2000 if name == "shadow" else 2001)
        p.add_argument("--timeout", type=int, default=900, help="Seconds to wait for enough traffic per stage")
        p.add_argument("--max-error-rate", type=float, default=0.01)
        p.add_argument("--max-p99-ms", type=float, default=50.0)
        p.add_argument("--max-flag-ratio", type=float, default=2.0)
        if name == "shadow":
            p.add_argument("--min-agreement", type=float, default=0.97)
        else:
            p.add_argument("--steps", default="5,25,50,100")
            p.add_argument("--promote", action="store_true", help="Promote automatically if every step passes")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
