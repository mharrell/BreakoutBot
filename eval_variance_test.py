"""
Evaluate policy-intrinsic vs. environment-intrinsic variance.

Runs the same model under four conditions to decompose score variance:
  (a) deterministic=True,  sticky=0.25  (current standard)
  (b) deterministic=False, sticky=0.25  (policy stochasticity + env noise)
  (c) deterministic=True,  sticky=0.0   (ALE state drift only)
  (d) deterministic=False, sticky=0.0   (policy stochasticity only)

By comparing unique-score counts and distributions across conditions,
we can measure:
  - How much variance comes from the policy's own action distribution
  - How much comes from sticky-action environmental noise
  - Whether deterministic=True is masking policy-intrinsic variance

This addresses FLAWS.md F-014: deterministic=True is used in all evaluations
with no stochastic-policy baseline.

Usage:
    python eval_variance_test.py [--model MODEL_PATH] [--games N]

Default: PPO_30b/final_model, 500 games per condition.
"""
import os
import csv
import time
import argparse
import numpy as np
from datetime import datetime

import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)


def run_eval(model, sticky, deterministic, n_games):
    """Run n_games and return list of scores."""
    env_kwargs = {"repeat_action_probability": 0.25 if sticky else 0.0}
    env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                          env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)

    obs = env.reset()
    scores = []
    episode = 0

    while episode < n_games:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        if done[0]:
            lives = info[0].get("lives", -1)
            if lives == 0:
                score = float(info[0].get("episode", {}).get("r", 0))
                scores.append(score)
                episode += 1
                obs = env.reset()
            else:
                obs, _, _, _ = env.step([0])

    env.close()
    return scores


def compute_metrics(scores, label):
    arr = np.array(scores)
    return {
        "condition": label,
        "n_games": len(arr),
        "unique_scores": len(set(scores)),
        "avg": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "zero_count": int(np.sum(arr == 0)),
        "zero_rate": np.sum(arr == 0) / len(arr) * 100,
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Decompose evaluation variance into policy and environment components"
    )
    parser.add_argument("--model", default="models/PPO_30b/final_model",
                        help="Path to model (without .zip)")
    parser.add_argument("--games", type=int, default=500,
                        help="Games per condition")
    args = parser.parse_args()

    if not os.path.exists(args.model + ".zip"):
        print(f"ERROR: Model not found at {args.model}.zip")
        return

    CONDITIONS = [
        # (label, sticky, deterministic)
        ("det=True_sticky=0.25", True, True),
        ("det=False_sticky=0.25", True, False),
        ("det=True_sticky=0.0", False, True),
        ("det=False_sticky=0.0", False, False),
    ]

    print("=" * 65)
    print("Evaluation Variance Decomposition")
    print(f"Model: {args.model}")
    print(f"Games per condition: {args.games}")
    print(f"Total games: {len(CONDITIONS) * args.games}")
    print("=" * 65)

    model = PPO.load(args.model)
    results = {}

    for label, sticky, det in CONDITIONS:
        print(f"\n  {label}...", end=" ", flush=True)
        start = time.time()

        scores = run_eval(model, sticky=sticky, deterministic=det,
                          n_games=args.games)
        metrics = compute_metrics(scores, label)
        results[label] = metrics

        elapsed = time.time() - start
        print(f"{len(scores)} games in {elapsed:.0f}s | "
              f"unique={metrics['unique_scores']} | "
              f"avg={metrics['avg']:.1f} | "
              f"zero={metrics['zero_rate']:.1f}%")

    # --- Analysis ---
    print(f"\n{'=' * 65}")
    print("RESULTS")
    print(f"{'=' * 65}")

    print(f"\n{'Condition':<25} {'Unique':>6} {'Avg':>7} {'Median':>6} "
          f"{'Zero%':>7} {'P95':>6} {'P99':>6}")
    print(f"{'-' * 65}")
    for label, sticky, det in CONDITIONS:
        m = results[label]
        print(f"{label:<25} {m['unique_scores']:>6} {m['avg']:>7.1f} "
              f"{m['median']:>6.0f} {m['zero_rate']:>6.1f}% "
              f"{m['p95']:>6.0f} {m['p99']:>6.0f}")

    print(f"\n{'-' * 50}")
    print("DECOMPOSITION")
    print(f"{'-' * 50}")

    # Isolate components
    a = results["det=True_sticky=0.25"]    # baseline (env noise + ALE drift)
    b = results["det=False_sticky=0.25"]   # baseline + policy stochasticity
    c = results["det=True_sticky=0.0"]     # ALE drift only
    d = results["det=False_sticky=0.0"]    # policy stochasticity + ALE drift

    # Policy stochasticity contribution (compare a vs b, or c vs d)
    policy_unique_diff = b["unique_scores"] - a["unique_scores"]
    print(f"\nPolicy stochasticity contribution to unique scores:")
    print(f"  With sticky:    det=False adds {policy_unique_diff} unique scores "
          f"({a['unique_scores']} → {b['unique_scores']})")
    policy_unique_diff_ns = d["unique_scores"] - c["unique_scores"]
    print(f"  Without sticky: det=False adds {policy_unique_diff_ns} unique scores "
          f"({c['unique_scores']} → {d['unique_scores']})")

    # Environment stochasticity contribution (compare c vs a)
    env_unique_diff = a["unique_scores"] - c["unique_scores"]
    print(f"\nSticky-action noise contribution to unique scores:")
    print(f"  det=True:  sticky adds {env_unique_diff} unique scores "
          f"({c['unique_scores']} → {a['unique_scores']})")
    env_unique_diff_stoch = b["unique_scores"] - d["unique_scores"]
    print(f"  det=False: sticky adds {env_unique_diff_stoch} unique scores "
          f"({d['unique_scores']} → {b['unique_scores']})")

    # Key question: is deterministic=True masking variance?
    if policy_unique_diff > 3 or policy_unique_diff_ns > 1:
        print(f"\n[!] Policy stochasticity adds meaningful variance. "
              f"deterministic=True may be UNDERESTIMATING the policy's true diversity.")
    else:
        print(f"\n[OK]Policy stochasticity adds minimal variance. "
              f"deterministic=True is representative of the policy's behavior.")

    if c["unique_scores"] <= 2:
        print(f"[OK]Without sticky actions, the policy collapses to ≤2 unique scores "
              f"(confirming memorization).")
    else:
        print(f"[!] Without sticky actions, the policy still produces "
              f"{c['unique_scores']} unique scores — possible genuine reactivity.")

    # Save detailed results
    output_path = os.path.join("recordings", "variance_decomposition.csv")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "condition", "n_games", "unique_scores", "avg", "median", "std",
            "min", "max", "zero_count", "zero_rate", "p95", "p99", "timestamp"
        ])
        for label, sticky, det in CONDITIONS:
            m = results[label]
            writer.writerow([
                label, m["n_games"], m["unique_scores"],
                round(m["avg"], 1), round(m["median"], 1), round(m["std"], 1),
                m["min"], m["max"], m["zero_count"], round(m["zero_rate"], 2),
                m["p95"], m["p99"], datetime.now().isoformat()
            ])

    print(f"\nDetailed results: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
