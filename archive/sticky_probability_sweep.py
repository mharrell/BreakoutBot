"""
Sticky probability sweep — test whether project findings are specific to p=0.25.

Evaluates trained models at multiple sticky probabilities to measure how
scores, zero-score rates, and variance change with stickiness level.

Loads PPO_30b/final_model and PPO_31b/final_model and runs 500-game
evaluations at sticky probabilities from 0.0 to 0.25. Outputs a CSV
and a printed summary showing the relationship between stickiness and
each metric.

This directly tests FLAWS.md F-007: the repeat_action_probability=0.25
value was never swept, so all conclusions about sticky actions may be
specific to this value.

Usage:
    python sticky_probability_sweep.py

Outputs:
    recordings/sticky_sweep_results.csv  — per-(model, p) metrics
    recordings/sticky_sweep_scores.csv   — per-game scores for plotting
"""
import os
import csv
import time
import numpy as np
from datetime import datetime

import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

# --- Configuration ---
MODELS = {
    "PPO_30b": "models/PPO_30b/final_model",
    "PPO_31b": "models/PPO_31b/final_model",
}
STICKY_PROBABILITIES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
N_GAMES = 500
OUTPUT_DIR = "recordings"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "sticky_sweep_results.csv")
SCORES_PATH = os.path.join(OUTPUT_DIR, "sticky_sweep_scores.csv")
# --------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_eval(model, sticky_p, n_games):
    """Run n_games at a given sticky probability, return list of scores."""
    env_kwargs = {"repeat_action_probability": sticky_p}
    env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                          env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)

    obs = env.reset()
    scores = []
    episode = 0

    while episode < n_games:
        action, _ = model.predict(obs, deterministic=True)
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


def compute_metrics(scores):
    """Compute summary metrics from a list of scores."""
    arr = np.array(scores)
    n = len(arr)
    unique = len(set(scores))
    zero_count = int(np.sum(arr == 0))

    return {
        "n_games": n,
        "unique_scores": unique,
        "avg": np.mean(arr),
        "median": np.median(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "zero_count": zero_count,
        "zero_rate": zero_count / n * 100,
        "p5": np.percentile(arr, 5),
        "p10": np.percentile(arr, 10),
        "p25": np.percentile(arr, 25),
        "p75": np.percentile(arr, 75),
        "p90": np.percentile(arr, 90),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
    }


def main():
    print("=" * 65)
    print("Sticky Probability Sweep")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Probabilities: {STICKY_PROBABILITIES}")
    print(f"Games per (model, p): {N_GAMES}")
    print(f"Total games: {len(MODELS) * len(STICKY_PROBABILITIES) * N_GAMES}")
    print("=" * 65)

    # Write headers
    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "sticky_p", "n_games", "unique_scores",
            "avg", "median", "std", "min", "max",
            "zero_count", "zero_rate",
            "p5", "p10", "p25", "p75", "p90", "p95", "p99",
            "timestamp"
        ])

    with open(SCORES_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "sticky_p", "game", "score", "timestamp"])

    all_metrics = {}

    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path + ".zip"):
            print(f"\nSKIPPING {model_name}: model not found at {model_path}.zip")
            continue

        print(f"\n{'-' * 50}")
        print(f"Loading {model_name} from {model_path}...")
        model = PPO.load(model_path)
        all_metrics[model_name] = {}

        for p in STICKY_PROBABILITIES:
            label = f"{model_name} p={p:.2f}"
            print(f"  {label}...", end=" ", flush=True)
            start = time.time()

            scores = run_eval(model, p, N_GAMES)
            metrics = compute_metrics(scores)
            all_metrics[model_name][p] = metrics

            elapsed = time.time() - start
            print(f"{len(scores)} games in {elapsed:.0f}s | "
                  f"avg={metrics['avg']:.1f} | unique={metrics['unique_scores']} | "
                  f"zero={metrics['zero_rate']:.1f}% | best={metrics['max']:.0f}")

            # Write results
            with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name, f"{p:.2f}", metrics["n_games"],
                    metrics["unique_scores"],
                    round(metrics["avg"], 1), round(metrics["median"], 1),
                    round(metrics["std"], 1), metrics["min"], metrics["max"],
                    metrics["zero_count"], round(metrics["zero_rate"], 2),
                    metrics["p5"], metrics["p10"], metrics["p25"],
                    metrics["p75"], metrics["p90"], metrics["p95"], metrics["p99"],
                    datetime.now().isoformat()
                ])

            # Write per-game scores
            with open(SCORES_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                timestamp = datetime.now().isoformat()
                for i, s in enumerate(scores):
                    writer.writerow([model_name, f"{p:.2f}", i + 1, int(s), timestamp])

    # --- Summary Table ---
    print("\n" + "=" * 65)
    print("SUMMARY: Score by Sticky Probability")
    print("=" * 65)

    for model_name in all_metrics:
        print(f"\n{model_name}:")
        print(f"{'p':>6}  {'avg':>7}  {'med':>6}  {'zero%':>7}  "
              f"{'uniq':>5}  {'P95':>6}  {'P99':>6}  {'best':>6}")
        print(f"{'-' * 60}")
        for p in STICKY_PROBABILITIES:
            if p in all_metrics[model_name]:
                m = all_metrics[model_name][p]
                print(f"{p:>4.2f}  {m['avg']:>7.1f}  {m['median']:>6.0f}  "
                      f"{m['zero_rate']:>6.1f}%  {m['unique_scores']:>5}  "
                      f"{m['p95']:>6.0f}  {m['p99']:>6.0f}  {m['max']:>6.0f}")

    # --- Key Finding ---
    print(f"\n{'-' * 50}")
    print("INTERPRETATION")
    print(f"{'-' * 50}")

    for model_name in all_metrics:
        p0 = all_metrics[model_name].get(0.0, {})
        p25 = all_metrics[model_name].get(0.25, {})
        if p0 and p25:
            print(f"\n{model_name} p=0.0 vs p=0.25:")
            print(f"  Unique scores: {p0['unique_scores']} -> {p25['unique_scores']}")
            print(f"  Average:       {p0['avg']:.1f} -> {p25['avg']:.1f}")
            print(f"  Zero-rate:     {p0['zero_rate']:.1f}% -> {p25['zero_rate']:.1f}%")
            print(f"  P99:           {p0['p99']:.0f} -> {p25['p99']:.0f}")

            if p0['unique_scores'] <= 2 and p25['unique_scores'] > 2:
                print(f"  -> Collapses without sticky, variance returns with sticky. "
                      f"Consistent with memorized policy + noise.")
            elif p0['unique_scores'] <= 2 and p25['unique_scores'] <= 2:
                print(f"  -> Collapses regardless of sticky setting.")
            else:
                print(f"  -> Survives without sticky — genuinely reactive policy.")

    print(f"\nFull results: {os.path.abspath(RESULTS_PATH)}")
    print(f"Per-game scores: {os.path.abspath(SCORES_PATH)}")


if __name__ == "__main__":
    main()
