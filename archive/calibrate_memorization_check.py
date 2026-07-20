"""
Calibrate the MemorizationCheckCallback's MEMORIZED/GENERALIZING threshold
for sticky-action environments.

The callback classifies models as MEMORIZED when <=2 unique scores appear
across 20 games. This threshold was calibrated for non-sticky environments,
where a dead policy produces exactly 1 unique score deterministically.

With repeat_action_probability=0.25, even a completely memorized fixed-action
policy will produce varied scores because 25% of steps are randomly repeated.
This script measures how many unique scores that noise alone produces, using
PPO_30a/final_model (confirmed MEMORIZED — 1-2 unique scores in all 10 checks
across 100M non-sticky steps, per recordings/PPO_30a_memorization_track.csv).

Output: a CSV at recordings/memorization_calibration.csv and a printed
summary showing the expected unique-score distribution from sticky noise,
so the Phase 2 GENERALIZING verdicts can be evaluated against a real baseline.

Usage:
    python calibrate_memorization_check.py [--model PATH] [--repetitions N]
"""
import os
import sys
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
MODEL_PATH = "models/PPO_30a/final_model"  # confirmed MEMORIZED in non-sticky eval
N_REPETITIONS = 20          # number of 20-game check batches (reduced from 50)
N_GAMES_PER_CHECK = 20      # matches MemorizationCheckCallback.n_games
MAX_STEPS_PER_GAME = 50000  # safety cap: ~14 min at 60fps, prevents runaway games
STICKY_VALUES = [False, True]  # test both for comparison
OUTPUT_DIR = "recordings"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "memorization_calibration.csv")
# --------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_check_batch(model, sticky, n_games):
    """Run one batch of n_games and return list of scores."""
    env_kwargs = {"repeat_action_probability": 0.25 if sticky else 0.0}
    env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                          env_kwargs=env_kwargs)
    env = VecFrameStack(env, n_stack=4)

    obs = env.reset()
    scores = []
    episode = 0
    steps_in_game = 0

    while episode < n_games:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        steps_in_game += 1

        if done[0]:
            lives = info[0].get("lives", -1)
            if lives == 0:
                score = float(info[0].get("episode", {}).get("r", 0))
                scores.append(score)
                episode += 1
                obs = env.reset()
                steps_in_game = 0
            else:
                obs, _, _, _ = env.step([0])
        elif steps_in_game >= MAX_STEPS_PER_GAME:
            # Safety cap: force-end runaway game
            score = float(info[0].get("episode", {}).get("r", 0))
            scores.append(score)
            episode += 1
            obs = env.reset()
            steps_in_game = 0

    env.close()
    return scores


def summarize(scores_list):
    """Compute per-batch unique-score stats from list of score arrays."""
    unique_counts = [len(set(batch)) for batch in scores_list]
    return {
        "mean_unique": np.mean(unique_counts),
        "std_unique": np.std(unique_counts),
        "min_unique": np.min(unique_counts),
        "max_unique": np.max(unique_counts),
        "p50_unique": np.percentile(unique_counts, 50),
        "p90_unique": np.percentile(unique_counts, 90),
        "p95_unique": np.percentile(unique_counts, 95),
        "p99_unique": np.percentile(unique_counts, 99),
    }


def main():
    # Parse optional args
    model_path = MODEL_PATH
    n_repetitions = N_REPETITIONS
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model_path = args[i + 1]; i += 2
        elif args[i] == "--repetitions" and i + 1 < len(args):
            n_repetitions = int(args[i + 1]); i += 2
        else:
            i += 1

    print("=" * 65)
    print("MemorizationCheckCallback Calibration")
    print(f"Model: {model_path} (confirmed MEMORIZED in non-sticky eval)")
    print(f"Repetitions: {n_repetitions} x {N_GAMES_PER_CHECK}-game batches")
    print(f"Testing sticky=False and sticky=True")
    print("=" * 65)

    # Load memorized model
    if not os.path.exists(model_path + ".zip"):
        print(f"\nERROR: Model not found at {model_path}.zip")
        print("Any confirmed-memorized model will work (e.g., PPO_30a/final_model).")
        return

    model = PPO.load(model_path)

    # Write CSV header
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sticky", "batch", "n_games", "unique_scores",
            "avg_score", "best_score", "worst_score", "scores_list",
            "timestamp"
        ])

    results = {}

    for sticky in STICKY_VALUES:
        label = "sticky=True (p=0.25)" if sticky else "sticky=False (p=0.0)"
        print(f"\n{'-' * 50}")
        print(f"Testing {label}")
        print(f"{'-' * 50}")

        all_batches = []
        start_time = time.time()

        for i in range(n_repetitions):
            scores = run_check_batch(model, sticky, N_GAMES_PER_CHECK)
            all_batches.append(scores)
            unique = len(set(scores))
            avg = np.mean(scores)
            best = max(scores)
            worst = min(scores)

            # Append to CSV
            with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    sticky, i + 1, N_GAMES_PER_CHECK, unique,
                    round(avg, 1), best, worst,
                    ",".join(str(int(s)) for s in sorted(scores)),
                    datetime.now().isoformat()
                ])

            # Progress every 10 batches
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) * N_GAMES_PER_CHECK / elapsed
                print(f"  Batch {i+1:>3}/{n_repetitions} | "
                      f"unique={unique} | avg={avg:.1f} | "
                      f"best={best:.0f} | worst={worst:.0f} | "
                      f"{rate:.0f} games/sec")

        elapsed = time.time() - start_time
        stats = summarize(all_batches)
        results[sticky] = stats

        print(f"\n  Completed {n_repetitions} batches in {elapsed:.0f}s "
              f"({n_repetitions * N_GAMES_PER_CHECK / elapsed:.0f} games/sec)")

    # --- Print calibration summary ---
    print("\n" + "=" * 65)
    print("CALIBRATION RESULTS")
    print("=" * 65)

    for sticky in STICKY_VALUES:
        label = "sticky=True (p=0.25)" if sticky else "sticky=False (p=0.0)"
        s = results[sticky]
        print(f"\n{label}:")
        print(f"  Unique scores per {N_GAMES_PER_CHECK}-game batch:")
        print(f"    Mean:  {s['mean_unique']:.1f}")
        print(f"    Std:   {s['std_unique']:.1f}")
        print(f"    Min:   {s['min_unique']:.0f}")
        print(f"    Max:   {s['max_unique']:.0f}")
        print(f"    P50:   {s['p50_unique']:.0f}")
        print(f"    P90:   {s['p90_unique']:.0f}")
        print(f"    P95:   {s['p95_unique']:.0f}")
        print(f"    P99:   {s['p99_unique']:.0f}")

    # --- Interpretation ---
    sticky_stats = results[True]
    nonsticky_stats = results[False]

    print(f"\n{'-' * 50}")
    print("INTERPRETATION")
    print(f"{'-' * 50}")

    print(f"\nNon-sticky baseline: a dead policy produces "
          f"{nonsticky_stats['mean_unique']:.1f} unique scores on average "
          f"(max {nonsticky_stats['max_unique']:.0f}).")

    print(f"\nSticky baseline: the SAME dead policy with p=0.25 sticky "
          f"produces {sticky_stats['mean_unique']:.1f} unique scores on average "
          f"(range {sticky_stats['min_unique']:.0f}-{sticky_stats['max_unique']:.0f}).")

    # Compare against PPO_30b/31b observed ranges
    ppo30b_min = 10  # from CSV: unique scores range 10-19
    ppo30b_max = 19
    ppo31b_min = 10
    ppo31b_max = 19

    p95 = sticky_stats['p95_unique']
    print(f"\nSticky-noise P95: {p95:.0f} unique scores in a 20-game batch.")
    print(f"PPO_30b observed range: {ppo30b_min}-{ppo30b_max} unique scores.")
    print(f"PPO_31b observed range: {ppo31b_min}-{ppo31b_max} unique scores.")

    if ppo30b_min <= p95:
        print(f"\n[!] WARNING: PPO_30b's minimum unique-score count ({ppo30b_min}) "
              f"is within the sticky-noise baseline (P95={p95:.0f}).")
        print(f"    The GENERALIZING verdict for PPO_30b is NOT reliably "
              f"distinguishable from a dead policy + sticky noise.")
    else:
        print(f"\n[OK]PPO_30b's unique-score counts exceed the sticky-noise P95. "
              f"Generalization is distinguishable from noise.")

    if ppo31b_min <= p95:
        print(f"\n[!] WARNING: PPO_31b's minimum unique-score count ({ppo31b_min}) "
              f"is within the sticky-noise baseline (P95={p95:.0f}).")
        print(f"    The GENERALIZING verdict for PPO_31b is NOT reliably "
              f"distinguishable from a dead policy + sticky noise.")
    else:
        print(f"\n[OK]PPO_31b's unique-score counts exceed the sticky-noise P95. "
              f"Generalization is distinguishable from noise.")

    print(f"\nRecommended calibrated MEMORIZED threshold for sticky models: "
          f"<={int(p95)} unique scores (P95 of noise distribution).")
    print(f"Previous threshold (<=2) was calibrated for non-sticky and is "
          f"inappropriate for sticky-action evaluation.")

    print(f"\nFull results saved to: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
