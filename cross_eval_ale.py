"""
Cross-evaluation: GymBreakout-trained model on ALE/Breakout-v5.

Quantifies the GymBreakout→ALE transfer gap — the central unvalidated risk
to all post-Experiment-4 conclusions (LOGICAL_AUDIT.md L-007).

Two evaluations:
  1. ALE/Breakout-v5 (frameskip=1, nosticky) — authentic Atari 2600
  2. GymBreakout(fixed=True) — custom engine, standard defaults

Both use identical preprocessing (84×84 grayscale, ClipReward, VecFrameStack=4)
and identical inference protocol (det=True + det=False, 100 games each).

Usage:
    python cross_eval_ale.py                          # PPO_35 best model
    python cross_eval_ale.py --run PPO_36             # PPO_36 best model
    python cross_eval_ale.py --run PPO_35 --games 200 # custom game count
    python cross_eval_ale.py --run PPO_35 --ale-only  # skip GymBreakout eval
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

import ale_py
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv

from gym_breakout import GymBreakout

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Observation preprocessing — identical for both engines
# ---------------------------------------------------------------------------

class GrayscaleResize(gym.ObservationWrapper):
    """Resize to (84, 84, 1) grayscale — matches train_ppo35.py eval pipeline."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[2] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif obs.ndim == 2:
            pass  # already grayscale
        resized = cv2.resize(obs, (self._width, self._height),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None] if resized.ndim == 2 else resized


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------

def make_ale_env():
    """ALE/Breakout-v5: authentic Atari, frameskip=1, no sticky actions."""
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_gymbreakout_env():
    """GymBreakout(fixed=True): custom engine, standard defaults."""
    env = GymBreakout(fixed=True)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episodes(model, env, n_games, deterministic, label=""):
    """Run n_games episodes and return scores."""
    scores = []
    obs = env.reset()

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
                if episode % 50 == 0:
                    sys.stdout.write(f"  [{label}] {episode}/{n_games} games...\n")
                    sys.stdout.flush()
                obs = env.reset()
            else:
                # Life lost — fire to respawn
                obs, _, _, _ = env.step([1])

    return scores


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_stats(scores):
    """Compute summary statistics for a score list."""
    if not scores:
        return {"unique": 0, "mean": 0, "median": 0, "std": 0, "min": 0, "max": 0}

    arr = np.array(scores, dtype=float)
    counter = Counter(scores)
    unique = len(counter)
    top3_pct = sum(c for _, c in counter.most_common(3)) / len(scores) * 100
    singletons = sum(1 for _, c in counter.items() if c == 1)
    singleton_ratio = singletons / unique if unique > 0 else 0
    zero_pct = 100 * counter.get(0, 0) / len(scores)

    # 95% bootstrap CI for the mean (10k resamples)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(10000):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(sample.mean()))
    means.sort()
    ci95 = (means[250], means[9750])

    return {
        "n_games": len(scores),
        "unique": unique,
        "mean": round(float(arr.mean()), 1),
        "median": round(float(np.median(arr)), 1),
        "std": round(float(arr.std()), 1),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "top3_pct": round(top3_pct, 1),
        "singleton_ratio": round(singleton_ratio, 3),
        "zero_score_pct": round(zero_pct, 1),
        "mean_ci95": [round(ci95[0], 1), round(ci95[1], 1)],
        "scores": [int(s) for s in scores],  # raw scores for later analysis
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-evaluate GymBreakout-trained model on ALE vs GymBreakout")
    parser.add_argument("--run", default="PPO_35",
                        help="Run name (default: PPO_35)")
    parser.add_argument("--model", default=None,
                        help="Model path (default: models/<RUN>/best_model.zip)")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games per mode (default: 100)")
    parser.add_argument("--ale-only", action="store_true",
                        help="Skip GymBreakout evaluation (ALE only)")
    parser.add_argument("--gym-only", action="store_true",
                        help="Skip ALE evaluation (GymBreakout only)")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (default: cuda)")
    args = parser.parse_args()

    RUN = args.run
    MODEL_PATH = args.model or f"models/{RUN}/best_model.zip"
    NUM_GAMES = args.games

    # Resolve model path relative to the ORIGINAL repo (not the worktree)
    # This script lives in the worktree, but models are in the original repo.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Walk up from worktree to original repo
    REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
    if not os.path.isabs(MODEL_PATH):
        MODEL_PATH = os.path.join(REPO_ROOT, MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print(f"  Script dir: {SCRIPT_DIR}")
        print(f"  Repo root:  {REPO_ROOT}")
        available = []
        models_dir = os.path.join(REPO_ROOT, "models")
        if os.path.isdir(models_dir):
            for d in os.listdir(models_dir):
                for fname in ["best_model.zip", "final_model.zip"]:
                    p = os.path.join(models_dir, d, fname)
                    if os.path.exists(p):
                        available.append(p)
        print("Available models:")
        for p in sorted(available):
            print(f"  {p}")
        sys.exit(1)

    print(f"Cross-Evaluation: {RUN} -> ALE + GymBreakout")
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Games:    {NUM_GAMES} per mode per environment")
    print(f"  Device:   {args.device}")
    print(f"  ALE:      frameskip=1, repeat_action_probability=0 (nosticky)")
    print(f"  GB:       GymBreakout(fixed=True), standard defaults")
    sys.stdout.flush()

    # Load model once
    print(f"\nLoading model...")
    sys.stdout.flush()
    model = PPO.load(MODEL_PATH, device=args.device, custom_objects={"n_envs": 1})
    print(f"Loaded. {model.num_timesteps:,} training steps.")
    sys.stdout.flush()

    results = {
        "run": RUN,
        "model_path": MODEL_PATH,
        "training_steps": int(model.num_timesteps),
        "n_games_per_mode": NUM_GAMES,
        "timestamp": datetime.now().isoformat(),
        "ale": {},
        "gymbreakout": {},
    }

    # -----------------------------------------------------------------------
    # ALE evaluation
    # -----------------------------------------------------------------------
    if not args.gym_only:
        print(f"\n{'='*60}")
        print(f"  ALE/Breakout-v5 -- authentic Atari 2600")
        print(f"{'='*60}")

        # det=True
        print(f"\n[ALE det=True]")
        sys.stdout.flush()
        ale_env = DummyVecEnv([make_ale_env])
        ale_env = VecFrameStack(ale_env, n_stack=4)
        model.set_env(ale_env)
        t0 = time.time()
        det_scores = run_episodes(model, ale_env, NUM_GAMES, deterministic=True,
                                  label="ALE det=True")
        det_elapsed = time.time() - t0
        results["ale"]["det_true"] = compute_stats(det_scores)
        results["ale"]["det_true"]["elapsed_s"] = round(det_elapsed, 1)
        ale_env.close()

        # det=False
        print(f"\n[ALE det=False]")
        sys.stdout.flush()
        ale_env = DummyVecEnv([make_ale_env])
        ale_env = VecFrameStack(ale_env, n_stack=4)
        model.set_env(ale_env)
        t0 = time.time()
        stoch_scores = run_episodes(model, ale_env, NUM_GAMES, deterministic=False,
                                    label="ALE det=False")
        stoch_elapsed = time.time() - t0
        results["ale"]["det_false"] = compute_stats(stoch_scores)
        results["ale"]["det_false"]["elapsed_s"] = round(stoch_elapsed, 1)
        ale_env.close()

    # -----------------------------------------------------------------------
    # GymBreakout evaluation
    # -----------------------------------------------------------------------
    if not args.ale_only:
        print(f"\n{'='*60}")
        print(f"  GymBreakout(fixed=True) -- custom engine")
        print(f"{'='*60}")

        # det=True
        print(f"\n[GymBreakout det=True]")
        sys.stdout.flush()
        gb_env = DummyVecEnv([make_gymbreakout_env])
        gb_env = VecFrameStack(gb_env, n_stack=4)
        model.set_env(gb_env)
        t0 = time.time()
        det_scores = run_episodes(model, gb_env, NUM_GAMES, deterministic=True,
                                  label="GB det=True")
        det_elapsed = time.time() - t0
        results["gymbreakout"]["det_true"] = compute_stats(det_scores)
        results["gymbreakout"]["det_true"]["elapsed_s"] = round(det_elapsed, 1)
        gb_env.close()

        # det=False
        print(f"\n[GymBreakout det=False]")
        sys.stdout.flush()
        gb_env = DummyVecEnv([make_gymbreakout_env])
        gb_env = VecFrameStack(gb_env, n_stack=4)
        model.set_env(gb_env)
        t0 = time.time()
        stoch_scores = run_episodes(model, gb_env, NUM_GAMES, deterministic=False,
                                    label="GB det=False")
        stoch_elapsed = time.time() - t0
        results["gymbreakout"]["det_false"] = compute_stats(stoch_scores)
        results["gymbreakout"]["det_false"]["elapsed_s"] = round(stoch_elapsed, 1)
        gb_env.close()

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  CROSS-EVALUATION RESULTS: {RUN}")
    print(f"{'='*60}")
    print()

    if not args.gym_only and not args.ale_only:
        # Side-by-side comparison
        ale_dt = results["ale"]["det_true"]
        ale_df = results["ale"]["det_false"]
        gb_dt = results["gymbreakout"]["det_true"]
        gb_df = results["gymbreakout"]["det_false"]

        header = f"{'Metric':<25} {'ALE det=T':>12} {'ALE det=F':>12} {'GB det=T':>12} {'GB det=F':>12}"
        print(header)
        print("-" * len(header))

        rows = [
            ("Unique scores", ale_dt["unique"], ale_df["unique"], gb_dt["unique"], gb_df["unique"]),
            ("Mean", ale_dt["mean"], ale_df["mean"], gb_dt["mean"], gb_df["mean"]),
            ("Std", ale_dt["std"], ale_df["std"], gb_dt["std"], gb_df["std"]),
            ("Min", ale_dt["min"], ale_df["min"], gb_dt["min"], gb_df["min"]),
            ("Max", ale_dt["max"], ale_df["max"], gb_dt["max"], gb_df["max"]),
            ("Top-3%", ale_dt["top3_pct"], ale_df["top3_pct"], gb_dt["top3_pct"], gb_df["top3_pct"]),
            ("Singleton ratio", ale_dt["singleton_ratio"], ale_df["singleton_ratio"], gb_dt["singleton_ratio"], gb_df["singleton_ratio"]),
            ("Zero-score%", ale_dt["zero_score_pct"], ale_df["zero_score_pct"], gb_dt["zero_score_pct"], gb_df["zero_score_pct"]),
        ]

        for label, a_dt, a_df, g_dt, g_df in rows:
            if isinstance(a_dt, float):
                print(f"{label:<25} {a_dt:>12.1f} {a_df:>12.1f} {g_dt:>12.1f} {g_df:>12.1f}")
            else:
                print(f"{label:<25} {a_dt:>12} {a_df:>12} {g_dt:>12} {g_df:>12}")

        print()
        print(f"{'95% CI (mean):':<25} "
              f"({ale_dt['mean_ci95'][0]}-{ale_dt['mean_ci95'][1]}) "
              f"({ale_df['mean_ci95'][0]}-{ale_df['mean_ci95'][1]}) "
              f"({gb_dt['mean_ci95'][0]}-{gb_dt['mean_ci95'][1]}) "
              f"({gb_df['mean_ci95'][0]}-{gb_df['mean_ci95'][1]})")

        # Transfer gap summary
        print()
        print(f"--- Transfer Gap ---")
        det_mean_diff = gb_dt["mean"] - ale_dt["mean"]
        stoch_mean_diff = gb_df["mean"] - ale_df["mean"]
        det_unique_diff = gb_dt["unique"] - ale_dt["unique"]
        stoch_unique_diff = gb_df["unique"] - ale_df["unique"]

        print(f"  det=True  mean delta:  {det_mean_diff:+.1f}  (GB - ALE)")
        print(f"  det=False mean delta:  {stoch_mean_diff:+.1f}  (GB - ALE)")
        print(f"  det=True  unique delta: {det_unique_diff:+}")
        print(f"  det=False unique delta: {stoch_unique_diff:+}")

        if abs(det_mean_diff) > gb_dt["mean"] * 0.5:
            print(f"  [!] LARGE TRANSFER GAP -- GymBreakout findings may not transfer to ALE.")
        elif abs(det_mean_diff) > gb_dt["mean"] * 0.2:
            print(f"  [!] MODERATE TRANSFER GAP -- interpret GymBreakout results with caution.")
        else:
            print(f"  Transfer gap within 20% -- reasonable agreement between engines.")

        if ale_dt["unique"] <= 2:
            print(f"  [!] ALE det=True: only {ale_dt['unique']} unique scores (memorized-like)")
        if ale_df["unique"] <= 2:
            print(f"  [!] ALE det=False: only {ale_df['unique']} unique scores (no entropy)")

    elif not args.gym_only:
        # ALE-only output
        for mode, label in [("det_true", "det=True"), ("det_false", "det=False")]:
            s = results["ale"][mode]
            print(f"  ALE {label}: unique={s['unique']}, mean={s['mean']}, "
                  f"std={s['std']}, range=[{s['min']},{s['max']}], "
                  f"top3={s['top3_pct']}%")

    elif not args.ale_only:
        # GymBreakout-only output
        for mode, label in [("det_true", "det=True"), ("det_false", "det=False")]:
            s = results["gymbreakout"][mode]
            print(f"  GB {label}:  unique={s['unique']}, mean={s['mean']}, "
                  f"std={s['std']}, range=[{s['min']},{s['max']}], "
                  f"top3={s['top3_pct']}%")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    out_path = os.path.join(SCRIPT_DIR, f"cross_eval_{RUN}_results.json")
    # Strip raw scores to keep JSON compact
    compact = {}
    for env_key in ["ale", "gymbreakout"]:
        if results[env_key]:
            compact[env_key] = {}
            for mode_key, stats in results[env_key].items():
                compact[env_key][mode_key] = {k: v for k, v in stats.items() if k != "scores"}
    results_compact = {**results, "ale": compact.get("ale", {}), "gymbreakout": compact.get("gymbreakout", {})}
    # Put back the full data for the raw scores
    results_compact["ale"] = results.get("ale", {})
    results_compact["gymbreakout"] = results.get("gymbreakout", {})

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(datetime.now().isoformat())
    sys.stdout.flush()
