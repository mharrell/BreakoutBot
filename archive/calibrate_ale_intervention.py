"""
Step 2 — ALE Dead-Model Calibration

Establishes the ALE dead-model baseline for intervention test retention.
Runs the intervention test on PPO_26 (ALE-trained, confirmed memorized).

Test protocol:
  - Normal:   ALE/Breakout-v5, fs=1, nosticky, NoopReset + FireReset +
              EpisodicLife + GrayscaleResize + ClipReward + Monitor,
              det=True, 100 games
  - Interven: Same + ALEBreakoutRandomized(teleport_prob=0.30),
              det=True, 100 games, interleaved pairwise

Pipeline: gym.make(ALE) -> NoopResetEnv -> [ALEBreakoutRandomized] ->
          FireResetEnv -> EpisodicLifeEnv -> GrayscaleResize ->
          ClipRewardEnv -> Monitor -> DummyVecEnv[1] -> VecFrameStack(4)

EpisodicLifeEnv ends episode on first life loss — standard Atari convention.
Same pipeline used for BOTH calibration and PPO_44 training — apples-to-apples.

This gives: "A known-dead ALE model retains X% score under intervention."
Without this, intervention retention numbers are uninterpretable (L-001).

Usage:
    python calibrate_ale_intervention.py
    python calibrate_ale_intervention.py --games 200
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
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv, NoopResetEnv,
                                                      FireResetEnv, EpisodicLifeEnv)

from ale_breakout_randomized import ALEBreakoutRandomized

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Observation preprocessing (matches PPO_44 training pipeline)
# ---------------------------------------------------------------------------

class GrayscaleResize(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[2] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(obs, (self._width, self._height),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None] if resized.ndim == 2 else resized


# ---------------------------------------------------------------------------
# Environment builder — shared pipeline
# ---------------------------------------------------------------------------

def build_env(teleport=False, teleport_prob=0.30):
    """Build a single ALE env with the standard pipeline.

    gym.make(ALE) -> NoopResetEnv -> [ALEBreakoutRandomized] ->
    FireResetEnv -> EpisodicLifeEnv -> GrayscaleResize -> ClipRewardEnv -> Monitor

    Args:
        teleport: If True, insert ALEBreakoutRandomized in the chain.
        teleport_prob: Teleport probability (if teleport=True).
    """
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    if teleport:
        env = ALEBreakoutRandomized(env, teleport_prob=teleport_prob)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_vec_env(teleport=False, teleport_prob=0.30):
    """Build a VecFrameStack-wrapped single env for inference."""
    env = build_env(teleport=teleport, teleport_prob=teleport_prob)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def find_model(run_name):
    repo_root = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    search_roots = [
        f"./models/{run_name}",
        os.path.join(repo_root, "models", run_name),
    ]
    for root in search_roots:
        for fname in ["best_model.zip", "final_model.zip"]:
            p = os.path.join(root, fname)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"No model found for {run_name}")


def load_model(run_name, device="cuda"):
    path = find_model(run_name)
    print(f"  Loading {path} ({device})...")
    sys.stdout.flush()
    return PPO.load(path, device=device)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, alpha=0.05):
    arr = np.array(data)
    n = len(arr)
    rng = np.random.default_rng(42)
    bootstrapped = np.array([
        statistic(rng.choice(arr, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(bootstrapped, 100 * alpha / 2)
    upper = np.percentile(bootstrapped, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Single game runner
# ---------------------------------------------------------------------------

def run_game(model, env, max_steps=50000):
    """Run a single episode to completion.

    With EpisodicLifeEnv, episode ends on first life loss.
    Monitor tracks score — read from info on done.

    Returns (score, n_interventions).
    """
    obs = env.reset()
    steps = 0
    while steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, info = env.step(action)
        steps += 1
        if done_arr[0]:
            score = float(info[0].get("episode", {}).get("r", 0))
            # Count interventions
            e = env.envs[0]
            n_int = 0
            while e is not None:
                if isinstance(e, ALEBreakoutRandomized):
                    n_int = e.intervention_count
                    break
                e = getattr(e, 'env', None)
            return score, n_int
    return 0.0, 0


# ---------------------------------------------------------------------------
# Intervention test
# ---------------------------------------------------------------------------

def run_intervention_test(run_name, n_games=100, device="cuda"):
    print(f"\n{'='*60}")
    print(f"  ALE Intervention Test: {run_name}")
    print(f"  {n_games} paired games, teleport_prob=0.30")
    print(f"  Pipeline: NoopReset -> [Teleport] -> FireReset -> Gray -> Clip -> Monitor")
    print(f"{'='*60}")
    sys.stdout.flush()

    model = load_model(run_name, device=device)

    normal_scores = []
    inter_scores = []
    all_interventions = []

    for i in range(n_games):
        # Normal game
        env_n = make_vec_env(teleport=False)
        sn, _ = run_game(model, env_n)
        normal_scores.append(sn)
        env_n.close()

        # Intervention game
        env_i = make_vec_env(teleport=True, teleport_prob=0.30)
        si, n_int = run_game(model, env_i)
        inter_scores.append(si)
        all_interventions.append(n_int)
        env_i.close()

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_games}] Normal={np.mean(normal_scores):.1f}, "
                  f"Inter={np.mean(inter_scores):.1f}, "
                  f"Avg int/game={np.mean(all_interventions):.1f}")
            sys.stdout.flush()

    normal = np.array(normal_scores)
    inter = np.array(inter_scores)
    deltas = normal - inter
    n_int = np.array(all_interventions)

    corr = (np.corrcoef(normal, inter)[0, 1]
            if normal.std() > 0 and inter.std() > 0 else 0)

    retention_pct = (inter.mean() / normal.mean() * 100) if normal.mean() > 0 else 0

    result = {
        "run": run_name,
        "n_games": n_games,
        "frameskip": 1,
        "teleport_prob": 0.30,
        "timestamp": datetime.now().isoformat(),
        "normal_mean": float(normal.mean()),
        "normal_median": float(np.median(normal)),
        "normal_std": float(normal.std()),
        "normal_unique": len(set(normal_scores)),
        "normal_min": int(normal.min()),
        "normal_max": int(normal.max()),
        "intervention_mean": float(inter.mean()),
        "intervention_median": float(np.median(inter)),
        "intervention_std": float(inter.std()),
        "intervention_unique": len(set(inter_scores)),
        "intervention_min": int(inter.min()),
        "intervention_max": int(inter.max()),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "retention_pct": round(retention_pct, 1),
        "correlation": round(corr, 3),
        "avg_interventions_per_game": float(n_int.mean()),
        "normal_mean_ci95": bootstrap_ci(normal_scores, np.mean),
        "intervention_mean_ci95": bootstrap_ci(inter_scores, np.mean),
        "normal_scores": [int(s) for s in normal_scores],
        "intervention_scores": [int(s) for s in inter_scores],
        "intervention_counts": [int(c) for c in all_interventions],
    }

    print(f"\n  RESULTS for {run_name}:")
    print(f"    Normal:       mean={result['normal_mean']:.1f} "
          f"med={result['normal_median']:.0f} std={result['normal_std']:.1f} "
          f"unique={result['normal_unique']} range=[{result['normal_min']},{result['normal_max']}]")
    print(f"    Intervention: mean={result['intervention_mean']:.1f} "
          f"med={result['intervention_median']:.0f} std={result['intervention_std']:.1f} "
          f"unique={result['intervention_unique']} range=[{result['intervention_min']},{result['intervention_max']}]")
    print(f"    Retention:    {retention_pct:.1f}%")
    print(f"    Correlation:  {corr:.3f}")
    print(f"    Delta:        mean={deltas.mean():.1f} std={deltas.std():.1f}")
    print(f"    Int/game:     {n_int.mean():.1f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ALE dead-model intervention calibration (Step 2)")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--runs", nargs="*", default=["PPO_26"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="calibration_ale_results.json")
    args = parser.parse_args()

    all_results = {}

    for run in args.runs:
        try:
            result = run_intervention_test(run, n_games=args.games, device=args.device)
            all_results[run] = result
        except FileNotFoundError as e:
            print(f"  SKIP {run}: {e}")
            continue

    # Save (strip raw scores)
    compact = {}
    for run, r in all_results.items():
        compact[run] = {k: v for k, v in r.items()
                        if k not in ("normal_scores", "intervention_scores",
                                     "intervention_counts")}
    with open(args.output, "w") as f:
        json.dump(compact, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    raw_path = args.output.replace(".json", "_raw.json")
    raw_data = {}
    for run, r in all_results.items():
        raw_data[run] = {
            "normal_scores": r.get("normal_scores", []),
            "intervention_scores": r.get("intervention_scores", []),
            "intervention_counts": r.get("intervention_counts", []),
        }
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw scores saved to {raw_path}")

    print(f"\n{'='*60}")
    print(f"  ALE DEAD-MODEL CALIBRATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Run':<12s} {'Normal':>8s} {'Inter':>8s} {'Retention':>10s} "
          f"{'Corr':>8s} {'Int/Game':>10s}")
    print(f"  {'-'*56}")
    for run, r in all_results.items():
        print(f"  {run:<12s} {r['normal_mean']:>8.1f} {r['intervention_mean']:>8.1f} "
              f"{r['retention_pct']:>9.1f}% {r['correlation']:>8.3f} "
              f"{r['avg_interventions_per_game']:>10.1f}")
    print(f"\n  Calibration baseline for ALE.")
    print(f"  If PPO_44 retention ~= this value: teleport alone insufficient.")
