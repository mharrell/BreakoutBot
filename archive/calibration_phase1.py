"""
Phase 1 Calibration Script — Logical Audit (2026-07-19)

Runs the tests that should have been run before breakthrough claims were written:

  1a. Intervention test on PPO_34 (confirmed argmax script) → dead-script baseline
  1b. Intervention test on PPO_35 (claimed-reactive) → fresh comparison data
  1c. Intervention test on PPO_36 (noise model) → noise-model comparison
  1d. Reactivity evaluation on PPO_34 → shape classifier calibration (known-dead)
  1e. Reactivity evaluation on PPO_35 → fresh comparison data
  1f. Reactivity evaluation on PPO_36 → noise-model comparison

All inference runs on CPU to avoid contending with GPU training runs.

Output: calibration_phase1_results.json with all raw data and computed metrics.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv

from gym_breakout import GymBreakout

# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------

class GrayscaleResize(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):
        resized = cv2.resize(obs, (self._width, self._height),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None]


class InterventionBreakout(gym.Wrapper):
    """After a paddle bounce, teleports ball (y-axis) and paddle (x-axis)."""

    def __init__(self, env, teleport_prob=0.3, ball_y_jitter=15,
                 paddle_x_jitter=30, seed=None):
        super().__init__(env)
        self.teleport_prob = teleport_prob
        self.ball_y_jitter = ball_y_jitter
        self.paddle_x_jitter = paddle_x_jitter
        self._rng = np.random.default_rng(seed)
        self.intervention_count = 0

    def reset(self, **kwargs):
        self.intervention_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        brk = self.env._env
        prev_vy = brk.ball_v[0] if brk else None
        obs, reward, terminated, truncated, info = self.env.step(action)
        cur_vy = brk.ball_v[0] if brk else None
        if (prev_vy is not None and cur_vy is not None
                and prev_vy > 0 and cur_vy < 0):
            if self._rng.random() < self.teleport_prob:
                ball = brk.ball
                paddle = brk.paddle
                dy = int(round(self._rng.normal(0, self.ball_y_jitter)))
                new_y = ball.pos[0] + dy
                new_y = max(15, min(paddle.pos[0] - 10, new_y))
                ball.pos[0] = new_y
                dx = int(round(self._rng.normal(0, self.paddle_x_jitter)))
                new_x = paddle.pos[1] + dx
                new_x = max(0, min(160 - paddle.size[1], new_x))
                paddle.pos[1] = new_x
                self.intervention_count += 1
        return obs, reward, terminated, truncated, info


def make_normal_env():
    env = GymBreakout(fixed=True)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_intervention_env(teleport_prob=0.3, ball_y_jitter=15, paddle_x_jitter=30):
    base = GymBreakout(fixed=True)
    env = InterventionBreakout(base, teleport_prob=teleport_prob,
                                ball_y_jitter=ball_y_jitter,
                                paddle_x_jitter=paddle_x_jitter)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env, base


# ---------------------------------------------------------------------------
# Model loading (CPU)
# ---------------------------------------------------------------------------

def find_model(run_name):
    """Find best_model.zip or a checkpoint for the given run."""
    # Check worktree first, then original repo
    search_roots = [
        f"./models/{run_name}",
        f"C:/Users/Silver Pangolin/PycharmProjects/breakoutBot/models/{run_name}",
    ]
    for root in search_roots:
        best = os.path.join(root, "best_model.zip")
        if os.path.exists(best):
            return best
        final = os.path.join(root, "final_model.zip")
        if os.path.exists(final):
            return final
    raise FileNotFoundError(f"No model found for {run_name}")


def load_model(run_name, device="cuda"):
    path = find_model(run_name)
    print(f"  Loading {path} ({device})...")
    return PPO.load(path, device=device)


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_episodes(model, vec_env, n_games, deterministic, max_frames=20000):
    """Run n_games episodes via model.predict. Returns list of scores."""
    scores = []
    obs = vec_env.reset()
    episode = 0
    frames = 0
    while episode < n_games and frames < max_frames * n_games:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done_arr, info = vec_env.step(action)
        frames += 1
        if done_arr[0]:
            lives = info[0].get("lives", -1)
            if lives == 0:
                score = float(info[0].get("episode", {}).get("r", 0))
                scores.append(score)
                episode += 1
                if episode % 50 == 0:
                    print(f"    [{episode}/{n_games}] games...")
                    sys.stdout.flush()
                obs = vec_env.reset()
            else:
                obs, _, _, _ = vec_env.step([1])
    return scores


def run_intervention_game(model, env, base_env, deterministic, max_frames=20000):
    """Run one game with intervention wrapper. Returns (score, n_interventions)."""
    obs = env.reset()
    brk = base_env._env
    done = False
    frames = 0
    ib = env.envs[0].env  # unwrap: VecFrameStack -> DummyVecEnv -> InterventionBreakout
    # Walk chain to find InterventionBreakout
    e = env.envs[0]
    while e is not None and not isinstance(e, InterventionBreakout):
        e = getattr(e, 'env', None)
    ib = e

    while not done and frames < max_frames:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done_arr, info = env.step(action)
        done = done_arr[0]
        frames += 1
    score = brk.score if brk else 0
    n_int = ib.intervention_count if ib else 0
    return score, n_int


# ---------------------------------------------------------------------------
# Distribution analysis (from eval_reactivity.py)
# ---------------------------------------------------------------------------

def analyze_distribution(scores):
    n = len(scores)
    unique = len(set(scores))
    counter = Counter(scores)
    top3_pct = sum(count for _, count in counter.most_common(3)) / n * 100
    singletons = sum(1 for _, count in counter.items() if count == 1)
    singleton_ratio = singletons / unique if unique > 0 else 0

    score_min, score_max = min(scores), max(scores)
    score_range = score_max - score_min
    gaps = 0
    max_gap = 0
    if score_range > 0:
        for s in range(int(score_min), int(score_max) + 1):
            if counter.get(s, 0) == 0:
                gaps += 1
                max_gap = max(max_gap, gaps)
            else:
                gaps = 0

    if top3_pct > 50 and singleton_ratio < 0.5:
        shape = "CLUSTERED"
    elif top3_pct < 35 or singleton_ratio > 0.6:
        shape = "CONTINUOUS"
    else:
        shape = "UNCLEAR"

    return {
        "unique": unique,
        "top3_pct": round(top3_pct, 1),
        "singleton_ratio": round(singleton_ratio, 3),
        "max_gap": max_gap,
        "score_range": score_range,
        "shape": shape,
        "n": n,
    }


def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, alpha=0.05):
    """Bootstrap confidence interval."""
    n = len(data)
    arr = np.array(data)
    bootstrapped = np.array([
        statistic(np.random.choice(arr, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(bootstrapped, 100 * alpha / 2)
    upper = np.percentile(bootstrapped, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Intervention test
# ---------------------------------------------------------------------------

def run_intervention_test(run_name, n_games=100, device="cuda"):
    print(f"\n{'='*60}")
    print(f"  Intervention Test: {run_name}  ({device}, {n_games} games)")
    print(f"{'='*60}")
    sys.stdout.flush()

    model = load_model(run_name, device=device)

    normal_scores = []
    inter_scores = []
    all_interventions = []

    for i in range(n_games):
        # Normal game
        base_n = GymBreakout(fixed=True)
        env_n = base_n
        env_n = GrayscaleResize(env_n, width=84, height=84)
        env_n = ClipRewardEnv(env_n)
        env_n = Monitor(env_n)
        env_n = DummyVecEnv([lambda: env_n])
        env_n = VecFrameStack(env_n, n_stack=4)

        obs = env_n.reset()
        brk = base_n._env
        done = False
        frames = 0
        while not done and frames < 20000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env_n.step(action)
            done = done_arr[0]
            frames += 1
        normal_scores.append(brk.score if brk else 0)
        env_n.close()

        # Intervention game
        base_i = GymBreakout(fixed=True)
        ib = InterventionBreakout(base_i)
        env_i = ib
        env_i = GrayscaleResize(env_i, width=84, height=84)
        env_i = ClipRewardEnv(env_i)
        env_i = Monitor(env_i)
        env_i = DummyVecEnv([lambda: env_i])
        env_i = VecFrameStack(env_i, n_stack=4)

        obs = env_i.reset()
        brk = base_i._env
        done = False
        frames = 0
        while not done and frames < 20000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env_i.step(action)
            done = done_arr[0]
            frames += 1
        inter_scores.append(brk.score if brk else 0)
        all_interventions.append(ib.intervention_count)
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
        "normal_mean": float(normal.mean()),
        "normal_median": float(np.median(normal)),
        "normal_std": float(normal.std()),
        "normal_unique": len(set(normal_scores)),
        "intervention_mean": float(inter.mean()),
        "intervention_median": float(np.median(inter)),
        "intervention_std": float(inter.std()),
        "intervention_unique": len(set(inter_scores)),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "retention_pct": round(retention_pct, 1),
        "correlation": round(corr, 3),
        "avg_interventions_per_game": float(n_int.mean()),
        "normal_mean_ci95": bootstrap_ci(normal_scores, np.mean),
        "intervention_mean_ci95": bootstrap_ci(inter_scores, np.mean),
    }

    print(f"\n  RESULTS for {run_name}:")
    print(f"    Normal:       mean={result['normal_mean']:.1f} "
          f"med={result['normal_median']:.0f} std={result['normal_std']:.1f} "
          f"unique={result['normal_unique']}")
    print(f"    Intervention: mean={result['intervention_mean']:.1f} "
          f"med={result['intervention_median']:.0f} std={result['intervention_std']:.1f} "
          f"unique={result['intervention_unique']}")
    print(f"    Retention:    {retention_pct:.1f}%")
    print(f"    Correlation:  {corr:.3f}")
    print(f"    Delta:        mean={deltas.mean():.1f} std={deltas.std():.1f}")
    print(f"    Int/game:     {n_int.mean():.1f}")

    return result


# ---------------------------------------------------------------------------
# Reactivity evaluation
# ---------------------------------------------------------------------------

def run_reactivity_test(run_name, n_games=100, device="cuda"):
    print(f"\n{'='*60}")
    print(f"  Reactivity Test: {run_name}  ({device}, {n_games} games each)")
    print(f"{'='*60}")
    sys.stdout.flush()

    model = load_model(run_name, device=device)

    results = {}
    for det, label in [(True, "det=True"), (False, "det=False")]:
        print(f"\n  [{label}] running...")
        sys.stdout.flush()

        base = GymBreakout(fixed=True)
        env = base
        env = GrayscaleResize(env, width=84, height=84)
        env = ClipRewardEnv(env)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, n_stack=4)

        scores = run_episodes(model, env, n_games, deterministic=det)
        env.close()

        dist = analyze_distribution(scores)
        ci_low, ci_high = bootstrap_ci(scores, np.mean)

        key = "det_true" if det else "det_false"
        results[key] = {
            "unique": dist["unique"],
            "mean": round(float(np.mean(scores)), 1),
            "median": round(float(np.median(scores)), 1),
            "std": round(float(np.std(scores)), 1),
            "min": int(min(scores)),
            "max": int(max(scores)),
            "top3_pct": dist["top3_pct"],
            "singleton_ratio": dist["singleton_ratio"],
            "max_gap": dist["max_gap"],
            "score_range": dist["score_range"],
            "shape": dist["shape"],
            "mean_ci95": [ci_low, ci_high],
            "zero_score_pct": round(100 * sum(1 for s in scores if s == 0) / len(scores), 1),
            "raw_scores": scores,
        }
        print(f"    unique={dist['unique']}, mean={results[key]['mean']:.1f}, "
              f"top3={dist['top3_pct']:.1f}%, shape={dist['shape']}")

    # Cross-check
    det = results["det_true"]
    stoch = results["det_false"]
    print(f"\n  SUMMARY {run_name}:")
    print(f"    det=True:  {det['unique']} unique, {det['shape']}")
    print(f"    det=False: {stoch['unique']} unique, {stoch['shape']}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 calibration — logical audit dead-model baselines")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per test (default 100)")
    parser.add_argument("--runs", nargs="*", default=["PPO_34", "PPO_35", "PPO_36"],
                        help="Runs to test (default: PPO_34 PPO_35 PPO_36)")
    parser.add_argument("--skip-intervention", action="store_true")
    parser.add_argument("--skip-reactivity", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="calibration_phase1_results.json")
    args = parser.parse_args()

    all_results = {}

    def save_incremental(results, suffix=""):
        """Save results so far, stripping raw scores for JSON size."""
        out = {}
        for key, val in results.items():
            if isinstance(val, dict):
                out[key] = {
                    k: v for k, v in val.items()
                    if k not in ("raw_scores", "det_true", "det_false")
                }
                if "det_true" in val:
                    out[key] = {
                        "det_true": {k: v for k, v in val["det_true"].items()
                                      if k != "raw_scores"},
                        "det_false": {k: v for k, v in val["det_false"].items()
                                       if k != "raw_scores"},
                    }
        fname = args.output.replace(".json", f"{suffix}.json")
        with open(fname, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"  [saved {fname}]")

    for run in args.runs:
        if not args.skip_intervention:
            result = run_intervention_test(run, n_games=args.games, device=args.device)
            all_results[f"{run}_intervention"] = result
            save_incremental(all_results, f"_partial_{run}_intervention")
        if not args.skip_reactivity:
            result = run_reactivity_test(run, n_games=args.games, device=args.device)
            all_results[f"{run}_reactivity"] = result
            save_incremental(all_results, f"_partial_{run}_reactivity")

    # Save results
    # Strip raw_scores for JSON (keep them in a separate file)
    json_results = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            json_results[key] = {
                k: v for k, v in val.items()
                if k not in ("raw_scores", "det_true", "det_false")
            }
            # Handle nested reactivity results
            if "det_true" in val:
                json_results[key] = {
                    "det_true": {k: v for k, v in val["det_true"].items()
                                  if k != "raw_scores"},
                    "det_false": {k: v for k, v in val["det_false"].items()
                                   if k != "raw_scores"},
                }

    with open(args.output, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Also save full raw scores separately
    raw_file = args.output.replace(".json", "_raw.json")
    raw_data = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            if "det_true" in val:
                raw_data[f"{key}_det_true"] = val["det_true"].get("raw_scores", [])
                raw_data[f"{key}_det_false"] = val["det_false"].get("raw_scores", [])
    with open(raw_file, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw scores saved to {raw_file}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"  CALIBRATION COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Run':<12s} {'Test':<16s} {'Retention':>10s} {'Corr':>8s}  "
          f"{'det=F unique':>14s} {'det=F shape':>14s}")
    print(f"  {'-'*70}")

    for run in args.runs:
        ik = f"{run}_intervention"
        rk = f"{run}_reactivity"
        if ik in all_results:
            ir = all_results[ik]
            print(f"  {run:<12s} {'intervention':<16s} "
                  f"{ir['retention_pct']:>9.1f}% {ir['correlation']:>8.3f}")
        if rk in all_results:
            rr = all_results[rk]
            print(f"  {run:<12s} {'reactivity':<16s} {'':>10s} {'':>8s}  "
                  f"{rr['det_false']['unique']:>14} {rr['det_false']['shape']:<14s}")

    print(f"\nDone.")
