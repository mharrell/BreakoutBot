"""
Reactivity evaluation — tests whether a GymBreakout-trained model plays
reactively or has memorized a fixed script.

The key diagnostic: deterministic=False (sampling from the policy distribution)
on GymBreakout(fixed=True). A reactive policy produces diverse scores because
different action samples lead to different game trajectories. A memorized
policy with near-one-hot action probabilities produces identical scores
regardless of sampling.

Also runs deterministic=True as a baseline (expected: identical scores in
this fully-deterministic environment, even for reactive policies).

Usage:
    python -u eval_reactivity.py                          # test PPO_35 best model
    python -u eval_reactivity.py --run PPO_34             # test PPO_34 best model
    python -u eval_reactivity.py --run PPO_35 --games 200 # custom game count
"""
import argparse
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


class GrayscaleResize(gym.ObservationWrapper):
    """Resize grayscale to (height, width, 1) — compatible with VecFrameStack."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )

    def observation(self, obs):
        resized = cv2.resize(obs, (self._width, self._height),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None]


def make_eval_env():
    """GymBreakout with fixed defaults — matches train_ppo35.py eval env."""
    env = GymBreakout(fixed=True)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


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
                obs, _, _, _ = env.step([1])

    return scores


def analyze_distribution(scores):
    """Analyze score distribution shape to distinguish script-switching from tracking.

    Script-switching produces CLUSTERED distributions: a few scores repeat many
    times (the scripts), plus noise around them. Ball-tracking produces CONTINUOUS
    distributions: scores spread across a range without sharp peaks.

    Returns dict with metrics and a shape verdict.
    """
    n = len(scores)
    unique = len(set(scores))
    counter = Counter(scores)

    # How concentrated are the top scores?
    top3_pct = sum(count for _, count in counter.most_common(3)) / n * 100

    # What fraction of unique scores appear exactly once? (singletons)
    singletons = sum(1 for _, count in counter.items() if count == 1)
    singleton_ratio = singletons / unique if unique > 0 else 0

    # Gap analysis: max consecutive values with zero occurrences
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

    # Shape verdict
    if top3_pct > 50 and singleton_ratio < 0.5:
        shape = "CLUSTERED (script-switching)"
        shape_detail = (f"Top 3 scores account for {top3_pct:.0f}% of games. "
                        f"Distribution has distinct peaks, not a continuous spread.")
    elif top3_pct < 35 or singleton_ratio > 0.6:
        shape = "CONTINUOUS (ball-tracking)"
        shape_detail = (f"Scores spread across range. "
                        f"Top 3 concentration={top3_pct:.0f}%, "
                        f"singleton ratio={singleton_ratio:.0%}.")
    else:
        shape = "UNCLEAR"
        shape_detail = (f"Top 3 concentration={top3_pct:.0f}%, "
                        f"singleton ratio={singleton_ratio:.0%}. "
                        f"Neither clearly clustered nor clearly continuous.")

    return {
        "unique": unique,
        "top3_pct": top3_pct,
        "singleton_ratio": singleton_ratio,
        "max_gap": max_gap,
        "score_range": score_range,
        "shape": shape,
        "shape_detail": shape_detail,
    }


def print_stats(scores, label):
    """Print summary statistics for a set of games."""
    info = analyze_distribution(scores)
    avg = float(np.mean(scores))
    std = float(np.std(scores))
    median = float(np.median(scores))

    print()
    print(f"{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Games:         {len(scores)}")
    print(f"  Unique scores: {info['unique']}")
    print(f"  Mean:          {avg:.1f}")
    print(f"  Median:        {median:.0f}")
    print(f"  Std:           {std:.1f}")
    print(f"  Min:           {min(scores):.0f}")
    print(f"  Max:           {max(scores):.0f}")

    # Horizontal bar chart — scores on x-axis, counts on y-axis
    counter = Counter(scores)
    score_min, score_max = int(min(scores)), int(max(scores))
    max_count = max(counter.values())
    n_cols = score_max - score_min + 1
    counts = [counter.get(s, 0) for s in range(score_min, score_max + 1)]

    print(f"  Score histogram (x = score, y = games, - = gap):")
    print()

    # Print rows from max_count down to 1
    for level in range(max_count, 0, -1):
        if level == max_count:
            label = f"{max_count:3d} |"
        elif level % 5 == 0 or level == 1:
            label = f"{level:3d} |"
        else:
            label = "    |"

        line = "".join(
            "#" if cnt >= level else ("-" if cnt == 0 and level == 1 else " ")
            for cnt in counts
        )
        print(f"  {label} {line}")

    # X-axis
    print(f"     +{'-' * n_cols}")

    # Score labels every 5
    label_line = "      "
    for s in range(score_min, score_max + 1):
        if s % 5 == 0:
            label_line += str(s)
        elif s == score_min:
            label_line += str(s)
        else:
            label_line += " "
    print(f"  {label_line}")
    print()

    # Distribution shape analysis
    print(f"  Distribution shape: {info['shape']}")
    print(f"    Top-3 concentration: {info['top3_pct']:.0f}% "
          f"(singleton ratio: {info['singleton_ratio']:.0%}, "
          f"max gap: {info['max_gap']}, range: {info['score_range']})")
    print(f"    {info['shape_detail']}")

    # Verdict
    zero_pct = 100 * sum(1 for s in scores if s == 0) / len(scores)
    print(f"  Zero-score rate: {zero_pct:.1f}%")

    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reactivity evaluation for GymBreakout models")
    parser.add_argument("--run", default="PPO_35", help="Run name (default: PPO_35)")
    parser.add_argument("--model", default=None,
                        help="Model path (default: models/<RUN>/best_model.zip)")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games per mode (default: 100)")
    args = parser.parse_args()

    RUN = args.run
    MODEL_PATH = args.model or f"models/{RUN}/best_model.zip"
    NUM_GAMES = args.games

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        available = []
        for d in os.listdir("models"):
            for fname in ["best_model.zip", "final_model.zip"]:
                p = os.path.join("models", d, fname)
                if os.path.exists(p):
                    available.append(p)
        print("Available models:")
        for p in sorted(available):
            print(f"  {p}")
        sys.exit(1)

    print(f"Reactivity Evaluation: {RUN}")
    print(f"  Model:    {MODEL_PATH}")
    print(f"  Games:    {NUM_GAMES} per mode")
    print(f"  Env:      GymBreakout(fixed=True) — standard defaults, no sticky")
    print(f"  Question: Does sampling from the policy produce diverse scores?")
    sys.stdout.flush()

    # Build eval environment
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Load model
    print(f"Loading model...")
    sys.stdout.flush()
    model = PPO.load(MODEL_PATH, device="cuda", custom_objects={"n_envs": 1})
    print(f"Loaded. {model.num_timesteps:,} training steps.")
    sys.stdout.flush()
    model.set_env(eval_env)

    # ---- Test 1: deterministic=True (baseline) ----
    print(f"\n[1/2] deterministic=True (argmax actions)...")
    sys.stdout.flush()
    t0 = time.time()
    det_scores = run_episodes(model, eval_env, NUM_GAMES,
                               deterministic=True, label="det=True")
    det_elapsed = time.time() - t0
    det_info = print_stats(det_scores,
        f"{RUN} — deterministic=True ({NUM_GAMES} games, {det_elapsed:.0f}s)")

    # Re-create env for clean state
    eval_env.close()
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    model.set_env(eval_env)

    # ---- Test 2: deterministic=False (the actual test) ----
    print(f"\n[2/2] deterministic=False (sampled actions)...")
    sys.stdout.flush()
    t0 = time.time()
    stoch_scores = run_episodes(model, eval_env, NUM_GAMES,
                                 deterministic=False, label="det=False")
    stoch_elapsed = time.time() - t0
    stoch_info = print_stats(stoch_scores,
        f"{RUN} — deterministic=False ({NUM_GAMES} games, {stoch_elapsed:.0f}s)")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  deterministic=True:   {det_info['unique']} unique, {det_info['shape']}")
    print(f"  deterministic=False:  {stoch_info['unique']} unique, {stoch_info['shape']}")

    # Conclusion logic combines diversity + distribution shape
    det_script = det_info['unique'] <= 2
    stoch_few = stoch_info['unique'] <= 2
    stoch_some = 3 <= stoch_info['unique'] <= 9
    stoch_diverse = stoch_info['unique'] >= 10
    stoch_continuous = stoch_info['shape'] == "CONTINUOUS (ball-tracking)"

    if det_script and stoch_continuous:
        print(f"  CONCLUSION: REACTIVE — policy tracks the ball under sampling.")
        print(f"  Argmax is a script, but the policy distribution is genuinely")
        print(f"  state-dependent (continuous score spread, not discrete modes).")
    elif det_script and stoch_diverse and not stoch_continuous:
        print(f"  CONCLUSION: SCRIPT-SWITCHING — policy has multiple scripts,")
        print(f"  not ball-tracking. Stochastic sampling selects among discrete")
        print(f"  scripts rather than varying continuously with game state.")
    elif det_script and stoch_few:
        print(f"  CONCLUSION: FULLY MEMORIZED — both argmax and sampling collapse")
        print(f"  to one or two scripts. No useful entropy.")
    elif det_script and stoch_some:
        print(f"  CONCLUSION: LOW DIVERSITY — argmax is a script, sampling produces")
        print(f"  {stoch_info['unique']} unique scores (limited variability). The policy has")
        print(f"  some entropy but not enough for script-switching classification.")
        print(f"  Continue training — may develop into script-switching or reactivity.")
    elif not det_script and stoch_continuous:
        print(f"  CONCLUSION: FULLY REACTIVE — both argmax and sampling track the")
        print(f"  ball. The policy has genuinely learned a closed-loop policy.")
    else:
        print(f"  CONCLUSION: See distribution analysis above — interpret with")
        print(f"  caution. Check score histograms for clustering vs continuity.")

    eval_env.close()
    print(f"Done. Total time: {det_elapsed + stoch_elapsed:.0f}s")
    sys.stdout.flush()
