"""
eval_intervention.py — Direct reactivity test via environment intervention.

After a paddle bounce, randomly teleports the ball (y-axis) and paddle (x-axis).
A memorized script keeps playing its timed sequence from unexpected positions
and fails. A reactive policy sees the new ball/paddle positions and adjusts.

Runs games with and without teleportation, compares score distributions.
"""

import argparse
import os
import time
import numpy as np
import cv2
import gymnasium as gym
from collections import defaultdict

from gym_breakout import GymBreakout
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.monitor import Monitor


# --- GymBreakout wrapper that injects teleportation on paddle bounces ---

class InterventionBreakout(gym.Wrapper):
    """After a paddle bounce, teleports ball (y-axis) and paddle (x-axis)."""

    def __init__(self, env, teleport_prob=0.3, ball_y_jitter=15,
                 paddle_x_jitter=30, seed=None):
        super().__init__(env)
        self.teleport_prob = teleport_prob
        self.ball_y_jitter = ball_y_jitter
        self.paddle_x_jitter = paddle_x_jitter
        self._rng = np.random.default_rng(seed)
        self.interventions = []
        self.intervention_count = 0   # survives auto-reset from DummyVecEnv

    def reset(self, **kwargs):
        self.intervention_count = len(self.interventions)
        self.interventions = []
        return self.env.reset(**kwargs)

    def step(self, action):
        # self.env is GymBreakout (we wrap it directly in make_env)
        # GymBreakout._env is the Breakout engine instance (created in reset())
        brk = self.env._env
        prev_vy = brk.ball_v[0] if brk else None

        obs, reward, terminated, truncated, info = self.env.step(action)

        cur_vy = brk.ball_v[0] if brk else None
        if (prev_vy is not None and cur_vy is not None
                and prev_vy > 0 and cur_vy < 0):
            if self._rng.random() < self.teleport_prob:
                ball = brk.ball
                paddle = brk.paddle
                # Ball y-teleport
                dy = int(round(self._rng.normal(0, self.ball_y_jitter)))
                new_y = ball.pos[0] + dy
                new_y = max(15, min(paddle.pos[0] - 10, new_y))
                ball.pos[0] = new_y
                # Paddle x-teleport
                dx = int(round(self._rng.normal(0, self.paddle_x_jitter)))
                new_x = paddle.pos[1] + dx
                new_x = max(0, min(160 - paddle.size[1], new_x))
                paddle.pos[1] = new_x
                # Record
                self.interventions.append(
                    (brk.step_count, new_y, new_x))

        return obs, reward, terminated, truncated, info


# --- Grayscale resize ---

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


# --- Model loading ---

def load_model(run_name):
    best = f"./models/{run_name}/best_model.zip"
    if os.path.exists(best):
        print(f"  Loading {best}")
        return PPO.load(best, device="cuda")
    import glob
    ckpt_dir = f"./models/{run_name}/checkpoint"
    if os.path.isdir(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "latest_checkpoint_*_steps.zip"))
        if ckpts:
            latest = max(ckpts, key=os.path.getmtime)
            print(f"  Loading {latest}")
            return PPO.load(latest, device="cuda")
    raise FileNotFoundError(f"No model found for {run_name}")


# --- Game runner ---

def run_game(env, base_env, ib, model, deterministic, max_frames=20000):
    """Run one game. Returns (score, n_interventions).
    base_env is the GymBreakout instance — we read _env.score directly
    because Monitor clips rewards via ClipRewardEnv (sign() only).
    ib is the InterventionBreakout wrapper (or None for normal games).

    IMPORTANT: We capture the Breakout engine reference right after reset().
    DummyVecEnv auto-resets on episode termination, which replaces
    base_env._env with a new Breakout instance (score=0). So we hold a
    reference to the original Breakout from before the reset."""
    obs = env.reset()
    brk = base_env._env  # capture before auto-reset replaces it
    done = False
    frames = 0
    while not done and frames < max_frames:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done_arr, info = env.step(action)
        done = done_arr[0]
        frames += 1
    score = brk.score if brk else 0
    n_int = ib.intervention_count if ib else 0
    return score, n_int


def make_env(intervene=False, teleport_prob=0.3, ball_y_jitter=15,
             paddle_x_jitter=30):
    """Build a vectorized env, optionally with intervention.
    Returns (vec_env, gym_breakout_base, intervention_wrapper_or_none)."""
    base = GymBreakout(fixed=True)
    ib = None
    if intervene:
        ib = InterventionBreakout(base, teleport_prob=teleport_prob,
                                  ball_y_jitter=ball_y_jitter,
                                  paddle_x_jitter=paddle_x_jitter)
        env = ib
    else:
        env = base
    env = GrayscaleResize(env)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env, base, ib


# --- Main ---

def run_intervention_test(run_name, n_games=100, teleport_prob=0.3,
                          ball_y_jitter=15, paddle_x_jitter=30,
                          deterministic=True, max_frames=20000):
    print(f"\n{'='*60}")
    print(f"  Intervention Test: {run_name}")
    print(f"  Games: {n_games}, teleport_p={teleport_prob}")
    print(f"  Ball y-jitter: {ball_y_jitter}, Paddle x-jitter: {paddle_x_jitter}")
    print(f"  deterministic={deterministic}")
    print(f"{'='*60}")

    model = load_model(run_name)

    normal_scores = []
    inter_scores = []
    all_interventions = []

    for i in range(n_games):
        # Normal game
        env_n, base_n, _ = make_env(intervene=False)
        s_normal, _ = run_game(env_n, base_n, None, model, deterministic, max_frames)
        normal_scores.append(s_normal)
        env_n.close()

        # Intervention game
        env_i, base_i, ib = make_env(intervene=True, teleport_prob=teleport_prob,
                                     ball_y_jitter=ball_y_jitter,
                                     paddle_x_jitter=paddle_x_jitter)
        s_inter, n_int = run_game(env_i, base_i, ib, model, deterministic, max_frames)
        inter_scores.append(s_inter)
        all_interventions.append(n_int)
        env_i.close()

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_games}] Normal={np.mean(normal_scores):.1f}, "
                  f"Inter={np.mean(inter_scores):.1f}, "
                  f"Avg int/game={np.mean(all_interventions):.1f}")

    normal = np.array(normal_scores)
    inter = np.array(inter_scores)
    deltas = normal - inter
    n_int = np.array(all_interventions)

    print(f"\n{'='*60}")
    print(f"  RESULTS: {run_name}")
    print(f"{'='*60}")
    print(f"\n  Scores (normal vs intervention):")
    print(f"    Normal:       mean={normal.mean():.1f} med={np.median(normal):.0f} "
          f"std={normal.std():.1f} min={normal.min():.0f} max={normal.max():.0f}")
    print(f"    Intervention: mean={inter.mean():.1f} med={np.median(inter):.0f} "
          f"std={inter.std():.1f} min={inter.min():.0f} max={inter.max():.0f}")
    print(f"    Delta (N-I):  mean={deltas.mean():.1f} med={np.median(deltas):.0f} "
          f"std={deltas.std():.1f}")
    print(f"  Unique: Normal={len(set(normal_scores))}, "
          f"Intervention={len(set(inter_scores))}")
    print(f"  Interventions/game: mean={n_int.mean():.1f} "
          f"min={n_int.min()} max={n_int.max()}")

    if normal.std() > 0 and inter.std() > 0:
        corr = np.corrcoef(normal, inter)[0, 1]
        print(f"\n  Score correlation: {corr:.3f}")
        print(f"    >0.7 = reactive    <0.3 = memorized")

    exact = (deltas == 0).sum()
    n_better = (deltas > 0).sum()
    i_better = (deltas < 0).sum()
    print(f"  Per-pair: {exact} same, {n_better} normal>inter, {i_better} inter>normal")

    # Distribution
    max_score = max(normal.max(), inter.max())
    min_score = min(normal.min(), inter.min())
    print(f"\n  Score distribution (N=normal, I=intervention):")
    n_bins = min(15, int(max_score - min_score + 1))
    bin_w = max(1, int((max_score - min_score) / n_bins))
    for lo in range(int(min_score), int(max_score) + 1, bin_w):
        hi = lo + bin_w - 1
        nc = int(((normal >= lo) & (normal <= hi)).sum())
        ic = int(((inter >= lo) & (inter <= hi)).sum())
        bar_n = '#' * max(1, nc) if nc > 0 else ''
        bar_i = ':' * max(1, ic) if ic > 0 else ''
        label = f"{lo}-{hi}" if bin_w > 1 else str(lo)
        print(f"    {label:>8s}: N {bar_n} ({nc})  I {bar_i} ({ic})")

    # Verdict
    corr = (np.corrcoef(normal, inter)[0, 1]
            if normal.std() > 0 and inter.std() > 0 else 0)
    print(f"\n  VERDICT:", end=" ")
    if n_int.mean() < 0.5:
        print("NO INTERVENTIONS — teleport failed to fire (investigate)")
    elif corr > 0.6 and abs(deltas.mean()) < 10:
        print("REACTIVE — intervention doesn't hurt the policy")
    elif deltas.mean() > 15:
        print("MEMORIZED — intervention causes large score drops")
    elif normal.std() < 2 and inter.std() < 2:
        print("MEMORIZED — deterministic script, both conditions identical")
    else:
        print("UNCLEAR — see distribution above")

    return {
        "run": run_name,
        "normal_mean": float(normal.mean()),
        "intervention_mean": float(inter.mean()),
        "delta_mean": float(deltas.mean()),
        "correlation": corr,
        "normal_unique": len(set(normal_scores)),
        "intervention_unique": len(set(inter_scores)),
        "avg_interventions": float(n_int.mean()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intervention reactivity test")
    parser.add_argument("--run", default="PPO_36")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--teleport-prob", type=float, default=0.3)
    parser.add_argument("--ball-y-jitter", type=int, default=15)
    parser.add_argument("--paddle-x-jitter", type=int, default=30)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--compare", nargs="*", default=None)
    args = parser.parse_args()

    runs = [args.run]
    if args.compare:
        runs.extend(args.compare)

    results = []
    for run in runs:
        result = run_intervention_test(
            run, n_games=args.games,
            teleport_prob=args.teleport_prob,
            ball_y_jitter=args.ball_y_jitter,
            paddle_x_jitter=args.paddle_x_jitter,
            deterministic=not args.stochastic)
        results.append(result)

    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Run':<12s} {'Normal':>8s} {'Inter':>8s} {'Delta':>8s} "
              f"{'Corr':>6s} {'Int/gm':>6s}")
        print(f"  {'-'*55}")
        for r in results:
            print(f"  {r['run']:<12s} {r['normal_mean']:8.1f} "
                  f"{r['intervention_mean']:8.1f} {r['delta_mean']:8.1f} "
                  f"{r['correlation']:6.3f} {r['avg_interventions']:6.1f}")

    print("\nDone.")
