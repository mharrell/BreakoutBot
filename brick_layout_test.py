"""
Brick Layout Generalization Test v2 -- multiple layout conditions.

Tests whether a policy adapts to novel brick layouts by clearing different
subsets of bricks via setRAM(). Extended from the original 3-condition test
to cover more layout variations.

Conditions:
  full          - standard Breakout (control)
  right_half    - clear RAM[0-17]  (right-side bricks removed)
  left_half     - clear RAM[18-35] (left-side bricks removed)
  top_half      - clear first half of each side (top rows)
  bottom_half   - clear second half of each side (bottom rows)
  checkerboard  - clear every other byte (even indices)
  sparse50      - randomly clear 50% of brick bytes

Usage:
    python brick_layout_test.py --model ./models/PPO_116/best_model.zip
    python brick_layout_test.py --model ./models/PPO_116/best_model.zip --layouts full,left_half,right_half,checkerboard
"""
import sys
import time
import re
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, NoopResetEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3

# Brick RAM: 36 bytes (0-35), bit-packed.
# Empirically: [0-17]=right half, [18-35]=left half
ALL_BRICK_ADDRS = list(range(36))
RIGHT_HALF_ADDRS = list(range(0, 18))
LEFT_HALF_ADDRS = list(range(18, 36))
TOP_HALF_ADDRS = list(range(0, 9)) + list(range(18, 27))
BOTTOM_HALF_ADDRS = list(range(9, 18)) + list(range(27, 36))
CHECKERBOARD_ADDRS = [i for i in range(36) if i % 2 == 0]

MAX_STEPS_PER_GAME = 10_000

# Layout definitions: name -> (description, list of bytes to clear, or None for random)
LAYOUT_DEFS = {
    "full":          ("standard layout", []),
    "right_half":    ("right-side bricks removed", RIGHT_HALF_ADDRS),
    "left_half":     ("left-side bricks removed", LEFT_HALF_ADDRS),
    "top_half":      ("top rows removed", TOP_HALF_ADDRS),
    "bottom_half":   ("bottom rows removed", BOTTOM_HALF_ADDRS),
    "checkerboard":  ("every other byte cleared", CHECKERBOARD_ADDRS),
    "sparse50":      ("50% bytes randomly cleared", "random_50"),
}
DEFAULT_LAYOUTS = ["full", "right_half", "left_half", "top_half",
                    "bottom_half", "checkerboard", "sparse50"]


# ---------------------------------------------------------------------------
# Wrappers
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


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs, info = self.env.reset()
        return obs, reward, terminated, truncated, info


class BrickClearWrapper(gym.Wrapper):
    """Zero out specified brick RAM addresses on every reset().

    If clear_addrs is the string "random_N", randomly clear N% of all 36
    brick bytes on each reset (different pattern every episode).
    """

    def __init__(self, env, clear_addrs=None, seed=None):
        super().__init__(env)
        self.clear_addrs = clear_addrs
        self._rng = np.random.default_rng(seed)
        self._random_pct = None
        if isinstance(clear_addrs, str) and clear_addrs.startswith("random_"):
            self._random_pct = int(clear_addrs.split("_")[1]) / 100.0
            self.clear_addrs = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._random_pct is not None:
            n_clear = max(1, int(36 * self._random_pct))
            addrs = list(self._rng.choice(36, size=n_clear, replace=False))
        else:
            addrs = self.clear_addrs
        for addr in addrs:
            self.unwrapped.ale.setRAM(addr, 0)
        return obs, info


# ---------------------------------------------------------------------------
# Env builders
# ---------------------------------------------------------------------------

def _base_env():
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    return env


def make_env_for_layout(layout_name):
    """Build an env factory for a given layout."""
    if layout_name not in LAYOUT_DEFS:
        raise ValueError(f"Unknown layout: {layout_name}. Options: {list(LAYOUT_DEFS)}")

    desc, addrs = LAYOUT_DEFS[layout_name]

    def factory():
        env = _base_env()
        if addrs:  # not empty list
            env = BrickClearWrapper(env, clear_addrs=addrs)
        env = EpisodicLifeEnv(env)
        env = GrayscaleResize(env, width=84, height=84)
        env = AutoResetWrapper(env)
        return env

    return factory


def make_vec_env(env_fn):
    env = DummyVecEnv([env_fn])
    env = VecFrameStack(env, n_stack=4)
    return env


# ---------------------------------------------------------------------------
# Dead baseline -- paddle-sweep script
# ---------------------------------------------------------------------------

def dead_sweep_scores(env_fn, n_games):
    SWEEP_SPEED = 2
    PADDLE_MIN, PADDLE_MAX = 8, 140
    scores = []
    for _ in range(n_games):
        env = env_fn()
        obs = env.reset()
        score = 0
        step = 0
        done = False
        direction = 1
        while not done and step < MAX_STEPS_PER_GAME:
            ram = env.unwrapped.ale.getRAM()
            px, by = int(ram[PADDLE_X]), int(ram[BALL_Y])
            if by > 180:
                action = FIRE
            else:
                if px >= PADDLE_MAX - SWEEP_SPEED:
                    direction = -1
                elif px <= PADDLE_MIN + SWEEP_SPEED:
                    direction = 1
                action = RIGHT if direction == 1 else LEFT
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            step += 1
            done = terminated or truncated
        scores.append(score)
        env.close()
    return scores


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def model_scores(model, env_fn, n_games):
    """Play n_games using one env instance (reuse across games)."""
    env = make_vec_env(env_fn)
    scores = []
    obs = env.reset()
    game_count = 0
    step = 0
    score = 0
    while game_count < n_games and step < MAX_STEPS_PER_GAME * n_games:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        score += reward[0]
        step += 1
        if dones[0]:
            scores.append(score)
            game_count += 1
            score = 0
    env.close()
    return scores


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def summarize(name, scores):
    arr = np.array(scores)
    return {
        'name': name, 'n': len(arr), 'mean': np.mean(arr),
        'std': np.std(arr), 'min': np.min(arr), 'max': np.max(arr),
        'unique': len(np.unique(arr)), 'zero_pct': np.mean(arr == 0) * 100,
        'values': sorted([float(x) for x in arr]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_107/best_model.zip"
    RUN_NAME = "PPO_107"
    N_GAMES = 20
    LAYOUTS = DEFAULT_LAYOUTS

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--model': MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == '--games': N_GAMES = int(args[i + 1]); i += 2
        elif args[i] == '--run-name': RUN_NAME = args[i + 1]; i += 2
        elif args[i] == '--layouts':
            LAYOUTS = args[i + 1].split(','); i += 2
        else: i += 1

    m = re.search(r'PPO_\d+[a-z]?', MODEL_PATH)
    if m:
        RUN_NAME = m.group(0)

    # Validate layouts
    for layout in LAYOUTS:
        if layout not in LAYOUT_DEFS:
            print(f"Error: unknown layout '{layout}'. Options: {list(LAYOUT_DEFS)}")
            sys.exit(1)

    print("=" * 70)
    print(f"Brick Layout Generalization Test v2 -- {RUN_NAME}")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Games per layout: {N_GAMES}")
    print(f"Layouts: {', '.join(LAYOUTS)}")
    print()

    # --- Dead baseline ---
    print("--- Dead baseline (paddle-sweep) ---")
    t0 = time.time()
    dead_results = {}
    for layout in LAYOUTS:
        desc, _ = LAYOUT_DEFS[layout]
        scores = dead_sweep_scores(make_env_for_layout(layout), N_GAMES)
        dead_results[layout] = summarize(f"Dead {layout}", scores)
        s = dead_results[layout]
        print(f"  Dead {layout:<16} avg={s['mean']:>6.1f}  std={s['std']:>5.1f}  "
              f"unique={s['unique']}  range=[{s['min']:.0f},{s['max']:.0f}]")
    print(f"  ({time.time() - t0:.0f}s)")
    print()

    # --- Model ---
    print(f"--- {RUN_NAME} ---")
    print("Loading model...")
    env = make_vec_env(make_env_for_layout("full"))
    model = PPO.load(MODEL_PATH, env=env, device="cuda")
    env.close()
    print(f"Loaded. Model step count: {model.num_timesteps:,}")
    print()

    t0 = time.time()
    model_results = {}
    for layout in LAYOUTS:
        scores = model_scores(model, make_env_for_layout(layout), N_GAMES)
        model_results[layout] = summarize(f"{RUN_NAME} {layout}", scores)
        s = model_results[layout]
        print(f"  {RUN_NAME} {layout:<16} avg={s['mean']:>6.1f}  std={s['std']:>5.1f}  "
              f"unique={s['unique']}  range=[{s['min']:.0f},{s['max']:.0f}]")
    print(f"  ({time.time() - t0:.0f}s)")
    print()

    # --- Score distributions ---
    print("=" * 70)
    print("SCORE DISTRIBUTIONS")
    print("=" * 70)
    for layout in LAYOUTS:
        print(f"  Dead {layout:<16} {dead_results[layout]['values']}")
    for layout in LAYOUTS:
        print(f"  {RUN_NAME} {layout:<16} {model_results[layout]['values']}")
    print()

    # --- Comparison table ---
    print("=" * 70)
    print("COMPARISON: Score Retention vs Full Layout")
    print("=" * 70)

    full_mean = model_results["full"]["mean"]
    dead_full_mean = dead_results["full"]["mean"]

    print(f"\n  {'Layout':<20} {'Dead':>10} {'Dead ret':>10} "
          f"{RUN_NAME:>10} {'Retention':>10} {'Unique':>8} {'Reactive?':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*12}")

    for layout in LAYOUTS:
        d = dead_results[layout]
        m = model_results[layout]
        dead_ret = d['mean'] / dead_full_mean * 100 if dead_full_mean > 0 else 0
        model_ret = m['mean'] / full_mean * 100 if full_mean > 0 else 0
        reactive = "YES" if m['unique'] > 2 else ("binary" if m['unique'] == 2 else "script")
        print(f"  {layout:<20} {d['mean']:>8.1f}  {dead_ret:>8.0f}% "
              f"{m['mean']:>8.1f}  {model_ret:>8.0f}%  {m['unique']:>6}  {reactive:>12}")

    print()
    print("Reactive = unique>2 on that layout (continuous adaptation, not binary succeed/fail)")
    print("Retention = score on this layout / score on full layout × 100%")
