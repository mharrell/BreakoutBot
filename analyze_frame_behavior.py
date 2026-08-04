"""
Per-frame behavioral analysis for split-watcher results.

Runs the same model on FULL vs ALTERED layouts and logs per-frame data:
  - FULL paddle_x, ball_x, ball_y
  - ALT paddle_x, ball_x, ball_y
  - FULL action, ALT action
  - Whether ALT paddle is closer to ALT ball than FULL ball (tracking signal)
  - Paddle-ball distance on both sides

This distinguishes genuine tracking from scrambled visual cues:
  - Genuine tracking: ALT paddle tracks ALT ball position (lower ALT distance)
  - Scrambled cues: ALT paddle diverges from FULL but doesn't track ALT ball

Output: CSV with one row per frame, per game.

Usage:
    python analyze_frame_behavior.py --model ./models/PPO_124/best_model.zip --games 5
"""
import sys
import re
import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import FireResetEnv, EpisodicLifeEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import ale_py
gym.register_envs(ale_py)

BALL_X, BALL_Y, PADDLE_X = 99, 101, 72
NOOP, FIRE, RIGHT, LEFT = 0, 1, 2, 3


class BrickClearWrapper(gym.Wrapper):
    def __init__(self, env, clear_addrs=None):
        super().__init__(env)
        self._static_addrs = None
        self._random_pct = None
        self._rng = np.random.default_rng()
        if isinstance(clear_addrs, str) and clear_addrs.startswith("random_"):
            self._random_pct = int(clear_addrs.split("_")[1]) / 100.0
        else:
            self._static_addrs = list(clear_addrs or [])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self._random_pct is not None:
            n_clear = max(1, int(36 * self._random_pct))
            addrs = list(self._rng.choice(36, size=n_clear, replace=False))
        else:
            addrs = self._static_addrs
        for addr in addrs:
            self.unwrapped.ale.setRAM(addr, 0)
        obs, _, _, _, _ = self.env.step(0)
        return obs, info


def make_raw_env(brick_addrs=None):
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = FireResetEnv(env)
    if brick_addrs is not None:
        env = BrickClearWrapper(env, clear_addrs=brick_addrs)
    env = EpisodicLifeEnv(env)
    return env


def get_ram(env):
    return env.unwrapped.ale.getRAM()


def initial_frame_stack(obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return [gray] * 4


def update_frame_stack(fs, obs):
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    fs.pop(0)
    fs.append(gray)
    return fs


if __name__ == "__main__":
    MODEL_PATH = "./models/PPO_124/best_model.zip"
    N_GAMES = 5
    MAX_FRAMES = 6000
    OUTPUT = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model":
            MODEL_PATH = args[i + 1]; i += 2
        elif args[i] == "--games":
            N_GAMES = int(args[i + 1]); i += 2
        elif args[i] == "--output":
            OUTPUT = args[i + 1]; i += 2
        else:
            i += 1

    if OUTPUT is None:
        m = re.search(r"PPO_\d+[a-z]?", MODEL_PATH)
        run_name = m.group(0) if m else "model"
        OUTPUT = f"recordings/{run_name}_frame_analysis.csv"

    m = re.search(r"PPO_\d+[a-z]?", MODEL_PATH)
    run_name = m.group(0) if m else "model"

    # Load model
    def _make_dummy():
        e = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
        e = FireResetEnv(e)
        e = EpisodicLifeEnv(e)
        return e
    dummy_env = DummyVecEnv([_make_dummy])
    dummy_env = VecFrameStack(dummy_env, n_stack=4)
    model = PPO.load(MODEL_PATH, env=dummy_env, device="cuda")
    dummy_env.close()

    print(f"Per-Frame Behavioral Analysis — {run_name}")
    print(f"  Model: {MODEL_PATH} @ {model.num_timesteps:,} steps")
    print(f"  Games per layout: {N_GAMES}")
    print(f"  Output: {OUTPUT}")
    print()

    LAYOUTS = [
        ("RIGHT_HALF", list(range(0, 18))),
        ("LEFT_HALF", list(range(18, 36))),
    ]

    # CSV header
    csv_lines = ["game,layout,frame,full_act,alt_act,"
                 "full_px,full_bx,full_by,"
                 "alt_px,alt_bx,alt_by,"
                 "full_pb_dist,alt_pb_dist,"
                 "alt_tracks_alt_better"]

    game_id = 0
    for layout_name, layout_addrs in LAYOUTS:
        for g in range(N_GAMES):
            game_id += 1
            env_full = make_raw_env(brick_addrs=None)
            env_alt = make_raw_env(brick_addrs=layout_addrs)

            obs_full, _info = env_full.reset()
            obs_alt, _info = env_alt.reset()

            fs_full = initial_frame_stack(obs_full)
            fs_alt = initial_frame_stack(obs_alt)

            done_full = False
            done_alt = False
            frame = 0

            while not (done_full and done_alt) and frame < MAX_FRAMES:
                frame += 1

                # Predict independently
                if not done_full:
                    left_obs = np.expand_dims(fs_full, axis=0)
                    left_action, _ = model.predict(left_obs, deterministic=True)
                    left_act = int(left_action[0])
                else:
                    left_act = NOOP

                if not done_alt:
                    right_obs = np.expand_dims(fs_alt, axis=0)
                    right_action, _ = model.predict(right_obs, deterministic=True)
                    right_act = int(right_action[0])
                else:
                    right_act = NOOP

                # RAM state
                full_ram = get_ram(env_full) if not done_full else None
                alt_ram = get_ram(env_alt) if not done_alt else None

                full_px = int(full_ram[PADDLE_X]) if full_ram is not None else -1
                full_bx = int(full_ram[BALL_X]) if full_ram is not None else -1
                full_by = int(full_ram[BALL_Y]) if full_ram is not None else -1
                alt_px = int(alt_ram[PADDLE_X]) if alt_ram is not None else -1
                alt_bx = int(alt_ram[BALL_X]) if alt_ram is not None else -1
                alt_by = int(alt_ram[BALL_Y]) if alt_ram is not None else -1

                # Paddle-ball distances
                full_dist = abs(full_px - full_bx) if (full_ram is not None) else -1
                alt_dist = abs(alt_px - alt_bx) if (alt_ram is not None) else -1

                # Tracking signal: is ALT paddle closer to ALT ball than FULL ball would be?
                alt_to_full_ball = abs(alt_px - full_bx) if (alt_ram is not None and full_ram is not None) else -1
                tracks_alt = 1 if (alt_dist >= 0 and alt_to_full_ball >= 0 and alt_dist < alt_to_full_ball) else 0

                csv_lines.append(
                    f"{game_id},{layout_name},{frame},{left_act},{right_act},"
                    f"{full_px},{full_bx},{full_by},"
                    f"{alt_px},{alt_bx},{alt_by},"
                    f"{full_dist},{alt_dist},{tracks_alt}"
                )

                # Step FULL
                if not done_full:
                    try:
                        needs_serve = int(full_ram[BALL_Y]) > 180
                    except Exception:
                        needs_serve = False
                    act = FIRE if needs_serve else left_act
                    obs, reward, terminated, truncated, info = env_full.step(act)
                    if terminated or truncated:
                        try:
                            is_game_over = env_full.unwrapped.ale.lives() == 0
                        except Exception:
                            is_game_over = True
                        if is_game_over:
                            done_full = True
                        else:
                            obs, info = env_full.reset()
                            fs_full = [cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)] * 4
                            fs_full = [cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA) for g in fs_full]
                            continue
                    else:
                        update_frame_stack(fs_full, obs)

                # Step ALT
                if not done_alt:
                    try:
                        needs_serve = int(alt_ram[BALL_Y]) > 180
                    except Exception:
                        needs_serve = False
                    act = FIRE if needs_serve else right_act
                    obs, reward, terminated, truncated, info = env_alt.step(act)
                    if terminated or truncated:
                        try:
                            is_game_over = env_alt.unwrapped.ale.lives() == 0
                        except Exception:
                            is_game_over = True
                        if is_game_over:
                            done_alt = True
                        else:
                            obs, info = env_alt.reset()
                            fs_alt = [cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)] * 4
                            fs_alt = [cv2.resize(g, (84, 84), interpolation=cv2.INTER_AREA) for g in fs_alt]
                            continue
                    else:
                        update_frame_stack(fs_alt, obs)

            env_full.close()
            env_alt.close()

    # Write CSV
    with open(OUTPUT, "w") as f:
        f.write("\n".join(csv_lines) + "\n")

    # Summary stats
    tracking_frames = sum(1 for line in csv_lines[1:] if line.endswith(",1"))
    total_frames = len(csv_lines) - 1
    pct_tracking = tracking_frames / total_frames * 100 if total_frames > 0 else 0

    print(f"  Total frames: {total_frames:,}")
    print(f"  Tracking frames (ALT closer to ALT ball than FULL ball): {tracking_frames:,} ({pct_tracking:.1f}%)")
    print(f"  Saved to: {OUTPUT}")
