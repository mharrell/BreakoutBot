"""
record_gameplay.py — Record representative gameplay videos for PPO models.

Records one game each of:
  - Normal mode (no interventions)
  - Intervention mode (ball/paddle teleports on paddle bounces)

For each mode, plays 5 games and saves the one closest to the median score.
Output: recordings/{model}_normal.mp4, recordings/{model}_intervention.mp4
"""

import sys
import os
import numpy as np
import cv2
import gymnasium as gym
from gym_breakout import GymBreakout
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.monitor import Monitor


# --- Wrappers ---------------------------------------------------------

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
    """Teleports ball (y-axis) and paddle (x-axis) after random paddle bounces."""

    def __init__(self, env, teleport_prob=0.3, ball_y_jitter=15,
                 paddle_x_jitter=30, seed=None):
        super().__init__(env)
        self.teleport_prob = teleport_prob
        self.ball_y_jitter = ball_y_jitter
        self.paddle_x_jitter = paddle_x_jitter
        self._rng = np.random.default_rng(seed)
        self.intervention_count = 0
        self.interventions = []
        self._just_teleported = False

    def reset(self, **kwargs):
        self.intervention_count = len(self.interventions)
        self.interventions = []
        self._just_teleported = False
        return self.env.reset(**kwargs)

    def step(self, action):
        brk = self.env._env
        prev_vy = brk.ball_v[0] if brk else None

        obs, reward, terminated, truncated, info = self.env.step(action)

        cur_vy = brk.ball_v[0] if brk else None
        self._just_teleported = False
        if (prev_vy is not None and cur_vy is not None
                and prev_vy > 0 and cur_vy < 0):
            if self._rng.random() < self.teleport_prob:
                ball = brk.ball
                paddle = brk.paddle
                dy = int(round(self._rng.normal(0, self.ball_y_jitter)))
                new_y = max(15, min(paddle.pos[0] - 10, ball.pos[0] + dy))
                ball.pos[0] = new_y
                dx = int(round(self._rng.normal(0, self.paddle_x_jitter)))
                new_x = max(0, min(160 - paddle.size[1], paddle.pos[1] + dx))
                paddle.pos[1] = new_x
                self.interventions.append((brk.step_count, new_y, new_x))
                self._just_teleported = True

        return obs, reward, terminated, truncated, info


# --- Helpers ----------------------------------------------------------

def load_model(run_name):
    import glob
    best = f"./models/{run_name}/best_model.zip"
    if os.path.exists(best):
        return PPO.load(best, device="cuda")
    ckpt_dir = f"./models/{run_name}/checkpoint"
    if os.path.isdir(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "latest_checkpoint_*_steps.zip"))
        if ckpts:
            latest = max(ckpts, key=os.path.getmtime)
            return PPO.load(latest, device="cuda")
    raise FileNotFoundError(f"No model found for {run_name}")


def build_env(intervene=False):
    """Build env chain. Returns (vec_env, breakout_ref, intervention_ref)."""
    base = GymBreakout(fixed=True)
    ib = None
    if intervene:
        ib = InterventionBreakout(base, teleport_prob=0.3,
                                  ball_y_jitter=15, paddle_x_jitter=30)
        env = GrayscaleResize(ib)
    else:
        env = GrayscaleResize(base)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env, base, ib


def play_game(model, intervene, deterministic):
    """Play one game. Returns (score, frames_list, interventions_list)."""
    env, base, ib = build_env(intervene=intervene)
    obs = env.reset()
    brk = base._env

    frames_list = []
    interventions_list = []
    done = False
    score = 0

    while not done:
        # Capture frame before stepping
        frame = base.render()
        if frame is not None:
            frames_list.append(frame.copy())

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done_arr, info = env.step(action)
        done = done_arr[0]

        if ib and ib._just_teleported:
            interventions_list.append(len(frames_list) - 1)

    score = brk.score if brk else 0
    n_int = ib.intervention_count if ib else 0
    env.close()
    return score, frames_list, interventions_list, n_int


def record_video(frames, output_path, fps=30, interventions=None):
    """Save frames to MP4 video. Overlays score info on each frame."""
    if not frames:
        print(f"  WARNING: No frames to record for {output_path}")
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    h, w = frames[0].shape
    display_h, display_w = 640, 480

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (display_w, display_h))

    for i, frame in enumerate(frames):
        display = cv2.resize(frame, (display_w, display_h),
                             interpolation=cv2.INTER_NEAREST)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Frame counter
        cv2.putText(display, f"Frame: {i}", (8, display_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1,
                    cv2.LINE_AA)

        # Flash on intervention frames
        if interventions and i in interventions:
            cv2.putText(display, "TELEPORT!",
                        (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                        (0, 0, 255), 3)

        out.write(display)

    out.release()
    return True


def record_representative(model, run_name, intervene, deterministic,
                          output_path, n_warmup=5):
    """Play n_warmup games, pick the one closest to median score, record it."""
    print(f"  Calibrating: {n_warmup} warmup games...", flush=True)
    scores = []
    all_frames = []
    all_interventions = []

    for i in range(n_warmup):
        score, frames, int_frames, n_int = play_game(
            model, intervene, deterministic)
        scores.append(score)
        all_frames.append(frames)
        all_interventions.append(int_frames)
        mode_str = "INT" if intervene else "NORM"
        det_str = "det" if deterministic else "stoch"
        print(f"    Game {i+1}: score={score:.0f}, int={n_int}, "
              f"frames={len(frames)} [{mode_str} {det_str}]", flush=True)

    scores = np.array(scores)
    median = np.median(scores)
    # Pick game closest to median
    best_idx = np.argmin(np.abs(scores - median))

    print(f"  Median score: {median:.0f}")
    print(f"  Recording game {best_idx+1} (score={scores[best_idx]:.0f}) "
          f"-> {output_path}", flush=True)

    success = record_video(all_frames[best_idx], output_path, fps=30,
                           interventions=all_interventions[best_idx])
    if success:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved: {output_path} ({size_mb:.1f} MB)", flush=True)
    else:
        print(f"  FAILED to save video!", flush=True)

    return scores[best_idx], median


# --- Main -------------------------------------------------------------

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "PPO_35"
    deterministic = "--stochastic" not in sys.argv
    out_dir = "recordings/videos"

    print(f"{'='*60}")
    print(f"  Recording: {run}")
    print(f"  deterministic={deterministic}")
    print(f"  Output: {out_dir}/")
    print(f"{'='*60}")

    model = load_model(run)

    # --- Normal mode ---
    print(f"\n--- Normal mode (no interventions) ---")
    out_normal = os.path.join(out_dir, f"{run}_normal.mp4")
    score_n, med_n = record_representative(
        model, run, intervene=False, deterministic=deterministic,
        output_path=out_normal, n_warmup=5)

    # --- Intervention mode ---
    print(f"\n--- Intervention mode (teleports) ---")
    out_inter = os.path.join(out_dir, f"{run}_intervention.mp4")
    score_i, med_i = record_representative(
        model, run, intervene=True, deterministic=deterministic,
        output_path=out_inter, n_warmup=5)

    print(f"\n{'='*60}")
    print(f"  DONE: {run}")
    print(f"  Normal representative: {score_n:.0f} (median of 5: {med_n:.0f})")
    print(f"  Intervention representative: {score_i:.0f} (median of 5: {med_i:.0f})")
    print(f"{'='*60}")
