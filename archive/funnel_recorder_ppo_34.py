"""
Funnel recorder for PPO_34 — Experiment 5B evaluation.
Evaluates the domain-randomized model on FIXED default Breakout parameters
(paddle_width=15, ball_speed=[4,2], paddle_speed=3).

This IS the generalization test: the agent trains on randomized physics
but must perform on standard defaults. If it generalizes, we see varied
scores. If it memorized, we see ≤2 unique scores.

No sticky actions in this environment — sticky-off verification is automatic.
"""
import os
import csv
import cv2
import time
import subprocess
import numpy as np
import gymnasium as gym
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from gym_breakout import GymBreakout
import cv2

RUN_NAME = "PPO_34"
MODEL_PATH = f"models/PPO_34/final_model"
NUM_GAMES = 10000
FUNNEL_THRESHOLD = 400
OUTPUT_DIR = "recordings"
LOG_PATH = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_funnel_log.csv")
PLAYBACK_FPS = 60
SLOW_FACTOR = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


class GrayscaleResize(gym.ObservationWrapper):
    """Resize grayscale observation to (height, width, 1)."""
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


def make_env():
    env = GymBreakout(fixed=True)         # standard defaults
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def get_frame(env):
    frame = env.render()
    if frame is None:
        return np.zeros((210, 160, 3), dtype=np.uint8)
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def write_video(frames, path):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), PLAYBACK_FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def slow_down_video(input_path, factor=0.5):
    slow_path = input_path.replace(".mp4", "_slow.mp4")
    result = subprocess.run([
        "ffmpeg", "-i", input_path,
        "-filter:v", f"setpts={1/factor}*PTS",
        "-y", slow_path
    ], capture_output=True)
    return slow_path if result.returncode == 0 else None


# --- Vectorized env for single-env eval ---
env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=4)

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
model = PPO.load(MODEL_PATH, custom_objects={"n_envs": 1})
model.set_env(env)

log_is_new = not os.path.exists(LOG_PATH)
log_file = open(LOG_PATH, "a", newline="")
log_writer = csv.writer(log_file)
if log_is_new:
    log_writer.writerow(["episode", "real_score", "is_funnel", "frame_count",
                          "agent_fps", "timestamp"])
    log_file.flush()

obs = env.reset()
episode = 1
scores = []
funnel_count = 0
frame_buffer = [get_frame(env.venv.envs[0])]
game_start_time = time.time()

print(f"Run: {RUN_NAME} | Fixed defaults (paddle=15, speed=[4,2], ps=3)")
print(f"  Training: random paddle_width/ball_speed/paddle_speed per episode")
print(f"  Threshold: {FUNNEL_THRESHOLD} | Cap: {NUM_GAMES}")
print(f"Per-game log: {LOG_PATH}")
print("-" * 60)

while episode <= NUM_GAMES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame_buffer.append(get_frame(env.venv.envs[0]))

    lives = info[0].get("lives", -1)

    if done[0]:
        if lives == 0:
            real_score = float(info[0].get("episode", {}).get("r", 0))
            scores.append(real_score)
            avg = sum(scores) / len(scores)
            best = max(scores)
            elapsed = time.time() - game_start_time
            game_frames = len(frame_buffer)
            agent_fps = game_frames / elapsed if elapsed > 0 else 60

            is_funnel = real_score >= FUNNEL_THRESHOLD
            if is_funnel:
                funnel_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    OUTPUT_DIR,
                    f"{RUN_NAME}_funnel_{int(real_score)}pts_{timestamp}.mp4"
                )
                write_video(frame_buffer, save_path)
                slow_path = slow_down_video(save_path, factor=SLOW_FACTOR)
                print(f"*** FUNNEL SAVED: {save_path} ***")
                if slow_path:
                    print(f"*** SLOW: {slow_path} ***")

            print(f"Game {episode:>5} | Score: {real_score:>6.0f} | Avg: {avg:>6.1f} | "
                  f"Best: {best:>6.0f} | Frames: {game_frames:>5} | "
                  f"Agent FPS: {agent_fps:>5.0f} | Funnels: {funnel_count}/{episode}")

            log_writer.writerow([episode, real_score, int(is_funnel), game_frames,
                                  round(agent_fps, 1), datetime.now().isoformat()])
            log_file.flush()

            episode += 1
            obs = env.reset()
            frame_buffer = [get_frame(env.venv.envs[0])]
            game_start_time = time.time()
        else:
            obs, _, _, _ = env.step([0])

env.close()
log_file.close()

# --- Results ---
print("-" * 60)
print(f"--- Final Results: {RUN_NAME} ({len(scores)} games) ---")
print(f"Average Score:    {sum(scores)/len(scores):.1f}")
print(f"Median Score:     {sorted(scores)[len(scores)//2]:.1f}")
print(f"Best Score:       {max(scores):.1f}")
print(f"Worst Score:      {min(scores):.1f}")
zero_count = sum(1 for s in scores if s == 0)
print(f"Zero-score games: {zero_count}/{len(scores)} ({100*zero_count/len(scores):.1f}%)")
print(f"Funnel Rate ({FUNNEL_THRESHOLD}+ pts): {funnel_count}/{len(scores)} "
      f"({100*funnel_count/len(scores):.2f}%)")

unique = len(set(scores))
print(f"\nUnique scores: {unique}")
if unique <= 2:
    verdict = "MEMORIZED"
    print(f"  ≤2 unique scores — deterministic script. Domain randomization failed at this stage.")
elif 3 <= unique <= 9:
    verdict = "PARTIAL"
    print(f"  3-9 unique scores — some variance but not full generalization.")
else:
    verdict = "GENERALIZING"
    print(f"  ≥10 unique scores — genuine score variance, likely reactive ball-tracking.")
