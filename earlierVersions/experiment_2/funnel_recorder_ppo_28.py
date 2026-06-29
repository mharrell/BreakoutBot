import ale_py
import gymnasium as gym
import os
import csv
import cv2
import time
import subprocess
import numpy as np
import stable_baselines3.common.utils as sb3_utils
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

# --- Run-specific config ---
# PPO_28 = PPO_26's weights continued, with sticky actions removed.
RUN_NAME = "PPO_28"
STICKY_ACTIONS = False
MODEL_PATH = f"../models/{RUN_NAME}/best_model"

# --- Measurement config ---
FUNNEL_THRESHOLD = 400
NUM_GAMES = 1000   # fresh env per game has overhead — 1k is enough for a clear picture
OUTPUT_DIR = "../../recordings"
LOG_PATH = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_funnel_log.csv")
PLAYBACK_FPS = 60
SLOW_FACTOR = 0.5
RENDER = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_env(seed):
    """Create a fresh environment with a specific seed.
    Recreating the env each game is the only reliable way to vary
    ball launch conditions — VecEnv wrapper chains swallow seed()
    calls before they reach the ALE in this version of SB3."""
    env_kwargs = {"repeat_action_probability": 0.25 if STICKY_ACTIONS else 0.0}
    if RENDER:
        env_kwargs["render_mode"] = "human"
    e = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=int(seed),
                       env_kwargs=env_kwargs)
    return VecFrameStack(e, n_stack=4)


def get_frame(env):
    return cv2.cvtColor(
        env.venv.envs[0].unwrapped.ale.getScreenRGB(), cv2.COLOR_RGB2BGR
    )


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
    if result.returncode == 0:
        return slow_path
    return None


log_is_new = not os.path.exists(LOG_PATH)
log_file = open(LOG_PATH, "a", newline="")
log_writer = csv.writer(log_file)
if log_is_new:
    log_writer.writerow(["episode", "real_score", "is_funnel", "frame_count",
                          "agent_fps", "seed", "timestamp"])
    log_file.flush()

# Load model once — reused across fresh envs each game
sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None
_probe_env = make_env(0)
model = PPO.load(MODEL_PATH, custom_objects={"n_envs": 1})
model.set_env(_probe_env)
_probe_env.close()

episode = 1
scores = []
funnel_count = 0

print(f"Run: {RUN_NAME} | Sticky: {STICKY_ACTIONS} | "
      f"Threshold: {FUNNEL_THRESHOLD} | Cap: {NUM_GAMES}")
print(f"Mode: fresh env per game (guaranteed seed variety)")
print(f"Per-game log: {LOG_PATH}")
print("-" * 60)

while episode <= NUM_GAMES:
    seed = np.random.randint(0, 2**31)
    env = make_env(seed)
    model.set_env(env)
    obs = env.reset()

    frame_buffer = [get_frame(env)]
    game_start_time = time.time()
    done_game = False

    while not done_game:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frame_buffer.append(get_frame(env))

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
                expected_duration = game_frames / PLAYBACK_FPS

                is_funnel = real_score >= FUNNEL_THRESHOLD
                if is_funnel:
                    funnel_count += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(
                        OUTPUT_DIR,
                        f"{RUN_NAME}_funnel_{int(real_score)}pts_{timestamp}.mp4"
                    )
                    write_video(frame_buffer, save_path)
                    print(f"*** FUNNEL SAVED: {save_path} ({expected_duration:.1f}s) ***")
                    slow_path = slow_down_video(save_path, factor=SLOW_FACTOR)
                    if slow_path:
                        print(f"*** SLOW: {slow_path} ***")

                funnel_rate = f"{funnel_count}/{episode} ({100*funnel_count/episode:.1f}%)"
                print(f"Game {episode:>5} | Score: {real_score:>6.0f} | Avg: {avg:>6.1f} | "
                      f"Best: {best:>6.0f} | Frames: {game_frames:>5} | "
                      f"Seed: {seed:>10} | Funnels: {funnel_rate}")

                log_writer.writerow([episode, real_score, int(is_funnel), game_frames,
                                      round(agent_fps, 1), seed, datetime.now().isoformat()])
                log_file.flush()

                episode += 1
                done_game = True
            else:
                obs, _, _, _ = env.step([0])

    env.close()

log_file.close()

print("-" * 60)
print(f"--- Final Results: {RUN_NAME} ({len(scores)} games, varied seeds) ---")
print(f"Average Score:    {sum(scores)/len(scores):.1f}")
print(f"Best Score:       {max(scores):.1f}")
print(f"Worst Score:      {min(scores):.1f}")
zero_count = sum(1 for s in scores if s == 0)
print(f"Zero-score games: {zero_count}/{len(scores)} ({100*zero_count/len(scores):.1f}%)")
print(f"Funnel Rate ({FUNNEL_THRESHOLD}+ pts): {funnel_count}/{len(scores)} "
      f"({100*funnel_count/len(scores):.1f}%)")
print(f"Unique score count: {len(set(scores))}  "
      f"(1-2 = memorization, many = genuine generalization)")
print(f"Full per-game log saved to: {LOG_PATH}")