import ale_py
import gymnasium as gym
import os
import csv
import cv2
import time
import subprocess
import stable_baselines3.common.utils as sb3_utils
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

# --- Run-specific config ---
RUN_NAME = "PPO_26"
STICKY_ACTIONS = True  # must match this run's training env config
MODEL_PATH = f"../models/{RUN_NAME}/best_model"

# --- Measurement config ---
FUNNEL_THRESHOLD = 400          # lowered from 500 per latest review (was missing some real funnel runs)
NUM_GAMES = 10000                # hard cap — script stops on its own, no manual kill needed
OUTPUT_DIR = "../recordings"
LOG_PATH = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_funnel_log.csv")
PLAYBACK_FPS = 60
SLOW_FACTOR = 0.5                # 0.5 = half speed, 0.25 = quarter speed

os.makedirs(OUTPUT_DIR, exist_ok=True)

env_kwargs = {"repeat_action_probability": 0.25} if STICKY_ACTIONS else {}

# Load model
sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None
env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None, env_kwargs=env_kwargs)
env = VecFrameStack(env, n_stack=4)

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
model = PPO.load(MODEL_PATH, custom_objects={"n_envs": 1})
model.set_env(env)


def get_frame(env):
    frame = env.venv.envs[0].unwrapped.ale.getScreenRGB()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def write_video(frames, path):
    """Writes a buffered list of frames to disk in one shot.
    Only called for games that clear the funnel threshold, so the vast
    majority of the 10,000 games never touch disk for video at all."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), PLAYBACK_FPS, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def slow_down_video(input_path, factor=0.5):
    """Re-encode video at slower playback speed using ffmpeg."""
    slow_path = input_path.replace(".mp4", "_slow.mp4")
    result = subprocess.run([
        "ffmpeg", "-i", input_path,
        "-filter:v", f"setpts={1/factor}*PTS",
        "-y", slow_path
    ], capture_output=True)
    if result.returncode == 0:
        return slow_path
    print(f"ffmpeg error: {result.stderr.decode()}")
    return None


# Per-game CSV log — survives crashes/interruptions, appends across re-runs
log_is_new = not os.path.exists(LOG_PATH)
log_file = open(LOG_PATH, "a", newline="")
log_writer = csv.writer(log_file)
if log_is_new:
    log_writer.writerow(["episode", "real_score", "is_funnel", "frame_count", "agent_fps", "timestamp"])
    log_file.flush()

obs = env.reset()
episode = 1
scores = []
funnel_count = 0
frame_buffer = [get_frame(env)]
game_start_time = time.time()

print(f"Run: {RUN_NAME} | Sticky actions: {STICKY_ACTIONS} | Funnel threshold: {FUNNEL_THRESHOLD} | Game cap: {NUM_GAMES}")
print(f"Per-game log: {LOG_PATH}")
print(f"Funnel videos saved to: {OUTPUT_DIR}")
print("-" * 60)

while episode <= NUM_GAMES:
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

            # Only funnel games get written to disk — frame_buffer is discarded
            # otherwise, which is what keeps this fast across 10,000 games.
            is_funnel = real_score >= FUNNEL_THRESHOLD
            if is_funnel:
                funnel_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_funnel_{int(real_score)}pts_{timestamp}.mp4")
                write_video(frame_buffer, save_path)
                print(f"*** FUNNEL SAVED: {save_path} ({expected_duration:.1f}s) ***")

                slow_path = slow_down_video(save_path, factor=SLOW_FACTOR)
                if slow_path:
                    print(f"*** SLOW VERSION: {slow_path} ({expected_duration / SLOW_FACTOR:.1f}s) ***")
                # No early shutdown on 600+ — every funnel in the full 10,000-game
                # sample is wanted this time, not just the first big one.

            funnel_rate = f"{funnel_count}/{episode} ({100*funnel_count/episode:.1f}%)"
            print(f"Game {episode:>5} | Score: {real_score:>6.0f} | Avg: {avg:>6.1f} | "
                  f"Best: {best:>6.0f} | Frames: {game_frames:>5} | Agent FPS: {agent_fps:>5.0f} | "
                  f"Funnels: {funnel_rate}")

            log_writer.writerow([episode, real_score, int(is_funnel), game_frames,
                                  round(agent_fps, 1), datetime.now().isoformat()])
            log_file.flush()

            episode += 1
            obs = env.reset()
            frame_buffer = [get_frame(env)]
            game_start_time = time.time()
        else:
            obs, _, _, _ = env.step([0])

env.close()
log_file.close()

print("-" * 60)
print(f"--- Final Results: {RUN_NAME} ({len(scores)} games) ---")
print(f"Average Score: {sum(scores)/len(scores):.1f}")
print(f"Best Score:    {max(scores):.1f}")
print(f"Worst Score:   {min(scores):.1f}")
print(f"Funnel Rate ({FUNNEL_THRESHOLD}+ pts): {funnel_count}/{len(scores)} ({100*funnel_count/len(scores):.1f}%)")
print(f"Full per-game log saved to: {LOG_PATH}")
