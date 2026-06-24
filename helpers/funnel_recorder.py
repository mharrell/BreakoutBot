import ale_py
import gymnasium as gym
import os
import cv2
import time
import shutil
import subprocess
import stable_baselines3.common.utils as sb3_utils
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import glob

def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "latest_checkpoint_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

gym.register_envs(ale_py)

RUN_NAME = "PPO_27"
# MODEL_PATH = f"../models/{RUN_NAME}/best_model"
MODEL_PATH = get_latest_checkpoint(f"../models/{RUN_NAME}/checkpoint")
FUNNEL_THRESHOLD = 500
OUTPUT_DIR = "../recordings"
TEMP_VIDEO = os.path.join(OUTPUT_DIR, "latest_game.mp4")
PLAYBACK_FPS = 60
SLOW_FACTOR = 0.5  # 0.5 = half speed, 0.25 = quarter speed

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None
# env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None)
env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                     env_kwargs={"repeat_action_probability": 0.25})
env = VecFrameStack(env, n_stack=4)

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
model = PPO.load(MODEL_PATH, custom_objects={"n_envs": 1})
model.set_env(env)

def get_frame(env):
    frame = env.venv.envs[0].unwrapped.ale.getScreenRGB()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def start_writer(frame):
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(
        TEMP_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        PLAYBACK_FPS,
        (w, h)
    )
    return writer

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
    else:
        print(f"ffmpeg error: {result.stderr.decode()}")
        return None

obs = env.reset()
episode = 1
scores = []
funnel_count = 0
game_frames = 0
game_start_time = time.time()

first_frame = get_frame(env)
writer = start_writer(first_frame)
writer.write(first_frame)
game_frames = 1

print(f"Recording started. Funnel threshold: {FUNNEL_THRESHOLD}")
print(f"Playback FPS: {PLAYBACK_FPS} | Slow factor: {SLOW_FACTOR}x")
print(f"Saving all games to: {TEMP_VIDEO}")
print(f"Funnel games saved to: {OUTPUT_DIR}")
print("-" * 50)

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    frame = get_frame(env)
    writer.write(frame)
    game_frames += 1

    lives = info[0].get("lives", -1)

    if done[0]:
        if lives == 0:
            real_score = float(info[0].get('episode', {}).get('r', 0))
            scores.append(real_score)
            avg = sum(scores) / len(scores)
            best = max(scores)

            elapsed = time.time() - game_start_time
            agent_fps = game_frames / elapsed if elapsed > 0 else 60
            expected_duration = game_frames / PLAYBACK_FPS

            writer.release()

            if real_score >= FUNNEL_THRESHOLD:
                funnel_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(OUTPUT_DIR, f"funnel_{int(real_score)}pts_{timestamp}.mp4")
                shutil.copy(TEMP_VIDEO, save_path)
                print(f"*** FUNNEL SAVED: {save_path} ({expected_duration:.1f}s) ***")

                slow_path = slow_down_video(save_path, factor=SLOW_FACTOR)
                if slow_path:
                    print(f"*** SLOW VERSION: {slow_path} ({expected_duration / SLOW_FACTOR:.1f}s) ***")
                    if real_score >= 600:
                        print("600+ point game captured! Shutting down.")
                        env.close()
                        exit()

            funnel_rate = f"{funnel_count}/{episode} ({100*funnel_count/episode:.1f}%)"
            print(f"Game {episode:>3} | Score: {real_score:>6.0f} | Avg: {avg:>6.1f} | Best: {best:>6.0f} | Frames: {game_frames:>5} | Agent FPS: {agent_fps:>5.0f} | Video: {expected_duration:.1f}s | Funnels: {funnel_rate}")

            episode += 1
            obs = env.reset()

            game_frames = 1
            game_start_time = time.time()
            next_frame = get_frame(env)
            writer = start_writer(next_frame)
            writer.write(next_frame)

        else:
            obs, _, _, _ = env.step([0])

env.close()