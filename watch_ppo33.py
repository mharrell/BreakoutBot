"""
Quick watch script — records the first PPO_33 game scoring 100+ points.
Uses standard frameskip=4, sticky=off (same as eval during training).
Saves MP4 to recordings/ and prints the score/frame count.
"""
import ale_py
import gymnasium as gym
import cv2
import os
import time
import numpy as np
import stable_baselines3.common.utils as sb3_utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

MODEL_PATH = "models/PPO_33/checkpoint"
CHECKPOINT = os.path.join(MODEL_PATH,
    "latest_checkpoint_19200000_steps.zip")  # latest at time of writing
TARGET_SCORE = 100
PLAYBACK_FPS = 30

os.makedirs("recordings", exist_ok=True)

env_kwargs = {"repeat_action_probability": 0.0}
sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None
env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None, env_kwargs=env_kwargs)
env = VecFrameStack(env, n_stack=4)

print(f"Loading: {CHECKPOINT}")
model = PPO.load(CHECKPOINT, custom_objects={"n_envs": 1})
model.set_env(env)

for game in range(1, 201):
    frames = []
    obs = env.reset()
    done = False
    score = 0

    while not done:
        frame = env.venv.envs[0].unwrapped.ale.getScreenRGB()
        frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        lives = info[0].get("lives", -1)
        if done[0]:
            if lives == 0:
                score = float(info[0].get("episode", {}).get("r", 0))
            else:
                obs, _, _, _ = env.step([0])
                done = False

    if score >= TARGET_SCORE:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = f"recordings/PPO_33_watch_{int(score)}pts_{len(frames)}f_{timestamp}.mp4"
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 PLAYBACK_FPS, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"Game {game}: {score:.0f} pts, {len(frames)} frames -> {path}")
        break
    else:
        print(f"Game {game}: {score:.0f} pts ({len(frames)}f) — skipping")

env.close()
