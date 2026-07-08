import ale_py
import gymnasium as gym
import os, glob
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

CHECKPOINT_DIR = r"C:\Users\Silver Pangolin\PycharmProjects\breakoutBot\models\PPO_31a\checkpoint"
VIDEO_DIR = r"C:\Users\Silver Pangolin\PycharmProjects\breakoutBot\recordings\videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

cp = r"C:\Users\Silver Pangolin\PycharmProjects\breakoutBot\models\PPO_31a\checkpoint\latest_checkpoint_124800000_steps"
print(f"Loading {os.path.basename(cp)}")

env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                     env_kwargs={"repeat_action_probability": 0.0})
env = VecFrameStack(env, n_stack=4)
model = PPO.load(cp, env=env)

obs = env.reset()
frames = []
done_flag = False
frame_count = 0

while not done_flag:
    frames.append(obs[0, :, :, -1].copy())  # last channel of frame stack
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, info = env.step(action)
    frame_count += 1
    if dones[0]:
        lives = info[0].get("lives", -1)
        if lives == 0:
            score = float(info[0].get("episode", {}).get("r", 0))
            done_flag = True

env.close()

print(f"Captured {len(frames)} frames, score: {score:.0f}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
upscaled = 336
out_path = os.path.join(VIDEO_DIR, f"ppo31a_381pts_{len(frames)}frames.mp4")
out = cv2.VideoWriter(out_path, fourcc, 30.0, (upscaled, upscaled))

for frame in frames:
    img = cv2.resize(frame, (upscaled, upscaled), interpolation=cv2.INTER_NEAREST)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out.write(img_bgr)

out.release()
print(f"Video saved: {out_path}")
print(f"Play with: start {out_path}")