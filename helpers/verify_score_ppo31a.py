import ale_py
import gymnasium as gym
import os, glob
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

cp = r"C:\Users\Silver Pangolin\PycharmProjects\breakoutBot\models\PPO_31a\checkpoint\latest_checkpoint_124800000_steps"

# Same setup as capture/eval — ClipRewardEnv applied
env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                     env_kwargs={"repeat_action_probability": 0.0})
env = VecFrameStack(env, n_stack=4)
model = PPO.load(cp, env=env)

obs = env.reset()
clipped_rewards = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, info = env.step(action)
    clipped_rewards += reward[0]
    if dones[0]:
        lives = info[0].get("lives", -1)
        if lives == 0:
            clipped_score = float(info[0].get("episode", {}).get("r", 0))
            break

env.close()
print(f"Clipped reward sum: {clipped_rewards:.0f}")
print(f"Clipped score (info): {clipped_score:.0f}")

# Now with raw game score (no ClipRewardEnv)
raw_env = gym.make("ALE/Breakout-v5", repeat_action_probability=0.0)
raw_env = gym.wrappers.AtariPreprocessing(raw_env, frame_skip=4, screen_size=84,
                                           grayscale_obs=True, scale_obs=False)
raw_env = gym.wrappers.FrameStackObservation(raw_env, 4)
model = PPO.load(cp, env=raw_env)

obs, _ = raw_env.reset()
raw_score = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = raw_env.step(action)
    raw_score += reward
    if terminated or truncated:
        break

raw_env.close()
print(f"Raw game score: {raw_score:.0f}")
print(f"Ratio (raw/clipped): {raw_score/clipped_score:.1f}x")