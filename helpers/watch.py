import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import os

gym.register_envs(ale_py)

RUN_NAME = "PPO_24"
MODEL_PATH = f"../models/{RUN_NAME}/best_model"

env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None, env_kwargs={"render_mode": "human"})
env = VecFrameStack(env, n_stack=4)

print(f"Loading model from: {os.path.abspath(MODEL_PATH)}")
model = PPO.load(MODEL_PATH, env=env)

obs = env.reset()
total_reward = 0
episode = 1

while True:
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]

    lives = info[0].get("lives", -1)

    if done[0]:
        if lives == 0:
            # Get real unclipped score from the ALE directly
            print(f"Info keys: {info[0]}")
            real_score = env.venv.envs[0].unwrapped.ale.getEpisodeFrameNumber()
            print(f"Game {episode} finished | Bricks: {total_reward:.0f} | Real Score: {info[0].get('episode', {}).get('r', '?')}")
            total_reward = 0
            episode += 1
            obs = env.reset()
        else:
            obs, _, _, _ = env.step([0])