import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

RUN_NAME = "PPO_22"
MODEL_PATH = f"../models/{RUN_NAME}/best_model"

env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
env = VecFrameStack(env, n_stack=4)

model = PPO.load(MODEL_PATH, env=env)

obs = env.reset()

total_reward = 0
episode = 1

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]

    if info[0].get("lives", 1) == 0 and done[0]:
        print(f"Game {episode} finished | Score: {total_reward:.0f}")
        total_reward = 0
        episode += 1
        obs = env.reset()
    elif done[0]:
        obs = env.reset()

env.close()