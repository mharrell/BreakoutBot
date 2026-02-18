import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=0, env_kwargs={"render_mode": "human"})
env = VecFrameStack(env, n_stack=4)

model = PPO.load("models/best_model")

obs = env.reset()
for _ in range(10_000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()


