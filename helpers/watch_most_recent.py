import ale_py
import gymnasium as gym
import glob
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

RUN_NAME = "PPO_26"
CHECKPOINT_PATH = f"../models/{RUN_NAME}/checkpoint"
DETERMINISTIC = True  # True = best play, False = natural/varied

def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "latest_checkpoint_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

model_path = get_latest_checkpoint(CHECKPOINT_PATH)
if not model_path:
    raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}")

env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None, env_kwargs={"render_mode": "human", "repeat_action_probability": 0.25})
env = VecFrameStack(env, n_stack=4)

print(f"Loading model from: {os.path.abspath(model_path)}")
model = PPO.load(model_path, env=env)

obs = env.reset()
episode = 1
scores = []

while True:
    action, _ = model.predict(obs, deterministic=DETERMINISTIC)
    obs, reward, done, info = env.step(action)

    lives = info[0].get("lives", -1)

    if done[0]:
        if lives == 0:
            real_score = info[0].get('episode', {}).get('r', '?')
            scores.append(real_score)
            avg = sum(scores) / len(scores)
            best = max(scores)
            print(f"Game {episode:>3} | Score: {real_score:>6} | Avg: {avg:>6.1f} | Best: {best:>6}")
            episode += 1
            obs = env.reset()
        else:
            obs, _, _, _ = env.step([0])