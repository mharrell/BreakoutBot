import ale_py
import gymnasium as gym
import os
import stable_baselines3.common.utils as sb3_utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

RUN_NAME = "PPO_25"
MODEL_PATH = f"../models/{RUN_NAME}/best_model"
NUM_GAMES = 50
FUNNEL_THRESHOLD = 200

env = make_atari_env("ALE/Breakout-v5", n_envs=64, seed=None)
env = VecFrameStack(env, n_stack=4)

sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None
model = PPO.load(MODEL_PATH, env=env, custom_objects={"n_envs": 64})

obs = env.reset()
episode = 1
scores = []
funnel_count = 0

while episode <= NUM_GAMES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    lives = info[0].get("lives", -1)

    if done[0]:
        if lives == 0:
            real_score = float(info[0].get('episode', {}).get('r', 0))
            scores.append(real_score)
            avg = sum(scores) / len(scores)
            best = max(scores)

            if real_score >= FUNNEL_THRESHOLD:
                funnel_count += 1
                funnel_tag = "*** FUNNEL ***"
            else:
                funnel_tag = ""

            funnel_rate = f"{funnel_count}/{episode} ({100*funnel_count/episode:.1f}%)"
            print(f"Game {episode:>3} | Score: {real_score:>6} | Avg: {avg:>6.1f} | Best: {best:>6} | Funnels: {funnel_rate} {funnel_tag}")

            episode += 1
            obs = env.reset()
        else:
            obs, _, _, _ = env.step([0] * env.num_envs)

env.close()
print(f"\n--- Final Results ({NUM_GAMES} games) ---")
print(f"Average Score: {sum(scores)/len(scores):.1f}")
print(f"Best Score:    {max(scores):.1f}")
print(f"Worst Score:   {min(scores):.1f}")
print(f"Funnel Rate:   {funnel_count}/{NUM_GAMES} ({100*funnel_count/NUM_GAMES:.1f}%)")