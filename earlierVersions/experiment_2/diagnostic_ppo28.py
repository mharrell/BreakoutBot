"""
Diagnostic: does PPO_28 actually play varied games?
Tests two approaches side by side:
  A) seed=None, single persistent env, just env.reset() between games
     (same as the original PPO_25/26/27 funnel recorders that worked)
  B) fresh env per game with explicit seed
     (what we've been using for 28/29, which produced identical games)
"""
import ale_py
import gymnasium as gym
import numpy as np
import stable_baselines3.common.utils as sb3_utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

RUN_NAME = "PPO_28"
MODEL_PATH = f"../models/{RUN_NAME}/best_model"
N_GAMES = 20

sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None

print("=" * 60)
print("APPROACH A: seed=None, single persistent env, just env.reset()")
print("(same approach that gave varied scores for PPO_25/26/27)")
print("=" * 60)

env_a = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                       env_kwargs={"repeat_action_probability": 0.0})
env_a = VecFrameStack(env_a, n_stack=4)
model = PPO.load(MODEL_PATH, custom_objects={"n_envs": 1})
model.set_env(env_a)

obs = env_a.reset()
scores_a = []
episode = 1
while episode <= N_GAMES:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env_a.step(action)
    if done[0]:
        lives = info[0].get("lives", -1)
        if lives == 0:
            score = float(info[0].get("episode", {}).get("r", 0))
            scores_a.append(score)
            print(f"  Game {episode:>3}: {score:.0f}")
            episode += 1
            obs = env_a.reset()
        else:
            obs, _, _, _ = env_a.step([0])
env_a.close()

print(f"\nA scores: {scores_a}")
print(f"A unique scores: {len(set(scores_a))}")

print()
print("=" * 60)
print("APPROACH B: fresh env per game with explicit seed")
print("(what we've been using — may not actually vary Breakout's game state)")
print("=" * 60)

scores_b = []
for i in range(1, N_GAMES + 1):
    seed = np.random.randint(0, 2**31)
    env_b = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=int(seed),
                           env_kwargs={"repeat_action_probability": 0.0})
    env_b = VecFrameStack(env_b, n_stack=4)
    model.set_env(env_b)
    obs = env_b.reset()
    done_game = False
    while not done_game:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env_b.step(action)
        if done[0]:
            lives = info[0].get("lives", -1)
            if lives == 0:
                score = float(info[0].get("episode", {}).get("r", 0))
                scores_b.append(score)
                print(f"  Game {i:>3}: {score:.0f} (seed={seed})")
                done_game = True
            else:
                obs, _, _, _ = env_b.step([0])
    env_b.close()

print(f"\nB scores: {scores_b}")
print(f"B unique scores: {len(set(scores_b))}")

print()
print("=" * 60)
print("VERDICT")
print("=" * 60)
if len(set(scores_a)) > 3:
    print("Approach A shows VARIED scores — model is NOT fully memorized.")
    print("The fresh-env seeding was the bug, not the model.")
elif len(set(scores_a)) == 1:
    print("Approach A also shows identical scores — model IS memorized.")
    print("Both approaches agree: genuine behavioral collapse.")
else:
    print(f"Approach A shows limited variance ({len(set(scores_a))} unique scores).")
    print("Possibly partial memorization or small number of ALE start states.")
