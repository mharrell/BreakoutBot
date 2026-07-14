"""
Quick sanity check: PPO_27 with sticky ON (p=0.25).
Verifies the model loads correctly and performs as expected.
If PPO_27 gets normal scores here (not 100% zeros), the nosticky
collapse is real behavior, not a loading/environment issue.

Expected: avg roughly in 20s-40s range, ~21% zero-score.
50 games, sticky on, deterministic=True.
"""
import ale_py
import gymnasium as gym
import stable_baselines3.common.utils as sb3_utils
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

MODEL_PATH = "models/PPO_27/final_model"

sb3_utils.check_for_correct_spaces = lambda *args, **kwargs: None
env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                     env_kwargs={"repeat_action_probability": 0.25})
env = VecFrameStack(env, n_stack=4)

print(f"Loading: {MODEL_PATH}")
model = PPO.load(MODEL_PATH, custom_objects={"n_envs": 1})
model.set_env(env)

scores = []
for game in range(1, 51):
    obs = env.reset()
    done = False
    frames = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        frames += 1
        if done[0]:
            lives = info[0].get("lives", -1)
            if lives == 0:
                score = float(info[0].get("episode", {}).get("r", 0))
                scores.append(score)
                print(f"Game {game:>3}: {score:>6.0f} pts, {frames:>5} frames | "
                      f"avg={sum(scores)/len(scores):.1f}")
            else:
                obs, _, _, _ = env.step([0])
                done = False

env.close()

zero_count = sum(1 for s in scores if s == 0)
unique = len(set(scores))
print(f"\nPPO_27 sticky-ON sanity check ({len(scores)} games):")
print(f"  Avg: {sum(scores)/len(scores):.1f}")
print(f"  Zero-score: {zero_count}/{len(scores)} ({100*zero_count/len(scores):.1f}%)")
print(f"  Unique scores: {unique}")
print(f"  Best: {max(scores):.0f}  Worst: {min(scores):.0f}")
print(f"\nExpected (from 10k eval): avg ~low 20s, ~21% zero-score")
print(f"If this matches: loading is fine, nosticky collapse is real.")
print(f"If 100% zeros here too: loading/env issue.")
