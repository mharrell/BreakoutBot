import ale_py
import gymnasium as gym
import glob
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

if __name__ == "__main__":
    RUN_NAME = "PPO_25"

    FUNNEL_THRESHOLD = 200

    model_path = f"../models/{RUN_NAME}/best_model"


    env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)

    print(f"Loading model from: {os.path.abspath(model_path)}")
    model = PPO.load(model_path, env=env)

    obs = env.reset()
    episode = 1
    scores = []
    funnel_count = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        lives = info[0].get("lives", -1)

        if done[0]:
            if lives == 0:
                real_score = info[0].get('episode', {}).get('r', '?')
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
                obs, _, _, _ = env.step([0])