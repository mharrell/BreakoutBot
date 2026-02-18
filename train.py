import ale_py
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback

gym.register_envs(ale_py)

# 8 parallel environments to collect experience faster
env = make_atari_env("ALE/Breakout-v5", n_envs=8, seed=42)
env = VecFrameStack(env, n_stack=4)

eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=123)
eval_env = VecFrameStack(eval_env, n_stack=4)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=10_000,  # changed from 50_000
    deterministic=True,
    render=False,
    verbose=1          # added this
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/",
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    learning_rate=2.5e-4,
    ent_coef=0.01,        # encourages exploration, default is 0.0
    vf_coef=0.5,          # how much to weight value function loss
    clip_range=0.1,       # tighter clipping for more stable updates
)


# Resume from checkpoint if it exists
checkpoint_path = "models/best_model.zip"
if os.path.exists(checkpoint_path):
    print("Loading existing model...")
    model = PPO.load(checkpoint_path, env=env,
                     custom_objects={"ent_coef": 0.01,
                                     "vf_coef": 0.5,
                                     "clip_range": 0.1})
else:
    print("Starting fresh...")

model.learn(
    total_timesteps=5_000_000,
    callback=eval_callback
)

model.save("breakout_ppo_final")
env.close()