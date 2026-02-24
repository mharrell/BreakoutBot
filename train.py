import ale_py
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from breakout_ram_env import BreakoutRamEnv, BreakoutRamEvalEnv

gym.register_envs(ale_py)

RUN_NAME = "PPO_18"

def make_env():
    return Monitor(BreakoutRamEnv())

def make_eval_env():
    return Monitor(BreakoutRamEvalEnv())

env = DummyVecEnv([make_env] * 32)
eval_env = DummyVecEnv([make_eval_env])

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/",
    log_path=f"./logs/{RUN_NAME}",
    eval_freq=50_000,
    deterministic=True,
    render=False,
    verbose=1,
)

model = PPO(
    "MlpPolicy",
    env,
    device='cpu',
    verbose=1,
    tensorboard_log=f"./tensorboard/{RUN_NAME}",
    n_steps=128,
    batch_size=1024,
    n_epochs=4,
    gamma=0.99,
    learning_rate=2.5e-4,
    ent_coef=0.006,
    vf_coef=0.5,
    clip_range=0.2,
)

# Resume from checkpoint if it exists
checkpoint_path = "models/best_model.zip"
if os.path.exists(checkpoint_path):
    print("Loading existing model...")
    model = PPO.load(checkpoint_path, env=env,
                     custom_objects={"ent_coef": 0.006,
                                     "vf_coef": 0.5,
                                     "clip_range": 0.2,
                                     "learning_rate": 2.5e-4},
                     device='cpu')
    model.set_logger(configure(f"./tensorboard/{RUN_NAME}", ["tensorboard", "stdout"]))
else:
    print("Starting fresh...")

model.learn(
    total_timesteps=10_000_000,
    callback=eval_callback
)

model.save("breakout_ppo_ram_final")
env.close()