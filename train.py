import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback

gym.register_envs(ale_py)

RUN_NAME = "PPO_21"


def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


env = make_atari_env("ALE/Breakout-v5", n_envs=32, seed=42)
env = VecFrameStack(env, n_stack=4)

eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=123)
eval_env = VecFrameStack(eval_env, n_stack=4)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./models/{RUN_NAME}",
    log_path=f"./logs/{RUN_NAME}",
    eval_freq=50_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1,
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=f"./tensorboard/{RUN_NAME}",
    n_steps=128,
    batch_size=1024,
    n_epochs=4,
    gamma=0.99,
    learning_rate=linear_schedule(2.5e-4, 1e-5),
    ent_coef=0.006,
    vf_coef=0.5,
    clip_range=linear_schedule(0.2, 0.05),
    policy_kwargs=dict(net_arch=[64, 64])
)

print(f"Starting fresh {RUN_NAME}...")

model.learn(
    total_timesteps=40_000_000,
    callback=eval_callback
)

model.save(f"./models/{RUN_NAME}/final_model")
env.close()