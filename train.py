import os
import glob
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
gym.register_envs(ale_py)

RUN_NAME = "PPO_23"
TOTAL_TIMESTEPS = 100_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule

def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "latest_checkpoint_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

env = make_atari_env("ALE/Breakout-v5", n_envs=64, seed=42)
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

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=CHECKPOINT_PATH,
    name_prefix="latest_checkpoint",
    save_replay_buffer=False,
    verbose=1,
)

callbacks = CallbackList([eval_callback, checkpoint_callback])

resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

if resume_path:
    print(f"Resuming {RUN_NAME} from {resume_path}...")
    model = PPO.load(resume_path, env=env, device="cuda")
    reset_num_timesteps = False
else:
    print(f"Starting fresh {RUN_NAME}...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./tensorboard/{RUN_NAME}",
        n_steps=128,
        batch_size=2048,
        n_epochs=4,
        gamma=0.99,
        learning_rate=linear_schedule(2.5e-4, 1e-5),
        ent_coef=0.006,
        vf_coef=0.5,
        clip_range=linear_schedule(0.2, 0.05),
        policy_kwargs=dict(net_arch=[64, 64])
    )
    reset_num_timesteps = True

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    reset_num_timesteps=reset_num_timesteps,
)

model.save(f"./models/{RUN_NAME}/final_model")
env.close()