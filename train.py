import os
import glob
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
gym.register_envs(ale_py)

RUN_NAME = "PPO_25"
TOTAL_TIMESTEPS = 400_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
PPO24_CHECKPOINT_PATH = "./models/PPO_24/checkpoint"

def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule

def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "latest_checkpoint_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

env = make_atari_env("ALE/Breakout-v5", n_envs=64, seed=None)
env = VecFrameStack(env, n_stack=4)

eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None)
eval_env = VecFrameStack(eval_env, n_stack=4)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"./models/{RUN_NAME}",
    log_path=f"./logs/{RUN_NAME}",
    eval_freq=50_000,
    n_eval_episodes=50,
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

# Priority 1: resume from a PPO_25 checkpoint (if this run has been interrupted)
resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

if resume_path:
    print(f"Resuming {RUN_NAME} from {resume_path}...")
    model = PPO.load(resume_path, env=env, device="cuda")
    reset_num_timesteps = False

else:
    # Priority 2: start from PPO_24's final checkpoint
    ppo24_path = get_latest_checkpoint(PPO24_CHECKPOINT_PATH)
    if not ppo24_path:
        raise FileNotFoundError(
            f"No PPO_24 checkpoint found at {PPO24_CHECKPOINT_PATH}. "
            f"Check the path and try again."
        )

    print(f"Starting {RUN_NAME} from PPO_24 final checkpoint: {ppo24_path}")
    model = PPO.load(ppo24_path, env=env, device="cuda")

    # Restore schedules — not saved in checkpoint, must be reassigned
    model.learning_rate = linear_schedule(2.5e-4, 1e-5)
    model.clip_range = linear_schedule(0.2, 0.05)
    model.ent_coef = 0.006
    model.tensorboard_log = f"./tensorboard/{RUN_NAME}"

    reset_num_timesteps = False  # Continue step count from PPO_24

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    reset_num_timesteps=reset_num_timesteps,
    tb_log_name=RUN_NAME,
)

model.save(f"./models/{RUN_NAME}/final_model")
env.close()