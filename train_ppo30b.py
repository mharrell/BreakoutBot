"""
PPO_30b — Phase 2 of Experiment 3 (sticky-action fine-tuning)
Loads PPO_30a's final_model and trains for 300M more steps WITH sticky actions.
LR schedule starts mid-range (1e-4 → 1e-5) rather than restarting from 2.5e-4
to avoid the aggressive early updates that contributed to PPO_28/29's collapse.

Run AFTER train_ppo30a.py has completed.
"""
import os
import glob
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

gym.register_envs(ale_py)

RUN_NAME = "PPO_30b"
SOURCE_MODEL = "./models/PPO_30a/final_model"
ADDITIONAL_STEPS = 300_000_000
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

# Sticky actions on for Phase 2
env = make_atari_env("ALE/Breakout-v5", n_envs=32, seed=None,
                     env_kwargs={"repeat_action_probability": 0.25})
env = VecFrameStack(env, n_stack=4)

eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                          env_kwargs={"repeat_action_probability": 0.25})
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

resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

if resume_path:
    print(f"Resuming {RUN_NAME} from checkpoint {resume_path}...")
    model = PPO.load(resume_path, env=env, device="cuda")
    reset_num_timesteps = False
else:
    if not os.path.exists(SOURCE_MODEL + ".zip"):
        raise FileNotFoundError(
            f"Source model not found at {SOURCE_MODEL}.zip\n"
            f"Make sure train_ppo30a.py has completed before running this script."
        )
    print(f"Starting {RUN_NAME} from {SOURCE_MODEL} with sticky actions added...")
    model = PPO.load(SOURCE_MODEL, env=env, device="cuda",
                     custom_objects={"n_envs": 32})

    # Mid-range LR restart — deliberately conservative.
    # PPO_28/29 restarted from 2.5e-4 (full high LR) and collapsed in ~30M steps.
    # Starting at 1e-4 gives the policy room to adapt without being pushed hard
    # enough to discover memorization attractors before stickiness can suppress them.
    model.learning_rate = linear_schedule(1e-4, 1e-5)
    model.clip_range = linear_schedule(0.15, 0.05)
    model.ent_coef = 0.006
    model.tensorboard_log = f"./tensorboard/{RUN_NAME}"
    reset_num_timesteps = False  # preserve step count from Phase 1

# Absolute step target — safe across restarts
TARGET_STEPS = model.num_timesteps + ADDITIONAL_STEPS
remaining = TARGET_STEPS - model.num_timesteps
print(f"{RUN_NAME}: current step {model.num_timesteps:,}, "
      f"training {remaining:,} more steps to reach {TARGET_STEPS:,}")

model.learn(
    total_timesteps=remaining,
    callback=callbacks,
    reset_num_timesteps=reset_num_timesteps,
    tb_log_name=RUN_NAME,
)

model.save(f"./models/{RUN_NAME}/final_model")
print(f"\n{RUN_NAME} Phase 2 complete at {model.num_timesteps:,} total steps.")
print(f"Run funnel_recorder_ppo_30b.py to evaluate single-env performance.")
env.close()
eval_env.close()
