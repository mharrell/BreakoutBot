"""
PPO_31a — Phase 1 of Experiment 3 (long non-sticky pretraining)
No sticky actions. Trains for 400M steps from scratch.
When complete, PPO_31b loads this checkpoint and adds sticky actions.

Part of the non-sticky pretraining duration sweep:
  PPO_30: 100M non-sticky → sticky (tests "basic competency is enough" hypothesis)
  PPO_31: 400M non-sticky → sticky (tests "depth matters" hypothesis)
"""
import os
import glob
import ale_py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from memorization_check_callback import MemorizationCheckCallback

gym.register_envs(ale_py)

RUN_NAME = "PPO_31a"
TARGET_STEPS = 400_000_000
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

# No sticky actions in Phase 1
env = make_atari_env("ALE/Breakout-v5", n_envs=32, seed=None,
                     env_kwargs={"repeat_action_probability": 0.0})
env = VecFrameStack(env, n_stack=4)

eval_env = make_atari_env("ALE/Breakout-v5", n_envs=1, seed=None,
                          env_kwargs={"repeat_action_probability": 0.0})
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

# Periodic in-memory check for behavioral collapse to a fixed action
# sequence — see EXPERIMENTS.md Experiment 2/3. sticky_actions=False here
# matches this script's training env config.
memorization_callback = MemorizationCheckCallback(
    run_name=RUN_NAME,
    sticky_actions=False,
    check_freq=10_000_000,
    n_games=20,
)

callbacks = CallbackList([eval_callback, checkpoint_callback, memorization_callback])

resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

if resume_path:
    print(f"Resuming {RUN_NAME} from {resume_path}...")
    model = PPO.load(resume_path, env=env, device="cuda")
    reset_num_timesteps = False
else:
    print(f"Starting {RUN_NAME} from scratch (no sticky actions)...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda",
        tensorboard_log=f"./tensorboard/{RUN_NAME}",
        n_steps=128,
        batch_size=1024,
        n_epochs=4,
        gamma=0.99,
        learning_rate=linear_schedule(2.5e-4, 1e-5),
        clip_range=linear_schedule(0.2, 0.05),
        ent_coef=0.006,
        vf_coef=0.5,
    )
    reset_num_timesteps = True

# Absolute step target — safe across restarts
remaining = TARGET_STEPS - model.num_timesteps
print(f"{RUN_NAME}: current step {model.num_timesteps:,}, "
      f"training {remaining:,} more steps to reach {TARGET_STEPS:,}")

if remaining <= 0:
    print("Target already reached. Nothing to do.")
else:
    model.learn(
        total_timesteps=remaining,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
        tb_log_name=RUN_NAME,
    )

model.save(f"./models/{RUN_NAME}/final_model")
print(f"\n{RUN_NAME} Phase 1 complete at {model.num_timesteps:,} steps.")
print(f"Next: run train_ppo31b.py to start the sticky-action phase.")
env.close()
eval_env.close()
