"""
PPO_33 — Experiment 5A: Frame Skip Randomization

Trains from scratch with RandomFrameSkip(min=2, max=8) and NO sticky actions.
Tests whether dynamics randomization (variable ball/paddle speed) forces
reactive ball-tracking, unlike sticky actions which only perturb the agent's
output and fail to prevent memorization in ConvNet policies.

Design (see EXPERIMENTS.md Experiment 5):
  Training:   RandomFrameSkip(2-8), frameskip=1 base, repeat_action_probability=0.0
  Evaluation: Standard frameskip=4, repeat_action_probability=0.0
  Target:     400M steps (matches PPO_32 from Experiment 4)

Key test: the model is trained with variable game speed but evaluated on
standard-speed Breakout. If it generalizes, it learned to track the ball.
If it collapses to a script, the frame skip variance wasn't enough to
break memorization.

Comparison groups (all at 400M total steps, single-phase):
  PPO_33:  RandomFrameSkip(2-8), p=0.0 (tested here)
  PPO_32:  Standard frameskip=4, p=0.05 (Experiment 4)
  PPO_30b: p=0.0 x 100M -> p=0.25 x 300M (confirmed memorized)
  PPO_31b: p=0.0 x 300M -> p=0.25 x 100M (confirmed memorized)
"""

import os
import glob
import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    WarpFrame,
    ClipRewardEnv,
)
from memorization_check_callback import MemorizationCheckCallback
from brick_counter import BrickCountingVecWrapper, BrickRolloutCallback
from random_frame_skip import RandomFrameSkip

gym.register_envs(ale_py)

RUN_NAME = "PPO_33"
TARGET_STEPS = 400_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

FRAME_SKIP_MIN = 2
FRAME_SKIP_MAX = 8


def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "latest_checkpoint_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# Training env: single-frame base + random frame skip + standard wrappers
# ---------------------------------------------------------------------------

def make_training_env():
    """Create a single training env with random frame skip and standard Atari
    preprocessing. The random frame skip replaces MaxAndSkipEnv — we build the
    wrapper stack manually instead of using make_atari_env's AtariWrapper."""
    env = gym.make(
        "ALE/Breakout-v5",
        frameskip=1,                       # single-frame mode so our wrapper
        repeat_action_probability=0.0,      # no sticky — testing dynamics, not actions
    )
    env = NoopResetEnv(env, noop_max=30)
    env = RandomFrameSkip(env, min_skip=FRAME_SKIP_MIN, max_skip=FRAME_SKIP_MAX)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Create a single eval env with STANDARD frameskip=4 and no sticky actions.
    The policy must generalize from variable-speed training to standard-speed
    evaluation — this IS the test of reactivity."""
    env = gym.make(
        "ALE/Breakout-v5",
        frameskip=4,                       # standard frameskip for evaluation
        repeat_action_probability=0.0,      # no sticky
    )
    env = NoopResetEnv(env, noop_max=30)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Vectorized environments
    # -----------------------------------------------------------------------

    print(f"Creating {RUN_NAME} training environments "
          f"(RandomFrameSkip {FRAME_SKIP_MIN}-{FRAME_SKIP_MAX}, 32 envs)...")
    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)
    env = BrickCountingVecWrapper(env)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

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

    # Memorization checks on STANDARD frameskip=4, sticky=off.
    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,
        check_freq=10_000_000,
        n_games=20,
        summary_lines=[
            f"PPO_33 — Experiment 5A (frame skip randomization)",
            f"Training: RandomFrameSkip({FRAME_SKIP_MIN}-{FRAME_SKIP_MAX}), p=0.0, 32 envs, from scratch",
            f"Eval: standard frameskip=4, p=0.0",
            "LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            "Memorization checks: standard frameskip=4, sticky=off",
            "Tests: dynamics randomization forces reactivity?",
        ],
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        memorization_callback,
        BrickRolloutCallback(),
    ])

    # -----------------------------------------------------------------------
    # Model setup (fresh or resume)
    # -----------------------------------------------------------------------

    resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

    if resume_path:
        print(f"Resuming {RUN_NAME} from {resume_path}...")
        model = PPO.load(resume_path, env=env, device="cuda")
        reset_num_timesteps = False
    else:
        print(f"Starting {RUN_NAME} from scratch (Experiment 5A: Frame Skip Randomization)...")
        print(f"  Training: RandomFrameSkip({FRAME_SKIP_MIN}-{FRAME_SKIP_MAX}), p=0.0")
        print(f"  Eval:     standard frameskip=4, p=0.0")
        print(f"  Target:   {TARGET_STEPS:,} steps")
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

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------

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
    print(f"\n{RUN_NAME} complete at {model.num_timesteps:,} total steps.")
    print("Next steps:")
    print(f"  1. Run funnel_recorder_ppo_33.py (10k games, standard frameskip=4, sticky=off)")
    print(f"     — this IS the nosticky verification since there are no sticky actions")
    print(f"  2. Compare against PPO_32 (p=0.05 single-phase) and PPO_30b/PPO_31b")
    print(f"  3. If MEMORIZED: move to Experiment 5B (RAM-parameterized physics)")
    env.close()
    eval_env.close()
