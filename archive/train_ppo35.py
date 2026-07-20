"""
PPO_35 — Experiment 5C: Continuous Mid-Game Physics Interpolation

Escalation from Experiment 5B (per-episode randomization, which produced a
memorized 64-point script). Instead of fixed params per episode, parameters
change continuously: every 60-300 frames, 0-3 of (paddle_width, paddle_speed,
ball_speed) smoothly interpolate to new random values over 30 frames.

The hypothesis: per-episode randomization wasn't enough because the agent
could detect its parameter set from the first few frames and select a
parameter-conditioned script. Continuous mid-game changes make that
impossible — the physics can shift at any moment, forcing genuine
moment-to-moment reactivity.

Training:   DynamicBreakout() — continuous parameter interpolation
Evaluation: GymBreakout(fixed=True) — standard defaults
Target:     400M steps
"""

import os
import glob
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from memorization_check_callback import MemorizationCheckCallback
from brick_counter import BrickCountingVecWrapper, BrickRolloutCallback
from gym_breakout import GymBreakout, DynamicBreakout, DEFAULT_PARAM_RANGES

RUN_NAME = "PPO_35"
TARGET_STEPS = 400_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"


class GrayscaleResize(gym.ObservationWrapper):
    """Resize grayscale to (height, width, 1) — compatible with VecFrameStack."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )
    def observation(self, obs):
        resized = cv2.resize(obs, (self._width, self._height),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None]


def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule


def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "latest_checkpoint_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


def make_training_env():
    """DynamicBreakout: parameters change continuously mid-game (Experiment 5C)."""
    env = DynamicBreakout()
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """GymBreakout with fixed defaults — the generalization test."""
    env = GymBreakout(fixed=True)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    pr = DEFAULT_PARAM_RANGES
    print(f"{RUN_NAME} — Experiment 5C: Continuous Mid-Game Physics Interpolation")
    print(f"  Training: DynamicBreakout — params change every 60-300 frames")
    print(f"    ranges: paddle_width=[{pr['paddle_width'][0]},{pr['paddle_width'][1]}], "
          f"ball_speed_y=[{pr['ball_speed_y'][0]},{pr['ball_speed_y'][1]}], "
          f"ball_speed_x=[{pr['ball_speed_x'][0]},{pr['ball_speed_x'][1]}], "
          f"paddle_speed=[{pr['paddle_speed'][0]},{pr['paddle_speed'][1]}]")
    print(f"  Eval:   GymBreakout(fixed=True) — standard defaults")
    print(f"  Note:   Memorization checks use ALE Breakout (not GymBreakout).")
    print(f"          Eval callback scores are the reliable behavioral data.")
    print(f"  Target: {TARGET_STEPS:,} steps")

    # -----------------------------------------------------------------------
    # Vectorized environments
    # -----------------------------------------------------------------------
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

    # Memorization checks use ALE Breakout (hardcoded in callback).
    # For PPO_35 (GymBreakout-trained), the eval callback is the
    # reliable behavioral data on the correct environment.
    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,
        check_freq=1_000_000,
        n_games=20,
        summary_lines=[
            f"PPO_35 — Experiment 5C (continuous mid-game physics interpolation)",
            f"Training: DynamicBreakout — params change every 60-300 frames",
            f"Eval: GymBreakout fixed defaults",
            "LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            "WARNING: memorization check uses ALE Breakout (not GymBreakout)",
            "Eval callback is the reliable behavioral metric for this experiment",
        ],
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        memorization_callback,
        BrickRolloutCallback(),
    ])

    # -----------------------------------------------------------------------
    # Model setup
    # -----------------------------------------------------------------------
    resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

    if resume_path:
        print(f"Resuming {RUN_NAME} from {resume_path}...")
        model = PPO.load(resume_path, env=env, device="cuda")
        reset_num_timesteps = False
    else:
        print(f"Starting {RUN_NAME} from scratch...")
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
    env.close()
    eval_env.close()
