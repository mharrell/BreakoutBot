"""
PPO_34 — Experiment 5B: Domain Randomization via Custom Breakout Physics

Trains on GymBreakout with randomized paddle width, ball speed, and paddle
speed (resampled each episode). Evaluates on fixed default parameters.

This is true physics randomization, not approximated via frame skip. Tests
whether varying the game's fundamental parameters forces reactive ball-tracking
in a way that sticky actions and frame skip randomization could not.

Design (see EXPERIMENTS.md Experiment 5B):
  Training:   GymBreakout(fixed=False) — random physics per episode
  Evaluation: GymBreakout(fixed=True)  — standard Breakout defaults
  Target:     400M steps

Comparison groups:
  PPO_34:  Domain randomization (physics params) — tested here
  PPO_33:  Frame skip randomization 2-8 (Experiment 5A)
  PPO_32:  Sticky p=0.05 single-phase (Experiment 4)
"""

import os
import glob
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
import cv2
import numpy as np
from memorization_check_callback import MemorizationCheckCallback
from brick_counter import BrickCountingVecWrapper, BrickRolloutCallback
from gym_breakout import GymBreakout, DEFAULT_PARAM_RANGES

RUN_NAME = "PPO_34"
TARGET_STEPS = 400_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"


class GrayscaleResize(gym.ObservationWrapper):
    """Resize grayscale observation from (H, W) to (height, width, 1).
    Outputs 3D with channel-last so VecFrameStack can concatenate on axis=-1."""
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
        return resized[:, :, None]  # add channel dim: (84, 84) -> (84, 84, 1)


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
    """Single training env: GymBreakout with randomized physics + standard
    Atari preprocessing (grayscale 84x84, clip reward, Monitor)."""
    env = GymBreakout(fixed=False)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Single eval env: GymBreakout with FIXED default parameters + same
    preprocessing. This IS the generalization test — the agent trains on
    randomized physics but must perform on standard Breakout defaults."""
    env = GymBreakout(fixed=True)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Parameter ranges (display only — actual ranges defined in gym_breakout)
    # -----------------------------------------------------------------------
    pr = DEFAULT_PARAM_RANGES
    print(f"{RUN_NAME} — Experiment 5B: Domain Randomization via Physics Parameters")
    print(f"  Training: paddle_width=[{pr['paddle_width'][0]},{pr['paddle_width'][1]}], "
          f"ball_speed_y=[{pr['ball_speed_y'][0]},{pr['ball_speed_y'][1]}], "
          f"ball_speed_x=[{pr['ball_speed_x'][0]},{pr['ball_speed_x'][1]}], "
          f"paddle_speed=[{pr['paddle_speed'][0]},{pr['paddle_speed'][1]}]")
    print(f"  Eval:     paddle_width=15, ball_speed=[4,2], paddle_speed=3")
    print(f"  Target:   {TARGET_STEPS:,} steps")

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

    # Memorization checks on fixed-parameter env, sticky=off.
    # This is the only reliable behavioral test.
    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,
        check_freq=1_000_000,
        n_games=20,
        summary_lines=[
            f"PPO_34 — Experiment 5B (physics domain randomization)",
            f"Training: random paddle_width/ball_speed/paddle_speed per episode",
            f"Eval: fixed defaults (paddle=15, speed=[4,2], paddle_speed=3)",
            "LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            "Memorization checks: fixed defaults, sticky=off",
            "Tests: does physics randomization force reactivity?",
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
    print("Next steps:")
    print(f"  1. Run funnel_recorder_ppo_34.py (10k games, fixed defaults)")
    print(f"  2. Compare against PPO_32 (p=0.05) and PPO_33 (frame skip)")
    env.close()
    eval_env.close()
