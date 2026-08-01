"""
PPO_123 -- Experiment 30: Extreme Bumper (2 bumpers, fast reposition)

PPO_120's single slow bumper produced a 83-pt script by 11M — the argmax
found paths around one obstacle. PPO_123 escalates:

  - 2 independent bumpers (double coverage, half the open lanes)
  - Faster reposition: 60-150 frames (vs 120-300)
  - Only 3+ brick shapes (3-9 bricks each, dropped 1-2 brick micro-shapes)
  - Clean old-bumper cleanup (no indestructible residue accumulation)

A script that works for one bumper configuration fails when either bumper
moves. With 2 bumpers each repositioning every 1-2.5 seconds, the number
of possible obstacle configurations is combinatorially vast. A script
cannot account for all of them.

Key test: eval on CLEAN Breakout (no bumper). If the policy learned to
navigate around unpredictable obstacles reactively, it should perform
well on clean Breakout — and the argmax should NOT be a single script.

Design:
  - Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv (5 lives)
              + ExtremeBumperWrapper(num_bumpers=2, reposition=(60,150))
  - Eval/Check: Standard Breakout (NO bumper) -- transfer test
  - FROM SCRATCH (seed=123)
  - Target: 25M steps
  - Standard PPO: NatureCNN, ent_coef=0.006
"""
import os
import numpy as np
import glob
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from memorization_check_callback import MemorizationCheckCallback
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback
from extreme_bumper_wrapper import ExtremeBumperWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_123"
TARGET_STEPS = 25_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 123

# Extreme bumper: 2 bumpers, fast reposition, 3+ brick shapes only
NUM_BUMPERS = 2
REPOSITION_RANGE = (60, 150)
ROW_RANGE = (4, 13)
BIT_RANGE = (0, 4)


class GrayscaleResize(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[2] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(obs, (self._width, self._height),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None] if resized.ndim == 2 else resized


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
    """Breakout + 2 fast-moving bumpers."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = ExtremeBumperWrapper(
        env, num_bumpers=NUM_BUMPERS,
        row_range=ROW_RANGE, bit_range=BIT_RANGE,
        reposition_range=REPOSITION_RANGE,
    )
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Standard Breakout WITHOUT bumper -- transfer test."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    return env


def make_check_env():
    """Standard Breakout WITHOUT bumper -- memcheck on clean physics."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


if __name__ == "__main__":
    print(f"{RUN_NAME} -- Experiment 30: Extreme Bumper")
    print(f"  Bumpers: {NUM_BUMPERS}")
    print(f"  Reposition range: {REPOSITION_RANGE} frames")
    print(f"  Row range: {ROW_RANGE}, Bit range: {BIT_RANGE}")
    print(f"  Shapes: 3+ bricks only (13 shapes)")
    print(f"  Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv + ExtremeBumper")
    print(f"  Eval/Check: Standard Breakout (NO bumper) -- transfer test")
    print(f"  FROM SCRATCH (seed={SEED}), target {TARGET_STEPS:,} steps")
    print(f"  Entropy coef: {ENT_COEF}")
    print()

    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=f"./models/{RUN_NAME}",
        log_path=f"./logs/{RUN_NAME}", eval_freq=100_000,
        n_eval_episodes=50, deterministic=True, render=False, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=156_250, save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint", save_replay_buffer=False, verbose=1)

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=20, max_check_steps=5_000_000, max_steps_per_game=10_000,
        make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"PPO_123 -- Experiment 30: Extreme Bumper",
            f"Bumpers: {NUM_BUMPERS}, Reposition: {REPOSITION_RANGE} frames",
            f"Shapes: 13 (3+ bricks), Rows: {ROW_RANGE}, Bits: {BIT_RANGE}",
            f"Training: ALE/Breakout-v5 + ExtremeBumperWrapper",
            f"Eval/Check: Standard Breakout (NO bumper) -- transfer test",
        ])

    label_callback = RunLabelCallback(RUN_NAME)
    callbacks = CallbackList([eval_callback, checkpoint_callback,
                              memorization_callback, label_callback])

    resume_path = get_latest_checkpoint(CHECKPOINT_PATH)
    if resume_path:
        print(f"Resuming {RUN_NAME} from {resume_path}...")
        model = PPO.load(resume_path, env=env, device="cuda")
        reset_num_timesteps = False
    else:
        print(f"Starting {RUN_NAME} from scratch (seed={SEED})...")
        model = PPO("CnnPolicy", env, verbose=1, device="cuda",
                    tensorboard_log=f"./tensorboard/{RUN_NAME}",
                    n_steps=128, batch_size=1024, n_epochs=4, gamma=0.99,
                    learning_rate=linear_schedule(2.5e-4, 1e-5),
                    clip_range=linear_schedule(0.2, 0.05),
                    ent_coef=ENT_COEF, vf_coef=0.5,
                    seed=SEED)
        reset_num_timesteps = True

    remaining = TARGET_STEPS - model.num_timesteps
    if remaining <= 0:
        print("Target already reached.")
    else:
        model.learn(total_timesteps=remaining, callback=callbacks,
                    reset_num_timesteps=reset_num_timesteps, tb_log_name=RUN_NAME)

    model.save(f"./models/{RUN_NAME}/final_model")
    print(f"\n{RUN_NAME} complete at {model.num_timesteps:,} total steps.")
    env.close(); eval_env.close()
