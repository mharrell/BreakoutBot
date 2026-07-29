"""
BeamRider Multi-Life — Phase 1: Test Hard Failure Mechanism

The original BeamRider probe (SEED=202) used EpisodicLifeEnv — life loss
ended the episode. Result: MULTIPLE_SCRIPTS (first reactive argmax in
project history).

This experiment removes EpisodicLifeEnv. The agent gets all 3 lives per
sector. Life loss is soft failure — the episode continues, progress is
preserved, the agent re-spawns. If the hard-failure thesis is correct,
this should produce SINGLE_SCRIPT.

Design:
  - Training: BeamRider WITHOUT EpisodicLifeEnv (soft failure, 3 lives/sector)
  - Eval/Check: Standard BeamRider WITH EpisodicLifeEnv (original config)
  - Standard PPO: NatureCNN, ent_coef=0.006
  - 9 actions (default ALE beam_rider action space)
  - No FireResetEnv (BeamRider may not need it — verify)
  - Target: 10M steps

Comparison:
  Original probe (SEED=202, EpisodicLifeEnv): 10/10 MULTIPLE_SCRIPTS
  This run   (SEED=205, no EpisodicLifeEnv): predicted SINGLE_SCRIPT
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
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, EpisodicLifeEnv
from memorization_check_callback import MemorizationCheckCallback
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "BEAMRIDER_MULTILIFE"
TARGET_STEPS = 10_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 205


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
    """BeamRider WITHOUT EpisodicLifeEnv — soft failure, 3 lives per sector."""
    env = gym.make("ALE/BeamRider-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    # NO EpisodicLifeEnv — life loss does NOT end the episode
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Standard BeamRider WITH EpisodicLifeEnv — hard failure checkpoint."""
    env = gym.make("ALE/BeamRider-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    return env


def make_check_env():
    env = gym.make("ALE/BeamRider-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


if __name__ == "__main__":
    print(f"{RUN_NAME} -- Phase 1: Test Hard Failure Mechanism")
    print(f"  Training: BeamRider WITHOUT EpisodicLifeEnv (soft failure, 3 lives/sector)")
    print(f"  Eval/Check: Standard BeamRider WITH EpisodicLifeEnv (hard failure)")
    print(f"  Original probe (SEED=202, EpisodicLifeEnv): MULTIPLE_SCRIPTS")
    print(f"  Prediction: soft failure -> SINGLE_SCRIPT")
    print()

    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=f"./models/{RUN_NAME}",
        log_path=f"./logs/{RUN_NAME}", eval_freq=50_000,
        n_eval_episodes=20, deterministic=True, render=False, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint", save_replay_buffer=False, verbose=1)

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=10, make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"BEAMRIDER_MULTILIFE -- Phase 1: Test Hard Failure Mechanism",
            f"Training: BeamRider WITHOUT EpisodicLifeEnv (soft failure)",
            f"Eval/Check: Standard BeamRider WITH EpisodicLifeEnv (hard failure)",
            f"Original (SEED=202, EpisodicLifeEnv): MULTIPLE_SCRIPTS at 10M",
            f"Prediction: soft failure removes hard constraint -> SINGLE_SCRIPT",
            f"Policy: NatureCNN, ent_coef={ENT_COEF}, 9 actions",
        ])

    label_callback = RunLabelCallback(RUN_NAME)
    callbacks = CallbackList([eval_callback, checkpoint_callback, memorization_callback, label_callback])

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
