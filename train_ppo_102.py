"""
PPO_102 — Experiment 11: Ball-Tracking Representation Supervision

BeamRider proved that hard failure constraints force reactive policies. But
Breakout's soft failure mode means PPO can always find a degenerate script.

Instead of changing the reward or environment, this experiment changes the
FEATURES THEMSELVES. A ball-position prediction head is jointly trained
with PPO on the shared CNN features. The CNN receives gradients from both:
  1. PPO policy/value loss — learn to play Breakout
  2. Ball-position MSE loss  — learn to see the ball

If ball position is baked into the features, any policy that uses those
features MUST be reactive. PPO can't "choose" to ignore the ball when the
features literally encode ball (x, y).

Unlike PPO_85 (frozen pre-trained features → collapse), the aux gradient
stays alive DURING policy learning, preventing the features from drifting
into a representation that supports blind scripts.

Design:
  - BallPositionWrapper reads ball (x,y) from RAM into info dict
  - BallPositionRecorder accumulates ball positions from VecEnv steps
  - BallTrackingCallback trains aux head after each PPO rollout
  - Aux head: CNN features (512) -> Linear(64) -> ReLU -> Linear(2) -> (x,y)
  - Separate Adam optimizer for CNN + aux head (lr=1e-4)
  - Standard PPO: NatureCNN, ent_coef=0.006
  - Eval/Check: Clean ALE (no aux supervision, standard wrappers)
  - Target: 50M steps
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
from ball_position_wrapper import BallPositionWrapper
from ball_tracking_callback import BallTrackingCallback, BallPositionRecorder
from run_label_callback import RunLabelCallback

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_102"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

AUX_LR = 1e-4
AUX_BATCH = 256
AUX_EPOCHS = 2
ENT_COEF = 0.006
SEED = 102


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
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    # BallPositionWrapper BEFORE wrappers that modify obs (reads RAM directly)
    env = BallPositionWrapper(env)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
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
    print(f"{RUN_NAME} — Experiment 11: Ball-Tracking Representation Supervision")
    print(f"  Aux LR: {AUX_LR} | Batch: {AUX_BATCH} | Epochs: {AUX_EPOCHS}")
    print(f"  Training: Clean ALE + ball-position aux supervision on CNN features")
    print(f"  Eval/Check: Clean ALE (no aux supervision)")
    print(f"  Hypothesis: ball position in features -> reactive policy by construction")
    print()

    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)
    # Wrap AFTER VecFrameStack so recorder sees the same 4-frame obs as the policy
    recorder = BallPositionRecorder(env)
    env = recorder

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=f"./models/{RUN_NAME}",
        log_path=f"./logs/{RUN_NAME}", eval_freq=50_000,
        n_eval_episodes=50, deterministic=True, render=False, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint", save_replay_buffer=False, verbose=1)

    ball_tracking_callback = BallTrackingCallback(
        recorder=recorder,
        aux_lr=AUX_LR,
        batch_size=AUX_BATCH,
        aux_epochs=AUX_EPOCHS,
        verbose=1,
    )

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=20, make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"PPO_102 — Experiment 11: Ball-Tracking Representation Supervision",
            f"Training: clean ALE + ball-position aux loss on shared CNN features",
            f"Aux: CNN(512)->Linear(64)->ReLU->Linear(2)->(ball_x,ball_y) | "
            f"lr={AUX_LR} batch={AUX_BATCH} epochs={AUX_EPOCHS}",
            f"Eval/Check: Clean ALE (no aux supervision)",
            f"Hypothesis: ball-encoding features -> reactive policy by construction",
            f"Policy: NatureCNN, ent_coef={ENT_COEF}",
        ])

    label_callback = RunLabelCallback(RUN_NAME)
    callbacks = CallbackList([
        eval_callback, checkpoint_callback, ball_tracking_callback,
        memorization_callback, label_callback,
    ])

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
