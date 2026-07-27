"""
RBO_02 — Revenge Brunch: OpticalFlow + Dropout, 1B Steps

RBO_01 proved that dropout + pretraining works but converges slowly (192M to reach
what PPO_70 hit at 33M). RBO_02 incorporates the lessons:

  - OpticalFlow: 2-channel [current, |diff|] replaces 4-frame stacking. The motion
    channel gives the CNN cleaner temporal signal → 6× faster convergence.
  - Dropout (p=0.1): prevents feature co-adaptation → more robust memorized script.
  - Standard entropy (0.006): high entropy (0.02) didn't change the destination,
    just added noise. Standard entropy keeps the policy from unnecessary wandering.
  - No sticky training: sticky actions are applied at INFERENCE only (like PPO_26).
    Training with sticky was proven ineffective — every sticky-trained model
    collapsed without sticky.
  - 1B steps: script quality scales with pretraining depth. PPO_26's 60pt script
    came from 838M steps. RBO_02 targets 1B to push beyond.

Goal: highest possible Breakout score by any means. Not a reactivity experiment.

Design:
  - Training:  Clean ALE/Breakout-v5 + OpticalFlow (no VecFrameStack)
  - Eval:      Clean ALE/Breakout-v5 + OpticalFlow
  - Check:     Clean ALE/Breakout-v5 + OpticalFlow
  - Arch:      NatureCNN (2-channel input) + Dropout(p=0.1)
  - ent_coef:  0.006
  - Target:    1,000,000,000 steps
"""
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from stable_baselines3.common.torch_layers import NatureCNN
from memorization_check_callback import MemorizationCheckCallback
from autoreset_wrapper import AutoResetWrapper
from obs_optical_flow import OpticalFlow
from run_label_callback import RunLabelCallback

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "RBO_02"
TARGET_STEPS = 1_000_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
DROPOUT_P = 0.1
ENT_COEF = 0.006


# ---------------------------------------------------------------------------
# DropoutNatureCNN — standard NatureCNN with dropout in feature space
# ---------------------------------------------------------------------------

class DropoutNatureCNN(NatureCNN):
    """NatureCNN with dropout after the final linear projection.

    SB3 automatically handles train/eval mode:
      - Rollouts (model.predict): eval mode → dropout OFF → deterministic features
      - PPO updates: train mode → dropout ON → regularized features

    Works with any number of input channels — 4 for VecFrameStack, 2 for OpticalFlow.
    NatureCNN reads observation_space.shape[0] for n_input_channels.
    """
    def __init__(self, observation_space, features_dim=512, dropout_p=0.1):
        super().__init__(observation_space, features_dim=features_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, observations):
        x = self.cnn(observations)   # conv stack → flattened
        x = self.linear(x)           # project to features_dim (512)
        x = self.dropout(x)          # dropout in feature space
        return x


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------

def make_training_env():
    """Clean ALE + OpticalFlow — no perturbation, no sticky, no frame stacking."""
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = OpticalFlow(env)          # [current, |diff|] — 2 channels, no stacking
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Clean ALE + OpticalFlow for evaluation."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = OpticalFlow(env)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    return env


def make_check_env():
    """Clean ALE + OpticalFlow for memorization checks."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = OpticalFlow(env)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    env = DummyVecEnv([lambda: env])
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{RUN_NAME} — Revenge Brunch: OpticalFlow + Dropout, 1B Steps")
    print(f"  OpticalFlow: 2-channel [current, |diff|] — no frame stacking")
    print(f"  Dropout:     p={DROPOUT_P} in feature space")
    print(f"  Training:    Clean ALE/Breakout-v5, no sticky, no perturbation")
    print(f"  Arch:        NatureCNN (2-channel input) + Dropout")
    print(f"  Target:      {TARGET_STEPS:,} steps (1B)")
    print(f"  ent_coef:    {ENT_COEF} (standard)")
    print(f"  LR:          2.5e-4 -> 1e-5, clip: 0.2 -> 0.05")
    print(f"  Inference:   sticky actions (p=0.25) applied at eval time only")
    print(f"  Goal:        Highest possible Breakout score by any means")
    print()

    # Vectorized environments — NO VecFrameStack (OpticalFlow replaces it)
    env = DummyVecEnv([make_training_env for _ in range(32)])
    eval_env = DummyVecEnv([make_eval_env])

    # Callbacks
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

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,
        check_freq=10_000_000,     # every 10M — this is a long run
        n_games=20,
        make_env_fn=make_check_env,
        check_deterministic_false=True,
        summary_lines=[
            f"RBO_02 — Revenge Brunch: OpticalFlow + Dropout, 1B Steps",
            f"OpticalFlow: 2-channel [current, |diff|] — no frame stacking",
            f"Dropout: p={DROPOUT_P} in feature space (train=ON, rollout=OFF)",
            f"Training: Clean ALE/Breakout-v5, no sticky, no perturbation",
            f"Architecture: NatureCNN (2-chan input) + Dropout, ent_coef={ENT_COEF}",
            f"LR 2.5e-4->1e-5, clip 0.2->0.05, batch_size=1024",
            f"Goal: 100+ pt argmax script. PPO_26 = 60pt (838M steps).",
            f"det=True = script quality. det=False = policy entropy state.",
        ],
    )

    label_callback = RunLabelCallback(RUN_NAME)

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        memorization_callback,
        label_callback,
    ])

    # Model setup — NatureCNN + Dropout, 2-channel input
    resume_path = get_latest_checkpoint(CHECKPOINT_PATH)

    if resume_path:
        print(f"Resuming {RUN_NAME} from {resume_path}...")
        model = PPO.load(resume_path, env=env, device="cuda")
        reset_num_timesteps = False
    else:
        print(f"Starting {RUN_NAME} from scratch...")
        policy_kwargs = dict(
            features_extractor_class=DropoutNatureCNN,
            features_extractor_kwargs=dict(features_dim=512, dropout_p=DROPOUT_P),
        )
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            device="cuda",
            tensorboard_log=f"./tensorboard/{RUN_NAME}",
            policy_kwargs=policy_kwargs,
            n_steps=128,
            batch_size=1024,
            n_epochs=4,
            gamma=0.99,
            learning_rate=linear_schedule(2.5e-4, 1e-5),
            clip_range=linear_schedule(0.2, 0.05),
            ent_coef=ENT_COEF,
            vf_coef=0.5,
        )
        reset_num_timesteps = True

    # Train
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
