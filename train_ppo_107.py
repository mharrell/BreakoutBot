"""
PPO_107 — Experiment 16: Adversarial Cursor (visible secondary agent)

After 106 experiments, PPO always memorizes in deterministic non-adversarial
environments. BeamRider proved adversarial entities with agency force reactivity.
This experiment ports the mechanism directly: a visible cursor that has its own
state machine, approaches the ball when the paddle isn't tracking, and attacks
(pushes the ball) if the paddle doesn't react.

The key innovation over PPO_105/106: the threat is a VISIBLE ENTITY WITH AGENCY.
The cursor moves across the screen, the agent sees it, and has warning frames
to react before the attack hits. This mirrors BeamRider's fundamental structure:
visible enemies → threat perception → evasive action → survival.

When the paddle tracks the ball, the cursor retreats and stays hidden. This
creates a natural reward gradient: track the ball → cursor stays away → score.
Don't track → cursor attacks → ball dodges → miss → lower score.

The cursor is only visible during THREATENING and ATTACK states. During
APPROACHING and COOLDOWN, it's invisible — the observation looks identical
to standard Breakout. This means in eval (standard Breakout, no wrapper),
there's no missing cursor because tracking keeps a hidden cursor at bay.

Design:
  - Training: ALE/Breakout-v5, frameskip=4, EpisodicLifeEnv (5 lives)
              + AdversarialCursorWrapper (visible stateful adversary)
  - Eval/Check: Standard ALE/Breakout-v5 (NO wrapper) — test transfer
  - Standard PPO: NatureCNN, ent_coef=0.006
  - Target: 50M steps (kill at 10M if SINGLE_SCRIPT at every checkpoint)
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
from adversarial_cursor_wrapper import AdversarialCursorWrapper
from autoreset_wrapper import AutoResetWrapper
from run_label_callback import RunLabelCallback

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_107"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"

ENT_COEF = 0.006
SEED = 107

# Cursor wrapper params (from calibration defaults)
CURSOR_PARAMS = dict(
    approach_speed=2.0,
    tracking_threshold=8,
    threat_radius=8,
    warning_frames=5,
    push_magnitude=4.0,
    cooldown_frames=60,
    cursor_size=4,
)


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
    """Breakout WITH AdversarialCursorWrapper, frameskip=4."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    # Cursor wrapper AFTER FireResetEnv, BEFORE Grayscale/ClipReward
    # (needs RGB observation to draw cursor + ALE for setRAM)
    env = AdversarialCursorWrapper(env, **CURSOR_PARAMS)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Standard Breakout WITHOUT cursor wrapper — test transfer."""
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
    """Standard Breakout WITHOUT cursor wrapper — test transfer."""
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
    pstr = ', '.join(f'{k}={v}' for k, v in CURSOR_PARAMS.items())
    print(f"{RUN_NAME} -- Experiment 16: Adversarial Cursor (visible secondary agent)")
    print(f"  Cursor params: {pstr}")
    print(f"  State machine: APPROACHING (inv) → THREATENING (vis, {CURSOR_PARAMS['warning_frames']}f warn)")
    print(f"                 → ATTACK (vis, ±{CURSOR_PARAMS['push_magnitude']}px push)")
    print(f"                 → COOLDOWN (inv, {CURSOR_PARAMS['cooldown_frames']}f) → respawn")
    print(f"  Training: ALE/Breakout-v5, fs=4, EpisodicLifeEnv, AdversarialCursorWrapper")
    print(f"  Eval/Check: Standard Breakout (NO cursor wrapper) — test transfer")
    print(f"  Key innovation: visible agent with agency. Cursor is only visible")
    print(f"    during THREATENING/ATTACK. Tracking paddle keeps cursor hidden.")
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
        save_freq=500_000, save_path=CHECKPOINT_PATH,
        name_prefix="latest_checkpoint", save_replay_buffer=False, verbose=1)

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME, sticky_actions=False, check_freq=1_000_000,
        n_games=20, make_env_fn=make_check_env, check_deterministic_false=True,
        summary_lines=[
            f"PPO_107 -- Experiment 16: Adversarial Cursor (visible secondary agent)",
            f"Cursor: state machine, visible during THREATENING/ATTACK only",
            f"Params: {pstr}",
            f"Training: ALE/Breakout-v5, fs=4, AdversarialCursorWrapper",
            f"Eval/Check: Standard Breakout (no cursor) — test transfer",
            f"Key: visible agent with agency. Tracking → cursor hides. Not tracking → attack.",
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
