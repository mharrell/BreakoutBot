"""
PPO_36 — Experiment 6: Per-Frame Ball Noise + Latent Dropout

Combines two anti-memorization mechanisms that attack the problem from
different angles:

1. Per-frame ball velocity noise (ball_noise_std=0.3):
   Gaussian perturbation N(0, 0.3) added to ball velocity each frame,
   rounded to nearest integer. ~10% of frames get a ±1 px/frame kick to
   one component; ~0.3% get ±2. Over hundreds of frames, the ball path
   becomes unpredictable. A memorized open-loop script that assumes
   "ball will be at (x,y) at frame N" will drift and miss. The only
   strategy that works across all noise realizations is to observe and react.

2. Latent dropout (DropoutNatureCNN, p=0.1):
   Dropout in the shared CNN feature space (after linear projection,
   before policy/value heads). During PPO updates, random features are
   zeroed out, preventing the network from encoding frame-precise timing
   information. Features that survive dropout must be robust (ball
   position) rather than brittle (exact frame count).

Training:   DynamicBreakout(ball_noise_std=0.05) — continuous physics + ball noise
Evaluation: GymBreakout(fixed=True) — standard defaults, no noise
Target:     400M steps

Hypothesis: Per-frame ball noise makes scripting provably worse than
tracking. Latent dropout denies the network the architectural capacity
for precise frame timing. Together: the environment resists scripting,
the network resists memorization.

Prediction table:
  det=False ≥20 unique, det=True ≥10 unique → SUCCESS (both reactive)
  det=False ≥20 unique, det=True ≤2 unique  → PARTIAL (like PPO_35, increase noise)
  det=False ≤5 unique,  det=True ≤2 unique  → FAILED (CNN too robust)
  det=False low scores, det=True low scores → CATASTROPHIC (reduce noise)
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
from dropout_features import DropoutNatureCNN

RUN_NAME = "PPO_36"
TARGET_STEPS = 400_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
BALL_NOISE_STD = 0.3   # ~10% of frames get ±1 velocity kick, ~0.3% get ±2
DROPOUT_P = 0.1


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
    """DynamicBreakout with per-frame ball velocity noise (Experiment 6).
    Continuous physics changes + unpredictable ball path."""
    env = DynamicBreakout(ball_noise_std=BALL_NOISE_STD)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """GymBreakout with fixed defaults — no noise, no randomization.
    This is the generalization test."""
    env = GymBreakout(fixed=True)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_check_env():
    """GymBreakout for memorization checks — matches eval env, VecFrameStack-wrapped."""
    env = GymBreakout(fixed=True)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


if __name__ == "__main__":
    pr = DEFAULT_PARAM_RANGES
    print(f"{RUN_NAME} — Experiment 6: Per-Frame Ball Noise + Latent Dropout")
    print(f"  Training: DynamicBreakout(ball_noise_std={BALL_NOISE_STD})")
    print(f"    Continuous physics changes every 60-300 frames")
    print(f"    Per-frame Gaussian ball velocity noise σ={BALL_NOISE_STD}")
    print(f"    ranges: paddle_width=[{pr['paddle_width'][0]},{pr['paddle_width'][1]}], "
          f"ball_speed_y=[{pr['ball_speed_y'][0]},{pr['ball_speed_y'][1]}], "
          f"ball_speed_x=[{pr['ball_speed_x'][0]},{pr['ball_speed_x'][1]}], "
          f"paddle_speed=[{pr['paddle_speed'][0]},{pr['paddle_speed'][1]}]")
    print(f"  Policy:  DropoutNatureCNN(dropout_p={DROPOUT_P})")
    print(f"    Dropout in shared feature space — active during PPO updates")
    print(f"    SB3 handles train/eval switching: dropout OFF during rollouts")
    print(f"  Eval:    GymBreakout(fixed=True) — standard defaults, no noise")
    print(f"  Check:   GymBreakout(fixed=True) — det=True + det=False every 1M steps")
    print(f"  Target:  {TARGET_STEPS:,} steps")
    print()
    print(f"  Hypothesis: Ball noise makes scripting worse than tracking.")
    print(f"  Latent dropout denies frame-precise timing to the network.")
    print(f"  Together: environment resists scripts, network resists memorization.")

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

    memorization_callback = MemorizationCheckCallback(
        run_name=RUN_NAME,
        sticky_actions=False,
        check_freq=1_000_000,
        n_games=20,
        make_env_fn=make_check_env,
        check_deterministic_false=True,
        summary_lines=[
            f"PPO_36 — Experiment 6 (per-frame ball noise + latent dropout)",
            f"Training: DynamicBreakout(ball_noise_std={BALL_NOISE_STD})",
            f"Policy: DropoutNatureCNN(dropout_p={DROPOUT_P})",
            f"Memorization check env: GymBreakout(fixed=True) — same as eval env",
            "LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            "Columns: det=True verdict + stoch_* columns for det=False reactivity check",
            "Experiment 6 hypothesis: ball noise + dropout prevent scripting",
        ],
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        memorization_callback,
        BrickRolloutCallback(),
    ])

    # -----------------------------------------------------------------------
    # Model setup — DropoutNatureCNN with p=0.1 dropout in feature space
    # -----------------------------------------------------------------------
    policy_kwargs = dict(
        features_extractor_class=DropoutNatureCNN,
        features_extractor_kwargs=dict(dropout_p=DROPOUT_P),
    )

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
            policy_kwargs=policy_kwargs,
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
