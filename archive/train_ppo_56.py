"""
PPO_56 — ALE Experiment 4c: X+Y Gentle (cooldown=60, prob=5% each)

PPO_54 tests X+Y at PPO_51's settings (cooldown=30, prob=10% each — ~2-4
interventions per axis per life). This run tests half that intensity: 60-frame
cooldowns at 5% probability per axis, giving ~1 intervention per axis per life.

The gentler rate asks: is the combined X+Y too disruptive at 10% each, or can
the model learn under lighter dual randomization? If PPO_56 converges to higher
scripts than PPO_54 with less collapse, the dual-axis approach works but needs
a lighter touch. If PPO_56 also stalls, the problem isn't the rate — it's that
two perturbation axes compound beyond what a CNN can track.

Design:
  - Training:  ALE/Breakout-v5 + ALEBreakoutXMirror(cooldown=60, prob=0.05)
                                + ALEBreakoutYPerturb(cooldown=60, prob=0.05, range=8)
               Dual perturbation at half the base rate.
  - Eval:      ALE/Breakout-v5 (clean, no perturbation)
  - Check:     ALE/Breakout-v5 (clean)
  - Arch:      NatureCNN (standard, no dropout)
  - Target:    50M steps

Hyperparams: n_envs=32, n_steps=128, batch_size=1024, n_epochs=4,
             gamma=0.99, lr=2.5e-4->1e-5, clip=0.2->0.05, ent_coef=0.006

Predecessors:
  PPO_51 (X-only, cooldown=30, prob=10%): 5-18 pt SINGLE_SCRIPT
  PPO_53 (X-only, cooldown=60, prob=5%): 3-5 pt script by 7M — too little
  PPO_54 (X+Y, cooldown=30, prob=10% each): running
  PPO_55 (Y-only, cooldown=30, prob=10%): running — clean ablation
  PPO_56 (this run): dual-axis at 5% — gentle version of PPO_54
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
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv
from memorization_check_callback import MemorizationCheckCallback
from ale_breakout_x_mirror import ALEBreakoutXMirror
from ale_breakout_y_perturb import ALEBreakoutYPerturb
from autoreset_wrapper import AutoResetWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_56"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
MIRROR_COOLDOWN = 60    # frames — guaranteed clean window between X-mirrors (1.0s)
MIRROR_PROB = 0.05      # per-frame probability once cooldown expires
PERTURB_COOLDOWN = 60   # frames — guaranteed clean window between Y-shifts (1.0s)
PERTURB_PROB = 0.05     # per-frame probability once cooldown expires
PERTURB_RANGE = 8       # ±pixels shift in ball Y


class GrayscaleResize(gym.ObservationWrapper):
    """Resize grayscale to (height, width, 1) — compatible with VecFrameStack."""
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
    """ALE/Breakout-v5 with X-mirror + Y-perturb at gentle rates.

    Pipeline: ALE -> NoopResetEnv -> ALEBreakoutXMirror -> ALEBreakoutYPerturb ->
              FireResetEnv -> EpisodicLifeEnv -> GrayscaleResize ->
              ClipRewardEnv -> Monitor
    """
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = ALEBreakoutXMirror(env, cooldown_frames=MIRROR_COOLDOWN, mirror_prob=MIRROR_PROB)
    env = ALEBreakoutYPerturb(env, cooldown_frames=PERTURB_COOLDOWN,
                              perturb_prob=PERTURB_PROB, perturb_range=PERTURB_RANGE)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Clean ALE/Breakout-v5 — no perturbation, the generalization test.

    Pipeline: ALE -> NoopResetEnv -> FireResetEnv -> EpisodicLifeEnv ->
              GrayscaleResize -> ClipRewardEnv -> Monitor -> AutoResetWrapper
    """
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
    """Clean ALE/Breakout-v5 for memorization checks.

    NO EpisodicLifeEnv — the MemorizationCheckCallback has its own life-loss
    handling (step([1]) to respawn) which is incompatible with EpisodicLifeEnv
    setting done=True on every life.

    Pipeline: ALE -> NoopResetEnv -> FireResetEnv ->
              GrayscaleResize -> ClipRewardEnv -> Monitor ->
              DummyVecEnv[1] -> VecFrameStack(4)
    """
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"{RUN_NAME} — ALE Experiment 4c: X+Y Gentle (cooldown=60, prob=5% each)")
    print(f"  X-Mirror:  cooldown={MIRROR_COOLDOWN}f, prob={MIRROR_PROB*100:.0f}%")
    print(f"  Y-Perturb: cooldown={PERTURB_COOLDOWN}f, prob={PERTURB_PROB*100:.0f}%, range=±{PERTURB_RANGE}px")
    print(f"  Training: ALE/Breakout-v5 + ALEBreakoutXMirror + ALEBreakoutYPerturb")
    print(f"    Both perturbations at half the base rate (PPO_54 = 10% each)")
    print(f"    Expected: ~1 X-mirror + ~1 Y-shift per life, clean windows between")
    print(f"  Eval:    ALE/Breakout-v5 — clean, no perturbation")
    print(f"  Check:   ALE/Breakout-v5 — clean, no perturbation")
    print(f"           det=True + det=False every 1M steps")
    print(f"  Arch:    NatureCNN (standard, no dropout)")
    print(f"  Target:  {TARGET_STEPS:,} steps (~50M)")
    print(f"  Envs:    32 parallel")
    print(f"  LR:      2.5e-4 -> 1e-5 (linear), clip: 0.2 -> 0.05 (linear)")
    print(f"  ent_coef=0.006, batch_size=1024, n_steps=128, n_epochs=4, gamma=0.99")
    print()
    print(f"  Hypothesis: If PPO_54 (10% each) is too disruptive, halving the rate")
    print(f"  may let the model learn under dual randomization. If PPO_56 also")
    print(f"  stalls, the dual-axis approach fundamentally breaks the learning")
    print(f"  signal regardless of rate.")
    print(f"  Predecessors:")
    print(f"    PPO_51 (X-only, cooldown=30, prob=10%): 5-18 pt SINGLE_SCRIPT")
    print(f"    PPO_53 (X-only, cooldown=60, prob=5%): 3-5 pt script — too little")
    print(f"    PPO_54 (X+Y, cooldown=30, prob=10% each): running")
    print(f"    PPO_55 (Y-only, cooldown=30, prob=10%): running — clean ablation")

    # -------------------------------------------------------------------
    # Vectorized environments
    # -------------------------------------------------------------------
    env = DummyVecEnv([make_training_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # -------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------
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
            f"PPO_56 — ALE Experiment 4c (X+Y gentle, cooldown=60f, prob=5% each)",
            f"X-Mirror: cooldown={MIRROR_COOLDOWN}f, prob={MIRROR_PROB}",
            f"Y-Perturb: cooldown={PERTURB_COOLDOWN}f, prob={PERTURB_PROB}, ±{PERTURB_RANGE}px",
            f"Training: ALE/Breakout-v5 + ALEBreakoutXMirror + ALEBreakoutYPerturb",
            f"Policy: NatureCNN (standard, no dropout)",
            f"Memorization check env: ALE/Breakout-v5 (clean — no perturbation)",
            f"LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            f"Gentle dual perturbation — half the rate of PPO_54",
            f"Predecessor PPO_54 (X+Y @ 10% each): running",
        ],
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        memorization_callback,
    ])

    # -------------------------------------------------------------------
    # Model setup — NatureCNN (standard, no custom feature extractor)
    # -------------------------------------------------------------------
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

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
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
