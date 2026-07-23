"""
PPO_58 — ALE Experiment 4b-REPLICATE-2: Y-Perturb Only (cooldown=30, prob=10%, ±8px, seed=58)

Exact duplicate of PPO_55 with a different RNG seed. PPO_55 showed the best
det=False diversity in the project (9 unique, avg 11.6, best 16 at 10M), but
we've never run the same config twice. This replicate answers two questions:

1. Run-to-run variance: how much does the same config differ with a different
   perturbation sequence and weight initialization?
2. Signal confirmation: is the 10M peak in PPO_55 real, or a noise spike?

Design:
  - Training:  ALE/Breakout-v5 + ALEBreakoutYPerturb(cooldown=30, prob=0.10, range=8, seed=58)
               Ball Y shifts by ±8px with 30-frame cooldown between perturbations.
               Different perturbation sequence from PPO_55 (seed=58 vs default).
  - Eval:      ALE/Breakout-v5 (clean, no perturbation)
  - Check:     ALE/Breakout-v5 (clean)
  - Arch:      NatureCNN (standard, no dropout)
  - Target:    50M steps

Hyperparams: n_envs=32, n_steps=128, batch_size=1024, n_epochs=4,
             gamma=0.99, lr=2.5e-4->1e-5, clip=0.2->0.05, ent_coef=0.006

Predecessors:
  PPO_55 (Y-only, cooldown=30, prob=10%, default seed): 9 unique, avg 11.6
    at 10M — best det=False diversity in the project
  PPO_58 (this run): exact replicate with seed=58 — quantifies run-to-run variance
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
from ale_breakout_y_perturb import ALEBreakoutYPerturb
from autoreset_wrapper import AutoResetWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_58"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
PERTURB_COOLDOWN = 30   # frames — guaranteed clean window between Y-shifts
PERTURB_PROB = 0.10     # per-frame probability once cooldown expires
PERTURB_RANGE = 8       # ±pixels shift in ball Y
SEED = 58               # different from PPO_55's default for run-to-run comparison

# Set seeds for reproducible but different initialization
np.random.seed(SEED)
import torch
torch.manual_seed(SEED)


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
    """ALE/Breakout-v5 with Y-perturb only (seed=58).

    Pipeline: ALE -> NoopResetEnv -> ALEBreakoutYPerturb ->
              FireResetEnv -> EpisodicLifeEnv -> GrayscaleResize ->
              ClipRewardEnv -> Monitor
    """
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = ALEBreakoutYPerturb(env, cooldown_frames=PERTURB_COOLDOWN,
                              perturb_prob=PERTURB_PROB, perturb_range=PERTURB_RANGE,
                              seed=SEED)
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
    print(f"{RUN_NAME} — ALE Experiment 4b-REPLICATE-2: Y-Perturb Only (seed={SEED})")
    print(f"  Y-Perturb: cooldown={PERTURB_COOLDOWN}f, prob={PERTURB_PROB*100:.0f}%, range=±{PERTURB_RANGE}px, seed={SEED}")
    print(f"  Training: ALE/Breakout-v5 + ALEBreakoutYPerturb")
    print(f"    Ball Y shifted ±{PERTURB_RANGE}px = 3-5 frame arrival timing change")
    print(f"    {PERTURB_COOLDOWN}f cooldown = clean trajectory windows between shifts")
    print(f"    No X-mirror — isolates the timing axis")
    print(f"    Seed={SEED} — replicate of PPO_55 for run-to-run variance estimate")
    print(f"  Eval:    ALE/Breakout-v5 — clean, no perturbation")
    print(f"  Check:   ALE/Breakout-v5 — clean, no perturbation")
    print(f"           det=True + det=False every 1M steps")
    print(f"  Arch:    NatureCNN (standard, no dropout)")
    print(f"  Target:  {TARGET_STEPS:,} steps (~50M)")
    print(f"  Envs:    32 parallel")
    print(f"  LR:      2.5e-4 -> 1e-5 (linear), clip: 0.2 -> 0.05 (linear)")
    print(f"  ent_coef=0.006, batch_size=1024, n_steps=128, n_epochs=4, gamma=0.99")
    print()
    print(f"  This is an EXACT REPLICATE of PPO_55 with a different seed.")
    print(f"  Purpose: measure run-to-run variance for the first time in this project.")
    print(f"  PPO_55 reached 9 unique, avg 11.6, best 16 at 10M (det=False).")
    print(f"  Does a different perturbation sequence produce different results?")
    print(f"  Predecessors:")
    print(f"    PPO_55 (Y-only, default seed): 9 unique, avg 11.6 at 10M")
    print(f"    PPO_58 (this run): replicate with seed=58")

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
            f"PPO_58 — ALE Experiment 4b-REPLICATE-2 (Y-perturb only, cooldown={PERTURB_COOLDOWN}f, seed={SEED})",
            f"Y-Perturb: cooldown={PERTURB_COOLDOWN}f, prob={PERTURB_PROB}, ±{PERTURB_RANGE}px, seed={SEED}",
            f"Training: ALE/Breakout-v5 + ALEBreakoutYPerturb (Y-only, no X-mirror)",
            f"Policy: NatureCNN (standard, no dropout)",
            f"Memorization check env: ALE/Breakout-v5 (clean — no perturbation)",
            f"LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            f"EXACT REPLICATE of PPO_55 with seed={SEED} — run-to-run variance measurement",
            f"PPO_55 (default seed): 9 unique, avg 11.6, best 16 at 10M det=False",
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
        print(f"Starting {RUN_NAME} from scratch (seed={SEED})...")
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
