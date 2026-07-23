"""
PPO_52 — ALE Experiment 3: X-Mirror Ball (cooldown=30, prob=20%)

Pairs with PPO_51 (prob=10%). Tests whether 20% post-cooldown mirror
probability forces horizontal tracking while the 30-frame cooldown
guarantees clean trajectory windows.

Follows PPO_49-50 which tested X-mirror at 60-80% per-frame without
cooldown and found the rates were too aggressive — the ball mirrored
every 1-2 frames, destroying the learning signal.

At 20% with 30-frame cooldown: mirrors fire faster after cooldown
(~5 frames expected wait) vs PPO_51's ~10 frames. Clean half-second
windows are the same. More aggressive test of tracking requirement.

Design:
  - Training:  ALE/Breakout-v5 + ALEBreakoutXMirror(cooldown=30, prob=0.20)
               Ball X reflects across center. 30-frame cooldown
               guarantees 0.5s clean trajectory windows between mirrors.
  - Eval:      ALE/Breakout-v5 (clean, no mirror)
  - Check:     ALE/Breakout-v5 (clean)
  - Arch:      NatureCNN (standard, no dropout)
  - Target:    50M steps

Hyperparams: n_envs=32, n_steps=128, batch_size=1024, n_epochs=4,
             gamma=0.99, lr=2.5e-4->1e-5, clip=0.2->0.05, ent_coef=0.006
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
from autoreset_wrapper import AutoResetWrapper

import ale_py
gym.register_envs(ale_py)

RUN_NAME = "PPO_52"
TARGET_STEPS = 50_000_000
CHECKPOINT_PATH = f"./models/{RUN_NAME}/checkpoint"
MIRROR_COOLDOWN = 30   # frames — guaranteed clean window between mirrors
MIRROR_PROB = 0.20     # per-frame probability once cooldown expires


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
    """ALE/Breakout-v5 with X-mirror at 20%.

    Pipeline: ALE -> NoopResetEnv -> ALEBreakoutXMirror ->
              FireResetEnv -> EpisodicLifeEnv -> GrayscaleResize ->
              ClipRewardEnv -> Monitor
    """
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = ALEBreakoutXMirror(env, cooldown_frames=MIRROR_COOLDOWN, mirror_prob=MIRROR_PROB)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    return env


def make_eval_env():
    """Clean ALE/Breakout-v5 — no mirror, the generalization test.

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
    print(f"{RUN_NAME} — ALE Experiment 3: X-Mirror (cooldown={MIRROR_COOLDOWN}f, prob={MIRROR_PROB*100:.0f}%)")
    print(f"  Training: ALE/Breakout-v5 + ALEBreakoutXMirror(cooldown={MIRROR_COOLDOWN}, prob={MIRROR_PROB})")
    print(f"    Ball X reflects across playfield center.")
    print(f"    {MIRROR_COOLDOWN}-frame cooldown = 0.5s guaranteed clean trajectory between mirrors.")
    print(f"    After cooldown: {MIRROR_PROB*100:.0f}% per-frame chance to mirror.")
    print(f"    Ball stays in open space — no fake brick scores.")
    print(f"    Vertical trajectory is natural — paddle approach always clean")
    print(f"    Camping on one side = miss the ball when it mirrors")
    print(f"  Eval:    ALE/Breakout-v5 — clean, no mirror")
    print(f"  Check:   ALE/Breakout-v5 — clean, no teleport")
    print(f"           det=True + det=False every 1M steps")
    print(f"  Arch:    NatureCNN (standard, no dropout)")
    print(f"  Target:  {TARGET_STEPS:,} steps (~50M)")
    print(f"  Envs:    32 parallel")
    print(f"  LR:      2.5e-4 -> 1e-5 (linear), clip: 0.2 -> 0.05 (linear)")
    print(f"  ent_coef=0.006, batch_size=1024, n_steps=128, n_epochs=4, gamma=0.99")
    print()
    print(f"  Hypothesis: X-mirror forces horizontal tracking. Unlike position")
    print(f"  teleport (PPO_47-48), the ball never appears inside bricks, so")
    print(f"  there are no fake scores. Unlike paddle-bounce teleport")
    print(f"  (PPO_44-46), hitting the ball is never punished. A stationary")
    print(f"  paddle works 0% of the time — the ball will mirror away.")
    print(f"  Predecessors:")
    print(f"    PPO_44-46 (paddle-bounce teleport 10-20%): all dead scripts")
    print(f"    PPO_47-48 (mid-flight position teleport 60-80%): degenerate")
    print(f"      stationary-paddle strategies with fake brick scores")
    print(f"    PPO_49-50 (X-mirror 60-80%): too aggressive, 0-1 pt scripts")

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
            f"PPO_52 — ALE Experiment 3 (X-mirror, cooldown={MIRROR_COOLDOWN}f, prob={MIRROR_PROB})",
            f"Training: ALE/Breakout-v5 + ALEBreakoutXMirror(cooldown={MIRROR_COOLDOWN}, prob={MIRROR_PROB})",
            f"Policy: NatureCNN (standard, no dropout)",
            f"Memorization check env: ALE/Breakout-v5 (clean — no mirror)",
            f"LR 2.5e-4->1e-5, clip 0.2->0.05, ent_coef=0.006, batch_size=1024",
            f"X-mirror w/ cooldown: 0.5s clean windows, ball reflected across center",
            f"Predecessors PPO_49-50 (60-80% X-mirror): too aggressive, learning signal destroyed",
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
