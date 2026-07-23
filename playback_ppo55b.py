"""
Interactive Playback — watch PPO_55b and manually launch the ball.

PPO_55b at ent_coef=0.02 has no functional deterministic argmax — it can't
even launch the ball when playing deterministically. This script lets you
watch it play and manually press FIRE to launch the ball, then observe
whether the model tracks it reactively or just flails.

Controls:
  SPACE or F   — force FIRE for one frame (launch the ball)
  A            — toggle auto-FIRE mode (keep firing every frame)
  Q or ESC     — quit
  Everything else — let the model choose the action (det=True, argmax)

Display (console):
  Frame number, score, lives remaining, model's chosen action, and whether
  the current action was a manual override.

At the end, saves a session log to recordings/playback_PPO_55b_<timestamp>.txt
with per-life breakdown: score, frames, manual fires, model FIREs, and
whether the model ever scored after a manual launch.

The first time the ball is in play, watch carefully: does the model move
the paddle toward the ball, or does it follow a fixed pattern?
"""
import sys
import time
import msvcrt
import cv2
import numpy as np
import gymnasium as gym
from collections import deque
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv, NoopResetEnv, FireResetEnv, EpisodicLifeEnv,
)
from autoreset_wrapper import AutoResetWrapper

import ale_py
gym.register_envs(ale_py)

DEFAULT_MODEL = "./models/PPO_55b/checkpoint/latest_checkpoint_32000000_steps.zip"
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
FRAME_STACK = 4
TOTAL_LIVES = 5

ACTION_NAMES = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}


# ---------------------------------------------------------------------------
# Preprocessing (matches training pipeline)
# ---------------------------------------------------------------------------
class PreprocessObs(gym.ObservationWrapper):
    """Grayscale + 84x84 resize -> (84, 84, 1) uint8."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.p_width = width
        self.p_height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[-1] == 3:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if obs.ndim == 2:
            obs = obs[:, :, None]
        obs = cv2.resize(obs, (self.p_width, self.p_height),
                         interpolation=cv2.INTER_AREA)
        if obs.ndim == 2:
            obs = obs[:, :, None]
        return obs


class SingleFrameStack(gym.Wrapper):
    """Frame stack for a single (non-vectorized) env. -> (84, 84, n_stack)."""
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self._frames = deque(maxlen=n_stack)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_shape[0], obs_shape[1], n_stack),
            dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self._frames.append(obs)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._stack(), reward, terminated, truncated, info

    def _stack(self):
        return np.concatenate(list(self._frames), axis=-1)


# ---------------------------------------------------------------------------
# Environment builder
# ---------------------------------------------------------------------------
def make_env():
    """Clean ALE Breakout with rendering."""
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0,
                   render_mode="human")
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = PreprocessObs(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = AutoResetWrapper(env)
    env = SingleFrameStack(env, n_stack=FRAME_STACK)
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  PPO_55b Interactive Playback — ent_coef=0.02, det=True")
    print("=" * 60)
    print(f"  Model: {MODEL_PATH}")
    print()
    print("  SPACE / F : force FIRE (launch ball)")
    print("  Q / ESC   : quit")
    print("  All other frames: model chooses action (det=True)")
    print()
    print("  Watch for: does the model track the ball after you launch it?")
    print("  Or does it follow a fixed paddle pattern regardless?")
    print("=" * 60)
    print()

    print("Loading model...")
    model = PPO.load(MODEL_PATH, device="cuda")
    print(f"Model steps: {model.num_timesteps:,}")
    print()

    env = make_env()

    obs, _ = env.reset()
    total_score = 0
    total_frames = 0
    life_score = 0
    life_frames = 0
    lives_remaining = TOTAL_LIVES
    manual_fires = 0
    model_fires = 0
    ball_was_in_play = False  # track whether the ball ever launched
    game_running = True
    last_action_name = "—"
    manual_override = False

    # Per-life tracking
    life_log = []           # list of dicts, one per life
    life_number = 0
    post_launch_scored = False   # did the model score after a manual launch this life?
    frames_since_manual = 0      # frames since last manual FIRE
    auto_fire = False            # toggle with 'A' key

    print("Waiting 2 seconds... focus the game window!")
    time.sleep(2)
    print("Starting. Press SPACE to launch the ball.\n")

    try:
        while game_running and lives_remaining > 0:
            # First frame of a new life — start tracking
            if life_frames == 0:
                life_number += 1
                life_manual_fires = 0
                life_model_fires = 0
                post_launch_scored = False
                frames_since_manual = 0

            # Check for keyboard input (non-blocking)
            manual_override = False
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key in (b' ', b'f', b'F'):
                    action = 1  # FIRE
                    manual_override = True
                    manual_fires += 1
                    life_manual_fires += 1
                    frames_since_manual = 0
                elif key in (b'a', b'A'):
                    auto_fire = not auto_fire
                    print(f"\n  *** Auto-FIRE: {'ON' if auto_fire else 'OFF'} ***")
                    if auto_fire:
                        action = 1
                        manual_override = True
                        manual_fires += 1
                        life_manual_fires += 1
                        frames_since_manual = 0
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                elif key in (b'q', b'Q', b'\x1b'):
                    print("\nQuit by user.")
                    break
                else:
                    if auto_fire:
                        action = 1
                        manual_override = True
                        manual_fires += 1
                        life_manual_fires += 1
                        frames_since_manual = 0
                    else:
                        action, _ = model.predict(obs, deterministic=True)
            elif auto_fire:
                action = 1
                manual_override = True
                manual_fires += 1
                life_manual_fires += 1
                frames_since_manual = 0
            else:
                action, _ = model.predict(obs, deterministic=True)

            # model.predict returns array([N]) — extract scalar
            if isinstance(action, np.ndarray):
                action = int(action.item())

            if action == 1 and not manual_override:
                model_fires += 1
                life_model_fires += 1

            obs, reward, terminated, truncated, info = env.step(action)

            total_frames += 1
            life_frames += 1
            frames_since_manual += 1
            total_score += int(reward)
            life_score += int(reward)

            if reward > 0:
                ball_was_in_play = True
                if life_manual_fires > 0:
                    post_launch_scored = True

            action_name = ACTION_NAMES.get(action, str(action))
            marker = " <<< MANUAL" if manual_override else ""

            # Print status every 30 frames or on important events
            if total_frames % 30 == 0 or manual_override or reward != 0:
                af_marker = " [AUTO]" if auto_fire else ""
                print(f"[frame {total_frames:>6}] action={action_name:>5}{marker:<15}{af_marker}"
                      f"  score={total_score:>4}  life_score={life_score:>3}  "
                      f"lives_left={lives_remaining}", end="")
                if reward != 0:
                    print(f"  *** SCORED {int(reward)} ***", end="")
                print()

            if terminated or truncated:
                # Log the life
                life_log.append({
                    "life": life_number,
                    "score": life_score,
                    "frames": life_frames,
                    "manual_fires": life_manual_fires,
                    "model_fires": life_model_fires,
                    "post_launch_scored": post_launch_scored,
                })
                print(f"  --- Life {life_number} lost. "
                      f"Score: {life_score:+d}, frames: {life_frames}, "
                      f"manual_fires: {life_manual_fires}, "
                      f"model_FIREs: {life_model_fires}, "
                      f"post_launch_scored: {post_launch_scored} ---")
                lives_remaining -= 1
                life_score = 0
                life_frames = 0

            # Safety timeout per life
            if life_frames > 50_000:
                life_log.append({
                    "life": life_number,
                    "score": life_score,
                    "frames": life_frames,
                    "manual_fires": life_manual_fires,
                    "model_fires": life_model_fires,
                    "post_launch_scored": post_launch_scored,
                    "timeout": True,
                })
                print(f"  !!! TIMEOUT: {life_frames} frames without completing life")
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    env.close()

    # -------------------------------------------------------------------
    # Session log
    # -------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"./recordings/playback_PPO_55b_{timestamp}.txt"

    print()
    print("=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  Total frames:       {total_frames}")
    print(f"  Total score:        {total_score}")
    print(f"  Manual FIREs:       {manual_fires}")
    print(f"  Model FIREs:        {model_fires}")
    print(f"  Ball ever in play:  {'YES' if ball_was_in_play else 'NO'}")
    print(f"  Lives played:       {len(life_log)}")
    print()
    if life_log:
        print("  Per-life breakdown:")
        print(f"  {'Life':<6} {'Score':>6} {'Frames':>8} {'Manual':>7} {'ModelFIRE':>10} {'ScoredAfter':>12}")
        for l in life_log:
            print(f"  {l['life']:<6} {l['score']:>+6d} {l['frames']:>8} "
                  f"{l['manual_fires']:>7} {l['model_fires']:>10} "
                  f"{str(l['post_launch_scored']):>12}")
        print()
        manual_launch_lives = [l for l in life_log if l['manual_fires'] > 0]
        scored_after = [l for l in manual_launch_lives if l['post_launch_scored']]
        if manual_launch_lives:
            print(f"  Lives with manual launch:    {len(manual_launch_lives)}")
            print(f"  Of those, scored after:      {len(scored_after)}")
            if scored_after:
                avg = sum(l['score'] for l in scored_after) / len(scored_after)
                print(f"  Avg score (scored-after lives): {avg:.1f}")
    print()
    if ball_was_in_play:
        print("  The ball launched and the model scored points.")
        print("  Did the model TRACK the ball or follow a FIXED PATTERN?")
    elif model_fires > 0:
        print("  The model pressed FIRE on its own but never scored.")
    else:
        print("  The model never pressed FIRE on its own — argmax has no")
        print("  concept of launching the ball. This confirms the INCOMPLETE")
        print("  det=True checks: the deterministic policy is non-functional.")

    # Write log file
    with open(log_path, "w") as f:
        f.write(f"PPO_55b Interactive Playback Session\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {MODEL_PATH}\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Total score: {total_score}\n")
        f.write(f"Manual FIREs: {manual_fires}\n")
        f.write(f"Model FIREs: {model_fires}\n")
        f.write(f"Ball ever in play: {ball_was_in_play}\n")
        f.write(f"Lives played: {len(life_log)}\n")
        f.write(f"\nPer-life breakdown:\n")
        f.write(f"{'Life':<6} {'Score':>6} {'Frames':>8} {'ManualFIREs':>12} {'ModelFIREs':>11} {'PostLaunchScored':>17}\n")
        for l in life_log:
            f.write(f"{l['life']:<6} {l['score']:>+6d} {l['frames']:>8} "
                    f"{l['manual_fires']:>12} {l['model_fires']:>11} "
                    f"{str(l['post_launch_scored']):>17}\n")
    print(f"\n  Session log saved to: {log_path}")
