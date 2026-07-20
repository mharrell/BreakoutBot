"""
watch_intervention.py — Watch PPO models play with and without ball/paddle teleportation.

Controls:
  N — normal game (no interventions)
  I — intervention game (teleports on random paddle bounces)
  T — toggle deterministic / stochastic
  Q — quit

Teleports flash red text on screen. Score, intervention count, and mode shown in title bar.
"""

import sys
import numpy as np
import cv2
import gymnasium as gym
from gym_breakout import GymBreakout
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.monitor import Monitor


# --- Wrappers ---------------------------------------------------------

class GrayscaleResize(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self._width = width
        self._height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, obs):
        resized = cv2.resize(obs, (self._width, self._height),
                             interpolation=cv2.INTER_AREA)
        return resized[:, :, None]


class InterventionBreakout(gym.Wrapper):
    """Teleports ball (y-axis) and paddle (x-axis) after random paddle bounces."""

    def __init__(self, env, teleport_prob=0.3, ball_y_jitter=15,
                 paddle_x_jitter=30, seed=None):
        super().__init__(env)
        self.teleport_prob = teleport_prob
        self.ball_y_jitter = ball_y_jitter
        self.paddle_x_jitter = paddle_x_jitter
        self._rng = np.random.default_rng(seed)
        self.intervention_count = 0
        self.interventions = []
        self._just_teleported = False

    def reset(self, **kwargs):
        self.intervention_count = len(self.interventions)
        self.interventions = []
        self._just_teleported = False
        return self.env.reset(**kwargs)

    def step(self, action):
        brk = self.env._env
        prev_vy = brk.ball_v[0] if brk else None

        obs, reward, terminated, truncated, info = self.env.step(action)

        cur_vy = brk.ball_v[0] if brk else None
        self._just_teleported = False
        if (prev_vy is not None and cur_vy is not None
                and prev_vy > 0 and cur_vy < 0):
            if self._rng.random() < self.teleport_prob:
                ball = brk.ball
                paddle = brk.paddle
                dy = int(round(self._rng.normal(0, self.ball_y_jitter)))
                new_y = max(15, min(paddle.pos[0] - 10, ball.pos[0] + dy))
                ball.pos[0] = new_y
                dx = int(round(self._rng.normal(0, self.paddle_x_jitter)))
                new_x = max(0, min(160 - paddle.size[1], paddle.pos[1] + dx))
                paddle.pos[1] = new_x
                self.interventions.append((brk.step_count, new_y, new_x))
                self._just_teleported = True

        return obs, reward, terminated, truncated, info


# --- Helpers ----------------------------------------------------------

def load_model(run_name):
    import os, glob
    best = f"./models/{run_name}/best_model.zip"
    if os.path.exists(best):
        print(f"Loading {best}")
        return PPO.load(best, device="cuda")
    ckpt_dir = f"./models/{run_name}/checkpoint"
    if os.path.isdir(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "latest_checkpoint_*_steps.zip"))
        if ckpts:
            latest = max(ckpts, key=os.path.getmtime)
            print(f"Loading {latest}")
            return PPO.load(latest, device="cuda")
    raise FileNotFoundError(f"No model found for {run_name}")


def build_env(intervene=False):
    """Build env chain. Returns (vec_env, breakout_ref, intervention_ref)."""
    base = GymBreakout(fixed=True)
    ib = None
    if intervene:
        ib = InterventionBreakout(base, teleport_prob=0.3,
                                  ball_y_jitter=15, paddle_x_jitter=30)
        env = GrayscaleResize(ib)
    else:
        env = GrayscaleResize(base)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env, base, ib


# --- Main -------------------------------------------------------------

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "PPO_35"
    model = load_model(run)

    print(f"\n{'='*60}")
    print(f"  {run} — Intervention Watch")
    print(f"  N=normal  I=intervention  T=toggle det/stochastic  Q=quit")
    print(f"  Starting in INTERVENTION mode, deterministic=True")
    print(f"{'='*60}\n")

    intervene = True
    deterministic = True
    env, base, ib = build_env(intervene=intervene)

    obs = env.reset()
    done = False
    frames = 0
    score = 0
    total_games = 0

    cv2.namedWindow("BreakoutBot — Intervention Watch", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BreakoutBot — Intervention Watch", 640, 480)

    while True:
        # Render
        frame = base.render()
        if frame is not None:
            display = cv2.resize(frame, (640, 480),
                                 interpolation=cv2.INTER_NEAREST)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

            # Overlays
            det_str = "DET" if deterministic else "STOCH"
            mode_str = "INTERVENTION" if intervene else "NORMAL"
            n_int = ib.intervention_count if ib else 0
            just_tp = ib._just_teleported if ib else False

            cv2.putText(display, f"{run} | {mode_str} | {det_str}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(display, f"Game: {total_games+1}  Score: {score:.0f}  "
                        f"Interventions: {n_int}  Frame: {frames}",
                        (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180), 1, cv2.LINE_AA)

            if just_tp:
                cv2.putText(display, "TELEPORT!",
                            (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                            (0, 0, 255), 3)

            cv2.imshow("BreakoutBot — Intervention Watch", display)

        # Input (33ms ≈ 30fps display, but game steps every frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            intervene = False
            done = True
        elif key == ord('i'):
            intervene = True
            done = True
        elif key == ord('t'):
            deterministic = not deterministic
            done = True

        # Restart game if switching modes or game ended
        if done:
            n_int = ib.intervention_count if ib else 0
            print(f"  Game {total_games+1}: score={score:.0f}, "
                  f"int={n_int}, frames={frames} "
                  f"[{'INT' if intervene else 'NORM'} "
                  f"{'det' if deterministic else 'stoch'}]")
            total_games += 1
            env.close()
            env, base, ib = build_env(intervene=intervene)
            obs = env.reset()
            done = False
            frames = 0
            score = 0
            continue

        # Step
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done_arr, info = env.step(action)
        done = done_arr[0]
        frames += 1
        if done:
            # Read real score from Breakout engine (Monitor clips rewards via ClipRewardEnv)
            score = base._env.score if base._env else 0

    cv2.destroyAllWindows()
    env.close()
    print(f"\nWatched {total_games} games. Done.")
