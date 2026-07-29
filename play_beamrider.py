"""
Play BeamRider with a trained model — render to screen.

Usage:
    python play_beamrider.py [model_path] [--games N] [--det/--stoch]

Default: loads BEAMRIDER_BASELINE final model, plays 5 games deterministic.
"""
import sys
import time
import cv2
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, EpisodicLifeEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import ale_py
gym.register_envs(ale_py)


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


class AutoResetWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs, _info = self.env.reset()
        return obs, reward, terminated, truncated, info


def make_env(noop_max=0, episodic_life=True):
    env = gym.make("ALE/BeamRider-v5", frameskip=4,
                   repeat_action_probability=0, render_mode="human")
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    if episodic_life:
        env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


if __name__ == "__main__":
    model_path = "./models/BEAMRIDER_BASELINE/final_model.zip"
    n_games = 5
    deterministic = True
    noop_max = 0
    episodic_life = True  # default: standard BeamRider (hard failure)

    # Parse args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--games":
            n_games = int(args[i + 1]); i += 2
        elif args[i] == "--stoch":
            deterministic = False; i += 1
        elif args[i] == "--det":
            deterministic = True; i += 1
        elif args[i] == "--noop":
            noop_max = int(args[i + 1]); i += 2
        elif args[i] == "--no-lives":
            episodic_life = False; i += 1
        elif not args[i].startswith("--"):
            model_path = args[i]; i += 1
        else:
            i += 1

    print(f"Loading {model_path}...")
    dummy_env = make_env(noop_max=noop_max, episodic_life=episodic_life)
    model = PPO.load(model_path, env=dummy_env, device="cuda")
    dummy_env.close()

    print(f"Model: {model.num_timesteps:,} steps")
    print(f"Mode: {'deterministic' if deterministic else 'stochastic'}")
    print(f"Noop max: {noop_max}")
    print(f"EpisodicLife: {episodic_life}")
    print(f"Playing {n_games} games...")
    print("Press Ctrl+C to stop early.\n")

    env = make_env(noop_max=noop_max, episodic_life=episodic_life)
    scores = []

    for game in range(n_games):
        obs = env.reset()
        score = 0.0
        frames = 0
        done = False
        t0 = time.time()

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_arr, _info = env.step(action)
            score += reward[0]
            frames += 1
            done = bool(done_arr[0])

        elapsed = time.time() - t0
        scores.append(score)
        print(f"  Game {game + 1}: {score:.0f} pts, {frames} frames, "
              f"{elapsed:.1f}s  ({'DET' if deterministic else 'STOCH'})")

    env.close()
    print(f"\nSummary: mean={np.mean(scores):.1f}, "
          f"min={np.min(scores):.0f}, max={np.max(scores):.0f}, "
          f"unique={len(set(round(s, 1) for s in scores))}")
