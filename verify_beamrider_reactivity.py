"""
Verify BEAMRIDER_BASELINE reactivity — no-noop test.

MULTIPLE_SCRIPTS means ≥3 unique scores, but NoopResetEnv(noop_max=30)
injects per-episode randomness. A fixed action script + varying start offset
can produce diverse scores — the actions are the same sequence, they just
hit different things at different times.

This test runs the model with noop_max=0 (no random offset) and compares
against the standard noop_max=30 control. If the policy is genuinely
reactive, score diversity persists. If it collapses, the "MULTIPLE_SCRIPTS"
verdict was an artifact of no-op noise.

Also tests with no-noop AND deterministic=True to check argmax collapse.
"""
import os
import time
import numpy as np
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, EpisodicLifeEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import ale_py
gym.register_envs(ale_py)

MODEL_PATH = "./models/BEAMRIDER_BASELINE/final_model.zip"
N_GAMES = 100


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
    """Auto-reset terminated/truncated envs so single-env VecEnv keeps going."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            obs, _info = self.env.reset()
        return obs, reward, terminated, truncated, info


def make_test_env(noop_max):
    """BeamRider with EpisodicLifeEnv, configurable noop_max."""
    env = gym.make("ALE/BeamRider-v5", frameskip=4, repeat_action_probability=0)
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
    env = AutoResetWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    return env


def run_games(model, env, n_games, deterministic):
    """Run n_games and return list of scores."""
    scores = []
    obs = env.reset()
    game_count = 0
    current_score = 0
    episode_frames = 0

    while game_count < n_games:
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        current_score += reward[0]
        episode_frames += 1

        if done[0]:
            scores.append(current_score)
            current_score = 0
            episode_frames = 0
            game_count += 1
            if game_count % 25 == 0:
                print(f"  {game_count}/{n_games} games...")

    return scores


def classify(unique_count):
    return "SINGLE_SCRIPT" if unique_count <= 2 else "MULTIPLE_SCRIPTS"


if __name__ == "__main__":
    print("=" * 60)
    print("BeamRider Reactivity Verification — No-Noop Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"Games per test: {N_GAMES}")
    print()
    print("Hypothesis: if MULTIPLE_SCRIPTS is genuine reactivity,")
    print("  noop_max=0 should preserve score diversity.")
    print("  If it collapses, the diversity was no-op noise.")
    print()

    # Load model once
    print("Loading model...")
    dummy_env = make_test_env(noop_max=0)
    model = PPO.load(MODEL_PATH, env=dummy_env, device="cuda")
    dummy_env.close()
    print(f"Model loaded. {model.num_timesteps:,} total timesteps.")
    print()

    results = {}

    for noop_max in [0, 30]:
        for det_label, deterministic in [("det=True", True), ("det=False", False)]:
            label = f"noop={noop_max}, {det_label}"
            print(f"--- {label} ---")
            env = make_test_env(noop_max=noop_max)
            t0 = time.time()
            scores = run_games(model, env, N_GAMES, deterministic)
            elapsed = time.time() - t0
            env.close()

            unique = len(set(round(s, 1) for s in scores))
            verdict = classify(unique)
            print(f"  {N_GAMES} games in {elapsed:.0f}s")
            print(f"  Scores: min={min(scores):.0f}, max={max(scores):.0f}, "
                  f"mean={np.mean(scores):.1f}, std={np.std(scores):.1f}")
            print(f"  Unique scores: {unique} -> {verdict}")
            print()

            results[label] = {
                "noop_max": noop_max,
                "deterministic": deterministic,
                "scores": np.array(scores),
                "unique": unique,
                "verdict": verdict,
            }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, r in results.items():
        print(f"  {label:25s}: unique={r['unique']:3d}  {r['verdict']}  "
              f"mean={np.mean(r['scores']):5.1f}  std={np.std(r['scores']):4.1f}  "
              f"range=[{np.min(r['scores']):.0f}, {np.max(r['scores']):.0f}]")

    # The critical comparison
    print()
    noop0_det = results["noop=0, det=True"]
    noop30_det = results["noop=30, det=True"]

    if noop0_det["unique"] <= 2:
        print("RESULT: noop_max=0 det=True is SINGLE_SCRIPT.")
        print("The MULTIPLE_SCRIPTS verdict is a no-op artifact. Model is MEMORIZED.")
    else:
        print(f"RESULT: noop_max=0 det=True has {noop0_det['unique']} unique scores -> "
              f"{noop0_det['verdict']}.")
        print("Score diversity survives no-noop. Evidence of GENUINE REACTIVITY.")
