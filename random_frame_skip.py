"""
Random Frame Skip wrapper for Atari Breakout domain randomization.

Varies the number of ALE frames each action is applied for (2-8 frames,
uniformly random per step). This randomizes:
  - Effective ball speed: 2-frame skip = ball moves ~2px/step, 8-frame = ~8px
  - Effective paddle responsiveness: 2 frames of LEFT = paddle moves ~2px,
    8 frames = ~8px

A timed open-loop script ("at frame 50, press LEFT") fails because the ball
is in a different place depending on the cumulative frame skip. The agent
MUST observe the ball position and react.

This is Experiment 5A — the cheapest dynamics randomization intervention.
If this doesn't force reactivity, escalate to RAM-parameterized physics
(Experiment 5B) or custom ROM (Experiment 5C).

Usage:
    from random_frame_skip import RandomFrameSkip
    env = gym.make("ALE/Breakout-v5", frameskip=1)
    env = RandomFrameSkip(env, min_skip=2, max_skip=8)
    # Then apply standard Atari wrappers (NoopResetEnv, WarpFrame, etc.)
"""

import numpy as np
from gymnasium import Wrapper


class RandomFrameSkip(Wrapper):
    """Vary frame skip uniformly in [min_skip, max_skip] each step.

    The underlying ALE env must be created with frameskip=1 (single-frame mode)
    for the randomization to have clean semantics. Using this on top of a
    frameskip=4 env would produce effective skip of 8-32 frames, which is
    too coarse.

    Does NOT apply max-pooling over frames (unlike MaxAndSkipEnv in SB3).
    Breakout has solid sprites (no flicker), so max-pooling is unnecessary
    for this game. Would need modification for games with flickering sprites.
    """

    def __init__(self, env, min_skip: int = 2, max_skip: int = 8):
        super().__init__(env)
        if min_skip < 1:
            raise ValueError(f"min_skip must be >= 1, got {min_skip}")
        if max_skip < min_skip:
            raise ValueError(f"max_skip ({max_skip}) < min_skip ({min_skip})")
        self.min_skip = min_skip
        self.max_skip = max_skip

    def step(self, action):
        skip = self.np_random.integers(self.min_skip, self.max_skip + 1)
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
