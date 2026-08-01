"""
ProximityRewardWrapper — directly reward keeping the paddle near the ball.

Every previous approach tried to force reactivity INDIRECTLY — through
environment perturbations, objective function regularization, or auxiliary
losses. This wrapper does the simplest possible thing: give the model a
reward when the paddle is horizontally close to the ball.

  bonus = scale * max(0, 1 - |paddle_x - ball_x| / max_distance)

When ball_y > 100 (descending toward paddle): paddle close to ball → bonus.
When ball_y <= 100 (ball in brick zone): no bonus (paddle position irrelevant).

This explicitly makes ball-tracking the optimal behavior at every timestep.
A center-hold script gets some proximity reward when the ball happens to pass
near center, but a reactive tracking policy gets the maximum bonus every step.

Eval on CLEAN Breakout (no proximity reward) tests whether the trained
ball-tracking behavior transfers — i.e., whether the policy learned to
track the ball or just learned to maximize the bonus.

RAM addresses (ALE 0.11.2, Breakout ROM):
  RAM 72: paddle_x (0-160ish)
  RAM 99: ball_x   (0-199, playfield ~0-160)
  RAM 101: ball_y  (0-210ish)

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = ProximityRewardWrapper(env)
    # ... standard wrappers ...
"""
import numpy as np
import gymnasium as gym


class ProximityRewardWrapper(gym.Wrapper):
    """Reward horizontal proximity between paddle and ball.

    During the descending phase (ball_y > 100), the paddle should be under
    the ball. This wrapper adds a per-step bonus proportional to how close
    the paddle is to the ball horizontally.

    A script that holds the paddle at center gets partial reward (ball passes
    near center sometimes). A reactive policy that tracks the ball gets the
    full proximity bonus at every step.

    Parameters
    ----------
    env : gym.Env
        ALE/Breakout-v5.
    scale : float
        Maximum per-step bonus when paddle is exactly at ball_x (default 0.05).
        At scale=0.05: ~25-50 bonus per game for tracking, ~5-15 for center-hold.
    max_distance : float
        Horizontal distance where bonus reaches zero (default 80).
        |paddle_x - ball_x| >= 80 → bonus = 0.
        |paddle_x - ball_x| = 0  → bonus = scale.
    descend_threshold : int
        Only apply bonus when ball_y > this value (ball descending toward paddle).
        Default 100 means bottom ~half of screen.
    """

    PADDLE_X_ADDR = 72
    BALL_X_ADDR = 99
    BALL_Y_ADDR = 101

    def __init__(
        self,
        env,
        scale=0.05,
        max_distance=80.0,
        descend_threshold=100,
        seed=None,
    ):
        super().__init__(env)
        self.scale = float(scale)
        self.max_distance = float(max_distance)
        self.descend_threshold = int(descend_threshold)
        self._total_bonus = 0.0
        self._bonus_steps = 0
        self._total_steps = 0

    def _get_ram(self, addr):
        return int(self.env.unwrapped.ale.getRAM()[addr])

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ball_y = self._get_ram(self.BALL_Y_ADDR)

        bonus = 0.0
        if ball_y > self.descend_threshold:
            paddle_x = self._get_ram(self.PADDLE_X_ADDR)
            ball_x = self._get_ram(self.BALL_X_ADDR)
            distance = abs(paddle_x - ball_x)
            bonus = self.scale * max(0.0, 1.0 - distance / self.max_distance)
            reward += bonus
            self._total_bonus += bonus
            self._bonus_steps += 1

        self._total_steps += 1

        if terminated or truncated:
            if isinstance(info, dict):
                info["proximity_bonus"] = self._total_bonus

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._total_bonus = 0.0
        self._bonus_steps = 0
        self._total_steps = 0
        return self.env.reset(**kwargs)

    def get_stats(self):
        return {
            "total_steps": self._total_steps,
            "bonus_steps": self._bonus_steps,
            "total_bonus": self._total_bonus,
            "mean_bonus_per_bonus_step": (
                self._total_bonus / max(self._bonus_steps, 1)
            ),
        }


# -----------------------------------------------------------------------
# Standalone smoke test
# -----------------------------------------------------------------------

if __name__ == "__main__":
    """Verify wrapper runs and bonus is computed."""
    import time
    import ale_py
    gym.register_envs(ale_py)

    print("ProximityRewardWrapper — Smoke Test")
    print(f"  Scale: 0.05, Max distance: 80, Descend threshold: 100")

    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env.reset()
    for _ in range(5):
        env.step(0)
    env.step(1)  # FIRE

    env = ProximityRewardWrapper(env)

    total_reward = 0.0
    for step in range(1000):
        action = np.random.choice([0, 2, 3])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 100 == 0:
            stats = env.get_stats()
            print(f"  Step {step}: bonus_steps={stats['bonus_steps']}, "
                  f"total_bonus={stats['total_bonus']:.2f}, "
                  f"mean_bonus={stats['mean_bonus_per_bonus_step']:.4f}")

        if terminated or truncated:
            obs, info = env.reset()
            for _ in range(5):
                env.step(0)
            env.step(1)

    stats = env.get_stats()
    print(f"\n  Final: total_reward={total_reward:.1f}, bonus_steps={stats['bonus_steps']}")
    print(f"  Stats: {stats}")
    env.close()
