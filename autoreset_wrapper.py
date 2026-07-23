"""
AutoResetWrapper — calls env.reset() before step() if the previous step
returned done=True, then presses FIRE to ensure the ball launches.

Fixes a hang in SB3's evaluate_policy(), which never calls env.reset()
between episodes. With EpisodicLifeEnv in the pipeline, after a life loss
the ball sits on the paddle waiting for FIRE. Two problems compound:

1. evaluate_policy() never calls env.reset() between episodes.
2. EpisodicLifeEnv.reset() uses NOOP (action=0) on life-loss autoreset,
   NOT FIRE. So even calling env.reset() doesn't launch the ball on a
   life-loss respawn. If the model's deterministic action isn't FIRE,
   the ball stays on the paddle forever and the eval hangs.

Fix: call env.reset(), then explicitly press FIRE (action=1). In Breakout,
FIRE is a no-op when the ball is already in play, so this is safe after
both life-loss and game-over autoresets.

Usage (place as the OUTERMOST wrapper, above Monitor):
    env = Monitor(env)
    env = AutoResetWrapper(env)
"""
import gymnasium as gym


class AutoResetWrapper(gym.Wrapper):
    """Calls env.reset() then FIRE before step() if the previous step returned
    done=True. This guarantees the ball launches regardless of what the model's
    deterministic action is.
    """

    def __init__(self, env):
        super().__init__(env)
        self._was_done = False

    def step(self, action):
        if self._was_done:
            self.env.reset()
            self._was_done = False
            # EpisodicLifeEnv.reset() uses NOOP (action=0) on life-loss
            # autoreset, which does NOT launch the ball. Explicitly press
            # FIRE (action=1) to guarantee the ball is in play. In Breakout,
            # FIRE is harmless when the ball is already in flight.
            self.env.step(1)  # FIRE
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._was_done = terminated or truncated
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._was_done = False
        return self.env.reset(**kwargs)
