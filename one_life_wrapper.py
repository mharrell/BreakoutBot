"""
OneLifeWrapper — end the episode on first life loss.

Replaces EpisodicLifeEnv. Instead of marking life loss as truncated and
continuing, this wrapper terminates the episode immediately. Breakout
becomes a hard-failure game: one ball lost = game over.

The hypothesis: hard failure makes memorized sweep scripts non-viable
because they can't accumulate enough points on one ball to be net-positive.
"""
import gymnasium as gym


class OneLifeWrapper(gym.Wrapper):
    """Terminate the episode on first life loss.

    Must be placed in the same position as EpisodicLifeEnv in the
    wrapper chain: after NoopResetEnv and FireResetEnv, before
    GrayscaleResize/ClipRewardEnv.
    """

    def __init__(self, env):
        super().__init__(env)
        self._lives = None

    def _get_lives(self):
        return self.env.unwrapped.ale.lives()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = self._get_lives()
        if self._lives is None:
            self._lives = current_lives
        if current_lives < self._lives:
            # Life lost — end the episode
            self._lives = current_lives
            terminated = True
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._lives = self._get_lives()
        return obs, info
