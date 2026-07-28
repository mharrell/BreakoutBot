"""
LifeLossPenalty — penalize PPO for losing the ball in Breakout.

BeamRider's hard failure constraint (one bullet = death) is what forces reactive
play. Breakout's failure is soft — lose the ball, bricks stay broken, just
re-serve. Scripts that break 3-5 bricks per life are locally optimal because
the Atari score never punishes inefficiency.

This wrapper adds a negative reward on every life loss, making scripts
net-negative. A paddle-sweep that breaks 4 bricks then loses the ball gets:
    +4 (bricks) - 10 (penalty) = -6 net reward

Doing nothing (0 reward) is better than that script. Only reactive play that
breaks 11+ bricks per life can be net-positive.

Design decisions:
  - Penalty: 10.0 per life loss (configurable). Chosen because ClipRewardEnv
    gives 1.0/brick, and scripts typically break 3-5 bricks/life. At 10.0,
    a script needs 11+ bricks/life to be net-positive.
  - Annealing: 5M steps linear ramp from 0→10.0 (configurable). Gives the
    agent time to learn basic gameplay before the penalty activates, without
    being so long that scripts become entrenched.
  - Detection: ale.lives() — the proper ALE API, not fragile RAM probing.
  - Wraps BEFORE EpisodicLifeEnv so the underlying ALE still reports full
    lives count (EpisodicLifeEnv sees life loss and terminates the episode).
"""
import gymnasium as gym


class LifeLossPenalty(gym.Wrapper):
    """Penalize life loss in Breakout to make memorized scripts net-negative.

    Tracks `ale.lives()` before and after each step. When lives decrease,
    subtracts an annealed penalty from the reward.

    Must be stacked BEFORE EpisodicLifeEnv in the wrapper chain so the
    underlying ALE still reports the full lives count. EpisodicLifeEnv
    will see the same life loss and terminate the episode normally.

    Args:
        env: Gymnasium environment (ALE/Breakout-v5 or similar).
        penalty: Base penalty per life loss (default 10.0).
        anneal_steps: Steps over which to linearly ramp penalty from 0→base.
                      (default 5_000_000, set to 0 for no annealing).
    """

    def __init__(self, env, penalty=10.0, anneal_steps=5_000_000):
        super().__init__(env)
        self.penalty = float(penalty)
        self.anneal_steps = int(anneal_steps)
        self._lives = None
        self._step_count = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._lives = self._get_lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        current_lives = self._get_lives()
        if self._lives is not None and current_lives < self._lives:
            # Life was lost — apply annealed penalty
            if self.anneal_steps > 0:
                anneal_frac = min(1.0, self._step_count / self.anneal_steps)
            else:
                anneal_frac = 1.0
            reward -= self.penalty * anneal_frac
            self._lives = current_lives

        return obs, reward, terminated, truncated, info

    def _get_lives(self):
        """Read lives from the underlying ALE interface."""
        # Walk the wrapper chain to find the base ALE env
        ale = self.env.unwrapped.ale
        return ale.lives()
