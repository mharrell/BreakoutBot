"""
AuxTrackingReward — add a tiny ball-tracking auxiliary reward via setRAM().

The perception POC proved that NatureCNN with 4-frame stacking can locate the
ball to 1.9px MAE (0.6px median). The conv features encode ball position with
near-perfect precision. PPO never discovers this signal on its own — 80+
experiments have all collapsed to SINGLE_SCRIPT.

This wrapper adds a small auxiliary reward for keeping the paddle near the
ball's horizontal position. It's 1/20th the scale of PPO_15's failed tracking
reward (0.005 vs 0.1). The hypothesis: at this scale, the auxiliary reward
surfaces the ball-position features PPO is otherwise blind to, without
becoming an exploitable optimization target.

Modes:
  - "proximity":  Continuous distance-based reward (closer = more)
  - "coarse":     Binary reward for being within a wide window (±40px)
  - "annealing":  Proximity reward that linearly decays to zero over training

RAM addresses (ALE 0.11.2, Breakout ROM):
  Ball X: 99, Ball Y: 101, Paddle X: 72

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = AuxTrackingReward(env, mode="proximity", scale=0.005)
"""
import numpy as np
import gymnasium as gym


class AuxTrackingReward(gym.Wrapper):
    """Add a ball-tracking auxiliary reward read from ALE RAM.

    Parameters
    ----------
    env : gym.Env
        ALE/Breakout-v5 (needs .unwrapped.ale for getRAM).
    mode : str
        "proximity" — continuous reward based on |paddle_x - ball_x|
        "coarse"    — binary reward when within `window` pixels
        "annealing" — like proximity but decays `scale` to 0 over `anneal_steps`
    scale : float
        Maximum auxiliary reward per frame (default 0.005 = 1/20th PPO_15).
    window : int
        For "coarse" mode: pixel window for binary reward (default 40).
    ball_y_threshold : int
        Only reward tracking when ball is below this Y (default 80).
        Prevents rewarding paddle-chasing when the ball is in the brick zone.
    anneal_steps : int
        For "annealing" mode: steps over which scale decays to 0.
    """

    def __init__(
        self,
        env,
        mode="proximity",
        scale=0.005,
        window=40,
        ball_y_threshold=80,
        anneal_steps=25_000_000,
        seed=None,
    ):
        super().__init__(env)
        self.mode = mode
        self.initial_scale = float(scale)
        self.scale = float(scale)
        self.window = int(window)
        self.ball_y_threshold = int(ball_y_threshold)
        self.anneal_steps = int(anneal_steps)
        self._rng = np.random.default_rng(seed)

        # Tracking
        self._step_count = 0
        self._total_aux_reward = 0.0
        self._total_game_reward = 0.0
        self._episode_aux_reward = 0.0
        self._episode_game_reward = 0.0
        self._episodes = 0

    # ------------------------------------------------------------------
    # RAM access
    # ------------------------------------------------------------------

    def _get_ram(self, addr):
        return int(self.env.unwrapped.ale.getRAM()[addr])

    def _get_ball_x(self):
        return self._get_ram(99)

    def _get_ball_y(self):
        return self._get_ram(101)

    def _get_paddle_x(self):
        return self._get_ram(72)

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_tracking_reward(self):
        """Compute auxiliary tracking reward for the current frame."""
        ball_y = self._get_ball_y()

        # Ball not in play or in brick zone — no reward
        if ball_y < self.ball_y_threshold:
            return 0.0

        ball_x = self._get_ball_x()
        paddle_x = self._get_paddle_x()
        distance = abs(paddle_x - ball_x)

        if self.mode == "coarse":
            # Binary: 1 if within window, 0 otherwise
            return self.scale if distance <= self.window else 0.0
        else:
            # "proximity" or "annealing": continuous, linear decay with distance
            # At distance=0: full scale. At distance=window: 0. Clamped.
            raw = max(0.0, 1.0 - distance / float(self.window))
            return self.scale * raw

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._episode_aux_reward = 0.0
        self._episode_game_reward = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, game_reward, terminated, truncated, info = self.env.step(action)

        # Compute auxiliary reward
        aux_reward = self._compute_tracking_reward()
        shaped_reward = game_reward + aux_reward

        # Track stats
        self._step_count += 1
        self._total_aux_reward += aux_reward
        self._total_game_reward += game_reward
        self._episode_aux_reward += aux_reward
        self._episode_game_reward += game_reward

        # Annealing: linearly decay scale
        if self.mode == "annealing" and self._step_count < self.anneal_steps:
            progress = self._step_count / self.anneal_steps
            self.scale = self.initial_scale * (1.0 - progress)

        if terminated or truncated:
            self._episodes += 1

        return obs, shaped_reward, terminated, truncated, info

    def get_stats(self):
        """Return cumulative tracking statistics."""
        return {
            "step_count": self._step_count,
            "episodes": self._episodes,
            "total_aux_reward": self._total_aux_reward,
            "total_game_reward": self._total_game_reward,
            "ratio": (self._total_aux_reward / max(self._total_game_reward, 0.001)),
        }
