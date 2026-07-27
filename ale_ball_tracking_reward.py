"""
ALEBallTrackingReward — reward ball-tracking behavior via setRAM().

The perception POC proved NatureCNN can locate the ball to 1.9px MAE, and
PPO_85 proved that even with those features frozen in, PPO converges to a
blind script. The Atari score signal is the dominant optimization attractor.

This wrapper adds auxiliary rewards that make ball-tracking behavior MORE
rewarding than the Atari score, trying to shift the optimization landscape
so that reactive policies occupy a higher local optimum than blind scripts.

Modes:
  "hit_only"             — +reward per paddle-ball contact
  "hit_double"           — +2× per paddle-ball contact
  "descending_proximity" — proximity reward ONLY when ball is descending
  "combined"             — hit + descending proximity + survival penalty

Hit detection (state machine):
  IDLE → ball enters paddle zone descending → PENDING →
    ball bounces up near paddle → HIT (reward + cooldown) → IDLE

  Detects the descending→ascending transition at paddle height with the
  ball X near paddle X. 15-frame cooldown prevents double-counting.

Descending proximity:
  Only active when ball_y > prev_ball_y (ball moving toward paddle).
  Rewards keeping paddle X near ball X during the approach phase.

Survival penalty (combined mode):
  Small per-frame penalty to discourage "just don't die" survival-mode
  attractor observed in PPO_68b, RBO_01, and PPO_79.

RAM addresses (ALE 0.11.2, Breakout ROM):
  Ball X: 99, Ball Y: 101, Paddle X: 72

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = BallTrackingReward(env, mode="combined")
    # ... standard wrappers ...
"""
import numpy as np
import gymnasium as gym


class BallTrackingReward(gym.Wrapper):
    """Add ball-tracking auxiliary rewards read from ALE RAM.

    Parameters
    ----------
    env : gym.Env
        ALE/Breakout-v5 (needs .unwrapped.ale for getRAM).
    mode : str
        "hit_only"             — +hit_reward per paddle-ball contact
        "hit_double"           — +2× hit_reward per paddle-ball contact
        "descending_proximity" — proximity reward only when ball descending
        "combined"             — hit + proximity + survival penalty
    hit_reward : float
        Reward per detected paddle-ball hit (default 1.0).
    proximity_scale : float
        Maximum proximity reward per frame when descending (default 0.005).
    proximity_window : int
        Pixel window for proximity reward (default 40).
    survival_penalty : float
        Per-frame penalty in "combined" mode (default -0.0001).
    hit_window : int
        Max |paddle_x - ball_x| for a paddle hit (default 12).
    paddle_zone_y : int
        Ball Y threshold — bounces below this Y are paddle/wall, not bricks.
    seed : int or None
        Seed for the internal RNG (not currently used for randomization).
    """

    # Paddle sits at Y ≈ 189-193 in TIA coordinates.
    # Bounces below Y=175 are assumed to be paddle/wall, not bricks (Y < 80).
    PADDLE_ZONE_Y = 175
    HIT_WINDOW = 12
    HIT_COOLDOWN = 15  # frames (~250ms at 60fps)
    PROXIMITY_WINDOW = 40
    DESCENDING_BALL_Y_MIN = 30  # ball must be below brick zone for proximity

    def __init__(
        self,
        env,
        mode="combined",
        hit_reward=1.0,
        proximity_scale=0.005,
        proximity_window=40,
        survival_penalty=-0.0001,
        hit_window=12,
        paddle_zone_y=175,
        seed=None,
    ):
        super().__init__(env)
        self.mode = mode
        self.hit_reward = float(hit_reward)
        self.proximity_scale = float(proximity_scale)
        self.proximity_window = int(proximity_window)
        self.survival_penalty = float(survival_penalty)
        self.hit_window = int(hit_window)
        self.paddle_zone_y = int(paddle_zone_y)
        self._rng = np.random.default_rng(seed)

        # Hit detection state machine
        self._hit_state = "idle"  # idle → pending → cooldown → idle
        self._cooldown_frames = 0
        self._prev_ball_y = 0
        self._prev_ball_x = 0

        # Stats
        self._step_count = 0
        self._total_aux_reward = 0.0
        self._total_game_reward = 0.0
        self._hit_count = 0
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
    # Hit detection (state machine)
    # ------------------------------------------------------------------

    def _detect_hit(self, ball_y, ball_x, paddle_x):
        """State machine for paddle-hit detection.

        IDLE → ball enters paddle zone while descending → PENDING →
          ball bounces up (ascending) → check paddle proximity →
          HIT (reward) → COOLDOWN → IDLE

        Returns hit_reward if a hit was detected this frame, else 0.0.
        """
        hit = 0.0
        ball_descending = ball_y > self._prev_ball_y
        ball_ascending = ball_y < self._prev_ball_y
        in_paddle_zone = ball_y >= self.paddle_zone_y
        near_paddle_x = abs(ball_x - paddle_x) <= self.hit_window

        if self._hit_state == "idle":
            if in_paddle_zone and ball_descending:
                self._hit_state = "pending"

        elif self._hit_state == "pending":
            if ball_ascending:
                # Ball bounced — was it off the paddle?
                if near_paddle_x:
                    hit = self.hit_reward
                    self._hit_count += 1
                self._hit_state = "cooldown"
                self._cooldown_frames = self.HIT_COOLDOWN
            elif not in_paddle_zone:
                # Ball left zone without bouncing (miss or passed paddle)
                self._hit_state = "idle"

        elif self._hit_state == "cooldown":
            self._cooldown_frames -= 1
            if self._cooldown_frames <= 0:
                self._hit_state = "idle"

        return hit

    # ------------------------------------------------------------------
    # Descending proximity
    # ------------------------------------------------------------------

    def _compute_descending_proximity(self, ball_y, ball_x, paddle_x):
        """Proximity reward ONLY when ball is descending toward paddle."""
        ball_descending = ball_y > self._prev_ball_y
        ball_in_play = ball_y >= self.DESCENDING_BALL_Y_MIN

        if not ball_descending or not ball_in_play:
            return 0.0

        distance = abs(paddle_x - ball_x)
        raw = max(0.0, 1.0 - distance / float(self.proximity_window))
        return self.proximity_scale * raw

    # ------------------------------------------------------------------
    # Combined reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, ball_y, ball_x, paddle_x):
        """Compute auxiliary reward based on mode."""
        reward = 0.0

        if self.mode in ("hit_only", "hit_double", "combined"):
            reward += self._detect_hit(ball_y, ball_x, paddle_x)

        if self.mode in ("descending_proximity", "combined"):
            reward += self._compute_descending_proximity(ball_y, ball_x, paddle_x)

        if self.mode == "combined":
            reward += self.survival_penalty

        return reward

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._hit_state = "idle"
        self._cooldown_frames = 0
        self._prev_ball_y = 0
        self._prev_ball_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Read current ball/paddle state from RAM
        ball_y = self._get_ball_y()
        ball_x = self._get_ball_x()
        paddle_x = self._get_paddle_x()

        # Compute auxiliary reward before stepping (uses pre-step positions)
        aux_reward = self._compute_reward(ball_y, ball_x, paddle_x)

        # Step the environment
        obs, game_reward, terminated, truncated, info = self.env.step(action)

        # Update state for next frame's hit/passing detection
        self._prev_ball_y = ball_y
        self._prev_ball_x = ball_x

        # Combine rewards
        shaped_reward = game_reward + aux_reward

        # Stats
        self._step_count += 1
        self._total_aux_reward += aux_reward
        self._total_game_reward += game_reward

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
            "hit_count": self._hit_count,
            "ratio": (
                self._total_aux_reward / max(self._total_game_reward, 0.001)
            ),
        }
