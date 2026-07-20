"""
ALEBreakoutRandomized — gym.Wrapper for ALE/Breakout-v5 dynamics randomization.

Implements ball teleportation on paddle bounce via ALE.setRAM(), matching
the GymBreakout InterventionBreakout pattern on authentic Atari Breakout.

Mechanism:
  1. Reads Ball Y (addr 101) and Ball X (addr 99) from ALE RAM each step.
  2. Tracks the frame-to-frame delta to detect direction changes.
  3. When ball transitions from falling (delta > 0) to rising (delta < 0)
     AND was in the paddle zone (ball_y > PADDLE_ZONE), a paddle bounce
     is detected.
  4. With `teleport_prob` probability, the ball is teleported to a random
     position in the upper portion of the playfield, breaking any timed
     script that depends on the ball being at a specific position.

RAM addresses (verified 2026-07-19, ALE 0.11.2):
  Ball X: 99   (read/write, range ~0-160)
  Ball Y: 101  (read/write, range ~0-200, higher = closer to paddle)
  Paddle X: 72 (read/write, range ~0-160)

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = ALEBreakoutRandomized(env, teleport_prob=0.30)
    # Then: GrayscaleResize -> ClipRewardEnv -> Monitor -> DummyVecEnv -> VecFrameStack
"""
import numpy as np
import gymnasium as gym


class ALEBreakoutRandomized(gym.Wrapper):
    """Teleports ball on paddle bounce with configurable probability.

    Detects paddle bounces by monitoring Ball Y (RAM addr 101) across
    consecutive frames. A bounce is: ball was falling (Y increasing)
    and transitions to rising (Y decreasing) while in the paddle zone.

    On detection, with `teleport_prob` probability, teleports the ball
    to a random (X, Y) in the playfield. No other randomization is
    applied — single variable for the first experiment.
    """

    # Playfield bounds (ALE Breakout coordinates, verified via probe)
    BALL_X_MIN = 10
    BALL_X_MAX = 150
    BALL_Y_MIN = 20
    BALL_Y_MAX = 140     # upper portion — well above the paddle
    PADDLE_ZONE = 170    # ball Y above this is considered "near paddle"

    # Ball sits at (0,0) before launch — ignore these frames
    PRELAUNCH_BALL_Y = 0

    def __init__(self, env, teleport_prob=0.30, seed=None):
        super().__init__(env)
        self.teleport_prob = float(teleport_prob)
        self._rng = np.random.default_rng(seed)

        # Per-episode state
        self.prev_ball_y = None
        self.prev_delta = None          # ball_y change from two frames ago
        self._frames_since_reset = 0
        self.intervention_count = 0

    # ------------------------------------------------------------------
    # RAM access helpers
    # ------------------------------------------------------------------

    def _get_ball_xy(self):
        """Read current ball position from RAM. Returns (x, y)."""
        ale = self.env.unwrapped.ale
        ram = ale.getRAM()
        return int(ram[99]), int(ram[101])

    def _set_ball_xy(self, x, y):
        """Write ball position to RAM."""
        ale = self.env.unwrapped.ale
        ale.setRAM(99, x)
        ale.setRAM(101, y)

    # ------------------------------------------------------------------
    # Bounce detection
    # ------------------------------------------------------------------

    def _detect_paddle_bounce(self, ball_y):
        """Return True if a paddle bounce just occurred.

        Criteria:
          1. Ball was near/at paddle zone in the previous frame.
          2. Ball was falling (delta > 0) between t-2 and t-1.
          3. Ball is now rising (delta < 0) between t-1 and t.

        Guard: skip first few frames after reset (ball launch).
        """
        if self._frames_since_reset < 6:
            return False
        if self.prev_ball_y is None or self.prev_delta is None:
            return False
        # Ball not yet launched
        if self.prev_ball_y == self.PRELAUNCH_BALL_Y:
            return False

        delta = ball_y - self.prev_ball_y
        was_falling = self.prev_delta > 0
        now_rising = delta < 0
        was_near_paddle = self.prev_ball_y > self.PADDLE_ZONE

        return was_falling and now_rising and was_near_paddle

    def _teleport_ball(self):
        """Move ball to a random position in the playfield."""
        new_x = int(self._rng.integers(self.BALL_X_MIN, self.BALL_X_MAX + 1))
        new_y = int(self._rng.integers(self.BALL_Y_MIN, self.BALL_Y_MAX + 1))
        self._set_ball_xy(new_x, new_y)
        self.prev_ball_y = new_y
        self.prev_delta = None  # invalidate delta after teleport
        self.intervention_count += 1

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self.prev_ball_y = None
        self.prev_delta = None
        self._frames_since_reset = 0
        self.intervention_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # Read ball position before the step
        ball_x, ball_y = self._get_ball_xy()

        # Detect paddle bounce from frame-to-frame ball movement
        if self._detect_paddle_bounce(ball_y):
            if self._rng.random() < self.teleport_prob:
                self._teleport_ball()
                # Re-read after teleport for tracking
                ball_x, ball_y = self._get_ball_xy()

        # Update tracking state for next frame
        if self.prev_ball_y is not None and ball_y != self.PRELAUNCH_BALL_Y:
            self.prev_delta = ball_y - self.prev_ball_y
        self.prev_ball_y = ball_y
        self._frames_since_reset += 1

        return self.env.step(action)
