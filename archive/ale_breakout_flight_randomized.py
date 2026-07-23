"""
ALEBreakoutFlightRandomized — mid-flight ball teleportation for ALE Breakout.

Unlike ALEBreakoutRandomized (which teleports on paddle bounce), this wrapper
teleports the ball while it's in flight through the upper/mid playfield. This
avoids the "punishment for success" problem — hitting the ball is never
penalized. The paddle-bounce and final approach are clean; only the ball's
trajectory through the upper field is randomized.

Mechanism:
  1. Reads Ball Y (addr 101) from ALE RAM each frame.
  2. If the ball is in the teleport zone (Y between 20 and 150, i.e. in flight
     through the upper/mid field), rolls for teleport with the configured
     probability.
  3. On teleport, moves the ball to a random (X, Y) within the teleport zone.
  4. Never teleports when the ball is near the paddle (Y >= 150) or during
     the pre-launch state (Y == 0).

This forces the policy to continuously track the ball through the upper field
rather than memorize timed trajectories. The ball's final approach to the
paddle is always clean, so reactive tracking is both necessary and sufficient.

RAM addresses (verified 2026-07-19, ALE 0.11.2):
  Ball X: 99   (read/write, range ~0-160)
  Ball Y: 101  (read/write, range ~0-200, higher = closer to paddle)
  Paddle X: 72 (read only, range ~0-160)

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = ALEBreakoutFlightRandomized(env, teleport_prob=0.60)
    # Then: FireResetEnv -> EpisodicLifeEnv -> GrayscaleResize ->
    #       ClipRewardEnv -> Monitor -> DummyVecEnv -> VecFrameStack
"""
import numpy as np
import gymnasium as gym


class ALEBreakoutFlightRandomized(gym.Wrapper):
    """Teleports ball mid-flight in upper/mid playfield.

    Only teleports when the ball is in flight through the upper portion
    of the screen. The paddle zone and ball-launch are never teleported,
    so hitting the ball is never penalized and the agent always gets a
    clean final approach to the paddle.
    """

    # Teleport zone bounds (ALE Breakout coordinates)
    BALL_X_MIN = 10
    BALL_X_MAX = 150
    BALL_Y_MIN = 20       # top of teleport zone
    BALL_Y_MAX = 140      # bottom of teleport zone (upper field)
    NO_TELEPORT_Y = 150   # at or below this = paddle zone, never teleport

    # Ball sits at (0,0) before launch — skip these frames
    PRELAUNCH_BALL_Y = 0

    def __init__(self, env, teleport_prob=0.60, seed=None):
        super().__init__(env)
        self.teleport_prob = float(teleport_prob)
        self._rng = np.random.default_rng(seed)

        # Per-episode state
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
    # Teleport logic
    # ------------------------------------------------------------------

    def _should_teleport(self, ball_y):
        """Check if ball is in the teleport zone (mid-flight, not near paddle)."""
        if self._frames_since_reset < 6:
            return False
        if ball_y == self.PRELAUNCH_BALL_Y:
            return False
        if ball_y >= self.NO_TELEPORT_Y:
            return False
        return True

    def _teleport_ball(self):
        """Move ball to a random position within the teleport zone."""
        new_x = int(self._rng.integers(self.BALL_X_MIN, self.BALL_X_MAX + 1))
        new_y = int(self._rng.integers(self.BALL_Y_MIN, self.BALL_Y_MAX + 1))
        self._set_ball_xy(new_x, new_y)
        self.intervention_count += 1

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._frames_since_reset = 0
        self.intervention_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        ball_x, ball_y = self._get_ball_xy()

        if self._should_teleport(ball_y):
            if self._rng.random() < self.teleport_prob:
                self._teleport_ball()

        self._frames_since_reset += 1
        return self.env.step(action)
