"""
ALEBreakoutXMirror — reflect ball X position across playfield center mid-flight.

Reflects the ball's horizontal coordinate across the playfield center:
X_new = CENTER*2 - X_old. The ball's natural velocity is preserved — it
continues in the same direction from the mirrored position. Vertical
descent (paddle timing) is intact. No fake brick scores.

Uses a cooldown mechanism: after a mirror, the ball's X is locked for a
minimum number of frames before another mirror can fire. This guarantees
clean, trackable trajectory windows between mirrors — the model can learn
cause-and-effect rather than chasing random X positions.

Mechanism:
  1. Read Ball Y (addr 101) from ALE RAM each frame.
  2. If the ball is in flight and cooldown has expired, roll for mirror
     with configured per-frame probability.
  3. On mirror: read Ball X (addr 99), reflect across center, write back.
  4. Reset cooldown counter — ball gets a guaranteed clean window.

RAM addresses (verified 2026-07-19, ALE 0.11.2):
  Ball X: 99   (read/write, range ~1-159)
  Ball Y: 101  (read/write, range 0-200+, higher = closer to paddle)

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = ALEBreakoutXMirror(env, cooldown_frames=30, mirror_prob=0.10)
    # Then: NoopResetEnv -> FireResetEnv -> EpisodicLifeEnv ->
    #       GrayscaleResize -> ClipRewardEnv -> Monitor
"""
import numpy as np
import gymnasium as gym


class ALEBreakoutXMirror(gym.Wrapper):
    """Mirrors ball X position mid-flight with cooldown-gated probability.

    After each mirror, a cooldown period (default 30 frames = 0.5 seconds
    at 60 fps) guarantees the ball follows a clean, trackable trajectory
    before another mirror can fire. Once cooldown expires, each frame
    rolls with `mirror_prob` for a reflection.

    At default settings (cooldown=30, prob=0.10), a life lasting ~2 seconds
    gets 2-4 mirrors, each followed by a clean half-second approach.
    """

    # Playfield bounds (ALE Breakout coordinates)
    BALL_X_MIN = 10
    BALL_X_MAX = 150
    NO_MIRROR_Y = 150     # at or below this = paddle zone, never mirror

    # Pre-launch state
    PRELAUNCH_BALL_Y = 0

    def __init__(self, env, cooldown_frames=30, mirror_prob=0.10, seed=None):
        super().__init__(env)
        self.cooldown_frames = int(cooldown_frames)
        self.mirror_prob = float(mirror_prob)
        self._rng = np.random.default_rng(seed)

        # Per-episode state
        self._frames_since_reset = 0
        self._frames_since_mirror = 0
        self.intervention_count = 0

    # ------------------------------------------------------------------
    # RAM access
    # ------------------------------------------------------------------

    def _get_ball_xy(self):
        """Read current ball position from RAM. Returns (x, y)."""
        ale = self.env.unwrapped.ale
        ram = ale.getRAM()
        return int(ram[99]), int(ram[101])

    def _set_ball_x(self, x):
        """Write ball X position to RAM."""
        ale = self.env.unwrapped.ale
        ale.setRAM(99, int(x))

    # ------------------------------------------------------------------
    # Mirror logic
    # ------------------------------------------------------------------

    def _mirror_x(self, current_x):
        """Reflect X across the playfield center, clamped to valid range.

        X=10 (far left)  → X=150 (far right)
        X=150 (far right) → X=10 (far left)
        X=80 (center)    → X=80 (unchanged)
        """
        center = (self.BALL_X_MIN + self.BALL_X_MAX) / 2.0  # 80.0
        new_x = 2.0 * center - float(current_x)
        return int(max(1.0, min(159.0, new_x)))

    def _in_mirror_zone(self, ball_y):
        """Check if ball is in a state where mirroring is valid."""
        if self._frames_since_reset < 6:
            return False         # grace period after reset
        if ball_y == self.PRELAUNCH_BALL_Y:
            return False         # pre-launch — ball waiting to be served
        if ball_y >= self.NO_MIRROR_Y:
            return False         # near paddle — preserve natural approach
        return True

    def _cooldown_expired(self):
        """Check if the mirror cooldown has elapsed."""
        return self._frames_since_mirror >= self.cooldown_frames

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._frames_since_reset = 0
        self._frames_since_mirror = 0
        self.intervention_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        ball_x, ball_y = self._get_ball_xy()

        if self._in_mirror_zone(ball_y) and self._cooldown_expired():
            if self._rng.random() < self.mirror_prob:
                new_x = self._mirror_x(ball_x)
                self._set_ball_x(new_x)
                self.intervention_count += 1
                self._frames_since_mirror = 0

        self._frames_since_reset += 1
        self._frames_since_mirror += 1
        return self.env.step(action)
