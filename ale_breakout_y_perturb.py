"""
ALEBreakoutYPerturb — shift ball Y position mid-flight by a random offset.

The X-mirror wrapper breaks *where* the ball arrives horizontally. This wrapper
breaks *when* the ball arrives: a ±DY pixel shift means the ball reaches paddle
height a few frames earlier or later, so a timed memorized sequence that works
at "frame 17" fails at frame 15 or 21.

Mechanism:
  1. Read Ball Y (addr 101) from ALE RAM each frame.
  2. If the ball is mid-flight in open space and cooldown has expired, roll
     for perturbation with configured per-frame probability.
  3. On perturb: add a random offset in [-range, +range], clamp to [1, 159]
     (ALE setRAM accepts unsigned bytes, 0-255).
  4. Reset cooldown counter — ball gets a guaranteed clean window.

RAM addresses (verified 2026-07-19, ALE 0.11.2):
  Ball Y: 101  (read/write, range 0-200+, higher = closer to paddle)

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = ALEBreakoutXMirror(env, cooldown_frames=30, mirror_prob=0.10)
    env = ALEBreakoutYPerturb(env, cooldown_frames=30, perturb_prob=0.10, perturb_range=8)
    # Then: FireResetEnv -> EpisodicLifeEnv -> GrayscaleResize -> ClipRewardEnv -> Monitor
"""
import numpy as np
import gymnasium as gym


class ALEBreakoutYPerturb(gym.Wrapper):
    """Shift ball Y position mid-flight by a random offset with cooldown-gated probability.

    After each perturbation, a cooldown period guarantees the ball follows a
    clean, trackable trajectory before another perturbation can fire. Once
    cooldown expires, each frame rolls with `perturb_prob` for a shift.

    The shift range of ±8 pixels corresponds to roughly 3-5 frames of arrival
    timing difference at typical ball vertical speeds — enough to break a
    timed script, not so large the ball leaves the playfield.
    """

    # Playfield bounds
    Y_MIN = 1        # top of screen (unsigned byte, can't be 0 or setRAM clamps)
    Y_MAX = 159      # bottom of playable area (unsigned byte max safe value)

    # Zone where perturbation is valid (open space, no bricks, no paddle)
    Y_PERTURB_MIN = 30   # below brick zone
    Y_PERTURB_MAX = 130  # above paddle zone

    # Pre-launch state
    PRELAUNCH_BALL_Y = 0

    def __init__(self, env, cooldown_frames=30, perturb_prob=0.10, perturb_range=8, seed=None):
        super().__init__(env)
        self.cooldown_frames = int(cooldown_frames)
        self.perturb_prob = float(perturb_prob)
        self.perturb_range = int(perturb_range)
        self._rng = np.random.default_rng(seed)

        # Per-episode state
        self._frames_since_reset = 0
        self._frames_since_perturb = 0
        self.intervention_count = 0

    # ------------------------------------------------------------------
    # RAM access
    # ------------------------------------------------------------------

    def _get_ball_y(self):
        """Read current ball Y position from RAM."""
        ale = self.env.unwrapped.ale
        ram = ale.getRAM()
        return int(ram[101])

    def _set_ball_y(self, y):
        """Write ball Y position to RAM, clamped to valid unsigned byte range."""
        ale = self.env.unwrapped.ale
        ale.setRAM(101, int(max(1, min(159, y))))

    # ------------------------------------------------------------------
    # Perturbation logic
    # ------------------------------------------------------------------

    def _in_perturb_zone(self, ball_y):
        """Check if ball is mid-flight in open space — safe to shift Y."""
        if self._frames_since_reset < 6:
            return False          # grace period after reset
        if ball_y == self.PRELAUNCH_BALL_Y:
            return False          # pre-launch — ball waiting to be served
        if ball_y < self.Y_PERTURB_MIN:
            return False          # still in/near brick zone
        if ball_y > self.Y_PERTURB_MAX:
            return False          # near paddle — preserve natural timing
        return True

    def _cooldown_expired(self):
        """Check if the perturbation cooldown has elapsed."""
        return self._frames_since_perturb >= self.cooldown_frames

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._frames_since_reset = 0
        self._frames_since_perturb = 0
        self.intervention_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        ball_y = self._get_ball_y()

        if self._in_perturb_zone(ball_y) and self._cooldown_expired():
            if self._rng.random() < self.perturb_prob:
                offset = self._rng.integers(-self.perturb_range, self.perturb_range + 1)
                self._set_ball_y(ball_y + offset)
                self.intervention_count += 1
                self._frames_since_perturb = 0

        self._frames_since_reset += 1
        self._frames_since_perturb += 1
        return self.env.step(action)
