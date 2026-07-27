"""
ALEBreakoutDynamicsRandomized — perturb ball and paddle positions via setRAM().

This is the ALE equivalent of DynamicBreakout (the custom engine wrapper that
produced the project's only intervention-robust policy, PPO_35). Instead of
smooth physics parameter interpolation (which requires engine-level control),
we directly teleport objects mid-game by writing to ALE RAM.

Three independent perturbation channels:

  1. Ball Y  (RAM 101): shift ball vertically   ±8px  — changes arrival timing
  2. Ball X  (RAM 99):  shift ball horizontally  ±6px  — changes lateral position
  3. Paddle X (RAM 72): shift paddle horizontally ±4px  — changes bounce angle

Each channel has its own cooldown timer and per-frame probability. Once a
channel's cooldown expires, each subsequent frame rolls with `perturb_prob`
until a perturbation fires, then the cooldown resets. Channels are independent
— sometimes one fires, sometimes two, rarely all three at once.

The cooldown (60 frames = 1 second at 60fps) guarantees the ball follows a
clean, trackable trajectory between perturbations. The agent always has time
to observe and react before the next disruption.

Why this should work (where perceptual noise didn't):
  - Timed scripts fail: the ball is at a different position than expected
  - Position-conditioned scripts fail: paddle shift → different bounce angle
  - Rhythm-learning fails: independent channels → no predictable pattern
  - The ONLY strategy that works across all perturbation realizations is to
    observe the ball's actual position and paddle's actual position each frame

RAM addresses (ALE 0.11.2, Breakout ROM):
  Ball Y:  101 — vertical position (0 = top, higher = lower on screen)
  Ball X:   99 — horizontal position
  Paddle X: 72 — horizontal position of player paddle

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=1, repeat_action_probability=0)
    env = ALEBreakoutDynamicsRandomized(env)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = EpisodicLifeEnv(env)
    env = GrayscaleResize(env, width=84, height=84)
    # ... VecFrameStack or OpticalFlow, ClipReward, Monitor, etc.
"""
import numpy as np
import gymnasium as gym


class ALEBreakoutDynamicsRandomized(gym.Wrapper):
    """Multi-channel dynamics randomization via ALE setRAM().

    Three independent perturbation channels (ball Y, ball X, paddle X),
    each with its own cooldown and per-frame probability. Perturbations
    only fire when the ball is in open space (not near bricks or paddle).

    Parameters
    ----------
    env : gym.Env
        Must be ALE/Breakout-v5 (needs .unwrapped.ale for setRAM).
    ball_y_prob : float
        Per-frame probability of ball Y perturbation after cooldown.
    ball_x_prob : float
        Per-frame probability of ball X perturbation after cooldown.
    paddle_x_prob : float
        Per-frame probability of paddle X perturbation after cooldown.
    cooldown_frames : int
        Minimum frames between perturbations (per channel).
    ball_y_range : int
        ±range for ball Y offset in pixels.
    ball_x_range : int
        ±range for ball X offset in pixels.
    paddle_x_range : int
        ±range for paddle X offset in pixels.
    seed : int or None
        Seed for the internal RNG.
    """

    # ------------------------------------------------------------------
    # Playfield bounds (unsigned byte range for setRAM)
    # ------------------------------------------------------------------

    # Ball Y: 0=top of screen. 1-159 is the safe unsigned byte range.
    BALL_Y_MIN = 1
    BALL_Y_MAX = 159

    # Ball X: horizontal position in TIA coordinates. Verified: ball can reach
    # X=180+ near right wall. Playfield is roughly 10-195 in TIA coords.
    BALL_X_MIN = 10
    BALL_X_MAX = 190

    # Paddle X: horizontal position. Paddle is ~15px wide.
    PADDLE_X_MIN = 8
    PADDLE_X_MAX = 140

    # Zone where ball perturbation is valid (open space only).
    Y_PERTURB_MIN = 30   # below brick zone
    Y_PERTURB_MAX = 130  # above paddle zone

    # Pre-launch state (ball waiting to be served).
    PRELAUNCH_BALL_Y = 0

    def __init__(
        self,
        env,
        ball_y_prob=0.01,
        ball_x_prob=0.01,
        paddle_x_prob=0.01,
        cooldown_frames=60,
        ball_y_range=8,
        ball_x_range=6,
        paddle_x_range=4,
        ball_noise_std=0.0,
        seed=None,
    ):
        super().__init__(env)
        self.ball_y_prob = float(ball_y_prob)
        self.ball_x_prob = float(ball_x_prob)
        self.paddle_x_prob = float(paddle_x_prob)
        self.cooldown_frames = int(cooldown_frames)
        self.ball_y_range = int(ball_y_range)
        self.ball_x_range = int(ball_x_range)
        self.paddle_x_range = int(paddle_x_range)
        self.ball_noise_std = float(ball_noise_std)
        self._rng = np.random.default_rng(seed)

        # Per-episode state
        self._frames_since_reset = 0

        # Cooldown counters — one per channel
        self._ball_y_cooldown = 0
        self._ball_x_cooldown = 0
        self._paddle_x_cooldown = 0

        # Perturbation counts
        self.ball_y_perturb_count = 0
        self.ball_x_perturb_count = 0
        self.paddle_x_perturb_count = 0

    # ------------------------------------------------------------------
    # RAM access
    # ------------------------------------------------------------------

    def _get_ram(self, addr):
        """Read a single byte from ALE RAM."""
        return int(self.env.unwrapped.ale.getRAM()[addr])

    def _set_ram(self, addr, value):
        """Write a single byte to ALE RAM."""
        self.env.unwrapped.ale.setRAM(addr, int(value))

    def _get_ball_y(self):
        return self._get_ram(101)

    def _set_ball_y(self, y):
        self._set_ram(101, int(max(self.BALL_Y_MIN, min(self.BALL_Y_MAX, y))))

    def _get_ball_x(self):
        return self._get_ram(99)

    def _set_ball_x(self, x):
        self._set_ram(99, int(max(self.BALL_X_MIN, min(self.BALL_X_MAX, x))))

    def _get_paddle_x(self):
        return self._get_ram(72)

    def _set_paddle_x(self, x):
        self._set_ram(72, int(max(self.PADDLE_X_MIN, min(self.PADDLE_X_MAX, x))))

    # ------------------------------------------------------------------
    # Zone gating
    # ------------------------------------------------------------------

    def _ball_in_perturb_zone(self):
        """Ball is mid-flight in open space — safe to shift X or Y."""
        ball_y = self._get_ball_y()
        if self._frames_since_reset < 6:
            return False
        if ball_y == self.PRELAUNCH_BALL_Y:
            return False
        if ball_y < self.Y_PERTURB_MIN:
            return False
        if ball_y > self.Y_PERTURB_MAX:
            return False
        return True

    # ------------------------------------------------------------------
    # Perturbation logic
    # ------------------------------------------------------------------

    def _try_perturb_ball_y(self):
        """Attempt ball Y perturbation if cooldown expired and zone is clear."""
        if self._ball_y_cooldown > 0:
            return
        if not self._ball_in_perturb_zone():
            return
        if self._rng.random() >= self.ball_y_prob:
            return
        offset = self._rng.integers(-self.ball_y_range, self.ball_y_range + 1)
        current = self._get_ball_y()
        self._set_ball_y(current + offset)
        self.ball_y_perturb_count += 1
        self._ball_y_cooldown = self.cooldown_frames

    def _try_perturb_ball_x(self):
        """Attempt ball X perturbation if cooldown expired and zone is clear."""
        if self._ball_x_cooldown > 0:
            return
        if not self._ball_in_perturb_zone():
            return
        if self._rng.random() >= self.ball_x_prob:
            return
        offset = self._rng.integers(-self.ball_x_range, self.ball_x_range + 1)
        current = self._get_ball_x()
        self._set_ball_x(current + offset)
        self.ball_x_perturb_count += 1
        self._ball_x_cooldown = self.cooldown_frames

    def _apply_ball_noise(self):
        """Add per-frame Gaussian noise to ball X and Y.

        At σ=0.5, ~32% of frames get a ±1px kick. After 100 frames the ball
        drifts ~3px from its natural path — not visible frame-to-frame, but
        compounding over hundreds of frames to make the exact ball position
        unpredictable. No zone gating — noise applies everywhere the ball moves.

        This is the mechanism DynamicBreakout used: continuous parameter drift
        that makes every ball path unique. Rare teleports (PPO_78) were too
        easy to ignore because 99% of frames followed the script exactly.
        """
        if self.ball_noise_std <= 0:
            return
        # Ball Y
        noise_y = int(round(float(self._rng.normal(0.0, self.ball_noise_std))))
        if noise_y != 0:
            self._set_ball_y(self._get_ball_y() + noise_y)
        # Ball X
        noise_x = int(round(float(self._rng.normal(0.0, self.ball_noise_std))))
        if noise_x != 0:
            self._set_ball_x(self._get_ball_x() + noise_x)

    def _try_perturb_paddle_x(self):
        """Attempt paddle X perturbation if cooldown expired.

        No zone gating — paddle is always on screen and always valid to shift.
        """
        if self._paddle_x_cooldown > 0:
            return
        if self._rng.random() >= self.paddle_x_prob:
            return
        offset = self._rng.integers(-self.paddle_x_range, self.paddle_x_range + 1)
        current = self._get_paddle_x()
        self._set_paddle_x(current + offset)
        self.paddle_x_perturb_count += 1
        self._paddle_x_cooldown = self.cooldown_frames

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._frames_since_reset = 0
        self._ball_y_cooldown = 0
        self._ball_x_cooldown = 0
        self._paddle_x_cooldown = 0
        self.ball_y_perturb_count = 0
        self.ball_x_perturb_count = 0
        self.paddle_x_perturb_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        # 1. Per-frame noise: tiny drift every frame, compounding over time.
        #    This is the key anti-memorization mechanism — the ball is NEVER
        #    exactly where a script expects it.
        self._apply_ball_noise()

        # 2. Large teleports: occasional big jumps the agent must recover from.
        self._try_perturb_ball_y()
        self._try_perturb_ball_x()
        self._try_perturb_paddle_x()

        # Decrement cooldowns (but not below 0)
        if self._ball_y_cooldown > 0:
            self._ball_y_cooldown -= 1
        if self._ball_x_cooldown > 0:
            self._ball_x_cooldown -= 1
        if self._paddle_x_cooldown > 0:
            self._paddle_x_cooldown -= 1

        self._frames_since_reset += 1
        return self.env.step(action)
