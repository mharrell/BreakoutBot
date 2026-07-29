"""
AdversarialCursorWrapper — visible cursor that attacks the ball when the paddle
isn't tracking it, then retreats when tracking resumes.

This is the first "secondary agent" in BreakoutBot — a scripted entity with
its own state machine that the PPO agent must learn to react to.

The key difference from all previous adversarial wrappers: the threat is VISIBLE.
A cursor appears near the ball and pulses before attacking. The agent sees it in
the observation and has time to react — exactly the pattern that makes BeamRider
work (visible enemies before they shoot).

State machine:
  APPROACHING (invisible): Cursor moves toward ball at approach_speed px/step.
    If paddle tracks ball (|px - bx| <= tracking_threshold): cursor retreats.
    Cursor reaches within threat_radius of ball + paddle not tracking → THREATENING.

  THREATENING (visible, pulsing): Warning for warning_frames steps.
    Paddle can abort by moving within tracking_threshold → COOLDOWN (no attack).
    If warning expires without abort → ATTACK.

  ATTACK (visible, bright flash): Push ball away from paddle by push_magnitude px.
    Also flips ball horizontal direction (RAM[105]) for natural-looking trajectory.
    → COOLDOWN.

  COOLDOWN (invisible): Wait cooldown_frames steps. Cursor respawns at random
    horizontal position. → APPROACHING.

Key design: cursor is only VISIBLE during THREATENING and ATTACK states.
When the paddle tracks the ball, the cursor stays hidden (either retreating
in APPROACHING or waiting in COOLDOWN). This means:

  - Training: cursor appears → agent sees threat → moves paddle → cursor hides.
    The agent learns the causal chain: "track ball → no visible threat."
  - Eval (standard Breakout, no wrapper): no cursor ever appears, but the
    agent still tracks the ball because tracking is the underlying behavior
    that was rewarded during training. The cursor was a training signal, not
    a permanent crutch.

Placement in wrapper chain:
  After FireResetEnv, BEFORE GrayscaleResize/ClipRewardEnv.
  Must have access to ale.getRAM() / ale.setRAM() and the RGB observation.
"""
import numpy as np
import gymnasium as gym


class AdversarialCursorWrapper(gym.Wrapper):
    """Visible cursor adversary that pushes the ball when paddle isn't tracking.

    Args:
        env: Gym environment with .unwrapped.ale (ALE interface)
        approach_speed: Pixels per step cursor moves toward/away from ball.
        tracking_threshold: |paddle_x - ball_x| <= this → paddle is tracking.
        threat_radius: Cursor within this distance of ball → threatening.
        warning_frames: Steps of visible pulsing before attack (time to react).
        push_magnitude: Pixels to push ball away from paddle on attack.
        cooldown_frames: Steps of invisibility after attack before cursor respawns.
        cursor_size: Half-width of the cursor square (4 → 9×9 square).
    """

    BALL_X_ADDR = 99
    BALL_Y_ADDR = 101
    PADDLE_X_ADDR = 72
    BALL_DIR_ADDR = 105  # 1=left, 255=right (unsigned)

    MIN_X = 8
    MAX_X = 152

    # State machine
    APPROACHING = 0
    THREATENING = 1
    ATTACK = 2
    COOLDOWN = 3

    def __init__(self, env, approach_speed=2.0, tracking_threshold=8,
                 threat_radius=8, warning_frames=5, push_magnitude=4.0,
                 cooldown_frames=60, cursor_size=4):
        super().__init__(env)
        self.approach_speed = float(approach_speed)
        self.tracking_threshold = float(tracking_threshold)
        self.threat_radius = float(threat_radius)
        self.warning_frames = int(warning_frames)
        self.push_magnitude = float(push_magnitude)
        self.cooldown_frames = int(cooldown_frames)
        self.cursor_size = int(cursor_size)

        # Internal state
        self._cursor_x = 80.0
        self._state = self.APPROACHING
        self._state_timer = 0
        self._total_push = 0.0
        self._attack_count = 0
        self._rng = np.random.RandomState()

    def _read_ram(self):
        ram = self.env.unwrapped.ale.getRAM()
        return int(ram[self.BALL_X_ADDR]), int(ram[self.BALL_Y_ADDR]), int(ram[self.PADDLE_X_ADDR])

    def _is_tracking(self, ball_x, paddle_x):
        return abs(ball_x - paddle_x) <= self.tracking_threshold

    def _clamp_x(self, x):
        return max(self.MIN_X, min(self.MAX_X, int(x)))

    def _draw_cursor(self, obs, bright=True):
        """Draw a bright square cursor on the RGB observation.

        obs shape is (210, 160, 3) — Atari screen.
        cursor_x, ball_y are in ALE RAM coordinates which map to
        (col, row) = (cursor_x, ball_y) in observation space.
        """
        cx = int(round(self._cursor_x))
        cy = int(round(self._cursor_y))
        half = self.cursor_size
        intensity = 255 if bright else 128

        y0 = max(0, cy - half)
        y1 = min(obs.shape[0], cy + half + 1)
        x0 = max(0, cx - half)
        x1 = min(obs.shape[1], cx + half + 1)

        if y1 > y0 and x1 > x0:
            obs[y0:y1, x0:x1, :] = intensity

    def step(self, action):
        # Step the environment first
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Read post-step state
        ball_x, ball_y, paddle_x = self._read_ram()
        self._cursor_y = float(ball_y)  # cursor tracks ball height exactly
        tracking = self._is_tracking(ball_x, paddle_x)

        push_applied = 0.0

        # --- State machine ---
        if self._state == self.APPROACHING:
            if tracking:
                # Retreat: move away from ball
                dx = self._cursor_x - ball_x
                if abs(dx) < 0.5:
                    direction = self._rng.choice([-1, 1])
                else:
                    direction = 1 if dx > 0 else -1
                self._cursor_x += direction * self.approach_speed
            else:
                # Approach: move toward ball
                dx = ball_x - self._cursor_x
                if abs(dx) > 0.5:
                    self._cursor_x += np.sign(dx) * self.approach_speed
                # If very close but not exactly at ball, snap
                else:
                    self._cursor_x = float(ball_x)

            # Check transition to THREATENING
            if not tracking and abs(self._cursor_x - ball_x) <= self.threat_radius:
                self._state = self.THREATENING
                self._state_timer = 0

        elif self._state == self.THREATENING:
            self._state_timer += 1
            # Cursor hovers near ball during warning
            # Paddle can abort by tracking
            if tracking:
                self._state = self.COOLDOWN
                self._state_timer = 0
            elif self._state_timer >= self.warning_frames:
                self._state = self.ATTACK
                self._state_timer = 0

        elif self._state == self.ATTACK:
            # Push ball away from paddle
            push_dir = 1 if ball_x >= paddle_x else -1
            push_amount = push_dir * self.push_magnitude
            new_ball_x = self._clamp_x(ball_x + push_amount)
            self.env.unwrapped.ale.setRAM(self.BALL_X_ADDR, new_ball_x)
            # Also flip ball direction for natural-looking trajectory
            current_dir = int(self.env.unwrapped.ale.getRAM()[self.BALL_DIR_ADDR])
            # 1 = left, 255 = right. Set direction away from paddle.
            new_dir = 255 if push_dir > 0 else 1
            self.env.unwrapped.ale.setRAM(self.BALL_DIR_ADDR, new_dir)

            push_applied = push_amount
            self._total_push += abs(push_amount)
            self._attack_count += 1

            self._state = self.COOLDOWN
            self._state_timer = 0

        elif self._state == self.COOLDOWN:
            self._state_timer += 1
            if self._state_timer >= self.cooldown_frames:
                # Respawn cursor at random horizontal position
                self._cursor_x = float(self._rng.randint(20, 141))
                self._cursor_y = float(ball_y)
                self._state = self.APPROACHING
                self._state_timer = 0

        # --- Draw cursor on observation if visible ---
        if self._state == self.THREATENING:
            # Pulsing: alternates bright/dim each step
            self._draw_cursor(obs, bright=(self._state_timer % 2 == 0))
        elif self._state == self.ATTACK:
            self._draw_cursor(obs, bright=True)

        # --- Surface info ---
        if info is None:
            info = {}
        info['adv_state'] = self._state
        info['adv_push'] = push_applied
        info['adv_cursor_x'] = self._cursor_x
        info['adv_attacks'] = self._attack_count

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ball_x, ball_y, paddle_x = self._read_ram()

        # Initialize cursor at random horizontal position, ball's height
        self._cursor_x = float(self._rng.randint(20, 141))
        self._cursor_y = float(ball_y)
        self._state = self.APPROACHING
        self._state_timer = 0
        self._total_push = 0.0
        self._attack_count = 0

        if info is None:
            info = {}
        info['adv_state'] = self._state
        info['adv_cursor_x'] = self._cursor_x
        info['adv_cursor_y'] = self._cursor_y

        return obs, info
