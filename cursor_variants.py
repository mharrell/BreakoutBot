"""
cursor_variants.py — Three next-generation adversarial cursor variants.

CursorAgent: Extracted state machine (shared by Variant C and future variants).
EpisodeRandomizedCursorWrapper: Parameters randomized per reset() (Variant A).
AlwaysVisibleCursorWrapper: Permanent spatial landmark, no attack (Variant B).
MultiCursorWrapper: Multiple independent asymmetric cursors (Variant C).

Original AdversarialCursorWrapper remains untouched as control baseline.
"""
import numpy as np
import gymnasium as gym


# ── CursorAgent — extracted state machine ──────────────────────────────

class CursorAgent:
    """Independent state machine for one cursor adversary.

    Extracted from AdversarialCursorWrapper so multiple cursors can share
    the same logic without duplicating code. Each CursorAgent tracks its
    own position, state, and parameters independently.

    The state machine is pixel-identical to the original wrapper:
      APPROACHING (0): move toward/away from ball, invisible.
      THREATENING (1): hover and pulse, visible. Aborts if tracking resumes.
      ATTACK (2): push ball away from paddle, visible bright flash.
      COOLDOWN (3): invisible wait, then respawn at random x.
    """

    APPROACHING = 0
    THREATENING = 1
    ATTACK = 2
    COOLDOWN = 3

    # Clamped x bounds for ball and cursor positioning
    MIN_X = 8
    MAX_X = 152

    def __init__(self, rng, params, agent_id=0, spawn_side=None):
        """
        Args:
            rng: numpy RandomState for reproducible randomness.
            params: dict with cursor parameters (approach_speed, etc.).
            agent_id: integer identifier (for info dict / debugging).
            spawn_side: None=anywhere, 'left'=[8,60), 'right'=[100,152].
        """
        self.rng = rng
        self.agent_id = agent_id
        self.spawn_side = spawn_side

        # Unpack parameters
        self.approach_speed = float(params.get('approach_speed', 2.0))
        self.tracking_threshold = float(params.get('tracking_threshold', 8))
        self.threat_radius = float(params.get('threat_radius', 8))
        self.warning_frames = int(params.get('warning_frames', 5))
        self.push_magnitude = float(params.get('push_magnitude', 4.0))
        self.cooldown_frames = int(params.get('cooldown_frames', 60))
        self.cursor_size = int(params.get('cursor_size', 4))

        # Internal state
        self.cursor_x = 80.0
        self.cursor_y = 100.0
        self.state = self.APPROACHING
        self.state_timer = 0
        self.attack_count = 0
        self.total_push = 0.0

    # ── spawn helpers ──────────────────────────────────────────────

    def _spawn_x(self):
        """Random x within spawn zone."""
        if self.spawn_side == 'left':
            return float(self.rng.uniform(self.MIN_X, 60))
        elif self.spawn_side == 'right':
            return float(self.rng.uniform(100, self.MAX_X + 1))
        else:
            return float(self.rng.randint(20, 141))

    # ── public API ─────────────────────────────────────────────────

    def reset(self, ball_x, ball_y):
        """Reset cursor state at episode start or cooldown expiry."""
        self.cursor_x = self._spawn_x()
        self.cursor_y = float(ball_y)
        self.state = self.APPROACHING
        self.state_timer = 0
        self.attack_count = 0
        self.total_push = 0.0

    def update(self, ball_x, ball_y, paddle_x, tracking):
        """Advance the state machine by one step.

        Args:
            ball_x, ball_y: ball position from ALE RAM.
            paddle_x: paddle position from ALE RAM.
            tracking: bool — is paddle tracking the ball?

        Returns dict with:
            push_applied: float (0.0 if no attack, signed push amount if attack).
            did_attack: bool.
            is_visible: bool — should cursor be drawn this step?
            brightness: bool or None — True=bright, False=dim, None=not visible.
            state: int — current state (for info/monitoring).
        """
        push_applied = 0.0
        did_attack = False

        # Cursor always tracks ball height
        self.cursor_y = float(ball_y)

        # ── APPROACHING ────────────────────────────────────────────
        if self.state == self.APPROACHING:
            if tracking:
                # Retreat: move away from ball
                dx = self.cursor_x - ball_x
                if abs(dx) < 0.5:
                    direction = self.rng.choice([-1, 1])
                else:
                    direction = 1 if dx > 0 else -1
                self.cursor_x += direction * self.approach_speed
            else:
                # Approach: move toward ball
                dx = ball_x - self.cursor_x
                if abs(dx) > 0.5:
                    self.cursor_x += np.sign(dx) * self.approach_speed
                else:
                    self.cursor_x = float(ball_x)

            # Transition to THREATENING: cursor close + paddle not tracking
            if not tracking and abs(self.cursor_x - ball_x) <= self.threat_radius:
                self.state = self.THREATENING
                self.state_timer = 0

        # ── THREATENING ────────────────────────────────────────────
        elif self.state == self.THREATENING:
            self.state_timer += 1
            # Paddle can abort by tracking
            if tracking:
                self.state = self.COOLDOWN
                self.state_timer = 0
            elif self.state_timer >= self.warning_frames:
                self.state = self.ATTACK
                self.state_timer = 0

        # ── ATTACK ─────────────────────────────────────────────────
        elif self.state == self.ATTACK:
            # Push ball away from paddle
            push_dir = 1 if ball_x >= paddle_x else -1
            push_applied = push_dir * self.push_magnitude
            self.total_push += abs(push_applied)
            self.attack_count += 1
            did_attack = True

            self.state = self.COOLDOWN
            self.state_timer = 0

        # ── COOLDOWN ───────────────────────────────────────────────
        elif self.state == self.COOLDOWN:
            self.state_timer += 1
            if self.state_timer >= self.cooldown_frames:
                self.cursor_x = self._spawn_x()
                self.cursor_y = float(ball_y)
                self.state = self.APPROACHING
                self.state_timer = 0

        # ── Visibility ─────────────────────────────────────────────
        is_visible = self.state in (self.THREATENING, self.ATTACK)
        brightness = None
        if self.state == self.THREATENING:
            brightness = (self.state_timer % 2 == 0)  # pulsing
        elif self.state == self.ATTACK:
            brightness = True

        return {
            'push_applied': push_applied,
            'did_attack': did_attack,
            'is_visible': is_visible,
            'brightness': brightness,
            'state': self.state,
        }


# ── Shared rendering helper ────────────────────────────────────────────

def _draw_cursor_on_obs(obs, cursor_x, cursor_y, cursor_size, bright):
    """Draw a single cursor square on an RGB observation in-place.

    Args:
        obs: numpy array shape (210, 160, 3), uint8.
        cursor_x, cursor_y: float pixel coordinates (ALE RAM space).
        cursor_size: int half-width (4 → 9×9 square).
        bright: bool — True=255 intensity, False=128.
    """
    cx = int(round(cursor_x))
    cy = int(round(cursor_y))
    half = cursor_size
    intensity = 255 if bright else 128

    y0 = max(0, cy - half)
    y1 = min(obs.shape[0], cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(obs.shape[1], cx + half + 1)

    if y1 > y0 and x1 > x0:
        obs[y0:y1, x0:x1, :] = intensity


# ── Variant A: Episode-Randomized Parameters ────────────────────────────

class EpisodeRandomizedCursorWrapper(gym.Wrapper):
    """Cursor with parameters randomized per reset() — un-anticipatable timing.

    Subclasses AdversarialCursorWrapper conceptually but implemented
    independently to avoid tight coupling. The state machine logic is
    identical; only the parameter values change between resets.

    Parameters are drawn from configurable distributions:
      - approach_speed, push_magnitude: log-uniform (multiplicative range)
      - cooldown_frames, warning_frames: uniform int (additive range)
      - tracking_threshold, threat_radius: uniform (additive range)
      - cursor_size: fixed (not randomized)
    """

    BALL_X_ADDR = 99
    BALL_Y_ADDR = 101
    PADDLE_X_ADDR = 72
    BALL_DIR_ADDR = 105

    MIN_X = 8
    MAX_X = 152

    APPROACHING = 0
    THREATENING = 1
    ATTACK = 2
    COOLDOWN = 3

    # Default parameter ranges: (min, max, distribution_type)
    DEFAULT_PARAM_RANGES = {
        'approach_speed': (1.0, 8.0, 'log_uniform'),
        'push_magnitude': (1.0, 16.0, 'log_uniform'),
        'cooldown_frames': (30, 150, 'uniform_int'),
        'warning_frames': (2, 12, 'uniform_int'),
        'tracking_threshold': (4, 20, 'uniform'),
        'threat_radius': (4, 20, 'uniform'),
    }

    def __init__(self, env, cursor_size=4, rng_seed=None, param_ranges=None):
        super().__init__(env)
        self._cursor_size = int(cursor_size)
        self._rng = np.random.RandomState(rng_seed)
        self.param_ranges = {**self.DEFAULT_PARAM_RANGES, **(param_ranges or {})}

        # Current params (set on first reset)
        self.approach_speed = 2.0
        self.tracking_threshold = 8.0
        self.threat_radius = 8.0
        self.warning_frames = 5
        self.push_magnitude = 4.0
        self.cooldown_frames = 60

        # Internal state
        self._cursor_x = 80.0
        self._cursor_y = 100.0
        self._state = self.APPROACHING
        self._state_timer = 0
        self._total_push = 0.0
        self._attack_count = 0
        self._current_params = {}

    def _sample_params(self):
        """Sample new parameter values from configured distributions."""
        new_params = {}
        for key, (lo, hi, dist_type) in self.param_ranges.items():
            if dist_type == 'log_uniform':
                log_lo, log_hi = np.log(lo), np.log(hi)
                val = np.exp(self._rng.uniform(log_lo, log_hi))
                new_params[key] = float(val)
            elif dist_type == 'uniform_int':
                new_params[key] = int(self._rng.randint(lo, hi + 1))
            elif dist_type == 'uniform':
                new_params[key] = float(self._rng.uniform(lo, hi))
        return new_params

    def _apply_params(self, params):
        """Set instance attributes from parameter dict."""
        self.approach_speed = float(params['approach_speed'])
        self.push_magnitude = float(params['push_magnitude'])
        self.cooldown_frames = int(params['cooldown_frames'])
        self.warning_frames = int(params['warning_frames'])
        self.tracking_threshold = float(params['tracking_threshold'])
        self.threat_radius = float(params['threat_radius'])

    def _read_ram(self):
        ram = self.env.unwrapped.ale.getRAM()
        return (int(ram[self.BALL_X_ADDR]),
                int(ram[self.BALL_Y_ADDR]),
                int(ram[self.PADDLE_X_ADDR]))

    def _is_tracking(self, ball_x, paddle_x):
        return abs(ball_x - paddle_x) <= self.tracking_threshold

    def _clamp_x(self, x):
        return max(self.MIN_X, min(self.MAX_X, int(x)))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ball_x, ball_y, paddle_x = self._read_ram()
        self._cursor_y = float(ball_y)
        tracking = self._is_tracking(ball_x, paddle_x)
        push_applied = 0.0

        if self._state == self.APPROACHING:
            if tracking:
                dx = self._cursor_x - ball_x
                if abs(dx) < 0.5:
                    direction = self._rng.choice([-1, 1])
                else:
                    direction = 1 if dx > 0 else -1
                self._cursor_x += direction * self.approach_speed
            else:
                dx = ball_x - self._cursor_x
                if abs(dx) > 0.5:
                    self._cursor_x += np.sign(dx) * self.approach_speed
                else:
                    self._cursor_x = float(ball_x)
            if not tracking and abs(self._cursor_x - ball_x) <= self.threat_radius:
                self._state = self.THREATENING
                self._state_timer = 0

        elif self._state == self.THREATENING:
            self._state_timer += 1
            if tracking:
                self._state = self.COOLDOWN
                self._state_timer = 0
            elif self._state_timer >= self.warning_frames:
                self._state = self.ATTACK
                self._state_timer = 0

        elif self._state == self.ATTACK:
            push_dir = 1 if ball_x >= paddle_x else -1
            push_amount = push_dir * self.push_magnitude
            new_ball_x = self._clamp_x(ball_x + push_amount)
            self.env.unwrapped.ale.setRAM(self.BALL_X_ADDR, new_ball_x)
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
                self._cursor_x = float(self._rng.randint(20, 141))
                self._cursor_y = float(ball_y)
                self._state = self.APPROACHING
                self._state_timer = 0

        if self._state == self.THREATENING:
            _draw_cursor_on_obs(obs, self._cursor_x, self._cursor_y,
                                self._cursor_size, bright=(self._state_timer % 2 == 0))
        elif self._state == self.ATTACK:
            _draw_cursor_on_obs(obs, self._cursor_x, self._cursor_y,
                                self._cursor_size, bright=True)

        if info is None:
            info = {}
        info['adv_state'] = self._state
        info['adv_push'] = push_applied
        info['adv_cursor_x'] = self._cursor_x
        info['adv_attacks'] = self._attack_count
        info['adv_params'] = self._current_params

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Sample new parameters BEFORE calling env.reset
        params = self._sample_params()
        self._apply_params(params)
        self._current_params = params

        obs, info = self.env.reset(**kwargs)
        ball_x, ball_y, paddle_x = self._read_ram()

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
        info['adv_params'] = self._current_params

        return obs, info


# ── Variant B: Always-Visible Cursor ──────────────────────────────────

class AlwaysVisibleCursorWrapper(gym.Wrapper):
    """Cursor always visible at ball position — purely informational, no attack.

    A permanent 9×9 square sits at (ball_x, ball_y) every frame.
    Intensity encodes tracking quality:
      - 255 (bright): paddle is tracking ball (|dx| <= tracking_threshold)
      - 128 (dim): paddle is not tracking ball

    No state machine, no ball pushing, no cooldown. The cursor is a spatial
    beacon that never disappears. Tests whether visual association alone,
    without punishment, can teach ball-tracking.

    If this fails (SINGLE_SCRIPT, 0% reversal), the fallback is B2: add
    an immediate attack when tracking is lost (no warning period).
    """

    BALL_X_ADDR = 99
    BALL_Y_ADDR = 101
    PADDLE_X_ADDR = 72

    def __init__(self, env, tracking_threshold=8, cursor_size=4):
        super().__init__(env)
        self.tracking_threshold = float(tracking_threshold)
        self.cursor_size = int(cursor_size)

    def _read_ram(self):
        ram = self.env.unwrapped.ale.getRAM()
        return (int(ram[self.BALL_X_ADDR]),
                int(ram[self.BALL_Y_ADDR]),
                int(ram[self.PADDLE_X_ADDR]))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ball_x, ball_y, paddle_x = self._read_ram()
        tracking = abs(ball_x - paddle_x) <= self.tracking_threshold

        # Always draw cursor at ball position
        _draw_cursor_on_obs(obs, float(ball_x), float(ball_y),
                            self.cursor_size, bright=tracking)

        if info is None:
            info = {}
        info['cursor_tracking'] = tracking

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if info is None:
            info = {}
        return obs, info


# ── Variant C: Multiple Independent Cursors ────────────────────────────

class MultiCursorWrapper(gym.Wrapper):
    """Multiple independent cursor adversaries with asymmetric behaviors.

    Each cursor is a CursorAgent with its own state machine, parameters,
    spawn zone, and attack timeline. Cursors operate independently —
    they can both be in APPROACHING, or one in ATTACK while the other
    is in COOLDOWN.

    Default configuration: 2 asymmetric cursors
      Cursor A (fast/light): speed=5, push=2, threshold=4, warning=3, cooldown=40
        Spawns left side. Forces constant precise tracking.
      Cursor B (slow/heavy): speed=1.5, push=8, threshold=16, warning=8, cooldown=80
        Spawns right side. Punishes sustained neglect.

    The policy must satisfy both simultaneously — a single sweeping script
    works for neither. This is the "un-anticipatable" variant.
    """

    BALL_X_ADDR = 99
    BALL_Y_ADDR = 101
    PADDLE_X_ADDR = 72
    BALL_DIR_ADDR = 105

    MIN_X = 8
    MAX_X = 152

    # Default asymmetric cursor configurations
    DEFAULT_CURSOR_CONFIGS = [
        {   # Cursor A: fast / light / tight tracking
            'approach_speed': 5.0,
            'tracking_threshold': 4,
            'threat_radius': 8,
            'warning_frames': 3,
            'push_magnitude': 2.0,
            'cooldown_frames': 40,
            'cursor_size': 4,
        },
        {   # Cursor B: slow / heavy / loose tracking
            'approach_speed': 1.5,
            'tracking_threshold': 16,
            'threat_radius': 8,
            'warning_frames': 8,
            'push_magnitude': 8.0,
            'cooldown_frames': 80,
            'cursor_size': 4,
        },
    ]

    def __init__(self, env, cursor_configs=None, max_push_per_step=20.0,
                 rng_seed=None):
        """
        Args:
            env: Gym environment with .unwrapped.ale interface.
            cursor_configs: list of param dicts, one per CursorAgent.
                If None, uses DEFAULT_CURSOR_CONFIGS (2 asymmetric cursors).
            max_push_per_step: cap on total ball displacement per step
                when multiple cursors attack simultaneously.
            rng_seed: seed for the wrapper's RNG.
        """
        super().__init__(env)
        self._rng = np.random.RandomState(rng_seed)
        self.max_push_per_step = float(max_push_per_step)

        if cursor_configs is None:
            cursor_configs = self.DEFAULT_CURSOR_CONFIGS

        spawn_sides = ['left', 'right']
        self.cursors = []
        for i, params in enumerate(cursor_configs):
            side = spawn_sides[i % len(spawn_sides)]
            agent = CursorAgent(self._rng, params, agent_id=i, spawn_side=side)
            self.cursors.append(agent)

    def _read_ram(self):
        ram = self.env.unwrapped.ale.getRAM()
        return (int(ram[self.BALL_X_ADDR]),
                int(ram[self.BALL_Y_ADDR]),
                int(ram[self.PADDLE_X_ADDR]))

    def _clamp_x(self, x):
        return max(self.MIN_X, min(self.MAX_X, int(x)))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ball_x, ball_y, paddle_x = self._read_ram()
        total_push = 0.0

        for cursor in self.cursors:
            tracking = abs(ball_x - paddle_x) <= cursor.tracking_threshold
            result = cursor.update(ball_x, ball_y, paddle_x, tracking)

            # Apply push if this cursor attacked
            if result['did_attack']:
                push_dir = 1 if ball_x >= paddle_x else -1
                push_amount = push_dir * cursor.push_magnitude

                # Cap total push per step
                remaining = self.max_push_per_step - abs(total_push)
                if abs(push_amount) > remaining:
                    push_amount = np.sign(push_amount) * remaining

                new_ball_x = self._clamp_x(ball_x + total_push + push_amount)
                self.env.unwrapped.ale.setRAM(self.BALL_X_ADDR, new_ball_x)
                new_dir = 255 if push_dir > 0 else 1
                self.env.unwrapped.ale.setRAM(self.BALL_DIR_ADDR, new_dir)
                total_push += push_amount

                # Re-read ball position so next cursor sees current state
                ball_x = int(self.env.unwrapped.ale.getRAM()[self.BALL_X_ADDR])

            # Draw cursor if visible
            if result['is_visible'] and result['brightness'] is not None:
                _draw_cursor_on_obs(obs, cursor.cursor_x, cursor.cursor_y,
                                    cursor.cursor_size, result['brightness'])

        if info is None:
            info = {}
        info['n_cursors'] = len(self.cursors)
        info['cursor_attacks'] = [c.attack_count for c in self.cursors]
        info['cursor_states'] = [c.state for c in self.cursors]
        info['adv_push'] = total_push

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ball_x, ball_y, paddle_x = self._read_ram()

        for cursor in self.cursors:
            cursor.reset(ball_x, ball_y)

        if info is None:
            info = {}
        info['n_cursors'] = len(self.cursors)

        return obs, info
