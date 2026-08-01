"""
RandomBounceWrapper — non-conditionable stochasticity on paddle bounces.

On every paddle bounce, randomly nudges the ball's X position by a small
Gaussian offset. This makes the post-bounce ball trajectory unpredictable
from any pixel or history — no fixed action sequence can consistently score.

The perturbation is applied AFTER env.step() returns, using ALE setRAM.
The current observation shows the unperturbed position; the NEXT observation
shows the perturbation's consequences. The model sees the effect but cannot
predict the random offset.

This is fundamentally different from:
- Sticky actions: noise on actions, Breakout forgives it
- Cursor wrapper: cursor position = f(paddle_position) → conditionable
- BeamRider enemies: shot timing = f(ship_position) → conditionable
- Dynamics randomization: fixed per episode → CNN conditions on first frames

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = RandomBounceWrapper(env, perturbation_std=3.0)  # AFTER FireResetEnv
    env = EpisodicLifeEnv(env)
"""
import numpy as np
import gymnasium as gym

# Safe ball X range (playfield edges, in TIA coordinates)
MIN_BALL_X = 10
MAX_BALL_X = 152

# Paddle covers roughly center ±8px; ball must be within this range
# of paddle center for a perturbation to be valid (ball actually on paddle)
PADDLE_HALF_WIDTH = 9


class RandomBounceWrapper(gym.Wrapper):
    """Randomly nudge ball X on paddle bounces to prevent memorization."""

    def __init__(self, env, perturbation_std=3.0, cooldown_frames=15,
                 paddle_zone_y=175, hit_window=20, draw_indicator=True,
                 perturbation_prob=1.0):
        super().__init__(env)
        self.perturbation_std = perturbation_std
        self.cooldown_frames = cooldown_frames
        self.paddle_zone_y = paddle_zone_y
        self.hit_window = hit_window
        self.draw_indicator = draw_indicator
        self.perturbation_prob = perturbation_prob

        self._rng = np.random.default_rng()
        self._prev_ball_y = None
        self._hit_state = "idle"         # idle → pending → cooldown → idle
        self._cooldown_remaining = 0
        self._n_perturbations = 0
        self._n_bounces = 0

    # ------------------------------------------------------------------
    # State reset
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_ball_y = None
        self._hit_state = "idle"
        self._cooldown_remaining = 0
        return obs, info

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Read post-step RAM
        ram = self.env.unwrapped.ale.getRAM()
        ball_x = int(ram[99])
        ball_y = int(ram[101])
        paddle_x = int(ram[72])

        # Detect paddle bounce
        bounced, was_paddle = self._detect_bounce(ball_x, ball_y, paddle_x)

        if bounced and was_paddle:
            self._n_bounces += 1
            # Apply perturbation with probability perturbation_prob
            if self._rng.random() < self.perturbation_prob:
                offset = int(round(self._rng.normal(0, self.perturbation_std)))
                if offset != 0:
                    new_x = int(np.clip(ball_x + offset, MIN_BALL_X, MAX_BALL_X))
                    self.env.unwrapped.ale.setRAM(99, new_x)
                    self._n_perturbations += 1

                    if self.draw_indicator:
                        self._draw_perturbation_indicator(obs, ball_x, new_x)

        # Surface stats
        if info is None:
            info = {}
        info['rb_perturbations'] = self._n_perturbations
        info['rb_bounces'] = self._n_bounces
        info['rb_offset'] = offset if (bounced and was_paddle and
                                        self._rng.random() < self.perturbation_prob and
                                        offset != 0) else 0

        self._prev_ball_y = ball_y
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Bounce detection (state machine)
    # ------------------------------------------------------------------

    def _detect_bounce(self, ball_x, ball_y, paddle_x):
        """State machine for paddle-bounce detection.

        IDLE → ball enters paddle zone while descending → PENDING →
          ball bounces (ascending) → check paddle proximity →
          HIT → COOLDOWN → IDLE

        Returns (bounced, was_paddle_hit).
        """
        bounced = False
        was_paddle = False

        if self._prev_ball_y is None:
            return False, False

        ball_descending = ball_y > self._prev_ball_y
        ball_ascending = ball_y < self._prev_ball_y
        in_paddle_zone = ball_y >= self.paddle_zone_y
        near_paddle_x = abs(ball_x - paddle_x) <= self.hit_window

        # Update cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        if self._hit_state == "idle":
            if in_paddle_zone and ball_descending:
                self._hit_state = "pending"

        elif self._hit_state == "pending":
            if ball_ascending:
                bounced = True
                was_paddle = near_paddle_x
                self._hit_state = "cooldown"
                self._cooldown_remaining = self.cooldown_frames
            elif not in_paddle_zone:
                # Ball left zone without bouncing (edge case: life loss, glitch)
                self._hit_state = "idle"

        elif self._hit_state == "cooldown":
            if self._cooldown_remaining <= 0:
                self._hit_state = "idle"

        return bounced, was_paddle

    # ------------------------------------------------------------------
    # Visual indicator (draws on the RGB observation)
    # ------------------------------------------------------------------

    def _draw_perturbation_indicator(self, obs, old_x, new_x):
        """Draw a small flash at the perturbation point.

        The perturbation happens at the bounce point (ball at paddle level),
        so we draw at the paddle Y position rather than the (now-changed)
        ball position.
        """
        try:
            if obs is None or obs.ndim != 3 or obs.shape[2] != 3:
                return
            h, w = obs.shape[:2]
            # Paddle is near the bottom of the screen
            y = h - 15  # ~paddle Y in rendered coordinates
            # Draw old position in red, new in green
            old_screen_x = min(max(old_x, 0), w - 1)
            new_screen_x = min(max(new_x, 0), w - 1)
            # Small 3x3 flash
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    py = y + dy
                    if 0 <= py < h:
                        if 0 <= old_screen_x + dx < w:
                            obs[py, old_screen_x + dx] = [255, 0, 0]  # red
                        if 0 <= new_screen_x + dx < w:
                            obs[py, new_screen_x + dx] = [0, 255, 0]  # green
        except Exception:
            pass  # Drawing is best-effort; never crash the game


# ------------------------------------------------------------------
# Standalone calibration
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import cv2
    import ale_py
    gym.register_envs(ale_py)
    from stable_baselines3.common.atari_wrappers import (
        FireResetEnv, NoopResetEnv, EpisodicLifeEnv
    )

    print("=" * 60)
    print("RandomBounceWrapper — Calibration")
    print("=" * 60)

    # Quick visual test: run a random agent with perturbations
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0,
                   render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    env = RandomBounceWrapper(env, perturbation_std=3.0, draw_indicator=True)
    env = EpisodicLifeEnv(env)

    obs, info = env.reset()
    total_reward = 0.0
    frames = 0
    n_perturbations = 0
    n_bounces = 0

    cv2.namedWindow("RandomBounceWrapper Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RandomBounceWrapper Test", 420, 320)

    print("Running 1 game with random agent (visual)...")
    print("RED flash = old ball X, GREEN flash = new ball X")
    print("Press ESC to skip, any other key to advance frame by frame")

    paused = True
    while True:
        if not paused:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frames += 1
            n_perturbations = info.get('rb_perturbations', 0)
            n_bounces = info.get('rb_bounces', 0)

            if terminated or truncated:
                obs, info = env.reset()

        display = cv2.resize(obs, (420, 320), interpolation=cv2.INTER_NEAREST)
        # Overlay stats
        cv2.putText(display, f"F:{frames} R:{total_reward:.0f} "
                    f"Bounces:{n_bounces} Perturbs:{n_perturbations}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("RandomBounceWrapper Test", display)

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            paused = not paused
        elif key != 255:
            # Any other key: step one frame
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frames += 1
            n_perturbations = info.get('rb_perturbations', 0)
            n_bounces = info.get('rb_bounces', 0)
            if terminated or truncated:
                obs, info = env.reset()

    cv2.destroyAllWindows()
    env.close()
    print(f"  Frames: {frames}, Score: {total_reward:.0f}")
    print(f"  Bounces detected: {n_bounces}")
    print(f"  Perturbations applied: {n_perturbations}")

    # ------------------------------------------------------------------
    # Dead baseline: center-hold script with perturbations
    # ------------------------------------------------------------------
    print()
    print("Dead baseline: center-hold script with std=3 perturbations")
    print("-" * 40)

    n_games = 10
    scores = []
    all_perturbs = []

    for g in range(n_games):
        env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
        env = NoopResetEnv(env, noop_max=30)
        env = FireResetEnv(env)
        env = RandomBounceWrapper(env, perturbation_std=3.0, draw_indicator=False)
        env = EpisodicLifeEnv(env)

        obs, info = env.reset()
        done = False
        score = 0.0
        while not done:
            ball_x = int(env.unwrapped.ale.getRAM()[99])
            paddle_x = int(env.unwrapped.ale.getRAM()[72])
            # Center-hold: move toward center
            if paddle_x < 80:
                action = 2  # RIGHT
            elif paddle_x > 80:
                action = 3  # LEFT
            else:
                action = 0  # NOOP

            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            if terminated or truncated:
                done = True
        env.close()
        scores.append(score)
        all_perturbs.append(info.get('rb_perturbations', 0))
        print(f"  Game {g+1}: {score:.0f} pts, {info.get('rb_perturbations', 0)} perturbations, "
              f"{info.get('rb_bounces', 0)} bounces")

    print(f"  Mean: {np.mean(scores):.1f}, Std: {np.std(scores):.1f}, "
          f"Unique: {len(set(round(s) for s in scores))}")
    print(f"  Perturbations/game: {np.mean(all_perturbs):.1f}")
    print()
    print("This is the noise floor — a dead script under random bounces.")
    print("A model that learns reactive tracking should substantially exceed this.")
