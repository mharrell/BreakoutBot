"""
AdversarialBallWrapper — make Breakout ball dodge a lazy paddle.

The BeamRider findings show that adversarial environments (enemies that
aim at you) break PPO's script-memorization attractor. This wrapper
injects adversarial dynamics into Breakout: the ball actively steers
away from the paddle when the paddle isn't tracking it.

Mechanism (proportional push with dead zone):
  - Read ball (x,y) and paddle (x) from ALE RAM each step
  - Track ball_y to determine if ball is heading downward (toward paddle)
  - When ball is heading down AND below paddle_zone_y:
      error = ball_x - paddle_x
      if |error| <= dead_zone: push = 0  (tracking is good enough)
      else:
          excess = |error| - dead_zone
          push = sign(error) * min(excess * proportional_gain, max_push)
      ball_x += push (via setRAM, clamped to screen bounds)
  - Paddle near ball: no push. Paddle far away: proportional shove.

  This creates a learnable gradient: moving toward the ball reduces push,
  making it easier to hit the ball and score.

Audit finding (July 28): The original constant-push design (±2.5 px
regardless of tracking error) created rapid error amplification at
frameskip=1. A 1px error → 2.5px push → 3.5px error → 2.5px push →
6px error ... within 10 frames the ball is 25px away regardless of
paddle movement. Proportional push with dead zone fixes this.

Placement in wrapper chain:
  After FireResetEnv, before GrayscaleResize/ClipRewardEnv.
  Must have access to ale.getRAM() / ale.setRAM().
"""
import numpy as np
import gymnasium as gym


class AdversarialBallWrapper(gym.Wrapper):
    """Push the ball away from the paddle when the paddle isn't tracking.

    Args:
        env: Gym environment with .unwrapped.ale (ALE interface)
        dead_zone: Pixels of tracking tolerance. |error| <= dead_zone → no push.
                   Default 4. Set to 0 for old constant-push behavior.
        proportional_gain: Push growth per px beyond dead_zone.
                           push = gain * (|error| - dead_zone), capped at max_push.
                           Default 0.5. At 20px error: push = 8px.
        paddle_zone_y: Ball Y threshold. Only apply push when ball is
                       below this line (heading toward paddle area).
                       Breakout screen is 210×160; paddle at ~y=190.
                       Default 140 means "last 50 pixels above paddle."
        max_push: Maximum push per step (default 15.0).
    """

    BALL_X_ADDR = 99
    BALL_Y_ADDR = 101
    PADDLE_X_ADDR = 72

    # Breakout ball screen bounds (approximate, from ALE game area)
    MIN_X = 8
    MAX_X = 152

    def __init__(self, env, dead_zone=4, proportional_gain=0.5,
                 paddle_zone_y=140, max_push=15.0):
        super().__init__(env)
        self.dead_zone = float(dead_zone)
        self.proportional_gain = float(proportional_gain)
        self.paddle_zone_y = int(paddle_zone_y)
        self.max_push = float(max_push)
        self._prev_ball_y = None
        self._total_push_this_episode = 0.0
        self._push_count_this_episode = 0

    def _read_ram(self):
        """Read ball and paddle positions from ALE RAM."""
        ram = self.env.unwrapped.ale.getRAM()
        ball_x = int(ram[self.BALL_X_ADDR])
        ball_y = int(ram[self.BALL_Y_ADDR])
        paddle_x = int(ram[self.PADDLE_X_ADDR])
        return ball_x, ball_y, paddle_x

    def _write_ball_x(self, x):
        """Write ball X position to ALE RAM, clamped to screen bounds."""
        x = max(self.MIN_X, min(self.MAX_X, int(x)))
        self.env.unwrapped.ale.setRAM(self.BALL_X_ADDR, x)

    def step(self, action):
        # Let the environment process this step first
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Read post-step ball and paddle state
        ball_x, ball_y, paddle_x = self._read_ram()

        # Determine if ball is heading downward (toward paddle)
        heading_down = False
        if self._prev_ball_y is not None:
            heading_down = ball_y > self._prev_ball_y

        self._prev_ball_y = ball_y

        # Apply adversarial push if conditions are met
        push_applied = 0.0
        if heading_down and ball_y > self.paddle_zone_y:
            error = ball_x - paddle_x
            # Proportional push with dead zone.
            # Paddle within dead_zone of ball → no push (tracking is good).
            # Paddle outside dead_zone → push proportional to excess error.
            abs_error = abs(error)
            if abs_error > self.dead_zone:
                excess = abs_error - self.dead_zone
                push_magnitude = min(excess * self.proportional_gain, self.max_push)
                # Sign: paddle left of ball (error > 0) → push ball right (+)
                #        paddle right of ball (error < 0) → push ball left (-)
                push_applied = np.sign(error) * push_magnitude

                new_ball_x = ball_x + push_applied
                self._write_ball_x(new_ball_x)

                self._total_push_this_episode += abs(push_applied)
                self._push_count_this_episode += 1

        # Surface push stats in info dict for monitoring
        if info is None:
            info = {}
        info['adv_push'] = push_applied
        info['adv_tracking_error'] = ball_x - paddle_x if heading_down and ball_y > self.paddle_zone_y else 0

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_ball_y = None
        self._total_push_this_episode = 0.0
        self._push_count_this_episode = 0

        # Read initial state for info
        ball_x, ball_y, paddle_x = self._read_ram()
        if info is None:
            info = {}
        info['adv_ball_x'] = ball_x
        info['adv_ball_y'] = ball_y
        info['adv_paddle_x'] = paddle_x

        return obs, info
