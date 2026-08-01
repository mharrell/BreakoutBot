"""
MovingBumperWrapper — an indestructible, randomly-repositioning brick shape.

Adds a small bumper (2-5 bricks in various shapes) that moves to a new random
position every 120-300 frames. The bumper bricks are restored every frame
after env.step(), making them indestructible — the ball bounces off but the
bricks never disappear.

Shapes include horizontal/vertical lines, squares, plus signs, and crosses.
Both the shape and its position are randomized, creating unpredictable
playfield geometry that a memorized script cannot account for.

Why this differs from previous approaches:
  - Changes the PLAYFIELD GEOMETRY (not ball position, not actions)
  - Small shapes (1-5 bricks) block paths without making the game unplayable
  - Random shape AND position → combinatorially many obstacle configurations
  - Different shapes create different ball deflection patterns
  - A script expecting clean geometry fails when a bumper changes the path

RAM addresses (ALE 0.11.2, Breakout ROM):
  Brick bytes: 0-35 (bit-packed, each byte = 8 bricks in one row on one side)
  [0-17] = right half, row 0 (top) to row 17 (bottom)
  [18-35] = left half,  row 0 (top) to row 17 (bottom)
  Bit 0 = innermost (near center), bit 7 = outermost (near edge)

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = MovingBumperWrapper(env, row_range=(4, 13))
    # ... standard wrappers ...

Standalone test:
    python moving_bumper_wrapper.py
"""
import numpy as np
import gymnasium as gym


# -----------------------------------------------------------------------
# Shape library — each shape is a set of (row_offset, bit_offset) tuples
# relative to an anchor (row, bit) position.
# -----------------------------------------------------------------------

SHAPES = {
    "H2":  [(0, 0), (0, 1)],                                   # 2×1 horizontal
    "H3":  [(0, 0), (0, 1), (0, 2)],                           # 3×1 horizontal
    "V2":  [(0, 0), (1, 0)],                                   # 1×2 vertical
    "V3":  [(0, 0), (1, 0), (2, 0)],                           # 1×3 vertical
    "V4":  [(0, 0), (1, 0), (2, 0), (3, 0)],                   # 1×4 vertical
    "SQ2": [(0, 0), (0, 1), (1, 0), (1, 1)],                   # 2×2 square
    "SQ3": [(0, 0), (0, 1), (0, 2),
             (1, 0), (1, 1), (1, 2),
             (2, 0), (2, 1), (2, 2)],                           # 3×3 square
    "PLUS":[(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],           # 5-brick plus
    "CROSS":[(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],          # 5-brick X
    "L":   [(0, 0), (1, 0), (2, 0), (2, 1)],                   # 4-brick L
    "L_REV":[(0, 1), (1, 1), (2, 0), (2, 1)],                  # 4-brick reverse L
    "STEP":[(0, 0), (0, 1), (1, 1), (1, 2)],                   # 4-brick stair
    "T":   [(0, 0), (0, 1), (0, 2), (1, 1)],                   # 4-brick T
    "CORNER":[(0, 0), (0, 1), (1, 0)],                          # 3-brick corner
    "DIAG":[(0, 0), (1, 1), (2, 2)],                            # 3-brick diagonal
}


class MovingBumperWrapper(gym.Wrapper):
    """Indestructible brick bumper with randomized shape and position.

    The bumper is a small shape (2-5 bricks) that occupies part of the
    playfield. Every 120-300 frames it teleports to a new random position
    with a new random shape. Bumper bricks are restored every frame after
    env.step(), making them indestructible.

    Parameters
    ----------
    env : gym.Env
        ALE/Breakout-v5 (needs .unwrapped.ale for getRAM/setRAM).
    row_range : (int, int)
        Min/max anchor row for bumper placement (0=top, 17=bottom).
        Default (4, 13) keeps the bumper in the middle ~half of the screen.
    bit_range : (int, int)
        Min/max anchor bit for bumper placement. Default (0, 4) leaves room
        for wider shapes (up to 3 bits wide) without going off-screen.
    reposition_range : (int, int)
        Min/max frames between bumper repositioning. Default (120, 300).
    shapes : list of str or None
        Shape names to use. None = all shapes. Default None.
    seed : int or None
        Seed for the internal RNG.
    """

    RIGHT_SIDE = 0     # bytes [0-17]
    LEFT_SIDE = 18     # bytes [18-35]

    def __init__(
        self,
        env,
        row_range=(4, 13),
        bit_range=(0, 4),
        reposition_range=(120, 300),
        shapes=None,
        seed=None,
    ):
        super().__init__(env)
        self.row_range = tuple(row_range)
        self.bit_range = tuple(bit_range)
        self.reposition_range = tuple(reposition_range)
        self._rng = np.random.default_rng(seed)

        # Validate
        if not (0 <= self.row_range[0] <= self.row_range[1] < 18):
            raise ValueError(f"row_range must be in [0, 17], got {row_range}")
        if not (0 <= self.bit_range[0] <= self.bit_range[1] < 8):
            raise ValueError(f"bit_range must be in [0, 7], got {bit_range}")

        # Shape selection
        self._shape_names = list(shapes) if shapes else list(SHAPES.keys())
        if not self._shape_names:
            raise ValueError("No shapes selected")

        # Current state
        self._anchor_row = None        # anchor row (0-17)
        self._anchor_bit = None        # anchor bit (0-7)
        self._anchor_side = None       # RIGHT_SIDE or LEFT_SIDE
        self._current_shape = None     # list of (row_off, bit_off) tuples
        self._shape_name = None        # string name
        self._reposition_counter = 0
        self._original_bricks = None   # captured on first reset
        self._step_count = 0
        self._move_count = 0

    # ------------------------------------------------------------------
    # RAM helpers
    # ------------------------------------------------------------------

    def _get_ram(self, addr):
        return int(self.env.unwrapped.ale.getRAM()[addr])

    def _set_ram(self, addr, value):
        self.env.unwrapped.ale.setRAM(addr, value)

    # ------------------------------------------------------------------
    # Bumper logic
    # ------------------------------------------------------------------

    def _pick_new_position(self):
        """Choose a new random shape, position, and side for the bumper."""
        # Pick shape
        self._shape_name = self._rng.choice(self._shape_names)
        self._current_shape = SHAPES[self._shape_name]

        # Compute valid anchor ranges given the shape's extent
        shape_max_row_off = max(off[0] for off in self._current_shape)
        shape_max_bit_off = max(off[1] for off in self._current_shape)

        max_row = min(self.row_range[1], 17 - shape_max_row_off)
        max_bit = min(self.bit_range[1], 7 - shape_max_bit_off)
        min_row = max(self.row_range[0], 0)
        min_bit = max(self.bit_range[0], 0)

        if max_row < min_row:
            max_row = min_row  # clamp, shape will clip at bottom
        if max_bit < min_bit:
            max_bit = min_bit  # clamp, shape will clip at edge

        self._anchor_row = int(self._rng.integers(min_row, max_row + 1))
        self._anchor_bit = int(self._rng.integers(min_bit, max_bit + 1))
        self._anchor_side = self.RIGHT_SIDE if self._rng.random() < 0.5 else self.LEFT_SIDE
        self._reposition_counter = int(self._rng.integers(self.reposition_range[0],
                                                           self.reposition_range[1]))
        self._move_count += 1

    def _get_shape_bytes(self):
        """Yield (byte_addr, bit_mask) for each brick in the current shape."""
        if self._anchor_row is None:
            return

        side_base = self._anchor_side
        for row_off, bit_off in self._current_shape:
            row = self._anchor_row + row_off
            bit = self._anchor_bit + bit_off

            # Skip if out of bounds
            if row < 0 or row >= 18 or bit < 0 or bit >= 8:
                continue

            byte_addr = side_base + row
            bit_mask = 1 << bit
            yield byte_addr, bit_mask

    def _write_bumper(self):
        """Write bumper bricks at the current position."""
        if self._anchor_row is None:
            return

        for byte_addr, bit_mask in self._get_shape_bytes():
            current = self._get_ram(byte_addr)
            self._set_ram(byte_addr, current | bit_mask)  # set bits (OR)

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Capture original brick layout on first reset
        if self._original_bricks is None:
            self._original_bricks = [self._get_ram(i) for i in range(36)]

        self._pick_new_position()
        self._write_bumper()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Restore bumper bricks (ball may have destroyed some during step)
        self._write_bumper()

        # Reposition timer
        self._reposition_counter -= 1
        if self._reposition_counter <= 0:
            self._pick_new_position()

        self._step_count += 1
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self):
        return {
            "step_count": self._step_count,
            "move_count": self._move_count,
            "shape": self._shape_name,
            "anchor_row": self._anchor_row,
            "anchor_bit": self._anchor_bit,
            "anchor_side": "right" if self._anchor_side == self.RIGHT_SIDE else "left",
            "reposition_counter": self._reposition_counter,
        }


# -----------------------------------------------------------------------
# Standalone visual test
# -----------------------------------------------------------------------

if __name__ == "__main__":
    """Visual test: watch the bumper in action."""
    import time
    import cv2
    import ale_py
    gym.register_envs(ale_py)

    print("=" * 60)
    print("Moving Bumper Wrapper — Visual Test")
    print("=" * 60)
    print(f"Shapes available: {list(SHAPES.keys())}")
    print("Watch for small indestructible brick shapes appearing in the")
    print("middle of the playfield, relocating every 2-5 seconds.")
    print("Press Ctrl+C to exit.")
    print()

    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0,
                   render_mode="human")

    # Manual FIRE to start
    env.reset()
    for _ in range(5):
        env.step(0)
    env.step(1)

    env = MovingBumperWrapper(env, row_range=(4, 13), bit_range=(0, 4),
                              reposition_range=(120, 300), seed=42)

    print(f"Row range: {env.row_range}")
    print(f"Bit range: {env.bit_range}")
    print(f"Reposition range: {env.reposition_range} frames")

    score = 0
    episode = 1

    try:
        for step in range(5000):
            action = np.random.choice([0, 2, 3], p=[0.3, 0.35, 0.35])
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward

            if env._reposition_counter == env.reposition_range[1] - 1:
                print(f"  Step {step:4d}: {env._shape_name:6s} @ "
                      f"row={env._anchor_row}, bit={env._anchor_bit}, "
                      f"side={['right','left'][env._anchor_side==env.LEFT_SIDE]}")

            if terminated or truncated:
                print(f"  Episode {episode}: score = {int(score)}")
                obs, info = env.reset()
                score = 0
                episode += 1
                for _ in range(5):
                    env.step(0)
                env.step(1)

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopped.")

    print(f"\nFinal stats: {env.get_stats()}")
    env.close()
