"""
ExtremeBumperWrapper — two independently-moving indestructible brick shapes.

An escalated version of MovingBumperWrapper (PPO_120). Two bumpers instead of
one, faster repositioning (60-150 frames vs 120-300), only 3+ brick shapes
for larger obstacles, and proper cleanup of old bumper positions.

Unlike the original wrapper, this CLEARS old bumper bricks when repositioning
so the playfield doesn't fill with indestructible residue over the episode.

RAM addresses (ALE 0.11.2, Breakout ROM):
  Brick bytes: 0-35 (bit-packed, each byte = 8 bricks in one row on one side)
  [0-17] = right half, row 0 (top) to row 17 (bottom)
  [18-35] = left half,  row 0 (top) to row 17 (bottom)
  Bit 0 = innermost (near center), bit 7 = outermost (near edge)

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = ExtremeBumperWrapper(env, num_bumpers=2)
    # ... standard wrappers ...
"""
import numpy as np
import gymnasium as gym


# Shape library — (row_offset, bit_offset) tuples relative to anchor
# Only 3+ brick shapes. H2 and V2 removed (too small).
SHAPES_3PLUS = {
    "H3":    [(0, 0), (0, 1), (0, 2)],                            # 3-brick horizontal
    "V3":    [(0, 0), (1, 0), (2, 0)],                            # 3-brick vertical
    "V4":    [(0, 0), (1, 0), (2, 0), (3, 0)],                    # 4-brick vertical
    "SQ2":   [(0, 0), (0, 1), (1, 0), (1, 1)],                    # 2x2 square
    "SQ3":   [(0, 0), (0, 1), (0, 2),
              (1, 0), (1, 1), (1, 2),
              (2, 0), (2, 1), (2, 2)],                             # 3x3 square
    "PLUS":  [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],            # 5-brick plus
    "CROSS": [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)],            # 5-brick X
    "L":     [(0, 0), (1, 0), (2, 0), (2, 1)],                    # 4-brick L
    "L_REV": [(0, 1), (1, 1), (2, 0), (2, 1)],                    # 4-brick rev L
    "STEP":  [(0, 0), (0, 1), (1, 1), (1, 2)],                    # 4-brick stair
    "T":     [(0, 0), (0, 1), (0, 2), (1, 1)],                    # 4-brick T
    "CORNER":[(0, 0), (0, 1), (1, 0)],                             # 3-brick corner
    "DIAG":  [(0, 0), (1, 1), (2, 2)],                             # 3-brick diagonal
}


def _new_bumper_state():
    """Return a fresh bumper state dictionary."""
    return {
        "shape_name": None,
        "shape": None,
        "anchor_row": None,
        "anchor_bit": None,
        "anchor_side": None,
        "counter": 0,
    }


class ExtremeBumperWrapper(gym.Wrapper):
    """Two independently-moving indestructible brick bumpers.

    Each bumper is a small shape (3-9 bricks) that occupies part of the
    playfield. Every 60-150 frames it teleports to a new random position
    with a new random shape. Bumper bricks are restored every frame after
    env.step(), making them indestructible. Old bumper positions are
    cleared when the bumper repositions.

    Parameters
    ----------
    env : gym.Env
        ALE/Breakout-v5 (needs .unwrapped.ale for getRAM/setRAM).
    num_bumpers : int
        Number of independently-moving bumpers (default 2).
    row_range : (int, int)
        Min/max anchor row for bumper placement (0=top, 17=bottom).
    bit_range : (int, int)
        Min/max anchor bit for bumper placement.
    reposition_range : (int, int)
        Min/max frames between bumper repositioning.
    seed : int or None
        Seed for the internal RNG.
    """

    RIGHT_SIDE = 0
    LEFT_SIDE = 18

    def __init__(
        self,
        env,
        num_bumpers=2,
        row_range=(4, 13),
        bit_range=(0, 4),
        reposition_range=(60, 150),
        seed=None,
    ):
        super().__init__(env)
        self.num_bumpers = int(num_bumpers)
        self.row_range = tuple(row_range)
        self.bit_range = tuple(bit_range)
        self.reposition_range = tuple(reposition_range)
        self._rng = np.random.default_rng(seed)
        self._shape_names = list(SHAPES_3PLUS.keys())
        self._bumpers = [_new_bumper_state() for _ in range(self.num_bumpers)]
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
    # Per-bumper helpers
    # ------------------------------------------------------------------

    def _get_shape_bytes(self, bumper):
        """Yield (byte_addr, bit_mask) for each brick in the bumper's shape."""
        if bumper["anchor_row"] is None:
            return
        side_base = bumper["anchor_side"]
        for row_off, bit_off in bumper["shape"]:
            row = bumper["anchor_row"] + row_off
            bit = bumper["anchor_bit"] + bit_off
            if row < 0 or row >= 18 or bit < 0 or bit >= 8:
                continue
            yield side_base + row, 1 << bit

    def _write_bumper(self, bumper):
        """Set bumper bricks at current position (OR)."""
        for byte_addr, bit_mask in self._get_shape_bytes(bumper):
            self._set_ram(byte_addr, self._get_ram(byte_addr) | bit_mask)

    def _clear_bumper(self, bumper):
        """Remove bumper bricks from old position (AND NOT)."""
        if bumper["anchor_row"] is None:
            return
        for byte_addr, bit_mask in self._get_shape_bytes(bumper):
            self._set_ram(byte_addr, self._get_ram(byte_addr) & ~bit_mask)

    def _pick_new_position(self, bumper):
        """Choose a new random shape, position, and side. Clears old position."""
        self._clear_bumper(bumper)

        bumper["shape_name"] = self._rng.choice(self._shape_names)
        bumper["shape"] = SHAPES_3PLUS[bumper["shape_name"]]

        shape_max_row_off = max(off[0] for off in bumper["shape"])
        shape_max_bit_off = max(off[1] for off in bumper["shape"])

        max_row = min(self.row_range[1], 17 - shape_max_row_off)
        max_bit = min(self.bit_range[1], 7 - shape_max_bit_off)
        min_row = max(self.row_range[0], 0)
        min_bit = max(self.bit_range[0], 0)

        if max_row < min_row:
            max_row = min_row
        if max_bit < min_bit:
            max_bit = min_bit

        bumper["anchor_row"] = int(self._rng.integers(min_row, max_row + 1))
        bumper["anchor_bit"] = int(self._rng.integers(min_bit, max_bit + 1))
        bumper["anchor_side"] = self.RIGHT_SIDE if self._rng.random() < 0.5 else self.LEFT_SIDE
        bumper["counter"] = int(self._rng.integers(*self.reposition_range))
        self._move_count += 1

        self._write_bumper(bumper)

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        for bumper in self._bumpers:
            self._pick_new_position(bumper)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Restore all bumper bricks (ball may have destroyed some during step)
        for bumper in self._bumpers:
            self._write_bumper(bumper)

        # Reposition each bumper whose timer expired
        for bumper in self._bumpers:
            bumper["counter"] -= 1
            if bumper["counter"] <= 0:
                self._pick_new_position(bumper)

        self._step_count += 1
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self):
        shapes = [b["shape_name"] for b in self._bumpers]
        return {
            "step_count": self._step_count,
            "move_count": self._move_count,
            "shapes": shapes,
        }


# -----------------------------------------------------------------------
# Standalone smoke test
# -----------------------------------------------------------------------

if __name__ == "__main__":
    """Verify wrapper runs without crashing and bumpers move."""
    import time
    import ale_py
    gym.register_envs(ale_py)

    print("ExtremeBumperWrapper — Smoke Test")
    print(f"  Bumpers: 2")
    print(f"  Shapes: {list(SHAPES_3PLUS.keys())}")
    print(f"  Reposition: 60-150 frames")

    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env.reset()
    for _ in range(5):
        env.step(0)
    env.step(1)  # FIRE

    env = ExtremeBumperWrapper(env, num_bumpers=2, seed=42)

    moves_logged = 0
    score = 0
    for step in range(2000):
        action = np.random.choice([0, 2, 3])
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

        # Log when a bumper moves
        for i, b in enumerate(env._bumpers):
            if b["counter"] == env.reposition_range[1] - 1:
                print(f"  Step {step}: bumper[{i}] -> {b['shape_name']} "
                      f"@ row={b['anchor_row']}, bit={b['anchor_bit']}, "
                      f"side={'R' if b['anchor_side']==0 else 'L'}")
                moves_logged += 1

        if terminated or truncated:
            obs, info = env.reset()
            score = 0
            for _ in range(5):
                env.step(0)
            env.step(1)

    print(f"\n  Total moves: {moves_logged}")
    print(f"  Stats: {env.get_stats()}")
    env.close()
