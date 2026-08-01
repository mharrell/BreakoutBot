"""
BrickPreclearWrapper — randomize the starting brick layout at each reset.

At env.reset(), clears 15-25 randomly-selected bricks from the wall. Each
env in the parallel VecEnv gets a different random pattern. The cleared
bricks appear as gaps in the initial observation — so the policy sees a
different brick layout each episode and must adapt.

A script that targets specific brick positions fails on episodes where those
bricks are already gone. To score consistently, the policy must read the
brick layout from the observation and adjust its targeting — a form of
visual reactivity that may generalize to ball-tracking.

Unlike per-episode dynamics randomization (PPO_33, which PPO conditioned on),
the brick layout IS the visual observation. The policy can't "condition and
ignore" because the brick layout IS what it must interact with.

Design decisions:
  - 1-life training (no EpisodicLifeEnv): more frequent resets = more layouts.
    eval/check uses standard EpisodicLifeEnv for transfer test.
  - After clearing bricks, takes one NOOP step to refresh the observation
    so the model sees the cleared wall (not the original full wall).
  - Only clears bricks that exist in the initial layout (reads RAM first).

RAM addresses (ALE 0.11.2, Breakout ROM):
  Brick bytes: 0-35 (bit-packed)
  [0-17] = right half, row 0 (top) to row 17 (bottom)
  [18-35] = left half,  row 0 (top) to row 17 (bottom)
  Bit 0 = innermost (near center), bit 7 = outermost (near edge)

Usage:
    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env = NoopResetEnv(env, noop_max=30)
    env = FireResetEnv(env)
    # NO EpisodicLifeEnv — 1-life episodes for more frequent pre-clearing
    env = BrickPreclearWrapper(env)
    env = GrayscaleResize(env, width=84, height=84)
    env = ClipRewardEnv(env)
    env = Monitor(env)
"""
import numpy as np
import gymnasium as gym


class BrickPreclearWrapper(gym.Wrapper):
    """Clear random bricks at reset to prevent layout-specific scripts.

    At each reset(), reads the current brick layout, randomly selects
    15-25 existing bricks, and clears them (sets their bits to 0 in RAM).
    Takes a NOOP step afterward to refresh the observation.

    Parameters
    ----------
    env : gym.Env
        ALE/Breakout-v5 (needs .unwrapped.ale for getRAM/setRAM).
        Must be applied BEFORE GrayscaleResize/ClipRewardEnv so the NOOP
        step's observation goes through the full wrapper chain.
    min_clear : int
        Minimum number of bricks to clear per reset (default 15).
    max_clear : int
        Maximum number of bricks to clear per reset (default 25).
    seed : int or None
        Seed for the internal RNG.
    """

    def __init__(self, env, min_clear=15, max_clear=25, seed=None):
        super().__init__(env)
        self.min_clear = int(min_clear)
        self.max_clear = int(max_clear)
        self._rng = np.random.default_rng(seed)
        self._total_cleared = 0
        self._reset_count = 0

    def _get_ram(self, addr):
        return int(self.env.unwrapped.ale.getRAM()[addr])

    def _set_ram(self, addr, value):
        self.env.unwrapped.ale.setRAM(addr, value)

    def _preclear_random_bricks(self):
        """Read current brick RAM, clear random subset of existing bricks."""
        # Build list of (byte_addr, bit) for all EXISTING bricks
        existing = []
        for byte_addr in range(36):
            current = self._get_ram(byte_addr)
            for bit in range(8):
                if current & (1 << bit):
                    existing.append((byte_addr, bit))

        if len(existing) == 0:
            return 0

        n_clear = self._rng.integers(
            min(self.min_clear, len(existing)),
            min(self.max_clear, len(existing)) + 1
        )

        chosen = self._rng.choice(len(existing), size=n_clear, replace=False)

        for idx in chosen:
            byte_addr, bit = existing[idx]
            current = self._get_ram(byte_addr)
            self._set_ram(byte_addr, current & ~(1 << bit))  # clear bit

        self._total_cleared += n_clear
        self._reset_count += 1
        return n_clear

    # ------------------------------------------------------------------
    # gym.Wrapper interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Clear random bricks and take one NOOP to refresh the observation
        # so the model sees the cleared wall, not the full wall.
        n_cleared = self._preclear_random_bricks()

        # Take a NOOP step to get a fresh observation showing cleared bricks.
        # The ALE processes one frame, ball moves slightly, observation updates.
        obs, _, _, _, _ = self.env.step(0)

        if isinstance(info, dict):
            info["precleared_bricks"] = n_cleared

        return obs, info

    def step(self, action):
        return self.env.step(action)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self):
        return {
            "reset_count": self._reset_count,
            "total_cleared": self._total_cleared,
            "mean_cleared_per_reset": (
                self._total_cleared / max(self._reset_count, 1)
            ),
        }


# -----------------------------------------------------------------------
# Standalone smoke test
# -----------------------------------------------------------------------

if __name__ == "__main__":
    """Verify wrapper runs and bricks are cleared."""
    import ale_py
    gym.register_envs(ale_py)

    print("BrickPreclearWrapper — Smoke Test")
    print(f"  Clear range: 15-25 bricks per reset")

    env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
    env.reset()
    for _ in range(5):
        env.step(0)
    env.step(1)  # FIRE

    env = BrickPreclearWrapper(env, min_clear=15, max_clear=25, seed=42)

    for episode in range(5):
        obs, info = env.reset()
        n_cleared = info.get("precleared_bricks", "?")
        print(f"  Episode {episode}: cleared {n_cleared} bricks")

        # Count remaining bricks post-preclear
        remaining = 0
        for byte_addr in range(36):
            ram_val = env.unwrapped.ale.getRAM()[byte_addr]
            remaining += bin(ram_val).count("1")
        print(f"    Bricks remaining after clear: {remaining}")

        # Run a few random steps
        for step in range(100):
            action = np.random.choice([0, 2, 3])
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    stats = env.get_stats()
    print(f"\n  Stats: {stats}")
    env.close()
