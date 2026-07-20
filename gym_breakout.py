"""
Gymnasium wrapper around breakout-env (vendored, patched).

Three modes:
  GymBreakout(fixed=True)   — default params, no randomization (eval)
  GymBreakout(fixed=False)  — random params per episode, static within episode
  DynamicBreakout()         — continuous mid-game parameter interpolation
                               (Experiment 5C: anti-memorization escalation)

The engine handles all 5 lives internally as one episode — no per-life dones.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from breakout_env_vendor.breakout_env import Breakout

# Default parameter ranges for randomization
DEFAULT_PARAM_RANGES = {
    "paddle_width": (8, 40),         # default: 15
    "ball_speed_y": (2, 8),          # default: 4
    "ball_speed_x": (1, 4),          # default: 2
    "paddle_speed":  (2, 6),         # default: 3
}

# Fixed defaults (for evaluation)
DEFAULT_PARAMS_FIXED = {
    "paddle_width": 15,
    "ball_speed": [4, 2],
    "paddle_speed": 3,
}


def _sample_params(ranges, rng):
    """Sample a parameter dict from ranges. Returns breakout-env config dict."""
    pw = int(rng.integers(ranges["paddle_width"][0], ranges["paddle_width"][1] + 1))
    bs_y = int(rng.integers(ranges["ball_speed_y"][0], ranges["ball_speed_y"][1] + 1))
    bs_x = int(rng.integers(ranges["ball_speed_x"][0], ranges["ball_speed_x"][1] + 1))
    ps = int(rng.integers(ranges["paddle_speed"][0], ranges["paddle_speed"][1] + 1))
    return {
        "paddle_width": pw,
        "ball_speed": [bs_y, bs_x],
        "paddle_speed": ps,
    }


class GymBreakout(gym.Env):
    """Gymnasium wrapper with optional per-episode parameter randomization.

    fixed=False: random params at reset, held constant for the episode.
    fixed=True:  default params, no randomization (evaluation mode).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, param_ranges=None, fixed=False):
        super().__init__()
        self.param_ranges = param_ranges or DEFAULT_PARAM_RANGES
        self.fixed = fixed

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

        self._env = None
        self._lives = 0
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self.fixed:
            params = dict(DEFAULT_PARAMS_FIXED)
        else:
            params = _sample_params(self.param_ranges, self._rng)

        self._env = Breakout(params)
        obs = self._env.reset()
        self._lives = self._env.conf["lifes"]
        return obs.astype(np.uint8), {}

    def step(self, action):
        if self._env is None:
            raise RuntimeError("Must call reset() before step().")

        obs, reward, terminal, _ = self._env.step(int(action))
        obs = obs.astype(np.uint8)
        lives = self._env.lifes
        life_lost = lives < self._lives
        self._lives = lives

        info = {"lives": lives}

        if life_lost and lives == 0:
            info["episode"] = {"r": float(self._env.score), "l": self._env.step_count}
            return obs, float(reward), True, False, info

        if terminal:
            info["episode"] = {"r": float(self._env.score), "l": self._env.step_count}
            return obs, float(reward), True, False, info

        return obs, float(reward), False, False, info

    def render(self):
        if self._env is None:
            return None
        return self._env.render()

    def close(self):
        self._env = None


# ---------------------------------------------------------------------------
# DynamicBreakout — continuous mid-game parameter interpolation (Experiment 5C)
# ---------------------------------------------------------------------------

# Parameter names and how to apply them to the engine
_PARAM_DEFS = {
    "paddle_width": {
        "range_key": "paddle_width",
        "get": lambda env: env.paddle.size[1],
        "set": lambda env, v: env.paddle.size.__setitem__(1, int(round(v))),
    },
    "paddle_speed": {
        "range_key": "paddle_speed",
        "get": lambda env: env.paddle_v[1],
        "set": lambda env, v: env.paddle_v.__setitem__(1, int(round(v))),
    },
    "ball_speed_y": {
        "range_key": "ball_speed_y",
        "get": lambda env: env.ball_v[0],
        "set": lambda env, v: env.ball_v.__setitem__(0, int(round(v))),
    },
    "ball_speed_x": {
        "range_key": "ball_speed_x",
        "get": lambda env: env.ball_v[1],
        "set": lambda env, v: env.ball_v.__setitem__(1, int(round(v))),
    },
}


class DynamicBreakout(gym.Env):
    """GymBreakout with continuous mid-game parameter interpolation.

    Every 60-300 frames (uniform random), 0-3 of the three parameter groups
    (paddle_width, paddle_speed, ball_speed) are randomly selected. Each
    selected parameter smoothly interpolates from its current value to a
    new random target over 30 frames.

    This is Experiment 5C: the anti-memorization escalation. Per-episode
    randomization (Experiment 5B) was not enough — the CNN memorized a
    parameter-conditioned 64-point script. Continuous mid-game changes
    force the agent to re-assess its physics model moment-to-moment.

    ball_noise_std (Experiment 6): If > 0, adds Gaussian noise N(0, σ)
    to ball velocity every frame. Tiny per-frame perturbations compound
    over hundreds of frames, making the ball path unpredictable. A
    memorized open-loop script that assumes the ball will be at position
    (x,y) at frame N will drift and miss. The only strategy that works
    across all noise realizations is to observe and react.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    INTERP_FRAMES = 30            # frames to smooth-interpolate each change
    CHANGE_INTERVAL = (60, 300)   # frames between change events (uniform)

    def __init__(self, param_ranges=None, ball_noise_std=0.0, dynamic_params=True):
        super().__init__()
        self.param_ranges = param_ranges or DEFAULT_PARAM_RANGES
        self.ball_noise_std = ball_noise_std
        self.dynamic_params = dynamic_params

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(210, 160), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

        self._env = None
        self._lives = 0
        self._rng = np.random.default_rng()

        # Interpolation state
        self._interps = {}         # param_name -> {"start", "target", "remaining"}
        self._frames_until_next = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        params = _sample_params(self.param_ranges, self._rng)
        self._env = Breakout(params)
        obs = self._env.reset()
        self._lives = self._env.conf["lifes"]

        # Start with all interpolations at current values (no pending changes)
        self._interps = {}
        self._schedule_next_change()
        return obs.astype(np.uint8), {}

    def step(self, action):
        if self._env is None:
            raise RuntimeError("Must call reset() before step().")

        # ---- Apply interpolation step (only when dynamic_params enabled) ----
        if self.dynamic_params:
            self._apply_interpolation_step()

        # ---- Per-frame ball velocity noise (Experiment 6) ----
        # Tiny Gaussian perturbation, rounded to nearest integer.
        # At σ=0.05, most frames get ±0 (no change); occasional frames get
        # ±1 px/frame kick. Over hundreds of frames, the ball path becomes
        # unpredictable → open-loop scripts that assume a predictable path
        # will drift and miss. The ONLY strategy is to observe and react.
        if self.ball_noise_std > 0:
            noise_y = int(round(float(self._rng.normal(0.0, self.ball_noise_std))))
            noise_x = int(round(float(self._rng.normal(0.0, self.ball_noise_std))))
            self._env.ball_v[0] += noise_y
            self._env.ball_v[1] += noise_x

        # ---- Step the engine ----
        obs, reward, terminal, _ = self._env.step(int(action))
        obs = obs.astype(np.uint8)
        lives = self._env.lifes
        life_lost = lives < self._lives
        self._lives = lives

        # ---- Check if it's time for new changes (only when dynamic_params enabled) ----
        if self.dynamic_params:
            self._frames_until_next -= 1
            if self._frames_until_next <= 0 and not self._interps:
                self._trigger_changes()
                self._schedule_next_change()

        info = {"lives": lives}

        if life_lost and lives == 0:
            info["episode"] = {"r": float(self._env.score), "l": self._env.step_count}
            return obs, float(reward), True, False, info

        if terminal:
            info["episode"] = {"r": float(self._env.score), "l": self._env.step_count}
            return obs, float(reward), True, False, info

        return obs, float(reward), False, False, info

    # ---- Internal ----

    def _schedule_next_change(self):
        lo, hi = self.CHANGE_INTERVAL
        self._frames_until_next = int(self._rng.integers(lo, hi + 1))

    def _trigger_changes(self):
        """Pick 0-3 params at random and start interpolating each to a new
        random target value."""
        param_names = list(_PARAM_DEFS.keys())
        n = int(self._rng.integers(0, 4))  # 0 to 3
        if n == 0:
            return
        chosen = list(self._rng.choice(param_names, size=n, replace=False))

        for name in chosen:
            pdef = _PARAM_DEFS[name]
            lo, hi = self.param_ranges[pdef["range_key"]]
            current = pdef["get"](self._env)
            target = float(self._rng.integers(lo, hi + 1))
            self._interps[name] = {
                "start": float(current),
                "target": target,
                "remaining": self.INTERP_FRAMES,
            }

    def _apply_interpolation_step(self):
        """Advance all active interpolations by one frame."""
        done = []
        for name, state in self._interps.items():
            state["remaining"] -= 1
            if state["remaining"] <= 0:
                # Final frame — snap to target
                _PARAM_DEFS[name]["set"](self._env, state["target"])
                done.append(name)
            else:
                # Linear interpolation
                t = 1.0 - (state["remaining"] / self.INTERP_FRAMES)
                val = state["start"] + (state["target"] - state["start"]) * t
                _PARAM_DEFS[name]["set"](self._env, val)
        for name in done:
            del self._interps[name]

    def render(self):
        if self._env is None:
            return None
        return self._env.render()

    def close(self):
        self._env = None


def make_dynamic_env():
    """Factory for DynamicBreakout — compatible with make_vec_env (Experiment 5C)."""
    return DynamicBreakout()
