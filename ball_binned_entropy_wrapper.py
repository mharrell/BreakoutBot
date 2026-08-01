"""
BallBinnedEntropyWrapper — reward action diversity conditioned on ball position.

TrajectoryEntropyWrapper's fatal flaw: a policy where half the envs go LEFT
and half go RIGHT looks diverse globally but is still a script — same action
for the same ball position every time. Global action-distribution entropy
rewards superficial diversity that doesn't require ball-tracking.

This wrapper conditions the cross-env action distribution on ball position:
  1. Read ball_x from ALE RAM 99 in each env
  2. Bin ball_x into LEFT / CENTER / RIGHT
  3. Compute action distribution WITHIN each bin
  4. bonus = scale × (1 - p(action | ball_bin))

A script takes the same action regardless of ball position:
  - Ball LEFT → LEFT, ball CENTER → LEFT, ball RIGHT → LEFT
  - p(LEFT | any_bin) = 1.0 → zero bonus in every bin

A reactive policy takes different actions for different ball positions:
  - Ball LEFT → RIGHT (move toward ball), ball RIGHT → LEFT
  - Different action distributions per bin → positive bonus

Unlike global trajectory entropy, this bonus does NOT vanish when the argmax
is a mixed-action script — the policy must diversify WITHIN ball-position bins
to earn the bonus.

Usage:
    env = DummyVecEnv([make_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)
    env = BallBinnedEntropyWrapper(env, entropy_scale=0.10, n_bins=3)
    model = PPO("CnnPolicy", env, ...)
"""
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class BallBinnedEntropyWrapper(VecEnvWrapper):
    """Add cross-env action-diversity bonus, conditioned on ball X position.

    At each step, reads ball_x from each env's ALE RAM, bins positions,
    and computes per-bin action distributions. Each env gets a bonus
    proportional to how rare its action is AMONG ENVS WITH SIMILAR BALL POSITION.

    A script (same action regardless of ball position) earns zero bonus
    because within each bin, every env takes the same action → p=1.0.
    A reactive policy (different actions for different ball positions)
    earns positive bonus because within-bin action diversity is high.

    Parameters
    ----------
    venv : VecEnv
        The vectorized environment (after VecFrameStack).
    entropy_scale : float
        Maximum per-step bonus when an action is unique within its bin.
        At 0.10 with uniform actions across 4 actions:
        p(action|bin) ~ 0.25 → bonus ~ 0.075/step → ~75/game.
    n_bins : int
        Number of ball-X bins (default 3: LEFT/CENTER/RIGHT).
        With 32 envs and 3 bins: ~10.7 envs/bin avg.
    playfield_width : int
        ALE Breakout playfield width in pixels (default 160).
    """

    def __init__(self, venv, entropy_scale=0.10, n_bins=3, playfield_width=160):
        super().__init__(venv)
        self.entropy_scale = float(entropy_scale)
        self.n_bins = int(n_bins)
        self.playfield_width = float(playfield_width)
        self._n_actions = self.action_space.n
        self._last_ball_x = None
        self._pending_bonuses = None
        self._total_bonus = 0.0
        self._step_count = 0

    def _get_ball_x(self):
        """Read ball_x (RAM 99) from each underlying ALE env.

        Navigates: BallBinnedEntropyWrapper -> VecFrameStack -> DummyVecEnv -> envs.
        Each env.unwrapped exposes the base ALE environment's .ale.getRAM().
        """
        inner = self.venv
        while hasattr(inner, 'venv'):
            inner = inner.venv
        if not hasattr(inner, 'envs'):
            # SubprocVecEnv or other — can't access RAM directly
            return np.zeros(self.num_envs, dtype=np.int32)
        ball_xs = []
        for env in inner.envs:
            try:
                ball_xs.append(env.unwrapped.ale.getRAM()[99])
            except Exception:
                ball_xs.append(80)  # fallback: center of playfield
        return np.array(ball_xs, dtype=np.int32)

    def _bin_ball_x(self, ball_xs):
        """Bin ball X positions into 0..n_bins-1."""
        bins = np.floor(ball_xs.astype(np.float64) * self.n_bins / self.playfield_width)
        return np.clip(bins, 0, self.n_bins - 1).astype(np.int32)

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.int64)

        if self._last_ball_x is not None and len(actions) > 1:
            bins = self._bin_ball_x(self._last_ball_x)
            bonuses = np.zeros(len(actions), dtype=np.float64)

            for b in range(self.n_bins):
                in_bin = bins == b
                n_in_bin = in_bin.sum()
                if n_in_bin <= 1:
                    continue  # need ≥2 envs in bin for diversity signal
                bin_actions = actions[in_bin]
                counts = np.bincount(bin_actions, minlength=self._n_actions)
                probs = counts / n_in_bin
                p_of_my_action = probs[bin_actions]
                bonuses[in_bin] = self.entropy_scale * (1.0 - p_of_my_action)

            self._pending_bonuses = bonuses.astype(np.float32)
        else:
            self._pending_bonuses = None

        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        if self._pending_bonuses is not None:
            rewards = rewards + self._pending_bonuses
            self._total_bonus += float(self._pending_bonuses.sum())
            self._pending_bonuses = None

        # Read ball positions from the CURRENT state (post-step).
        # These will be used when the NEXT step_async fires with PPO's
        # actions for this state — the ball position the actions are taken IN.
        self._last_ball_x = self._get_ball_x()
        self._step_count += len(rewards)

        # Surface bonus in info for Monitor ep_info logging
        for i in range(len(rewards)):
            if dones[i]:
                infos[i] = infos[i] or {}
                if isinstance(infos[i], dict):
                    infos[i].setdefault("ball_binned_bonus", 0.0)

        return obs, rewards, dones, infos

    def reset(self):
        self._last_ball_x = None
        self._pending_bonuses = None
        return self.venv.reset()

    def get_stats(self):
        """Return cumulative statistics."""
        return {
            "step_count": self._step_count,
            "total_bonus": self._total_bonus,
            "mean_bonus_per_step": (
                self._total_bonus / max(self._step_count, 1)
            ),
        }


if __name__ == "__main__":
    """Smoke test: verify wrapper runs and ball_x reads work."""
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    import ale_py
    gym.register_envs(ale_py)

    def _make():
        env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
        return env

    venv = DummyVecEnv([_make for _ in range(8)])
    venv = VecFrameStack(venv, n_stack=4)
    venv = BallBinnedEntropyWrapper(venv, entropy_scale=0.10, n_bins=3)

    obs = venv.reset()
    # Fire to launch ball
    for _ in range(20):
        obs, rewards, dones, infos = venv.step(np.array([1] * 8))

    # Now run some steps with random actions and check ball_x is read
    ball_x_samples = []
    for step in range(200):
        actions = np.random.randint(0, 4, size=8)
        obs, rewards, dones, infos = venv.step(actions)
        if venv._last_ball_x is not None:
            ball_x_samples.extend(venv._last_ball_x.tolist())

    print(f"Smoke test: 200 random steps, 8 envs")
    print(f"  Ball X range: [{min(ball_x_samples)}, {max(ball_x_samples)}]")
    print(f"  Ball X mean: {np.mean(ball_x_samples):.1f}")
    print(f"  Unique ball X: {len(set(ball_x_samples))}")
    print(f"  Stats: {venv.get_stats()}")
    venv.close()
