"""
TrajectoryEntropyWrapper — penalize action-sequence scripts via cross-env entropy.

PPO's built-in ent_coef penalizes per-step distribution flatness ("be uncertain").
This wrapper penalizes cross-episode action identity ("don't do the same thing
in every episode at the same step").

A script: all 32 envs take the SAME action at step t → no bonus.
A reactive policy: different ball positions → different actions at step t → bonus.

The reward is per-env: bonus[i] = scale × (1 - p(action_i))
  - If you did what everyone else did → bonus ≈ 0
  - If you were the only env to press LEFT → bonus ≈ scale × (31/32)

This is computed at the VecEnv level (after VecFrameStack), where we see all
32 actions simultaneously. The modified rewards flow into PPO's rollout buffer
and from there into the loss — no SB3 internals hacking needed.

Usage:
    env = DummyVecEnv([make_env for _ in range(32)])
    env = VecFrameStack(env, n_stack=4)
    env = TrajectoryEntropyWrapper(env, entropy_scale=0.01)
    model = PPO("CnnPolicy", env, ...)
"""
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


class TrajectoryEntropyWrapper(VecEnvWrapper):
    """Add cross-env action-diversity bonus to rewards.

    At each step, computes the action distribution across all envs. Each env
    gets a bonus proportional to how RARE its action was: the more envs that
    took a different action, the higher the bonus for the contrarians.

    This directly attacks the script property: a memorized action sequence
    produces identical actions across all envs → p(action) ≈ 1.0 → zero bonus.
    A reactive policy produces different actions based on different game states
    → p(action) < 1.0 → positive bonus.

    Parameters
    ----------
    venv : VecEnv
        The vectorized environment (after VecFrameStack).
    entropy_scale : float
        Maximum per-step bonus when an action is unique (default 0.01).
        At 0.01 and ~4000 frames/game: ~20 bonus for a diverse population
        vs ~0 for a script. Game reward is ~60.
    """

    def __init__(self, venv, entropy_scale=0.01):
        super().__init__(venv)
        self.entropy_scale = float(entropy_scale)
        self._last_actions = None
        self._total_bonus = 0.0
        self._step_count = 0
        self._n_actions = venv.action_space.n  # 4 for Breakout

    def step_async(self, actions):
        self._last_actions = np.asarray(actions, dtype=np.int64)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        if self._last_actions is not None and len(self._last_actions) > 1:
            actions = self._last_actions

            # Action distribution across envs at this step
            counts = np.bincount(actions, minlength=self._n_actions)
            probs = counts / len(actions)  # p(action) for each action

            # Per-env bonus: reward actions that are rare in the population
            # If all 32 envs took action 0: p(0)=1.0 → bonus = scale×(1-1) = 0
            # If 1 env took action 0: p(0)=1/32 → bonus = scale×(1-1/32) ≈ 0.97×scale
            p_of_my_action = probs[actions]       # shape (n_envs,)
            bonuses = self.entropy_scale * (1.0 - p_of_my_action)

            rewards = rewards + bonuses
            self._total_bonus += float(bonuses.sum())
        else:
            bonuses = np.zeros(len(rewards))

        self._step_count += len(rewards)

        # Log per-step stats to info (surfaces in Monitor's ep_info)
        for i in range(len(rewards)):
            if dones[i]:
                infos[i] = infos[i] or {}
                if isinstance(infos[i], dict):
                    infos[i].setdefault("trajectory_bonus", 0.0)

        return obs, rewards, dones, infos

    def reset(self):
        self._last_actions = None
        return self.venv.reset()

    def get_stats(self):
        """Return cumulative trajectory-entropy statistics."""
        return {
            "step_count": self._step_count,
            "total_bonus": self._total_bonus,
            "mean_bonus_per_step": (
                self._total_bonus / max(self._step_count, 1)
            ),
        }


if __name__ == "__main__":
    """Smoke test: verify wrapper runs without crashing."""
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    import ale_py
    gym.register_envs(ale_py)

    def _make():
        env = gym.make("ALE/Breakout-v5", frameskip=4, repeat_action_probability=0)
        return env

    venv = DummyVecEnv([_make for _ in range(4)])
    venv = VecFrameStack(venv, n_stack=4)
    venv = TrajectoryEntropyWrapper(venv, entropy_scale=0.01)

    obs = venv.reset()
    total_bonus = 0.0
    for step in range(500):
        actions = np.random.randint(0, 4, size=4)
        obs, rewards, dones, infos = venv.step(actions)
        bonus_this_step = rewards.sum()  # game rewards are clipped, so bonus dominates sum
        total_bonus += bonus_this_step

    print(f"Smoke test: 500 random steps, cumulative bonus ~ {total_bonus:.1f}")
    print(f"Stats: {venv.get_stats()}")
    venv.close()
