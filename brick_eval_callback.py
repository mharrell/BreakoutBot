"""
BrickEvalCallback — extends SB3's EvalCallback to log bricks_cleared alongside
raw game score. Requires BrickCountingVecWrapper on the eval env to inject
bricks_cleared into the episode info dict.
"""
import numpy as np
from stable_baselines3.common.callbacks import EvalCallback


class BrickEvalCallback(EvalCallback):
    def _log_success_callback(self, locals_, globals_):
        pass

    def _on_step(self) -> bool:
        continue_training = True
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            import time
            start = time.time()

            episode_rewards = []
            episode_bricks = []
            episode_lengths = []

            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = np.array([False] * self.eval_env.num_envs)
                while not done.all():
                    action, states = self.model.predict(
                        obs, state=None, episode_start=None,
                        deterministic=self.deterministic
                    )
                    obs, rewards, dones, infos = self.eval_env.step(action)
                    for i in range(self.eval_env.num_envs):
                        if dones[i] and not done[i]:
                            lives = infos[i].get("lives", -1)
                            if lives == 0:
                                ep_info = infos[i].get("episode", {})
                                episode_rewards.append(ep_info.get("r", 0))
                                episode_bricks.append(ep_info.get("bricks_cleared", 0))
                                episode_lengths.append(ep_info.get("l", 0))
                            done[i] = dones[i] and lives == 0

            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                mean_bricks = np.mean(episode_bricks) if episode_bricks else 0
                mean_length = np.mean(episode_lengths)

                self.last_mean_reward = mean_reward
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(mean_reward)
                self.evaluations_length.append(mean_length)

                if self.log_path is not None:
                    import os
                    kwargs = {}
                    if not hasattr(self, "_brick_header_written"):
                        self._brick_header_written = True
                    np.savez(
                        self.log_path,
                        timesteps=np.array(self.evaluations_timesteps),
                        results=np.array(self.evaluations_results),
                        ep_lengths=np.array(self.evaluations_length),
                        bricks=np.array(getattr(self, "evaluations_bricks",
                                     [mean_bricks] if not hasattr(self, "evaluations_bricks") else
                                     list(self.evaluations_bricks) + [mean_bricks])),
                    )
                if not hasattr(self, "evaluations_bricks"):
                    self.evaluations_bricks = []
                self.evaluations_bricks.append(mean_bricks)

                if self.best_model_save_path is not None:
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        if self.verbose >= 1:
                            print(f"Saving new best model to {self.best_model_save_path}")
                        self.model.save(os.path.join(self.best_model_save_path, "best_model"))

                if self.verbose >= 1:
                    elapsed = time.time() - start
                    print(f"Eval num_timesteps={self.num_timesteps}, "
                          f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                    if mean_bricks > 0:
                        ratio = mean_reward / mean_bricks
                        print(f"  Bricks: {mean_bricks:.0f} avg, "
                              f"{max(episode_bricks)} best  |  "
                              f"Ratio: {ratio:.1f} pts/brick")
                    print(f"Elapsed: {elapsed:.2f}s")

        return continue_training