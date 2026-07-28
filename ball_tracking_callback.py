"""
BallTrackingCallback — auxiliary supervision that forces CNN features to encode
ball position from ALE RAM.

While PPO_85 proved that frozen ball-tracking features collapse (PPO ignores
them), joint training keeps the ball-tracking gradient alive DURING policy
learning. After each PPO rollout, we train a small prediction head to regress
ball (x,y) from the shared CNN features, backpropagating through the CNN.

The CNN receives gradients from TWO sources every update:
  1. PPO policy/value loss (learn to play Breakout)
  2. Ball-position MSE loss (learn to see the ball)

If ball position is literally baked into the features, any policy that uses
those features is reactive by construction.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper


class BallPositionRecorder(VecEnvWrapper):
    """Wraps a VecEnv to record ball positions from info dicts.

    Accumulates (ball_x, ball_y) tuples for each env step so the callback
    can pair them with observations from the rollout buffer.

    Must wrap the VecEnv AFTER VecFrameStack (the top-level training env).
    """

    def __init__(self, venv, max_buffer=200_000):
        super().__init__(venv)
        self.max_buffer = max_buffer
        # Ring buffer: two parallel lists
        self.ball_xs = []
        self.ball_ys = []

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for info in infos:
            self.ball_xs.append(int(info.get('ball_x', 0)))
            self.ball_ys.append(int(info.get('ball_y', 0)))
        # Trim oldest entries
        overflow = len(self.ball_xs) - self.max_buffer
        if overflow > 0:
            self.ball_xs = self.ball_xs[overflow:]
            self.ball_ys = self.ball_ys[overflow:]
        return obs, rewards, dones, infos

    def get_ball_positions(self, n=None):
        """Return (n, 2) numpy array of recent ball positions.

        Args:
            n: Number of most recent entries to return. None = all.
        """
        if n is None:
            n = len(self.ball_xs)
        n = min(n, len(self.ball_xs))
        if n == 0:
            return np.zeros((0, 2), dtype=np.float32)
        recent_x = self.ball_xs[-n:]
        recent_y = self.ball_ys[-n:]
        return np.stack([recent_x, recent_y], axis=1).astype(np.float32)

    def reset(self):
        obs = self.venv.reset()
        # Reset doesn't provide infos, so we can't record ball positions here.
        # The callback handles the one-frame offset (see _train_aux docstring).
        return obs


class BallTrackingCallback(BaseCallback):
    """Trains a ball-position prediction head on top of the shared CNN features.

    After each PPO rollout, samples observations from the rollout buffer and
    corresponding ball positions from the recorder, then trains a small MLP
    to predict ball (x, y) from the CNN features. Gradients flow back into
    the CNN, jointly training it for both PPO and ball-tracking.

    The aux head architecture:
        NatureCNN features (512) -> Linear(64) -> ReLU -> Linear(2) -> (x, y)

    Args:
        recorder: BallPositionRecorder wrapping the training VecEnv.
        aux_weight: Not used directly (the aux optimizer handles scale).
                    Kept for future loss-weighting experiments.
        aux_lr: Learning rate for the aux optimizer (Adam).
        batch_size: Minibatch size for aux training.
        aux_epochs: Number of epochs over the rollout data for aux training.
    """

    def __init__(self, recorder, aux_weight=1.0, aux_lr=1e-4,
                 batch_size=256, aux_epochs=2, verbose=0):
        super().__init__(verbose)
        self.recorder = recorder
        self.aux_weight = aux_weight
        self.aux_lr = aux_lr
        self.batch_size = batch_size
        self.aux_epochs = aux_epochs
        self.aux_head = None
        self.aux_optimizer = None
        self._last_rollout = -1

    def _init_callback(self):
        # Create aux prediction head on the same device as the model
        device = self.model.device
        n_features = 512  # NatureCNN output dim

        self.aux_head = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        ).to(device)

        # Optimizer updates CNN features + aux head
        cnn_params = list(self.model.policy.features_extractor.parameters())
        head_params = list(self.aux_head.parameters())
        self.aux_optimizer = torch.optim.Adam(
            cnn_params + head_params, lr=self.aux_lr,
        )

        # Track rollout boundaries to trigger aux training once per rollout
        self._steps_per_rollout = self.model.n_steps * self.model.n_envs
        self._last_rollout = -1

    def _on_step(self) -> bool:
        # Train aux head once after each rollout collection completes.
        # _on_step fires after every env step (n_envs steps at once), so
        # num_timesteps advances by n_envs each call. We detect rollout
        # boundaries by watching num_timesteps cross multiples of
        # n_steps * n_envs.
        current_rollout = self.num_timesteps // self._steps_per_rollout
        if current_rollout > self._last_rollout:
            self._last_rollout = current_rollout
            if self.aux_head is not None:
                self._train_aux()
        return True

    def _train_aux(self):
        """Train the aux head on the most recent rollout data.

        Alignment note: the rollout buffer stores observations BEFORE each
        env step, while the recorder stores ball positions AFTER each step.
        This creates a one-frame offset (ball position for obs[t] is recorded
        at step t-1). For a regularization signal, this small temporal offset
        is acceptable noise — the ball moves ~2-4 px/frame, and the aux loss
        only needs the features to encode approximate ball location.
        """
        rollout_buffer = self.model.rollout_buffer
        n_envs = self.model.n_envs
        n_steps_buffered = rollout_buffer.size()  # number of rollout steps (128)
        n_transitions = n_steps_buffered * n_envs  # total env transitions (4096)

        # Get ball positions (one per env transition).
        # The recorder stores in step-major order:
        #   env0_s0, env1_s0, ..., env31_s0, env0_s1, env1_s1, ..., env31_s1, ...
        ball_pos = self.recorder.get_ball_positions(n_transitions)
        if len(ball_pos) < self.batch_size:
            return

        # Reshape ball positions to env-major order to match observations
        # From (n_steps*n_envs, 2) step-major → (n_steps, n_envs, 2)
        # → swapaxes → (n_envs, n_steps, 2) → flatten to (n_transitions, 2) env-major
        ball_pos_2d = ball_pos.reshape(n_steps_buffered, n_envs, 2)
        ball_pos = ball_pos_2d.swapaxes(0, 1).reshape(n_transitions, 2)

        # Observations from the rollout buffer
        # rollout_buffer.observations has shape (n_steps, n_envs, C, H, W)
        # = (128, 32, 4, 84, 84). Flatten to env-major order to match targets:
        # env0_s0, env0_s1, ..., env0_s127, env1_s0, ..., env31_s127
        observations_env = rollout_buffer.observations[-n_steps_buffered:]
        observations = observations_env.swapaxes(0, 1).reshape(
            n_transitions, *observations_env.shape[2:],
        )

        # Normalize ball positions to [0, 1]
        ball_x = ball_pos[:, 0] / 160.0
        ball_y = ball_pos[:, 1] / 210.0
        targets = np.stack([ball_x, ball_y], axis=1)

        n = min(len(observations), len(targets))

        device = self.model.device
        total_loss = 0.0
        n_batches = 0

        for _ in range(self.aux_epochs):
            perm = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]

                obs_batch = torch.as_tensor(
                    observations[idx], dtype=torch.float32, device=device,
                )
                target_batch = torch.as_tensor(
                    targets[idx], dtype=torch.float32, device=device,
                )

                # Forward through shared CNN -> aux head
                features = self.model.policy.features_extractor(obs_batch)
                pred = self.aux_head(features)
                loss = F.mse_loss(pred, target_batch)

                self.aux_optimizer.zero_grad()
                loss.backward()
                self.aux_optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        if n_batches > 0 and self.verbose:
            avg_loss = total_loss / n_batches
            print(f"[BallTracking] step={self.num_timesteps:,} "
                  f"aux_mse={avg_loss:.4f} "
                  f"px_err_x={np.sqrt(avg_loss) * 160:.1f}px "
                  f"px_err_y={np.sqrt(avg_loss) * 210:.1f}px")
