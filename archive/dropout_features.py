"""
Custom feature extractors for BreakoutBot experiments.

DropoutNatureCNN — NatureCNN with dropout in the feature space (Experiment 6+).
WideDropoutNatureCNN — 2× wider CNN channels + dropout (Experiment 8, PPO_39).

SB3 automatically handles train/eval mode switching:
  - Rollouts (model.predict): eval mode → dropout OFF → deterministic features
  - PPO updates: train mode → dropout ON → regularized feature learning
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import NatureCNN, BaseFeaturesExtractor


class DropoutNatureCNN(NatureCNN):
    """NatureCNN with dropout after the final convolutional layer.

    Architecture (same as NatureCNN, with dropout added):
      conv1 (32 filters, 8×8, stride 4) → ReLU
      conv2 (64 filters, 4×4, stride 2) → ReLU
      conv3 (64 filters, 3×3, stride 1) → ReLU
      flatten → linear(features_dim) → DROPOUT  ← new
      → features_dim-dimensional output shared by policy & value heads

    Args:
        observation_space: gym space (handled by parent)
        features_dim: output dimension (default 512, same as NatureCNN)
        dropout_p: dropout probability (default 0.1)
    """

    def __init__(self, observation_space, features_dim=512, dropout_p=0.1):
        super().__init__(observation_space, features_dim=features_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, observations):
        x = self.cnn(observations)   # conv stack → flattened
        x = self.linear(x)           # project to features_dim (512)
        x = self.dropout(x)          # dropout in feature space
        return x


class WideDropoutNatureCNN(BaseFeaturesExtractor):
    """2× wider NatureCNN with dropout in feature space.

    Channels doubled vs standard NatureCNN: (32,64,64) → (64,128,128).
    ~3.5M parameters vs ~1.7M for NatureCNN.

    Architecture:
      conv1 (64 filters, 8×8, stride 4) → ReLU
      conv2 (128 filters, 4×4, stride 2) → ReLU
      conv3 (128 filters, 3×3, stride 1) → ReLU
      flatten → linear(features_dim) → ReLU → DROPOUT
      → features_dim-dimensional output

    Args:
        observation_space: gym space
        features_dim: output dimension (default 512)
        dropout_p: dropout probability (default 0.1)
    """

    def __init__(self, observation_space, features_dim=512, dropout_p=0.1):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, observations):
        x = self.cnn(observations)
        x = self.linear(x)
        x = self.dropout(x)
        return x
