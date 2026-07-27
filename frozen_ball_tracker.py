"""
FrozenBallTracker — use the pretrained ball-tracking CNN as a frozen feature
extractor for PPO.

The perception POC proved NatureCNN can locate the ball to 1.9px MAE (0.6px
median). This class loads those pretrained conv + linear weights and freezes
them, so PPO's policy/value heads receive features that already encode ball
position with near-perfect precision. The policy only has to learn "what action
to take given these features" — the hard perceptual work is done.

Architecture matches NatureCNN exactly:
  Conv: 4ch->32(8x8/4)->64(4x4/2)->64(3x3/1)  [FROZEN]
  Flatten
  Linear(conv_out, 512)                          [FROZEN]
  ReLU
  -> 512-dim features -> SB3 policy/value heads   [TRAINED]

Usage:
    policy_kwargs = dict(
        features_extractor_class=FrozenBallTracker,
        features_extractor_kwargs=dict(
            pretrained_path="perception_poc_4frame_model.pt",
        ),
    )
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, ...)
"""
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class FrozenBallTracker(BaseFeaturesExtractor):
    """NatureCNN-compatible feature extractor with frozen ball-tracking weights.

    Loads the conv stack and Linear(conv_out, 512) from the perception POC
    model, freezes them, and exposes 512-dim features to SB3's policy/value nets.

    Parameters
    ----------
    observation_space : gym.Space
        Must be Box(4, 84, 84) — 4-frame stacked grayscale.
    features_dim : int
        Output feature dimension (default 512, matches NatureCNN).
    pretrained_path : str
        Path to the perception_poc_4frame_model.pt checkpoint.
    freeze_conv : bool
        Freeze conv layers (default True).
    freeze_linear : bool
        Freeze the linear projection layer (default True).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        pretrained_path: str = "perception_poc_4frame_model.pt",
        freeze_conv: bool = True,
        freeze_linear: bool = True,
    ):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        # Same conv stack as NatureCNN + BallTrackerCNN4Frame
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, n_input_channels, 84, 84)
            conv_out = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(conv_out, features_dim),
            nn.ReLU(),
        )

        # Load pretrained weights
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # Freeze
        if freeze_conv:
            for p in self.cnn.parameters():
                p.requires_grad = False
        if freeze_linear:
            for p in self.linear.parameters():
                p.requires_grad = False

        # Report
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"FrozenBallTracker: {total:,} total params, "
              f"{trainable:,} trainable ({100*trainable/total:.0f}%)")

    def _load_pretrained(self, path):
        """Load conv and linear weights from the perception POC checkpoint."""
        state_dict = torch.load(path, map_location="cpu")

        # Map BallTrackerCNN4Frame keys to our keys
        # BallTrackerCNN4Frame: cnn.X.weight, head.0.weight, head.2.weight
        # FrozenBallTracker:     cnn.X.weight, linear.0.weight
        mapping = {}
        for key in state_dict:
            if key.startswith("cnn."):
                mapping[key] = key  # same name
            elif key.startswith("head.0."):
                # head.0 = Linear(conv_out, 512) -> linear.0
                new_key = key.replace("head.0.", "linear.0.")
                mapping[key] = new_key
            # Skip head.2 (the regression output layer — we don't use it)

        loaded = 0
        skipped = 0
        own_state = self.state_dict()
        for src_key, dst_key in mapping.items():
            if dst_key in own_state:
                if own_state[dst_key].shape == state_dict[src_key].shape:
                    own_state[dst_key].copy_(state_dict[src_key])
                    loaded += 1
                else:
                    print(f"  Shape mismatch: {dst_key} "
                          f"{own_state[dst_key].shape} vs {state_dict[src_key].shape}")
                    skipped += 1
            else:
                skipped += 1

        print(f"FrozenBallTracker: loaded {loaded} tensors from {path} "
              f"(skipped {skipped})")

    def forward(self, observations):
        return self.linear(self.cnn(observations))
