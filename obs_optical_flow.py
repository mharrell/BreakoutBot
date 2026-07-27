"""
Optical Flow observation wrapper — replaces 4-frame stacking with explicit velocity.

Instead of feeding 4 raw frames and forcing the CNN to learn temporal
patterns in deeper layers, this wrapper computes the absolute difference
between consecutive frames and stacks it with the current frame:

    output = [current_frame, abs(current_frame - previous_frame)]

This is a 2-channel observation in channels-first format (2×84×84) instead
of the standard 4-channel frame stack. Channel 0 is the current visual state.
Channel 1 is the motion map — every pixel that moved is lit up, directly
encoding velocity. Channels-first is required by NatureCNN.

Why this helps:
  The CNN's deeper layers are where temporal patterns get learned. Those
  patterns are the foundation of memorized scripts — the network discovers
  "this sequence of 4 frames → paddle at X." By providing velocity in the
  first layer (channel 1 IS motion), the CNN doesn't need to build temporal
  feature detectors. No temporal detectors = no basis for timed sequences.

  This doesn't prevent the model from learning a script — it just removes
  the mechanism that makes scripts easy. The CNN gets motion for free, so
  the optimizer has no reason to invest capacity in the infrastructure
  that scripts depend on.

Pipeline: after GrayscaleResize, replaces VecFrameStack entirely.
"""
import numpy as np
import gymnasium as gym


class OpticalFlow(gym.ObservationWrapper):
    """Replace frame stacking with [current, |diff|].

    Output shape: (2, 84, 84) channels-first — NatureCNN expects
    observation_space.shape[0] to be the channel count. Without this,
    shape[0]=84 and the first Conv2d gets 84 input channels instead of 2.
    No VecFrameStack needed.
    """
    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape  # e.g. (84, 84, 1)
        # 2 channels (channels-first for NatureCNN): current frame + diff
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(2, h, w), dtype=np.uint8)
        self._prev_frame = None

    def reset(self, **kwargs):
        self._prev_frame = None
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs):
        if obs.ndim == 3 and obs.shape[2] == 1:
            obs = obs[:, :, 0]       # squeeze channel dim for arithmetic

        if self._prev_frame is None:
            diff = np.zeros_like(obs)
        else:
            diff = np.abs(obs.astype(np.int16) - self._prev_frame.astype(np.int16))
            diff = diff.astype(np.uint8)

        self._prev_frame = obs.copy()
        return np.stack([obs, diff], axis=0)   # (2, 84, 84) channels-first
