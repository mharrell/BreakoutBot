# BreakoutBot 🎮

A reinforcement learning agent trained to play Atari Breakout using PPO (Proximal Policy Optimization) via Stable-Baselines3.

## Current Best Performance
- **Peak Eval Score: 85.4** (at 19.2M timesteps)
- **Best Individual Game: 43**
- **Total Steps Trained: 25M**

## Results

After 9 experimental runs and 25 million training steps, the agent has learned to actively chase the ball and develop strategies well beyond the random baseline of 0-2 points per game.

### Training Progress (Best Run — PPO_13)

| Timestep | Eval Reward |
|----------|-------------|
| 1.6M | 32.2 |
| 4.8M | 39.0 |
| 6.4M | 47.4 |
| 19.2M | **85.4** 🏆 |

## Setup

```bash
# Clone the repo
git clone https://github.com/mharrell/BreakoutBot
cd BreakoutBot

# Install dependencies
pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom torch

# Accept ROM license
AutoROM --accept-license
```

## Usage

### Train
```bash
python train.py
```

### Watch the agent play
```bash
python watch.py
```

### Check evaluation history
```bash
python get_eval_logs.py
```

### Monitor training in TensorBoard
```bash
tensorboard --logdir ./tensorboard/
```

## Best Hyperparameters (PPO_13)

```python
PPO(
    "CnnPolicy",
    n_envs=32,
    n_steps=128,
    batch_size=1024,
    n_epochs=4,
    gamma=0.99,
    learning_rate=2.5e-4,
    ent_coef=0.006,
    vf_coef=0.5,
    clip_range=0.2,
    net_arch=[64, 64]
)
```

## Experiment History

| Run | Key Change | Peak Eval | Outcome |
|-----|-----------|-----------|---------|
| PPO_5 | Baseline defaults | ~30 | Entropy collapsed at 2M steps |
| PPO_6 | ent_coef=0.01 | ~26 | Too aggressive, unstable |
| PPO_7 | ent_coef=0.003 | ~31 | Best small-net run |
| PPO_8 | net=[512,512] | ~22 | Large network underperformed |
| PPO_9 | lr=1.25e-4 | ~21 | Still entropy collapsed |
| PPO_10 | ent_coef=0.006 | ~25 | Better but network limiting |
| PPO_11 | Back to [64,64] | ~20 | lr too low for small network |
| PPO_12 | n_envs=32 | ~6 | batch_size too small for 32 envs |
| PPO_13 | batch=1024, lr=2.5e-4 | **85.4** | Best run ✅ |

## Key Lessons Learned

- Larger networks are not always better — [64,64] outperformed [512,512] for Breakout
- Learning rate must match network size — small networks need higher learning rates
- When increasing n_envs, scale batch_size proportionally
- Entropy coefficient of 0.006 maintained healthy exploration through 25M steps
- approx_kl spikes are recoverable — the run stabilized after each spike

## Reference

See [RL_REFERENCE.md](RL_REFERENCE.md) for a full guide to PPO hyperparameters, training metrics, and the decision framework used throughout this project.

## What's Next

- Continue training toward consistent 85+ eval scores
- Experiment with reward shaping
- Apply curriculum learning approach to StarCraft II economic agent
