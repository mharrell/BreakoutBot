# BreakoutBot 🎮

A reinforcement learning agent trained to play Atari Breakout using PPO (Proximal Policy Optimization) via Stable-Baselines3. Built as a first foray into RL — no pretrained models or copied hyperparameters, everything discovered through systematic experimentation.

## Current Best Performance
<img src="assets/tunnel_exploit.png" alt="Tunnel exploit in action" width="600">

- **Peak Eval Score: 119.80** (pixel-based, PPO_23 at 217.6M timesteps)
- **Best Individual Game: 43**
- **Total Steps Trained: 244M+**
- **Current Run: PPO_23 complete** — pixel-based with linear LR and clip_range decay, 244M steps. New all-time best.

## Approach

This project went through two distinct phases:

### Phase 1: Pixel-Based Training (PPO_5 through PPO_14)
The agent learned directly from stacked game frames using a CNN policy. After 9 runs and 35M+ steps, a previous best eval score of **85.4** was achieved with the following configuration:
- `n_envs=32`, `batch_size=1024`, `lr=2.5e-4`, `ent_coef=0.006`, `net_arch=[64,64]`

### Phase 2: RAM-Based Training with Reward Shaping (PPO_15–PPO_19)
Switched to reading the Atari RAM directly, which gives access to exact game state values. This enabled faster training (1400+ fps) and precise reward shaping, but the approach was ultimately abandoned after underperforming pixel runs.

### Phase 3: Back to Pixels with Linear Decay (PPO_20+)
Returned to pixel-based training with two key improvements:
- **n_envs=64** and **batch_size=2048** for faster experience collection
- **Linear decay** for both `learning_rate` (2.5e-4 → 1e-5) and `clip_range` (0.2 → 0.05) to prevent catastrophic forgetting after peak performance

PPO_22 set a new record of **87.2** at 57.6M steps. PPO_23 extended training to 244M steps with automatic checkpoint resuming and pushed the record to **119.80**, with a consistent 90-110+ eval floor in the final third of training.

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
# Edit RUN_NAME in the script first
python get_eval_logs.py
```

### Probe Atari RAM addresses
```bash
python probe_ram.py
```

### Monitor training in TensorBoard
```bash
tensorboard --logdir ./tensorboard/
```

## Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `breakout_ram_env.py` | RAM observation wrapper with reward shaping |
| `watch.py` | Watch the trained agent play |
| `get_eval_logs.py` | Parse evaluation checkpoint history |
| `probe_ram.py` | Discover Atari RAM addresses for game state |

## RAM Addresses (Breakout)

Discovered by probing RAM during live gameplay:

| Address | Value |
|---------|-------|
| 70 | Paddle x position (0-191) |
| 72 | Ball x position (0-191) |
| 90 | Ball y position |

## Reward Shaping

PPO_15 adds a ball tracking bonus on top of the standard brick-hit reward:

```python
# Small bonus for keeping paddle aligned with ball
# Only when ball is in lower half of screen (heading toward paddle)
if ball_y > 128:
    tracking_reward = 1.0 - abs(int(paddle_x) - int(ball_x)) / 191.0
else:
    tracking_reward = 0.0

shaped_reward = game_reward + 0.1 * tracking_reward
```

## Experiment History

| Run | Obs Type | Key Parameters | Peak Eval | Notes |
|-----|----------|----------------|-----------|-------|
| PPO_5 | Pixel | Baseline defaults, n_envs=8 | ~30 | Entropy collapsed at 2M steps |
| PPO_6 | Pixel | ent_coef=0.01 | ~26 | Too aggressive, unstable |
| PPO_7 | Pixel | ent_coef=0.003 | ~31 | Best small-network run |
| PPO_8 | Pixel | net=[512,512] | ~22 | Large network underperformed |
| PPO_9 | Pixel | lr=1.25e-4 | ~21 | Still entropy collapsed |
| PPO_10 | Pixel | ent_coef=0.006 | ~25 | Better entropy, network limiting |
| PPO_11 | Pixel | Back to [64,64] | ~20 | lr too low for small network |
| PPO_12 | Pixel | n_envs=32 | ~6 | batch_size too small |
| PPO_13 | Pixel | batch=1024, lr=2.5e-4 | 85.4 | Previous best — peaked at 19.2M then collapsed |
| PPO_14 | Pixel | lr=1.25e-4 | ~59 | Lower lr, still oscillating |
| PPO_15 | RAM | RAM obs + ball tracking reward | 56.8 | Good peak, then degraded |
| PPO_16 | RAM | RAM obs + reward shaping | 56.4 | Short run, ended near peak |
| PPO_17 | RAM | RAM obs | 0.0 | Completely broken (unknown cause) |
| PPO_18 | RAM | RAM obs + paddle hit reward | 19.0 | Collapsed badly |
| PPO_19 | RAM | RAM obs | 36.0 | Full run, mediocre. RAM abandoned |
| PPO_20 | Pixel | n_envs=64, batch=2048 | 50.0 | Cut short |
| PPO_21 | Pixel | Linear LR decay 2.5e-4→1e-5, n_envs=32 | ~47 | LR decay confirmed to help |
| PPO_22 | Pixel | Linear LR + clip_range decay, n_envs=64, batch=2048, 60M steps | 87.2 | Previous best at 57.6M steps |
| PPO_23 | Pixel | Same as PPO_22, n_eval_episodes=20, checkpoint resuming, 244M steps total | **119.80** ✅ | New all-time best at 217.6M steps. Consistent 90-110+ floor in final stretch |

## Key Lessons Learned

- Larger networks are not always better — [64,64] outperformed [512,512] for Breakout
- Learning rate must match network size — small networks need higher learning rates
- Scale batch_size proportionally when increasing n_envs
- RAM observations train ~4x faster than pixel observations for MlpPolicy
- MlpPolicy runs better on CPU — pass `device='cpu'` explicitly
- Reward shaping for ball tracking caused the agent to mirror the ball without scoring
- Constant LR causes catastrophic forgetting after peak — fix with linear decay
- Decaying both learning_rate AND clip_range together produces more stable late-training behavior
- More timesteps matter significantly — PPO_23 didn't hit its ceiling until 170M+ steps
- Checkpoint resuming with `reset_num_timesteps=False` lets LR decay track correctly across restarts
- Fixed `progress_remaininga` typo in linear_schedule before PPO_23 — always verify schedule function
- Eval seed consistency matters — agent trained on seed=42 plays best on similar seed distributions. PPO_24 should address seed diversity so agent handles both ball launch directions

## Reference

See [RL_REFERENCE.md](RL_REFERENCE.md) for a full guide to PPO hyperparameters, training metrics, RAM addresses, reward shaping design, and the decision framework used throughout this project.