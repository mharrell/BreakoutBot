# PPO Training Reference Guide
### Breakout RL Agent — Levers, Outputs & Experiment History

**Current Best: 85.4 eval (pixel-based) | Best Individual Game: 43 | Total Steps Trained: 35M+**

---

## Part 1: Training Levers

These are the parameters you control in `train.py`. Each one affects a different aspect of how the agent learns. Key principle: **change one thing at a time** so you can interpret the results.

| Parameter | What It Does | Effect on Training |
|-----------|-------------|-------------------|
| `ent_coef` | Entropy coefficient — adds bonus reward for taking varied actions | Higher = more exploration, prevents early convergence. Too high = unstable training |
| `learning_rate` | How big each update step is | Lower = more stable but slower. Larger networks need lower rates. Too low = entropy collapse |
| `clip_range` | Limits how much the policy can change in one update | Lower = more conservative updates. Watch approx_kl for signs it's too high |
| `n_steps` | How many steps to collect before each update | Higher = more data per update, smoother gradients but slower iteration |
| `batch_size` | How much data to use per gradient update | Scale up proportionally when increasing n_envs |
| `n_epochs` | How many times to reuse collected data per update | Higher = more efficient data use but risk of overfitting |
| `gamma` | Discount factor — how much to value future vs immediate rewards | Higher = agent thinks further ahead. Lower = more focused on immediate rewards |
| `net_arch` | Size of fully connected layers (pixel runs only) | Larger = more capacity but needs lower learning rate and more careful tuning |
| `vf_coef` | How much to weight value function loss vs policy loss | Higher = more focus on accurate reward prediction |
| `n_envs` | Number of parallel game environments | More = faster experience collection. Scale batch_size proportionally |
| `total_timesteps` | Total training duration | More = more time to learn. Can extend by restarting with checkpoint |
| `device` | CPU vs GPU | MlpPolicy (RAM runs) runs faster on CPU. CnnPolicy (pixel runs) benefits from GPU |

---

## Part 2: Observation Modes

This project has used two fundamentally different observation approaches:

### Pixel-Based (Runs 1-14)
- Agent sees stacked frames of the game screen as images
- Uses `CnnPolicy` — convolutional neural network processes visual input
- Uses `make_atari_env` with `VecFrameStack(n_stack=4)`
- Slower training (~300-600 fps with 32 envs)
- More generalizable but harder to add custom reward shaping

### RAM-Based (Run 15+)
- Agent sees the 128-byte Atari RAM directly as structured data
- Uses `MlpPolicy` — simple fully connected network
- Uses custom `BreakoutRamEnv` wrapper
- Much faster training (~1400+ fps with 32 envs)
- Enables reward shaping using exact game state values
- Run on CPU (MlpPolicy doesn't benefit from GPU)

### Key RAM Addresses (Breakout)
Discovered by probing RAM during live gameplay:

| Address | Value |
|---------|-------|
| 70 | Paddle x position (0-191) |
| 72 | Ball x position (0-191) |
| 90 | Ball y position (increments each step) |

---

## Part 3: Reward Shaping

### Ball Tracking Reward (PPO_15+)
Added a small bonus reward to encourage the paddle to track the ball:

```python
# Only reward tracking when ball is in lower half of screen
if ball_y > 128:
    tracking_reward = 1.0 - abs(int(paddle_x) - int(ball_x)) / 191.0
else:
    tracking_reward = 0.0

shaped_reward = game_reward + 0.1 * tracking_reward
```

**Design decisions:**
- `ball_y > 128` threshold — only reward tracking when ball is heading toward paddle, not when hitting bricks at top
- `0.1` scaling — keeps tracking as a nudge, not the primary objective
- `int()` casting — prevents uint8 overflow when subtracting RAM values

---

## Part 4: Training Outputs

| Output | What It Means | Healthy Range / What to Watch |
|--------|--------------|-------------------------------|
| `ep_rew_mean` | Average score per episode during training (noisy) | Should climb over time. Dips normal during exploration phases |
| `ep_len_mean` | Average game length in frames | Longer = agent keeping ball alive longer. Sudden drops signal a problem |
| `eval/mean_reward` | Average score during deterministic evaluation — most reliable metric | Your ground truth. Should trend upward. Plateau or decline = time to intervene |
| `fps` | Frames processed per second across all environments | RAM runs: 1000-2500+. Pixel runs: 300-600. Drops suggest throttling |
| `entropy_loss` | How much the agent is exploring vs exploiting | Early: -1.0 to -1.5. Mid: -0.3 to -0.5. Late: -0.1 to -0.3. Near 0 = collapsed |
| `explained_variance` | How accurately the agent predicts future rewards | Closer to 1.0 is better. Below 0.5 = value function is struggling |
| `approx_kl` | How much the policy changed this update | Should stay below 0.05. Spikes above 0.1 = watch closely. Above 0.15 = concern |
| `clip_fraction` | How often PPO's safety clamp triggered | 0.05-0.15 is healthy. Consistently high = learning rate may be too aggressive |
| `loss` | Combined training loss | Should generally decrease over time. Spikes can indicate instability |
| `value_loss` | How wrong the reward predictions were | Should decrease over time. Climbing = agent seeing unfamiliar situations |
| `policy_gradient_loss` | How much the policy is being pushed to change | Should be small and stable. Large values indicate aggressive updates |
| `n_updates` | Total number of gradient updates performed | Useful for tracking how much training has happened. Confirms checkpoint loaded correctly |

### Healthy Entropy Loss by Training Stage

| Stage | Steps | Healthy Range | What It Means |
|-------|-------|--------------|---------------|
| Early | 0 - 500k | -1.0 to -1.5 | Agent exploring freely, not committed to any strategy |
| Mid | 500k - 2M | -0.3 to -0.5 | Narrowing down good strategies while still exploring |
| Late | 2M+ | -0.1 to -0.3 | Some confidence is fine as long as reward is still climbing |
| 🚨 Red flag | Any | Near 0.0 before 1M | Premature convergence — agent got confident too early |

---

## Part 5: Experiment History

| Run | TensorBoard | Obs Type | Key Parameters | Peak Eval | Notes |
|-----|-------------|----------|----------------|-----------|-------|
| 1 | PPO_5 | Pixel | Default: ent_coef=0.0, lr=2.5e-4, net=[64,64], n_envs=8 | ~30 | Entropy collapsed ~2M steps |
| 2 | PPO_6 | Pixel | ent_coef=0.01, clip_range=0.1 | ~26 | Too much entropy, unstable |
| 3 | PPO_7 | Pixel | ent_coef=0.003, clip_range=0.2 | ~31 | Best small-network pixel run |
| 4 | PPO_8 | Pixel | Added net=[512,512] | ~22 | Large network underperformed |
| 5 | PPO_9 | Pixel | net=[512,512], lr=1.25e-4 | ~21 | Entropy still collapsed early |
| 6 | PPO_10 | Pixel | net=[512,512], lr=1.25e-4, ent_coef=0.006 | ~25 | Better entropy, network limiting |
| 7 | PPO_11 | Pixel | net=[64,64], ent_coef=0.006, lr=1.25e-4 | ~20 | lr too low for small network |
| 8 | PPO_12 | Pixel | n_envs=32, batch=1024, lr=1.25e-4 | ~6 | 32 envs + low lr = collapse |
| 9 | PPO_13 | Pixel | n_envs=32, batch=1024, lr=2.5e-4, ent_coef=0.006 | **85.4** | Best pixel run. 25M steps |
| 10 | PPO_14 | Pixel | Same as PPO_13, lr=1.25e-4 | ~59 | Lower lr, still oscillating |
| 11 | PPO_15 | RAM | RAM obs, reward shaping, MlpPolicy, CPU | In progress | First RAM run |

### PPO_13 Eval Score History (Best Pixel Run)

| Timestep | Eval Reward |
|----------|-------------|
| 1,600,000 | 32.2 |
| 3,200,000 | 33.4 |
| 4,800,000 | 39.0 |
| 6,400,000 | **47.4** |
| 8,000,000 | 32.0 |
| 9,600,000 | 35.8 |
| 11,200,000 | 37.6 |
| 12,800,000 | 37.0 |
| 14,400,000 | 33.8 |
| 16,000,000 | 27.6 |
| 17,600,000 | 42.6 |
| 19,200,000 | **85.4** 🏆 |
| 20,800,000 | 38.6 |
| 22,400,000 | 41.2 |
| 24,000,000 | 31.2 |

### Key Lessons Learned

1. **Larger networks are not always better** — net=[512,512] consistently underperformed net=[64,64].
2. **Learning rate must match network size** — small networks need higher learning rates.
3. **Scale batch_size with n_envs** — n_envs=32 with batch_size=256 caused extreme entropy collapse.
4. **Best pixel config**: ent_coef=0.006, n_envs=32, batch_size=1024, lr=2.5e-4, net=[64,64].
5. **RAM observations are much faster** — 1400+ fps vs 300-600 fps for pixel runs.
6. **RAM enables precise reward shaping** — exact ball and paddle positions available directly.
7. **MlpPolicy runs better on CPU** — no benefit from GPU for non-CNN policies.
8. **Reward shaping requires care** — tracking bonus scaled to 0.1 to avoid overwhelming brick reward.

---

## Decision Framework

| What You See | What It Means | What To Do |
|-------------|--------------|-----------|
| Reward climbing steadily | Healthy training | Keep going, don't touch anything |
| Reward plateau, healthy entropy | May need more time | Wait it out, check again in 500k steps |
| Reward plateau, collapsed entropy | Premature convergence | Increase `ent_coef` or restart with new params |
| Reward declining after peak | Instability or collapse | Check if both rollout AND eval are declining. If yes, stop and adjust |
| `approx_kl` consistently above 0.1 | Updates too aggressive | Monitor closely. Above 0.15 = consider reducing learning rate |
| fps suddenly drops | Hardware throttling | Close background apps, check GPU temp with `nvidia-smi` |
| best_model.zip not updating | Eval hasn't beaten previous best | Check eval logs with `get_eval_logs.py` |
| Both rollout and eval declining | Real regression | Stop training, diagnose before continuing |
| Rollout dips but eval holds | Exploration phase | Normal, wait it out |

---

## Useful Commands

```bash
# Check GPU status and temperature
nvidia-smi

# Monitor training in TensorBoard
tensorboard --logdir ./tensorboard/

# Parse evaluation log history (edit RUN_NAME in script first)
python get_eval_logs.py

# Watch trained agent play
python watch.py

# Probe Atari RAM addresses
python probe_ram.py
```