# PPO Training Reference Guide
### Breakout RL Agent — Levers, Outputs & Experiment History

**Current Best: 124.00 eval (pixel-based, PPO_24) | Best Individual Game: 397 (real score) | Total Steps Trained: 550M+**

---

## Part 1: Training Levers

These are the parameters you control in `train.py`. Each one affects a different aspect of how the agent learns. Key principle: **change one thing at a time** so you can interpret the results.

| Parameter | What It Does | Effect on Training |
|-----------|-------------|-------------------|
| `ent_coef` | Entropy coefficient — adds bonus reward for taking varied actions | Higher = more exploration, prevents early convergence. Too high = unstable training |
| `learning_rate` | How big each update step is | Lower = more stable but slower. Larger networks need lower rates. Too low = entropy collapse. Pass a callable for linear decay |
| `clip_range` | Limits how much the policy can change in one update | Lower = more conservative updates. Watch approx_kl for signs it's too high. Decay linearly alongside learning_rate for best results |
| `n_steps` | How many steps to collect before each update | Higher = more data per update, smoother gradients but slower iteration |
| `batch_size` | How much data to use per gradient update | Scale up proportionally when increasing n_envs |
| `n_epochs` | How many times to reuse collected data per update | Higher = more efficient data use but risk of overfitting |
| `gamma` | Discount factor — how much to value future vs immediate rewards | Higher = agent thinks further ahead. Lower = more focused on immediate rewards |
| `net_arch` | Size of fully connected layers (pixel runs only) | Larger = more capacity but needs lower learning rate and more careful tuning |
| `vf_coef` | How much to weight value function loss vs policy loss | Higher = more focus on accurate reward prediction |
| `n_envs` | Number of parallel game environments | More = faster experience collection. Scale batch_size proportionally |
| `total_timesteps` | Total training duration | More = more time to learn. PPO_23 showed major gains didn't arrive until 170M+ steps |
| `device` | CPU vs GPU | MlpPolicy (RAM runs) runs faster on CPU. CnnPolicy (pixel runs) benefits from GPU |

### Linear Schedule Pattern (Confirmed Working in PPO_22, PPO_23, PPO_24)

Decay both `learning_rate` and `clip_range` from full value to near-zero over the full run:

```python
def linear_schedule(start: float, end: float):
    def schedule(progress_remaining: float) -> float:
        return end + (start - end) * progress_remaining
    return schedule

learning_rate=linear_schedule(2.5e-4, 1e-5),
clip_range=linear_schedule(0.2, 0.05),
```

When resuming from a checkpoint, use `reset_num_timesteps=False` so the schedule continues from where it left off rather than resetting to the start values.

---

## Part 2: Observation Modes

This project has used two fundamentally different observation approaches:

### Pixel-Based (PPO_5–14, PPO_20–24) ✅ Current approach
- Agent sees stacked frames of the game screen as images
- Uses `CnnPolicy` — convolutional neural network processes visual input
- Uses `make_atari_env` with `VecFrameStack(n_stack=4)`
- Training speed: ~350-620 fps with 64 envs (varies with system state — see Hardware Notes)
- More generalizable. Best results in this project.

### RAM-Based (PPO_15–19) — Abandoned
- Agent sees the 128-byte Atari RAM directly as structured data
- Uses `MlpPolicy` — simple fully connected network
- Uses custom `BreakoutRamEnv` wrapper
- Much faster training (~1400+ fps with 32 envs)
- Enables reward shaping using exact game state values
- **Ultimately abandoned** — never surpassed pixel-based performance despite speed advantage

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

**Result:** Backfired — agent learned to mirror the ball without actually scoring points.

### Important Note on ClipRewardEnv
`make_atari_env` automatically applies `ClipRewardEnv`, which clips all rewards to [-1, 1] during training. This means every brick hit returns exactly 1.0 regardless of actual point value. The agent learns entirely from clipped signals, yet still discovers high-value strategies like the tunnel exploit. Eval scores reported during training reflect clipped rewards — real in-game scores are much higher. A clipped eval score of ~124 corresponds to real game scores of 200-400+ when the tunnel strategy executes.

---

## Part 4: Training Outputs

| Output | What It Means | Healthy Range / What to Watch |
|--------|--------------|-------------------------------|
| `ep_rew_mean` | Average score per episode during training (noisy, clipped rewards) | Should climb over time. Dips normal during exploration phases. 100+ rollout mean = exceptional |
| `ep_len_mean` | Average game length in frames | Longer = agent keeping ball alive longer. Sudden drops signal a problem. 1400-1500 = healthy |
| `eval/mean_reward` | Average score during deterministic evaluation — most reliable metric | Your ground truth. Should trend upward. Plateau or decline = time to intervene |
| `fps` | Frames processed per second across all environments | RAM runs: 1000-2500+. Pixel runs: 350-620 with 64 envs. Drops suggest throttling or sleep/wake cycle |
| `entropy_loss` | How much the agent is exploring vs exploiting | Early: -1.0 to -1.5. Mid: -0.3 to -0.5. Late: -0.1 to -0.3. Near 0 = collapsed |
| `explained_variance` | How accurately the agent predicts future rewards | Closer to 1.0 is better. Below 0.5 = value function is struggling. May drop when agent reaches new high-score territory |
| `approx_kl` | How much the policy changed this update | Should stay below 0.05. Spikes above 0.1 = watch closely. Above 0.15 = concern. Near 0 at end of run = policy frozen |
| `clip_fraction` | How often PPO's safety clamp triggered | 0.05-0.15 is healthy. Very low at end of long runs = policy fully stabilized |
| `loss` | Combined training loss | Should generally decrease over time. Spikes can indicate instability |
| `value_loss` | How wrong the reward predictions were | Should decrease over time. Climbing late in training = agent reaching score ranges it rarely saw before — not necessarily a crisis |
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

## Part 5: Hardware Notes

**Test System:** Intel i5-13600K (14 cores / 20 threads), 32GB RAM, NVIDIA GeForce RTX 3060 Ti (8GB VRAM)

### Training Speed
- Pixel runs with 64 envs: **350-620 fps** depending on system state
- fps drops significantly after sleep/wake cycles — Windows deprioritizes the process. Restart the script to recover full speed
- Running from PyCharm adds slight overhead vs running `python train.py` directly from PowerShell

### GPU Utilization Pattern
PPO with 64 Atari environments is **CPU-bound during rollout collection** and GPU-bound only during the update step. This creates a bursty pattern: GPU spikes to ~40-50% during updates, then drops near 0% while the CPU steps all 64 environments. Average GPU utilization appears low (~4-6%) but this is normal and expected — not a misconfiguration.

To verify training is running correctly, use `nvidia-smi -l 1` and watch for periodic GPU spikes. If GPU never spikes above 10%, something is wrong.

### Approximate Wall Clock Time (RTX 3060 Ti)
| Steps | Approximate Time |
|-------|-----------------|
| 100M | 4-8 hours |
| 300M | 13-24 hours |
| 400M | 18-32 hours |

Range reflects fps variance from system state (fresh start vs post-sleep-cycle).

### Monitoring Commands
```bash
# Snapshot GPU status
nvidia-smi

# Watch GPU live (refreshes every second)
nvidia-smi -l 1

# Monitor training in TensorBoard
tensorboard --logdir ./tensorboard/
```

---

## Part 6: Experiment History

| Run | Obs Type | Key Parameters | Peak Eval | Notes |
|-----|----------|----------------|-----------|-------|
| PPO_5 | Pixel | Default: ent_coef=0.0, lr=2.5e-4, net=[64,64], n_envs=8 | ~30 | Entropy collapsed ~2M steps |
| PPO_6 | Pixel | ent_coef=0.01, clip_range=0.1 | ~26 | Too much entropy, unstable |
| PPO_7 | Pixel | ent_coef=0.003, clip_range=0.2 | ~31 | Best small-network pixel run |
| PPO_8 | Pixel | net=[512,512] | ~22 | Large network underperformed |
| PPO_9 | Pixel | net=[512,512], lr=1.25e-4 | ~21 | Entropy still collapsed early |
| PPO_10 | Pixel | net=[512,512], lr=1.25e-4, ent_coef=0.006 | ~25 | Better entropy, network limiting |
| PPO_11 | Pixel | net=[64,64], ent_coef=0.006, lr=1.25e-4 | ~20 | lr too low for small network |
| PPO_12 | Pixel | n_envs=32, batch=1024, lr=1.25e-4 | ~6 | 32 envs + low lr = collapse |
| PPO_13 | Pixel | n_envs=32, batch=1024, lr=2.5e-4, ent_coef=0.006 | 85.4 | Previous best. Peaked 19.2M then collapsed |
| PPO_14 | Pixel | Same as PPO_13, lr=1.25e-4 | ~59 | Lower lr, still oscillating |
| PPO_15 | RAM | RAM obs, ball tracking reward, MlpPolicy, CPU | 56.8 | Good peak at 4.8M, then degraded |
| PPO_16 | RAM | RAM obs, reward shaping | 56.4 | Short run, ended near peak |
| PPO_17 | RAM | RAM obs | 0.0 | Completely broken (unknown cause) |
| PPO_18 | RAM | RAM obs, paddle hit reward | 19.0 | Collapsed badly |
| PPO_19 | RAM | RAM obs | 36.0 | Full run, mediocre. RAM approach abandoned |
| PPO_20 | Pixel | n_envs=64, batch=2048, lr=2.5e-4 (const) | 50.0 | Cut short |
| PPO_21 | Pixel | n_envs=32, batch=1024, linear LR 2.5e-4→1e-5, 40M steps | ~47 | LR decay confirmed helpful |
| PPO_22 | Pixel | n_envs=64, batch=2048, linear LR 2.5e-4→1e-5, linear clip 0.2→0.05, 60M steps | 87.2 | Previous best at 57.6M steps |
| PPO_23 | Pixel | Same as PPO_22, n_eval_episodes=20, checkpoint resuming, 244M steps total | 119.80 | All-time best at 217.6M steps. Consistent 90-110+ eval floor in final stretch |
| PPO_24 | Pixel | Same as PPO_23, seed=None, n_eval_episodes=50, ~300M steps | **124.00** ✅ | New all-time best at 265.6M steps. Confirmed tunnel exploit (397 real points observed). TensorBoard shows upward trend at run end — more steps warranted |

### PPO_24 Eval Score History (Current Best Run)

*Note: Early eval logs (0–169M) lost to unexpected system restart. Data available from 169M onward.*

| Timestep | Eval Reward |
|----------|-------------|
| 169,600,000 | 96.18 |
| 172,800,000 | 86.30 |
| 176,000,000 | 89.18 |
| 179,200,000 | 80.32 |
| 182,400,000 | 72.34 |
| 185,600,000 | 72.96 |
| 188,800,000 | 76.18 |
| 192,000,000 | 95.22 |
| 195,200,000 | 77.84 |
| 198,400,000 | 70.48 |
| 201,600,000 | 91.48 |
| 204,800,000 | 77.68 |
| 208,000,000 | 94.34 |
| 211,200,000 | 104.88 |
| 214,400,000 | 86.12 |
| 217,600,000 | 90.56 |
| 220,800,000 | 101.76 |
| 224,000,000 | 81.48 |
| 227,200,000 | 84.34 |
| 230,400,000 | 88.04 |
| 233,600,000 | 79.04 |
| 236,800,000 | 96.14 |
| 240,000,000 | 84.96 |
| 243,200,000 | 86.26 |
| 246,400,000 | 68.30 |
| 249,600,000 | 83.98 |
| 252,800,000 | 98.80 |
| 256,000,000 | 93.20 |
| 259,200,000 | 78.84 |
| 262,400,000 | 89.64 |
| 265,600,000 | **124.00** 🏆 |
| 268,800,000 | 73.54 |
| 272,000,000 | 117.76 |
| 275,200,000 | 105.18 |
| 278,400,000 | 83.22 |
| 281,600,000 | 94.66 |
| 284,800,000 | 91.62 |
| 288,000,000 | 82.48 |
| 291,200,000 | 83.42 |
| 294,400,000 | 91.48 |
| 297,600,000 | 101.12 |
| 300,800,000 | 117.14 |

### PPO_23 Eval Score History

| Timestep | Eval Reward |
|----------|-------------|
| 99,200,000 | 65.65 |
| 102,400,000 | 82.65 |
| 105,600,000 | 74.60 |
| 108,800,000 | 65.35 |
| 112,000,000 | 79.10 |
| 115,200,000 | 50.60 |
| 118,400,000 | 66.35 |
| 121,600,000 | 69.25 |
| 124,800,000 | 64.80 |
| 128,000,000 | 73.20 |
| 131,200,000 | 50.35 |
| 134,400,000 | 82.85 |
| 137,600,000 | 78.10 |
| 140,800,000 | 63.95 |
| 144,000,000 | 83.65 |
| 147,200,000 | 76.40 |
| 150,400,000 | 71.70 |
| 153,600,000 | 63.15 |
| 156,800,000 | 77.85 |
| 160,000,000 | 82.20 |
| 163,200,000 | 86.20 |
| 166,400,000 | 78.70 |
| 169,600,000 | 118.70 |
| 172,800,000 | 61.75 |
| 176,000,000 | 70.85 |
| 179,200,000 | 79.25 |
| 182,400,000 | 79.70 |
| 185,600,000 | 117.20 |
| 188,800,000 | 60.05 |
| 192,000,000 | 82.20 |
| 195,200,000 | 97.65 |
| 198,400,000 | 90.55 |
| 201,600,000 | 94.50 |
| 204,800,000 | 102.95 |
| 208,000,000 | 75.30 |
| 211,200,000 | 97.55 |
| 214,400,000 | 97.85 |
| 217,600,000 | **119.80** |
| 220,800,000 | 83.90 |
| 224,000,000 | 105.70 |
| 227,200,000 | 108.40 |
| 230,400,000 | 86.55 |
| 233,600,000 | 100.45 |
| 236,800,000 | 89.95 |
| 240,000,000 | 116.45 |
| 243,200,000 | 108.65 |

### PPO_22 Eval Score History

| Timestep | Eval Reward |
|----------|-------------|
| 3,200,000 | 25.5 |
| 6,400,000 | 22.9 |
| 9,600,000 | 30.2 |
| 12,800,000 | 31.7 |
| 16,000,000 | 32.3 |
| 19,200,000 | 35.3 |
| 22,400,000 | 35.7 |
| 25,600,000 | 52.1 |
| 28,800,000 | 28.0 |
| 32,000,000 | 44.7 |
| 35,200,000 | 43.9 |
| 38,400,000 | 47.7 |
| 41,600,000 | 52.7 |
| 44,800,000 | 50.0 |
| 48,000,000 | 62.3 |
| 51,200,000 | 60.8 |
| 54,400,000 | 65.4 |
| 57,600,000 | **87.2** |

### PPO_13 Eval Score History (First Major Breakthrough)

| Timestep | Eval Reward |
|----------|-------------|
| 1,600,000 | 32.2 |
| 3,200,000 | 33.4 |
| 4,800,000 | 39.0 |
| 6,400,000 | 47.4 |
| 8,000,000 | 32.0 |
| 9,600,000 | 35.8 |
| 11,200,000 | 37.6 |
| 12,800,000 | 37.0 |
| 14,400,000 | 33.8 |
| 16,000,000 | 27.6 |
| 17,600,000 | 42.6 |
| 19,200,000 | **85.4** |
| 20,800,000 | 38.6 |
| 22,400,000 | 41.2 |
| 24,000,000 | 31.2 |

---

## Part 7: Key Lessons Learned

1. **Larger networks are not always better** — net=[512,512] consistently underperformed net=[64,64].
2. **Learning rate must match network size** — small networks need higher learning rates.
3. **Scale batch_size with n_envs** — n_envs=32 with batch_size=256 caused extreme entropy collapse.
4. **Best pixel config**: ent_coef=0.006, n_envs=64, batch_size=2048, linear LR 2.5e-4→1e-5, linear clip 0.2→0.05, net=[64,64].
5. **RAM observations are much faster** — 1400+ fps vs 350-620 fps for pixel runs.
6. **RAM reward shaping backfired** — agent learned to mirror the ball without scoring points.
7. **MlpPolicy runs better on CPU** — no benefit from GPU for non-CNN policies.
8. **Constant LR causes catastrophic forgetting** — PPO_13's 85.4 peak was immediately followed by collapse.
9. **Decay both LR and clip_range together** — confirmed across PPO_22, PPO_23, and PPO_24 to prevent late-run collapse.
10. **More timesteps matter significantly** — PPO_23 didn't reach its ceiling until 170M+ steps. PPO_24's TensorBoard showed an upward trend right at the 300M cutoff. Don't stop early.
11. **Checkpoint resuming works** — use `reset_num_timesteps=False` so LR decay tracks continuously across restarts.
12. **Seed diversity improves generalization** — PPO_23 trained on seed=42 played best on similar seed distributions. PPO_24 used `seed=None` for both training and eval environments, confirmed to improve generalization across ball launch directions without hurting performance.
13. **Eval scores understate real performance** — ClipRewardEnv clips training rewards to [-1, 1] per brick. A clipped eval mean of 124 corresponds to real in-game scores of 200-400+ when the tunnel strategy executes. Always verify with `watch.py` using `info[0]['episode']['r']` for true scores.
14. **The tunnel exploit is real and learnable** — PPO_24 confirmed a real in-game score of 397 in a single observed game, with the ball trapped behind the brick wall. The strategy is discovered but not yet consistent. More training time should improve consistency.
15. **High eval variance is meaningful** — PPO_24's ±88 standard deviation around a 105 mean reflects the bimodal nature of the tunnel strategy: either the agent finds it (very high score) or it doesn't (normal score). Variance is a signal, not noise.
16. **GPU utilization looks deceptively low** — with 64 Atari envs on this hardware, the GPU only fires during PPO update steps. Average utilization of 4-6% with spikes to ~50% is normal and expected. The bottleneck is Python IPC overhead between environment workers, not GPU capacity.
17. **fps drops after sleep/wake cycles** — Windows deprioritizes long-running background processes after sleep. Restarting the script restores full speed. Disable sleep during long training runs.

---

## Decision Framework

| What You See | What It Means | What To Do |
|-------------|--------------|-----------|
| Reward climbing steadily | Healthy training | Keep going, don't touch anything |
| Reward plateau, healthy entropy | May need more time | Wait it out, check again in 500k steps |
| Reward plateau, collapsed entropy | Premature convergence | Increase `ent_coef` or restart with new params |
| Reward declining after peak | Instability or collapse | Check if both rollout AND eval are declining. If yes, stop and adjust |
| `approx_kl` consistently above 0.1 | Updates too aggressive | Monitor closely. Above 0.15 = consider reducing learning rate |
| `approx_kl` near 0.0 late in run | Policy frozen, LR exhausted | Fine to stop — best_model.zip already captured the peak |
| fps suddenly drops | Hardware throttling or sleep/wake cycle | Restart script. Close background apps. Check GPU with `nvidia-smi` |
| best_model.zip not updating | Eval hasn't beaten previous best | Check eval logs with `get_eval_logs.py` |
| Both rollout and eval declining | Real regression | Stop training, diagnose before continuing |
| Rollout dips but eval holds | Exploration phase | Normal, wait it out |
| value_loss climbing late in run | Agent reaching new score territory | Not a crisis — value function catching up to new behavior |
| Strong improvement in final steps | Model had more to give | Run longer next time |
| TensorBoard eval curve still rising at run end | Run was cut short of its ceiling | Continue as next run with reset_num_timesteps=False |
| High variance in eval scores (±80 or more) | Bimodal strategy — tunnel found sometimes | More training time should improve consistency |

---

## Useful Commands

```bash
# Check GPU status and temperature
nvidia-smi

# Watch GPU utilization live
nvidia-smi -l 1

# Monitor training in TensorBoard
tensorboard --logdir ./tensorboard/

# Parse evaluation log history (edit RUN_NAME in script first)
python get_eval_logs.py

# Watch trained agent play (use info[0]['episode']['r'] for real scores)
python watch.py

# Probe Atari RAM addresses
python probe_ram.py
```

### watch.py Real Score Note
Training uses `ClipRewardEnv` so `reward` values in watch.py are clipped to [-1, 1]. To see true Atari scores, read from the Monitor wrapper's episode info:

```python
if done[0] and lives == 0:
    real_score = info[0]['episode']['r']
    print(f"Game {episode} finished | Real Score: {real_score:.0f}")
```