# PPO Training Reference Guide
### Breakout RL Agent — Levers, Outputs & Experiment History

**Current Best: 143.36 eval (pixel-based, sticky actions, PPO_27 at 804.8M steps) | Previous Pre-Sticky Record: 140.94 (PPO_25 @ 838M) | Best Individual Game: 600+ (real score) | Total Steps Trained: 2B+ combined across PPO_25/26/27**

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
| `repeat_action_probability` | Sticky actions — chance a step repeats the previous action regardless of agent's choice | Forces reactive ball-tracking over positional memorization. Improves stability and general play, but reduces precise/optimal-execution strategies like the tunnel exploit |

### Linear Schedule Pattern (Confirmed Working in PPO_22, PPO_23, PPO_24, PPO_26, PPO_27)

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

### Pixel-Based (PPO_5–14, PPO_20–27) ✅ Current approach
- Agent sees stacked frames of the game screen as images
- Uses `CnnPolicy` — convolutional neural network processes visual input
- Uses `make_atari_env` with `VecFrameStack(n_stack=4)`
- Training speed: ~350-620 fps with 64 envs (varies with system state — see Hardware Notes)
- More generalizable. Best results in this project.
- PPO_26 and PPO_27 add `repeat_action_probability=0.25` (sticky actions) on top of this approach — see Part 6 and EXPERIMENTS.md for findings.

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
`make_atari_env` automatically applies `ClipRewardEnv`, which clips all rewards to [-1, 1] during training. This means every brick hit returns exactly 1.0 regardless of actual point value. The agent learns entirely from clipped signals, yet still discovers high-value strategies like the tunnel exploit. Eval scores reported during training reflect clipped rewards — real in-game scores are much higher. A clipped eval score of ~140 corresponds to real game scores of 300-600+ when the tunnel strategy executes. Single-env watch scripts average ~56 per game, while training eval means of 100-140 reflect the parallel sampling of 50 episodes across 64 environments.

**Commentary — what the agent has actually been optimizing for:** Because `ClipRewardEnv` flattens every brick hit to exactly 1.0, every run in this project (PPO_5 through PPO_27) has never directly trained on raw point score. The true training signal has always been **bricks removed**, not score — a brick worth 1 point and a brick worth 7 points contribute identically to the training reward. Real score is a downstream readout used for eval and comparison between runs, not the thing being optimized during gradient updates. This matters for interpreting results: the project's apparent successes (and the tunnel exploit specifically) emerged from a process-style objective — "clear bricks" — rather than an outcome-style objective — "maximize point total." The fact that this also happens to produce excellent real-score outcomes is a useful existence proof that a well-chosen process metric, here forced by an environment wrapper rather than deliberately designed, can outperform optimizing the raw outcome directly. It does **not** mean every process proxy works this well — see the paddle-bounce/episode-length reward-hacking analysis in EXPERIMENTS.md (Experiment 1, Option D discussion) for a process proxy that fails badly, and contrast with why "bricks removed" succeeds where "bounces + survival time" would not: there is no cheap way to rack up brick-clear reward without actually clearing bricks, whereas bounces and episode length can both be maximized by a degenerate strategy (e.g. a stable vertical ball loop) that never touches a brick at all.

### Sticky Actions as an Environment-Level Intervention (PPO_26, PPO_27)
Unlike reward shaping, `repeat_action_probability=0.25` doesn't touch the reward signal at all — it modifies the environment dynamics so 25% of steps repeat the previous action regardless of what the agent chose. This is the Machado et al. (2018) standard mitigation for memorization in deterministic Atari environments. Confirmed effects:
- Improves training stability and general single-env play consistency
- Reduces tunnel exploit completion rate (precise execution becomes harder when actions are occasionally overridden)
- Full comparative results in EXPERIMENTS.md

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
- PPO_26 (64 envs) and PPO_27 (32 envs) trained simultaneously, sharing one GPU — neither ran at full available speed during that period

### GPU Utilization Pattern
PPO with 64 Atari environments is **CPU-bound during rollout collection** and GPU-bound only during the update step. This creates a bursty pattern: GPU spikes to ~40-50% during updates, then drops near 0% while the CPU steps all 64 environments. Average GPU utilization appears low (~4-6%) but this is normal and expected — not a misconfiguration.

To verify training is running correctly, use `nvidia-smi -l 1` and watch for periodic GPU spikes. If GPU never spikes above 10%, something is wrong.

### Approximate Wall Clock Time (RTX 3060 Ti)
| Steps | Approximate Time |
|-------|-----------------|
| 100M | 4-8 hours |
| 300M | 13-24 hours |
| 400M | 18-32 hours |
| 1B+ | Multiple weeks (with system restarts) |

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
| PPO_24 | Pixel | Same as PPO_23, seed=None, n_eval_episodes=50, ~300M steps | 124.00 | New all-time best at 265.6M steps. Confirmed tunnel exploit (397 real points observed). TensorBoard shows upward trend at run end — more steps warranted |
| PPO_25 | Pixel | Continued from PPO_24 checkpoint, same config, 1B+ steps, no sticky actions | 140.94 | Held all-time record until surpassed by PPO_27. Real game scores of 600+ observed. Tunnel exploit fires at ~3-5% rate in single-env play |
| PPO_26 | Pixel | Continued from PPO_25 best_model + `repeat_action_probability=0.25`, 64 envs, fresh step count | 134.16 | New PPO_26 high at 905.6M steps. Highly oscillating training — latest batch swung 71.50–134.16. Inherited weights gave early advantage but advantage eroded over time (see EXPERIMENTS.md) |
| PPO_27 | Pixel | Fresh agent, `repeat_action_probability=0.25` from step one, 32 envs | **143.36** ✅ | **Current all-time record across all runs**, set at 804.8M steps — surpasses PPO_25's prior 140.94. Most stable of the three: latest batch averaged 112.3 vs PPO_26's 104.2, with tighter oscillation |

### Eval Score History — Condensed Summaries

*Full per-checkpoint logs for each run live in `logs/<RUN_NAME>/evaluations.npz` and can be regenerated anytime with `python helpers/get_eval_logs.py`. The tables below summarize the shape of each run rather than reproducing every logged step.*

| Run | Steps Covered | Batch/Run Avg | Peak Eval | Low Point | Notes |
|-----|---------------|---------------|-----------|-----------|-------|
| PPO_13 | 1.6M – 24.0M | ~37 | **85.4** @ 19.2M | 27.6 @ 16.0M | First major breakthrough; collapsed immediately after peak (no LR decay yet) |
| PPO_22 | 3.2M – 57.6M | ~44 | **87.2** @ 57.6M | 22.9 @ 6.4M | Steady upward climb with linear LR/clip decay confirmed working |
| PPO_23 | 99.2M – 243.2M | ~82 | **119.80** @ 217.6M | 50.35 @ 131.2M | Consistent 90-110+ floor in final stretch |
| PPO_24 | 169.6M – 300.8M | ~89 | **124.00** @ 265.6M | 68.30 @ 246.4M | Confirmed tunnel exploit (397 real points). Eval curve still rising at cutoff |
| PPO_25 | 342.4M – 1,004.8M | ~104 | **140.94** @ 838.4M | 71.72 @ 345.6M | Held all-time record until surpassed by PPO_27. Real game scores of 600+ observed. Large log gaps (390.4M–412.8M, 464.0M–489.6M, 652.8M–681.6M, 860.8M–918.4M) from system restarts |
| PPO_26 (latest batch) | 889.6M – 947.2M | 104.2 | **134.16** @ 905.6M | 71.50 @ 947.2M | Batch ends on a downward note right after its own record — flagged for the PPO_26 tail-diagnosis investigation in EXPERIMENTS.md |
| PPO_27 (latest batch) | 798.4M – 856.0M | 112.3 | **143.36** @ 804.8M 🏆 | 88.34 @ 800.0M | **Current all-time record across all runs.** Tighter, more consistent batch than PPO_26's equivalent window |

**All-time record progression:** PPO_13 (85.4) → PPO_22 (87.2) → PPO_23 (119.80) → PPO_24 (124.00) → PPO_25 (140.94) → **PPO_27 (143.36, current)**

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
13. **Eval scores reflect parallel env sampling** — ClipRewardEnv clips training rewards to [-1, 1] per brick. Training eval runs 50 episodes across 64 parallel envs, producing a different score distribution than single-env sequential play. Single-env watch scripts average ~56 per game; training eval means of 100-140 reflect this sampling difference. Always verify with `watch.py` using `info[0]['episode']['r']` for true scores.
14. **The tunnel exploit is real, learnable, and measurable** — PPO_25 achieves tunnel completion in ~3-5% of single-env games (roughly 1 in 30-45 games), with real scores of 200-600+ when it fires. The agent attempts multiple partial tunnels simultaneously rather than committing to one clean hole — improving tunnel consistency is the current open problem.
15. **High eval variance is meaningful** — wide standard deviation around the eval mean reflects the bimodal nature of the tunnel strategy: either the agent finds it (very high score) or it doesn't (normal score). Variance is a signal, not noise.
16. **GPU utilization looks deceptively low** — with 64 Atari envs on this hardware, the GPU only fires during PPO update steps. Average utilization of 4-6% with spikes to ~50% is normal and expected. The bottleneck is Python IPC overhead between environment workers, not GPU capacity.
17. **fps drops after sleep/wake cycles** — Windows deprioritizes long-running background processes after sleep. Restarting the script restores full speed. Disable sleep during long training runs.
18. **Training can continue indefinitely via restart behavior** — with `reset_num_timesteps=False` and checkpoint loading, each system restart effectively resets the step budget against the saved checkpoint's step count. PPO_25 reached 1B+ steps this way, well past the intended 400M target. Intentional or not, this enabled discovering the performance ceiling more thoroughly.
19. **Fresh sticky-action training eventually overtook inherited weights in peak performance** — PPO_27 (fresh agent, sticky actions from step one) surpassed both PPO_26 (inherited PPO_25 weights + sticky actions) and the original PPO_25 record to set a new all-time high of 143.36 @ 804.8M steps. This resolves the open question from Experiment 1: total training time on a clean, environment-consistent policy foundation matters more in the long run than inheriting a billion steps of prior (mismatched) experience.
20. **PPO_27's stability advantage compounds over time** — in directly comparable recent batches, PPO_27's eval mean (112.3) now exceeds PPO_26's (104.2), with visibly tighter oscillation. PPO_26's most recent batch ended on a downward note (71.50 after a 134.16 peak just a few checkpoints prior) — worth watching for a real regression vs. normal noise.
21. **Sticky actions trade tunnel-exploit frequency for general consistency and ceiling** — the tradeoff identified in Experiment 1 still holds directionally, but PPO_27's results show that, given enough training, the "general consistency" side of that trade can also produce a higher absolute peak score than the no-sticky-action baseline (PPO_25). The funnel-rate cost remains the open problem (see EXPERIMENTS.md).
22. **The project has effectively been training on a process metric all along, not raw score** — `ClipRewardEnv` flattens every brick hit to 1.0 regardless of point value, so the real training signal across every run has been "bricks removed," with raw score only ever used as an eval/comparison readout. This is a useful real-world data point for process-vs-outcome reward design: a well-chosen process proxy ("clear bricks") produced excellent outcome results ("high score") here, while a different process proxy tried explicitly (paddle bounces + episode length, see EXPERIMENTS.md Experiment 1 Option D discussion) would not have, because it admits a degenerate high-reward strategy that never engages with the actual task.

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
