# PPO Training Reference Guide
### Breakout RL Agent — Levers, Outputs & Experiment History

**SUMMARY:** Comprehensive PPO reference for this project. Part 1 = training parameters, Part 2 = observation modes, Part 3 = reward shaping + sticky actions, Part 4 = training metric diagnostics, Part 5 = hardware notes, Part 6 = full experiment history, Part 7 = 40 lessons learned. **Single most important finding (2026-07-14): No model in this project has ever genuinely generalized.** Every sticky-trained model tested without sticky actions collapsed: PPO_26 (60-pt script), PPO_30b (zero script), PPO_31b (31-pt script), PPO_27 (noise-dependent degenerate). Sticky actions mask memorization with noise; they do not prevent or cure it. This independently confirms Zhang et al. (2018), who found ConvNets are naturally noise-robust and memorize through sticky perturbations. The nosticky-verification protocol is the only reliable behavioral test. See Lessons #32, #36-39 and FLAWS.md. **Active experiment:** Experiment 4 — PPO_32 (low-sticky single-phase, p=0.05 from scratch).

**Eval Record: 147.02 (PPO_27 @ 867M, sticky) | Note: PPO_27 confirmed noise-dependent degenerate without sticky (100% zeros). All sticky-trained models are memorized. See Lesson #39 and FLAWS.md.**

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
| `repeat_action_probability` | Sticky actions — chance a step repeats the previous action regardless of agent's choice | **Does NOT force reactive ball-tracking in deep ConvNet policies.** Zhang et al. (2018) showed CNNs are naturally noise-robust and memorize through sticky perturbations. This project confirmed it across 5 models (PPO_26/27/28/29/30b/31b) — every sticky-trained model collapsed to a deterministic script or noise-dependent degenerate policy without sticky actions. Sticky actions mask memorization; they do not prevent or cure it. See Lesson #39. |

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

### Sticky Actions as an Environment-Level Intervention (PPO_26, PPO_27, PPO_28, PPO_29)
Unlike reward shaping, `repeat_action_probability=0.25` doesn't touch the reward signal at all — it modifies the environment dynamics so 25% of steps repeat the previous action regardless of what the agent chose. This is the Machado et al. (2018) standard mitigation for memorization in deterministic Atari environments. Confirmed effects across all four runs:
- Improves training stability and general single-env play consistency (higher average and median score) — **but only when combined with substantial prior non-sticky training.** PPO_26 (deep + sticky) shows this clearly; PPO_27 (sticky from step one, no deep prior training) does not.
- **Completely eliminates** the deterministic zero-score launch-angle failure mode — but again, only in combination with deep prior training (PPO_26: 0%). Sticky actions alone do not fix this: PPO_27's zero-score rate (21.27%) statistically matches PPO_25's non-sticky baseline (20.0%) — see Lesson #22.
- Does **not** reduce funnel exploit completion rate — the earlier hypothesis to that effect was based on a small mismatched-threshold sample (PPO_26's funnel rate is higher than PPO_25's: 0.07% vs 0.02%).
- **Must remain on at inference time.** Removing sticky actions from any trained model in this project (PPO_28, PPO_29) caused rapid memorization collapse to fixed open-loop action sequences within tens of millions of steps. Sticky actions are not a training phase — they are a permanent environmental constraint that prevents memorization at every step. See Lessons #28-29 and EXPERIMENTS.md Experiment 2.
- Full comparative results in EXPERIMENTS.md.

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
| PPO_25 | Pixel | Continued from PPO_24 checkpoint, same config, 1B+ steps, no sticky actions | 140.94 | Held all-time eval record until surpassed by PPO_27. Real game scores of 600+ observed. Single-env (10,000 games, matched script): avg 34.6, **20.0% zero-score games** — see Lesson #22 and Part 3 commentary |
| PPO_26 | Pixel | Continued from PPO_25 best_model + `repeat_action_probability=0.25`, 64 envs | 134.16 (eval) | **Completed at 1,001,828,352 total steps.** Single-env (10,000 games): avg **54.3**, 0% zero-score, funnel rate 0.07% — best single-env model. See EXPERIMENTS.md Experiment 1 |
| PPO_27 | Pixel | Fresh agent, `repeat_action_probability=0.25` from step one, 32 envs | **147.02** ✅ | **Current all-time eval record** @ 867.2M steps. But single-env (10,000 games): avg 27.95, 21.27% zero-score — worst single-env performer despite best eval score. See Lesson #27 |
| PPO_28 | Pixel | PPO_26 weights + `repeat_action_probability=0.0` (sticky removed), 32 envs | ~366 (artifact) | **Memorization collapse confirmed.** Plays fixed 383-frame, 104-point script from any starting condition. Eval scores (340-366) and rollout stats (healthy-looking entropy/EV) were misleading — see EXPERIMENTS.md Experiment 2. Run terminated. |
| PPO_29 | Pixel | PPO_27 weights + `repeat_action_probability=0.0` (sticky removed), 32 envs | ~418 (artifact) | **Memorization collapse confirmed.** Plays fixed 395-frame, 355-point script from any seed. Eval "record" of 418.88 is 50 repetitions of same memorized sequence. Run terminated. See EXPERIMENTS.md Experiment 2 and Lesson #28. |

### Eval Score History — Condensed Summaries

*Full per-checkpoint logs for each run live in `logs/<RUN_NAME>/evaluations.npz` and can be regenerated anytime with `python helpers/get_eval_logs.py`. The tables below summarize the shape of each run rather than reproducing every logged step.*

| Run | Steps Covered | Batch/Run Avg | Peak Eval | Low Point | Notes |
|-----|---------------|---------------|-----------|-----------|-------|
| PPO_13 | 1.6M – 24.0M | ~37 | **85.4** @ 19.2M | 27.6 @ 16.0M | First major breakthrough; collapsed immediately after peak (no LR decay yet) |
| PPO_22 | 3.2M – 57.6M | ~44 | **87.2** @ 57.6M | 22.9 @ 6.4M | Steady upward climb with linear LR/clip decay confirmed working |
| PPO_23 | 99.2M – 243.2M | ~82 | **119.80** @ 217.6M | 50.35 @ 131.2M | Consistent 90-110+ floor in final stretch |
| PPO_24 | 169.6M – 300.8M | ~89 | **124.00** @ 265.6M | 68.30 @ 246.4M | Confirmed tunnel exploit (397 real points). Eval curve still rising at cutoff |
| PPO_25 | 342.4M – 1,004.8M | ~104 | **140.94** @ 838.4M | 71.72 @ 345.6M | Held all-time eval record until surpassed by PPO_27. Real game scores of 600+ observed. Large log gaps (390.4M–412.8M, 464.0M–489.6M, 652.8M–681.6M, 860.8M–918.4M) from system restarts |
| PPO_26 | 566.4M – 1,001.8M (completed) | ~90.5 (full run) | **134.16** @ 905.6M | 46.06 @ 998.4M (new all-time eval low) | **Run complete.** A real, sustained dip occurred at ~670-755M steps (42% of evals below 70) before recovering. Ended healthy — final two evals (109.34, 112.10) followed immediately after the 46.06 low, with no instability signature in approx_kl/entropy_loss at the tail |
| PPO_27 | 481.6M – 867.2M+ (ongoing) | ~104 (growing) | **147.02** @ 867.2M 🏆 | 80.26 @ 640.0M | **Current all-time eval record across all runs.** Both floor and ceiling have risen steadily across quarters of training (Q1 mean 97→Q4 mean 112; Q1 max 114→Q4 max 147), genuine improvement rather than noise |

**All-time eval record progression:** PPO_13 (85.4) → PPO_22 (87.2) → PPO_23 (119.80) → PPO_24 (124.00) → PPO_25 (140.94) → **PPO_27 (147.02, current)**

### Single-Env Matched Funnel Comparison (10,000 games each, identical script/threshold)

*This is the metric that reflects actual sequential gameplay, as opposed to the parallel-sampled eval score above — see Lesson #13. Full methodology and discussion in EXPERIMENTS.md.*

| Metric | PPO_25 | PPO_26 | PPO_27 |
|--------|--------|--------|--------|
| Average score | 34.6 | **54.3** 🏆 | 27.95 (worst) |
| Median score | 30.0 | **46.0** 🏆 | 23.0 (worst) |
| Zero-score games | 1,998 (20.0%) | **0 (0.0%)** 🏆 | 2,127 (21.27%, worst) |
| Funnel rate (400+) | 0.02% | **0.07%** 🏆 | 0.01% (worst) |
| Best single-env score | 406 | **415** 🏆 | 406 |

PPO_26 wins every single-env metric. PPO_27 — the all-time eval-score leader — is the *worst* single-env performer of the three, with a zero-score rate statistically matching PPO_25's despite having sticky actions. The eval-score leaderboard and the single-env leaderboard are inverted, not just different — see Lesson #23 and #27.

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
18. **Training can continue indefinitely via restart behavior — but this was a bug, now understood and fixed for future runs.** With `reset_num_timesteps=False` and a hardcoded `TOTAL_TIMESTEPS` constant, SB3's `learn()` treats that constant as "train N *more* steps from wherever I am," not an absolute target. Every restart with the same script pushed the goalpost another `TOTAL_TIMESTEPS` further out. This is exactly why PPO_26 stopped at ~960M (a restart bug, not deliberate) and PPO_27 stopped at ~880M on its own — both landed almost exactly `TOTAL_TIMESTEPS` past their respective resume points. The fix going forward (starting with PPO_28): compute `remaining = ABSOLUTE_TARGET - model.num_timesteps` and pass that to `learn()`, so the run always converges on a true target regardless of how many times it's restarted.
19. **Fresh sticky-action training eventually overtook inherited weights in eval-score performance** — PPO_27 (fresh agent, sticky actions from step one) surpassed both PPO_26 (inherited PPO_25 weights + sticky actions) and the original PPO_25 record to set a new all-time eval high, now at 147.02 @ 867.2M steps. This resolves the open question from Experiment 1: total training time on a clean, environment-consistent policy foundation matters more in the long run than inheriting a billion steps of prior (mismatched) experience — at least on eval score (see Lesson #23 for why single-env performance tells a different story).
20. **PPO_26 had a real, sustained mid-run instability dip — not just noise — but recovered cleanly.** Steps ~670-755M showed 42% of evals below 70 (vs. ~10-16% in surrounding quarters), a genuine regression rather than ordinary variance. It recovered on its own within the next quarter and went on to set its own peak (134.16) afterward. The earlier-flagged "tail dip" (134.16→71.50 at the very end of an early data batch) turned out, once the fuller run came in, to be ordinary noise — a separate and much milder thing from this real dip.
21. **Sticky actions do NOT trade funnel-exploit frequency for general consistency — that original hypothesis was wrong.** Measured properly (matched script, matched 400-point threshold, full 10,000-game samples for both), PPO_26 has both a higher single-env average (54.3 vs 34.6) AND a higher funnel rate (0.07% vs 0.02%) than PPO_25. The original tradeoff conclusion was based on a small, inconsistently-thresholded snapshot and doesn't survive a fair comparison.
22. **PPO_25's "zero-score" failure mode is real, large, structural — and requires BOTH sticky actions AND deep prior training to fix, not either alone.** PPO_25 (deep, no sticky): 20.0% zero-score. PPO_26 (deep + sticky): 0.0% zero-score — completely eliminated. PPO_27 (sticky, fresh — no deep prior training): 21.27% zero-score — statistically the same as PPO_25, despite having sticky actions. Sticky actions alone do not fix this. The deep inherited training appears to be the necessary ingredient; sticky actions can only leverage it once present.
23. **Eval score and single-env score are not just "different leaderboards" — they're inverted across these three models.** PPO_27 holds the all-time eval record (147.02) and is the *worst* single-env performer of the three on every metric measured (average, median, zero-score rate, funnel rate). PPO_26 has a middling eval peak (134.16) and the *best* single-env performance. Do not use eval score alone to judge a model's real-gameplay quality — always verify against a single-env run.
24. **RAM coordinate convention for paddle/ball x-position runs opposite to naive intuition.** Empirically, the `LEFT` action increases `paddle_x` (and `RIGHT` decreases it) — the reverse of assuming `x=0` is the left wall. This was discovered while investigating a "panic" pattern in PPO_26's worst games and isn't a bug; it's just how the RAM byte happens to be laid out. Worth remembering before doing any further RAM-based behavioral analysis on this project.
25. **Frame-level "is the action moving toward the ball" is not a reliable diagnostic for game outcome.** Tested across a 6-game quick-death cluster (61.3% direction-correct) vs. a 40-game unfiltered control sample (55.3%), including a velocity-aware variant (54.6%) — none of these showed a meaningful gap correlated with score. This kind of instantaneous per-frame position check doesn't seem to capture whatever actually distinguishes good games from bad ones in this project; worth knowing before reaching for it again as a debugging tool.
26. **The project has effectively been training on a process metric all along, not raw score** — `ClipRewardEnv` flattens every brick hit to 1.0 regardless of point value, so the real training signal across every run has been "bricks removed," with raw score only ever used as an eval/comparison readout. This is a useful real-world data point for process-vs-outcome reward design: a well-chosen process proxy ("clear bricks") produced excellent outcome results ("high score") here, while a different process proxy tried explicitly (paddle bounces + episode length, see EXPERIMENTS.md Experiment 1 Option D discussion) would not have, because it admits a degenerate high-reward strategy that never engages with the actual task.
27. **Total training steps alone don't explain PPO_26 vs. PPO_27's single-env gap — the *type* of early training matters.** By the time of its funnel test, PPO_27 had accumulated roughly as many total steps as PPO_26, entirely under sticky actions, and still showed PPO_25-level zero-score failure. This suggests the non-sticky phase specifically (not just raw step count) is doing the important work — possibly building basic visual/control competency that sticky-action training later refines, rather than sticky actions teaching reactive tracking from a blank slate. Motivates the next planned experiment: sweep the *duration* of non-sticky pretraining before switching to sticky actions, to find out how much is actually necessary (see EXPERIMENTS.md Planned Next Steps).
28. **Removing sticky actions from any trained model in this project causes rapid memorization collapse.** Both PPO_28 (PPO_26 lineage) and PPO_29 (PPO_27 lineage) converged to fixed open-loop action sequences within tens of millions of steps of stickiness removal. A deterministic policy in a deterministic environment will always find a fixed-sequence solution faster than a general reactive one, because the fixed sequence is a vastly simpler optimization target. PPO_26's non-sticky pretraining foundation did not protect against this. The two-phase sticky-then-off recipe does not work for these models.
29. **Sticky actions are required at inference time, not just during training.** Their role is not "teach reactive tracking during formative training, then remove." They are an ongoing environmental constraint that makes memorization unreliable as a strategy at every step. Remove them from a converged model and it finds a fixed sequence within hundreds of gradient updates.
30. **Training metrics cannot be trusted to detect memorization collapse.** During PPO_28/29's collapse: `ep_rew_mean` jumped from ~265 to 409; eval scores hit 418.88; `explained_variance` hit 1.0; `value_loss` dropped to 0.001; `loss` went negative. PPO_28 even showed *apparently healthy* metrics (entropy -0.45 to -0.55, EV 0.76-0.92, value_loss 1-4) because 32 envs settling into different fixed loops looked like genuine diversity in aggregate. The only reliable diagnostic is behavioral: play the model against varied starting conditions and check whether scores vary. `deterministic=False` producing identical results to `deterministic=True` is the earliest detectable signal. **However: the behavioral test itself (MemorizationCheckCallback, ≤2 unique scores = MEMORIZED) has NOT been calibrated for sticky-action environments. See Lesson #32.**
31. **In Atari Breakout specifically, the ALE random seed does not meaningfully vary the ball launch direction.** Ball direction appears to be determined by an internal game-frame timer, not the ALE seed. Evaluation scripts relying on `seed=int` variation to produce diverse starting conditions will produce identical games instead. The correct approach for producing varied game-to-game outcomes in a single-env evaluation loop (for sticky-action-free models) is `seed=None` with a single persistent environment and plain `env.reset()` between episodes — which lets natural ALE internal state accumulation drive variation. This is distinct from models trained with sticky actions, where action-override randomness drives variation independently of the ALE seed.
32. **The MemorizationCheckCallback "GENERALIZING" verdict is uncalibrated for sticky-action models — CONFIRMED with data (2026-07-14).** Calibration using PPO_30a (confirmed MEMORIZED, 2 unique scores non-sticky): with p=0.25 sticky, the SAME dead policy produces 8-14 unique scores per 20-game batch (mean 11.3, P95=14). Both PPO_30b (10-19) and PPO_31b (10-19) overlap with this noise baseline. Nosticky verification confirms both collapse to ≤2 unique scores. Sticky probability sweep shows that even p=0.05 produces 55-63 unique scores from these memorized policies. **The GENERALIZING verdict is not a reliable indicator of reactive behavior for any sticky-action model.** The ≤2 threshold is valid for non-sticky only. See FLAWS.md F-001 and `recordings/memorization_calibration.csv`.
33. **Anti-correlated experimental variables cannot be causally separated.** Experiment 3's design (PPO_30b: 100M pretrain + 300M sticky; PPO_31b: 300M pretrain + 100M sticky) makes pretraining duration and sticky-step count perfectly anti-correlated at the 400M total budget. The conclusion "sticky steps drive right-tail performance; non-sticky pretraining suppresses failure" is underdetermined — the data is equally consistent with a single-variable explanation ("more sticky training improves everything" or "less pretraining hurts the floor"). Before attributing an outcome to variable X in any experiment, list every other variable that changed between groups. If the list has >1 item, the attribution is underdetermined. See FLAWS.md F-002.
34. **Funnel rate comparisons require statistical caution at N=10,000.** With observed counts of 0-7 funnel events in 10,000 games, 95% binomial confidence intervals overlap heavily. A difference of 5 events (e.g., 7 vs. 2) yields p ≈ 0.18 by Fisher's exact test — not significant at conventional thresholds. Always report binomial CIs for rare-event metrics. Treat funnel rate differences of <5 events as directional indicators, not settled differences. For reliable comparison with 80% power at α=0.05, detecting a 2× difference in funnel rate would require ~50,000 games per model. See FLAWS.md F-010.
35. **Always verify evaluation completion before reporting results.** PPO_31b's 10k-game evaluation was reported as complete with final statistics, but the funnel log contained only 9,247 rows (753 short). Large-sample statistics from 9,247 games are unlikely to differ meaningfully from 10,000, but presenting them as "final" without verifying the row count is a process failure. After every evaluation: verify the funnel log has exactly 10,000 data rows (plus header), recompute statistics from the verified data, and only then mark the evaluation complete. See FLAWS.md F-004.
36. **The argmax decision rule (deterministic=True) can mask real policy learning.** PPO_30b with deterministic=True and sticky=off: 2 unique scores, 99.8% zeros — appears completely dead. Same model with deterministic=False and sticky=off: **43 unique scores, avg 23.5** — the policy has learnable action preferences that the argmax suppresses. The policy is not dead; the argmax is collapsing to a deterministic attractor. Always test both deterministic and stochastic inference before concluding a policy has "collapsed." See `eval_variance_test.py` and `recordings/variance_decomposition.csv`.
37. **Sticky probability drives unique-score count, not policy quality.** The sticky probability sweep showed that at p=0.05 (just 5% action-repeat), both PPO_30b and PPO_31b jump from 2 to 55-63 unique scores. The relationship between sticky probability and unique-score count is mechanical, not evidential. Higher unique scores at higher sticky probabilities do not indicate better policies — they indicate more environmental noise. Never use unique-score count as a quality metric for sticky-action models without controlling for sticky probability.
38. **A memorized script that happens to score non-zero is not "robustness."** PPO_31b's 2.4% zero-score rate was interpreted as robustness from deeper pretraining. Nosticky verification revealed the truth: PPO_31b plays a fixed 178-frame, 31-point script that happens to never score zero. Its "robustness" is an artifact of which script got memorized, not of genuine reactivity. PPO_30b's script zeros 99.8% of the time — it's not "fragile," it just memorized a different script. When a model shows unexpectedly low zero-score rates, verify with nosticky evaluation before attributing to robustness.
40. **Environmental dynamics randomization, not perceptual noise, forces reactivity.** CNNs are naturally robust to pixel-level perturbations — translation invariance is built into the architecture via convolutional filters and max-pooling. Obfuscating the agent's visual field (data augmentation, observation noise) doesn't prevent memorization; it just forces the CNN to learn to see through the noise, and it arrives at the same feature-level representations regardless. What breaks memorized open-loop scripts is unpredictability in the environment's *dynamics*: variable ball speed, variable frame skip, variable paddle responsiveness. If the ball arrives at the paddle's row at an unpredictable time, no timed script works — the agent must observe where the ball actually is and react accordingly. This is domain randomization (the standard sim-to-real transfer technique) applied to the memorization problem: randomize the environment parameters a memorized policy depends on, and the policy must learn features that generalize across those parameters. Later Breakout-style games implicitly use this principle via powerups, speed changes, and challenge modes that vary what the player can depend on.

39. **Sticky actions do not prevent memorization in deep ConvNet RL agents — and this was known in 2018.** Zhang et al. (2018), "A Study on Overfitting in Deep Reinforcement Learning," published months after Machado et al., directly tested whether sticky actions prevent memorization in DQN and other deep RL agents. Their finding: *"Stochasticity could neither prevent deep RL agents from serious overfitting nor detect overfitted agents effectively."* ConvNets are "automatically robust" to sticky perturbations — a memorized action sequence still works through 25% noise because the perturbation is small relative to the network's tolerance. This project independently confirmed Zhang et al.'s finding across 5 PPO models spanning p=0.0, p=0.25, and both combined: **every sticky-trained model tested without sticky actions collapsed to a deterministic script (PPO_26: 60 pts, PPO_31b: 31 pts, PPO_30b: 0 pts) or a noise-dependent degenerate policy (PPO_27: 100% zeros, 19-frame deaths).** The field adopted sticky actions as standard evaluation protocol; Zhang et al.'s warning that they don't work for deep RL appears to have been largely overlooked. The project's nosticky-verification protocol is the correct diagnostic. Sources: [Zhang et al. 2018](http://bengio.abracadoudou.com/cv/publications/pdf/zhang_2018_arxiv.pdf), [Machado et al. 2018](https://jair.org/index.php/jair/article/download/11182/26388).

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
| `ep_rew_mean` jumps 2-4x in under 5M steps; `explained_variance` → 1.0; `value_loss` → ~0; `loss` goes negative | Memorization collapse — policy has converged to a fixed action sequence, not genuine skill | Verify with behavioral test: run `deterministic=False` and check if scores vary across games. If identical → confirmed collapse. Kill the run — it will not recover. |
| `deterministic=False` produces same scores as `deterministic=True` | Policy entropy is functionally zero — the model is outputting the same action from every state it encounters | Confirmed memorization. The training entropy_loss reading may still look healthy (common in multi-env runs) — don't trust it. Behavioral test is definitive. |
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