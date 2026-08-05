# Next Steps — August 2026

## Current State

PPO_124 (proximity reward, scale=0.05) is the first verified reactive PPO argmax on Atari Breakout: 0/240 perfect transfers, 100% no-timing ALT retention, intervention AUC 0.421. Every prior model (123 experiments) was confirmed memorized by split-watcher.

**Since this doc was first drafted (Aug 3-4):**

- **Scale sweep complete:** PPO_127 (scale=0.10) and PPO_128 (scale=0.025) both ran 25M steps. Results: 0.05 is the unambiguous sweet spot. Higher scale (0.10) suppresses both score and reactivity — proximity overwhelms game reward. Lower scale (0.025) produces high scores but script-dominated — game reward overpowers tracking gradient. The relationship is non-monotonic and tightly tuned.

- **PPO_126 oscillation discovered (NOT regression):** Split-watcher on all 12 checkpoints (5M→50M) revealed PPO bounces between reactive and script-dominated phases with ~10-15M period. Two competing basins of nearly equal value at scale=0.05. The 50M checkpoint is in a script-dominated trough, not permanently regressed — it would likely oscillate back. FULL unique=1 for every checkpoint (expected — deterministic reactive policy on deterministic layout).

- **Experiment 35 complete:** Fading (PPO_131) and step-down (PPO_132a→132b) both trained and analyzed. All proximity-reward models verified reactive via ball-teleport split-watcher. Fading is the best variant — 428 points, AUC 0.402, px_corr 0.025. Step-down retains reactivity but scores lower.
  - Ball-teleport split-watcher built (`ball_teleport_split_watcher.py`) — replaces broken BrickClearWrapper, uses ball X teleport for reliable argmax reactivity measurement.
  - Per-frame analysis updated and run — PPO_131 shows 72.5% frame-level ball tracking.

However, publishing efforts (Medium, Reddit) got little traction. Key weaknesses to address before arXiv submission: **single seed per config**, **no zero-scale ablation**.

## Priority 1: Multi-Seed Replication

Run 3 additional seeds of PPO_124 with identical config:

| Run | Seed | Config | Steps | Expected |
|-----|------|--------|-------|----------|
| PPO_124a | 1241 | ProximityReward(scale=0.05) | 25M | 0 perfect transfers, high ALT retention |
| PPO_124b | 1242 | ProximityReward(scale=0.05) | 25M | same |
| PPO_124c | 1243 | ProximityReward(scale=0.05) | 25M | same |

Training time: ~6.5 hours each on RTX 3060 Ti. Can run two in parallel on 8GB VRAM (32 envs each, ~870MB per instance).

Script: clone `train_ppo_124.py`, change SEED to 1241/1242/1243, change RUN_NAME to PPO_124a/b/c.

Verification per seed:
1. Split-watcher no-timing (60 games) — expect 0 perfect transfers, 100% ALT retention
2. Intervention gradient — expect AUC > 0.3, clean dose-response

## Priority 2: Zero-Scale Control (Ablation)

Run one seed with proximity scale=0.0:

| Run | Seed | Config | Steps | Expected |
|-----|------|--------|-------|----------|
| PPO_124_control | 1240 | ProximityReward(scale=0.0) | 25M | MEMORIZED (perfect transfers, SINGLE_SCRIPT) |

Training time: ~6.5 hours.

This proves the proximity reward *caused* the reactivity — without it, the same setup produces a memorized script. Essential for causal claims.

Script: clone `train_ppo_124.py`, set PROXIMITY_SCALE=0.0, RUN_NAME="PPO_124_control".

## Priority 3: Scale Sensitivity Sweep ✅ DONE

Completed Aug 3-4 with PPO_127 (scale=0.10) and PPO_128 (scale=0.025). Both ran 25M steps.

| Run | Seed | Scale | Score | Divergence | Verdict |
|-----|------|-------|-------|------------|---------|
| PPO_128 | 128 | 0.025 | 395pt | 39.1% | Script-dominated — game reward overwhelms tracking |
| PPO_124 | 124 | 0.05 | ~350pt | ~70% | Sweet spot — balanced basins, oscillates |
| PPO_127 | 127 | 0.10 | 250pt | 5.9% | Strongly script-dominated — proximity overwhelms game |

Conclusion: scale=0.05 is the unambiguous sweet spot. No further scale values needed unless exploring very fine gradations (0.04, 0.06).

## Priority 4: Oscillation Control (Fading & Step-Down) ✅ DONE

Completed August 4, 2026. Ball-teleport split-watcher results (10 games):

| Run | Config | px_corr | Div | Track | FULL | ALT | AUC |
|-----|--------|---------|-----|-------|------|-----|-----|
| PPO_131 | Fading 0.05→0.0, 25M | 0.025 | 71% | 73% | 428 | 428 | 0.402 |
| PPO_132a | scale=0.05, 15M | -0.027 | 63% | 81% | 85 | 38 | 0.357 |
| PPO_132b | Step-down 0.05→0.0, 25M | 0.150 | 61% | 71% | 186 | 307 | 0.312 |

**Fading is the clear winner.** Highest scores (428), highest AUC (0.402), strongest tracking signal. Step-down retains reactivity but scores lower. All proximity-reward models pass the ball-teleport test — the reward reliably produces reactive argmax policies.

## Priority 5: Pong Transfer

Apply proximity reward to Pong. Same principle: reward paddle for vertical closeness to ball. Tests whether the approach generalizes beyond Breakout.

Needs: Pong RAM addresses (paddle Y, ball X/Y), adapted wrapper, training script.

`train_pong_baseline.py` exists for reference. This is the strongest evidence for "proximity reward as a general method."

## Priority 6: Statistical Rigor

- Run 100+ games per layout in split-watcher (currently 20-30)
- Bootstrap confidence intervals on px_corr and ALT retention
- Formal statistical test on perfect transfer count
- Per-frame tracking analysis via `analyze_frame_behavior.py` (tool built, not yet run)

## Priority 7: Per-Frame Behavioral Analysis ✅ DONE

`analyze_frame_behavior.py` updated to use ball teleport (replacing broken BrickClearWrapper) and run on PPO_131. Result: **72.5% tracking** over 28,410 frames — ALT paddle consistently closer to teleported ALT ball than FULL ball. Confirms frame-level argmax reactivity.

Run against PPO_124 best checkpoint with 20+ games to characterize the tracking signal at the frame level.

## Files to Create

- `train_ppo_124a.py` through `train_ppo_124c.py` — multi-seed replication
- `train_ppo_124_control.py` — zero-scale ablation
- `pong_proximity_wrapper.py` + `train_pong_proximity.py` — Pong transfer
- `verify_split_watcher_large.py` — 100-game verification with bootstrap CIs

## Already Created (Aug 3-4)

- `train_ppo_127.py` / `train_ppo_128.py` — scale sweep (complete at 25M)
- `train_ppo_129.py` / `train_ppo_130.py` — continuation scripts (127→50M, 128→50M; deferred)
- `train_ppo_131.py` — fading experiment (complete)
- `train_ppo_132a.py` / `train_ppo_132b.py` — step-down experiment (complete)
- `fading_proximity_wrapper.py` — decaying scale wrapper
- `analyze_frame_behavior.py` — per-frame tracking analysis
- `batch_split_watcher.py` / `run_batch.ps1` / `run_batch_fast.ps1` — batch split-watcher tooling

## Publishing Plan (after replication)

1. Multi-seed results confirm PPO_124 wasn't a fluke
2. Zero-scale control proves causation
3. Update paper with replication data
4. Try arXiv again (need endorser still: https://arxiv.org/auth/endorse?x=MUM8BP)
5. Consider submitting to a workshop (less gatekeepy than main conference, still gets you a citable publication)
   - AAAI Workshop on Reinforcement Learning
   - NeurIPS Workshop on Deep RL
   - ALA (Adaptive and Learning Agents) Workshop at AAMAS

## Notes

- BreakoutBot repo: `github.com/mharrell/BreakoutBot`
- Clean repo: `github.com/mharrell/breakout-reactive-ppo`
- Paper: `breakout-reactive-ppo/paper.tex`
- Blog post (Medium): `breakout-reactive-ppo/BLOG_POST_MEDIUM.md`
- arXiv endorsement link: https://arxiv.org/auth/endorse?x=MUM8BP
- Email: mikey.harrell@gmail.com
- YouTube split-watcher: https://www.youtube.com/watch?v=6ixVwQm7u5Y
