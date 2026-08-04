# Next Steps — August 2026

## Current State

PPO_124 (proximity reward) is the first verified reactive PPO argmax on Atari Breakout: 0/240 perfect transfers, 100% no-timing ALT retention, intervention AUC 0.421. Every prior model (123 experiments) was confirmed memorized by split-watcher.

However, publishing efforts (Medium, Reddit) got little traction. Key weakness to address before arXiv submission: **single seed per config.**

## Priority 1: Multi-Seed Replication

Run 3 additional seeds of PPO_124 with identical config:

| Run | Seed | Config | Steps | Expected |
|-----|------|--------|-------|----------|
| PPO_124a | 1241 | ProximityReward(scale=0.05) | 25M | 0 perfect transfers, high ALT retention |
| PPO_124b | 1242 | ProximityReward(scale=0.05) | 25M | same |
| PPO_124c | 1243 | ProximityReward(scale=0.05) | 25M | same |

Training time: ~6.5 hours each on RTX 3060 Ti. Can run sequentially or in parallel if GPU memory allows (32 envs each, ~870MB per instance — two might fit in 8GB VRAM).

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

## Priority 3: Scale Sensitivity Sweep

| Run | Seed | Scale | Expected |
|-----|------|-------|----------|
| PPO_127 | 127 | 0.01 | Too weak? May not shift optimum |
| PPO_128 | 128 | 0.10 | Too strong? May overwhelm game reward |
| PPO_129 | 129 | 0.25 | Stress test |

Shows there's a sweet spot. Maps the relationship between scale and reactivity.

## Priority 4: Pong Transfer

Apply proximity reward to Pong. Same principle: reward paddle for vertical closeness to ball. Tests whether the approach generalizes beyond Breakout.

Needs: Pong RAM addresses (paddle Y, ball X/Y), adapted wrapper, training script.

## Priority 5: Statistical Rigor

- Run 100+ games per layout in split-watcher (currently 20)
- Bootstrap confidence intervals on px_corr and ALT retention
- Formal statistical test on perfect transfer count

## Priority 6: Checkpoint Selection Analysis

PPO_126 showed regression at 50M (best checkpoint was at 47.4M). Investigate:
- Save checkpoints every 1M steps
- Split-watcher each checkpoint
- Find the "reactive window" — when does reactivity emerge and when does it collapse?

## Files to Create

- `train_ppo_124a.py` through `train_ppo_124c.py` — multi-seed replication
- `train_ppo_124_control.py` — zero-scale ablation
- `train_ppo_127.py` through `train_ppo_129.py` — scale sweep
- `pong_proximity_wrapper.py` + `train_pong_*.py` — Pong transfer
- `verify_split_watcher_large.py` — 100-game verification with bootstrap CIs

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
