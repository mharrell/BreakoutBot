# Replication Package Sketch

## "The Memorization Attractor: PPO Never Learns to React in Deterministic Atari"

---

## 1. Experiment Grid

### 1.1 Core Conditions (Breakout — 5 seeds each)

| # | Condition | Description | Hypothesized Outcome |
|---|-----------|-------------|---------------------|
| C1 | Baseline PPO | Clean ALE, no interventions, frameskip=4 | SINGLE_SCRIPT, 0-5 pts |
| C2 | Sticky p=0.10 | C1 + repeat_action_probability=0.10 | SINGLE_SCRIPT nosticky, diverse stoch |
| C3 | Sticky p=0.25 | C1 + repeat_action_probability=0.25 | SINGLE_SCRIPT nosticky, highly diverse stoch |
| C4 | Y-perturb 10% | Ball Y ±8px, p=0.10, 30f cooldown | SINGLE_SCRIPT (confirmed: PPO_55/57/58) |
| C5 | Y-perturb 25% | Ball Y ±8px, p=0.25, 30f cooldown | SINGLE_SCRIPT (hypothesis from PPO_39-43) |
| C6 | Y-perturb 50% | Ball Y ±8px, p=0.50, 30f cooldown | UNCERTAIN — PPO_97 pending |
| C7 | Ball-hit reward | +1.0/hit via RAM detection | SINGLE_SCRIPT (confirmed: PPO_92) |
| C8 | C6 + C7 combined | 50% Y-perturb + 1.0/hit | UNCERTAIN — PPO_100 pending |
| C9 | Dead model | Random-initialized PPO, deterministic inference | SINGLE_SCRIPT, 0 pts (calibration baseline) |

**Total: 9 conditions × 5 seeds = 45 experiments**

### 1.2 Multi-Env Transfer (5 seeds each, baseline PPO only)

| Game | Why | Determinism Level |
|------|-----|-------------------|
| Breakout | Primary | Fully deterministic |
| Pong | Two-agent but deterministic opponent | Fully deterministic |
| Space Invaders | Random initial positions, deterministic after | Partially deterministic |
| Beam Rider | Enemy patterns, deterministic setup | Mostly deterministic |

**Total: 3 additional envs × 5 seeds = 15 experiments**

### 1.3 Perception POC Replication

| # | Description |
|---|-------------|
| P1 | Frozen NatureCNN ball-tracking — replicate 1.9px MAE finding |
| P2 | PPO_85 replication — frozen-in features → collapse anyway |
| P3 | Multi-seed P1 (3 seeds) — measurement stability |

**Total: 5 experiments**

### Grand Total: ~65 experiments

At ~8 hours each (50M steps, RTX 3060 Ti), that's ~520 GPU-hours. At 4 concurrent runs: **~5-6 days of wall-clock time.**

---

## 2. Standardized Protocol Per Experiment

### 2.1 Training

```python
# Fixed across all conditions
n_envs = 32
n_steps = 128
batch_size = 1024
n_epochs = 4
gamma = 0.99
ent_coef = 0.006
learning_rate = linear(2.5e-4 → 1e-5)
clip_range = linear(0.2 → 0.05)

# Training env (frameskip=1 for setRAM experiments, 4 otherwise)
# Eval env: always frameskip=4, clean ALE, ClipRewardEnv

target_steps = 50_000_000
```

### 2.2 Checkpoints & Metrics

Every experiment produces:
- **Checkpoint every 100K steps** (500 checkpoints/run)
- **EvalCallback every 50K steps** — 50 episodes, det=True, clean ALE
- **MemorizationCheckCallback every 1M steps** — 20 games det=True, 20 det=False, clean ALE
- **TensorBoard logs** — rollout metrics, losses, grad norms

### 2.3 Nosticky Verification (Final Checkpoint)

For every experiment, at the final checkpoint:
```python
# 10,000 games, clean ALE, det=True, sticky_actions=False
# Verdict: SINGLE_SCRIPT (≤2 unique scores) vs MULTIPLE_SCRIPTS
# Record: unique scores, score distribution, game lengths
```

This is the **primary endpoint**. Every model that passes nosticky verification gets:
- Intervention test (ball/paddle perturbation)
- Frame-level action analysis (do actions covary with ball position?)
- Cross-seed consistency check (do all 5 seeds pass?)

---

## 3. Statistical Framework

### 3.1 Primary Claim

> "PPO argmax policy converges to a single fixed action sequence in deterministic Breakout."

**Null hypothesis:** P(free of SINGLE_SCRIPT | baseline PPO) ≥ 0.5
**Test:** Binomial test across 5 baseline seeds

If 5/5 seeds are SINGLE_SCRIPT: p = 0.5^5 = 0.031 (one-sided)
If we want p < 0.001: need 10/10 seeds (0.5^10 = 0.00098)

### 3.2 Intervention Effectiveness

> "No intervention reduces the SINGLE_SCRIPT rate."

**Null:** Each intervention reduces SINGLE_SCRIPT rate vs baseline.
**Test:** Fisher's exact test per intervention vs baseline (C1).
**Correction:** Bonferroni-Holm across 8 interventions.

### 3.3 Sticky Actions Mask Memorization

> "p=0.25 sticky actions produce MULTIPLE_SCRIPTS verdicts that revert to SINGLE_SCRIPT when sticky is removed."

**Test:** McNemar's test — each seed is its own control (sticky vs nosticky).
**Metric:** SINGLE_SCRIPT under sticky vs SINGLE_SCRIPT under nosticky.

### 3.4 Dead-Model Calibration

Every diagnostic must be calibrated against C9 (dead model):
- MemorizationCheckCallback verdicts
- Intervention test retention %
- eval_reactivity.py shape classifier
- Score diversity metrics

### 3.5 Multi-Env Generalization

> "The effect is not unique to Breakout."

Same protocol on Pong, Space Invaders, Beam Rider.
If all 4 envs show SINGLE_SCRIPT in 5/5 seeds: p = (0.5^5)^4 ≈ 9.5×10⁻⁷

---

## 4. Data Products (Per Experiment)

```
runs/{condition}/{seed}/
├── checkpoints/
│   └── latest_checkpoint_*.zip         (500 files)
├── eval/
│   └── evaluations.npz                 (1000 entries: step × mean_score × std)
├── memorization/
│   └── memorization_track.csv          (50 rows: step × det_true_verdict × det_false_verdict)
├── nosticky/
│   └── funnel_*.csv                    (10,000 rows: episode × score × length)
├── tensorboard/
│   └── events.out.tfevents.*           (full training metrics)
├── config.json                         (all hyperparameters)
└── metadata.json                       (wall time, GPU, git hash, env versions)
```

### Reproducibility anchors:
- `requirements.txt` with exact versions (stable-baselines3, ale-py, gymnasium, torch)
- `environment.yml` with CUDA version
- Dockerfile (optional but good practice)
- All wrapper code versioned and archived (Zenodo DOI)
- Raw RAM dumps for intervention test verification

---

## 5. Figures & Tables

### Table 1: Primary Results
| Condition | Seeds | SINGLE_SCRIPT Rate | Median Score | Score Range |
|-----------|-------|--------------------|--------------|-------------|
| C1 Baseline | 5 | 100% | — | — |
| C2 Sticky p=0.10 | 5 | 100% (nosticky) | — | — |
| ... | ... | ... | ... | ... |

### Figure 1: Perception-Policy Gap
Side-by-side: NatureCNN ball-tracking accuracy (1.9px MAE subplot) vs PPO_85 frozen-feature collapse (score → 0 subplot).

### Figure 2: Sticky Masking
Histogram: C2 sticky-on score distribution (diverse, 8-14 unique) vs C2 sticky-off (1 unique). Same seed, same checkpoint.

### Figure 3: Intervention Landscape
Heatmap: (y-perturb prob × aux reward scale) → SINGLE_SCRIPT rate. Each cell = mean across seeds.

### Figure 4: Cross-Env Replication
4-panel: score distributions for Breakout, Pong, Space Invaders, Beam Rider. Same protocol, same outcome.

### Table 2: Dead-Model Calibration
| Diagnostic | Dead (C9) | Live (C1) | Distinguishable? |
|------------|-----------|-----------|------------------|
| Memorization verdict | SINGLE_SCRIPT | SINGLE_SCRIPT | ✗ |
| Intervention retention | 47.7% | varies | Only if >47.7% |
| eval_reactivity shape | CLUSTERED | CLUSTERED | ✗ |
| Score diversity (det=False) | 8-14 unique | 1 unique | ✓ (but only for non-sticky) |

---

## 6. Artifacts to Publish

### Required:
1. **Paper** (8-12 pages, NeurIPS/ICLR format)
2. **All training scripts** (versioned, with seeds)
3. **All checkpoint files** (Zenodo — likely 20-50GB for key checkpoints)
4. **All raw data CSVs** (evaluations, memorization tracks, nosticky funnels)
5. **Analysis notebooks** — reproduce every figure and table from raw CSVs
6. **Perception POC code + data** — the 1.9px MAE claim

### Nice-to-have:
7. **Docker image** with full environment
8. **Video gallery** — 30s gameplay clips showing SINGLE_SCRIPT vs reactive
9. **Interactive dashboard** — explore any checkpoint's score distribution
10. **Pre-registration** — OSF registration with hypotheses before running multi-seed

---

## 7. Writing Outline

### Abstract
We show that PPO's argmax policy in deterministic Atari Breakout converges to a single fixed action sequence — a memorized script — regardless of intervention. Across 100+ experiments spanning dynamics randomization, reward shaping, sticky actions, and perceptual augmentation, no model has ever learned to react to ball position. Sticky actions (the standard Atari benchmark convention) mask this failure: they produce score diversity that mimics reactivity but collapses to a single script when removed. We formalize this as the memorization attractor problem, provide a standardized nosticky verification protocol, and show the effect replicates across multiple deterministic Atari environments. Our results challenge the interpretation of PPO benchmark scores in deterministic domains and establish a new baseline for measuring genuine reactivity.

### Sections
1. **Introduction** — The standard Atari benchmark, sticky actions, and what "solved" means
2. **The Memorization Attractor** — Defining SINGLE_SCRIPT, the perception-policy gap, why score gradients drive it
3. **Methods** — Environment, PPO configuration, memorization check protocol, nosticky verification
4. **Experiment 1: Baseline PPO** — 5 seeds, all SINGLE_SCRIPT
5. **Experiment 2: Sticky Actions Mask Memorization** — p=0.10/0.25 appear reactive, collapse nosticky
6. **Experiment 3: Dynamics Randomization** — Y-perturb across probability sweep
7. **Experiment 4: Reward Shaping** — Auxiliary tracking rewards
8. **Experiment 5: Combined Interventions** — Randomization + reward shaping
9. **Experiment 6: Multi-Env Replication** — Pong, Space Invaders, Beam Rider
10. **Experiment 7: Perception POC** — NatureCNN sees the ball, PPO ignores it
11. **Dead-Model Calibration** — Why standard diagnostics fail
12. **Discussion** — Implications for Atari benchmarking, RL in deterministic environments, what WOULD prove reactivity
13. **Conclusion** — PPO doesn't learn to react in deterministic Breakout. Prove us wrong.

---

## 8. What Would Actually Prove Us Wrong

The paper should specify, clearly:
1. **A nosticky verification protocol** that any lab can run
2. **A passing threshold**: "≥3 unique scores on det=True nosticky at final checkpoint across all 5 seeds"
3. **A replication budget**: "here's the GPU-hours, here's the code"
4. **A leaderboard**: any team that produces a reactive PPO policy for deterministic Breakout gets listed

This turns "PPO can't play Breakout" from a complaint into a challenge.
