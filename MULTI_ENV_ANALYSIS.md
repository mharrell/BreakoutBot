# Multi-Env Replication Analysis

## The Question

Is SINGLE_SCRIPT unique to Breakout, or does PPO always memorize a single
action sequence in deterministic Atari environments?

## Candidate Games

### Tier 1: Fully Deterministic, Small Action Space (strongest tests)

| Game | Actions | Genre | Determinism | Why It Matters |
|------|---------|-------|-------------|----------------|
| **Pong** | 6 | Paddle | Full (fixed AI opponent) | Closest analog to Breakout. If Pong is SINGLE_SCRIPT too, it's not about bricks. |
| **Space Invaders** | 6 | Shooter | Partial (UFOs random) | Tests whether one random element prevents memorization. |
| **BeamRider** | 9 | Shooter | Full (fixed waves) | Like Space Invaders but fully deterministic. The pure case. |
| **Freeway** | 3 | Action | Full (car patterns) | Minimal action space (3). If a 3-action game can't escape SINGLE_SCRIPT... |
| **Q\*bert** | 6 | Puzzle | Full (enemy patterns) | Different genre. Diagonal movement. Tests genre generality. |

### Tier 2: Partially Random (intermediate cases — probe if Tier 1 all positive)

| Game | Actions | Random elements | What it tests |
|------|---------|-----------------|---------------|
| **MsPacman** | 9 | Ghost decision stochasticity | Classic — does moderate randomness suffice? |
| **Phoenix** | 8 | Wave patterns, boss timing | If SINGLE_SCRIPT here, partial randomness isn't enough. |

### Tier 3: Fully Deterministic, Larger Action Spaces (harder to learn)

| Game | Actions | Genre | Notes |
|------|---------|-------|-------|
| **Enduro** | 9 | Racing | Deterministic traffic. Visual complexity. |
| **Boxing** | 18 | Sports | Deterministic AI. Large action space may slow scripting. |

## Recommended Probing Order

### Phase 1: Quick Probes (already written)

```
train_pong_baseline.py        ALE/Pong-v5         10M steps  ~2 hrs
train_space_invaders_baseline.py  ALE/SpaceInvaders-v5  10M steps  ~2 hrs
```

These answer: is the effect Breakout-specific?

### Phase 2: If both Phase 1 are SINGLE_SCRIPT

```
train_beamrider_baseline.py   ALE/BeamRider-v5    10M steps  ~2 hrs
train_freeway_baseline.py     ALE/Freeway-v5       10M steps  ~2 hrs
```

These answer: is the effect universal across deterministic Atari?

### Phase 3: Only if evidence suggests a boundary

```
train_mspacman_baseline.py    ALE/MsPacman-v5     10M steps  ~2 hrs
train_phoenix_baseline.py     ALE/Phoenix-v5       10M steps  ~2 hrs
```

These probe where the boundary is — how much randomness breaks memorization?

## What Each Outcome Means

### All-deterministic games are SINGLE_SCRIPT
→ The paper's claim generalizes: PPO memorization is a property of
  deterministic RL environments, not Breakout specifically.
→ Implication: any Atari benchmark result on deterministic games should
  include nosticky verification.
→ p-value: 4 games, 5 seeds each = (0.5^5)^4 ≈ 9.5×10⁻⁷

### Breakout + Pong are SINGLE_SCRIPT, Space Invaders is not
→ One random element (UFOs) is sufficient to break memorization.
→ The paper's claim narrows to fully-deterministic games.
→ Implication: the threshold is surprisingly low — even minimal
  stochasticity prevents SINGLE_SCRIPT.

### Only Breakout is SINGLE_SCRIPT
→ The effect is game-specific. The paper narrows to "why Breakout?"
→ Hypothesis: Breakout's score gradient (flat reward per brick) combined
  with simple physics creates a uniquely scriptable landscape.
→ Would need to explain why Pong (similarly simple, flat reward) escapes.

### Nothing is SINGLE_SCRIPT (including Breakout re-test)
→ Something changed (ALE version, PPO version, etc.) and the original
  finding doesn't replicate. This would be the most interesting outcome
  but contradicts 100 experiments.

## Wrapper Differences Per Game

| Game | FireResetEnv | EpisodicLifeEnv | Notes |
|------|-------------|-----------------|-------|
| Breakout | ✓ | ✓ | Standard |
| Pong | ✗ | ✗ | No FIRE, no lives (21-point games) |
| Space Invaders | ✓ | ✓ | Standard |
| BeamRider | ✓ | ✓ | Standard shooter setup |
| Freeway | ✗ | ✗ | No FIRE, no lives (timed game) |
| Q\*bert | ✗ | ✓ (lives) | No FIRE needed to start |

## Script Template

All probe scripts follow the same pattern — copy `train_pong_baseline.py` and:
1. Change env name
2. Adjust wrappers (FireResetEnv, EpisodicLifeEnv per table above)
3. Change RUN_NAME and SEED
4. Everything else identical: 32 envs, NatureCNN, ent_coef=0.006, 10M steps,
   MemorizationCheckCallback every 1M, EvalCallback every 50K

## Cost

| Phase | Scripts | GPU hours | Wall clock (2 concurrent) |
|-------|---------|-----------|---------------------------|
| 1 (Pong + Space Invaders) | 2 | ~4 | ~2 hrs |
| 2 (BeamRider + Freeway) | 2 | ~4 | ~2 hrs |
| 3 (MsPacman + Phoenix) | 2 | ~4 | ~2 hrs |
| **Total (all 3 phases)** | **6** | **~12** | **~6 hrs** |

## For the Paper

If all 6 probes are SINGLE_SCRIPT, that's 6 deterministic/partially-deterministic
Atari games across 4 genres (paddle, shooter, action, puzzle) — strong evidence
that the memorization attractor is a property of PPO + deterministic environments,
not a quirk of Breakout.
