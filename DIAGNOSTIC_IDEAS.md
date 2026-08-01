# Diagnostic Ideas — July 30, 2026

Three new reactivity diagnostics to build. These fill the biggest remaining gap in the project's measurement toolkit: **no frame-level behavioral metrics exist.** Every current diagnostic looks at score outcomes; none measure whether actions are conditioned on game state (LOGICAL_AUDIT.md L-012).

---

## Idea 1: SCAD Probe — State-Conditioned Action Distribution (HIGHEST PRIORITY)

**The gap:** We've never measured whether PPO actions depend on ball position. All 100+ experiments judge reactivity through scores, score diversity, or perturbation response — never through direct action analysis.

**What it does:** During normal gameplay (no teleportation), record per-frame: `(frame, ball_x, ball_y, paddle_x, action, action_probs)`. Then compute conditional action probabilities.

**Key metrics:**
- **Tracking probability:** P(action moves paddle toward ball | ball is not centered under paddle)
  - Dead sweep script ≈ 50% (random direction relative to ball)
  - Reactive policy > 70% (consistently moves toward ball)
- **Mutual information:** I(action ; sign(ball_x - paddle_x))
  - Dead ≈ 0 bits (action independent of ball position)
  - Higher = stronger conditioning on ball position

**Dead baseline:** Sweep script and center-hold script — run through same analysis. Both should produce MI ≈ 0 and tracking probability ≈ 50%.

**Both inference modes:** det=True and det=False. The gap matters (Critical Rule #9 — argmax can collapse while distribution retains useful entropy). A model with tracking-prob=40% on det=True but 65% on det=False is "close" — the distribution sees the ball, the mode doesn't.

**Implementation:** Cheap — RAM reads during regular evaluation games, same pattern as `probe_107_intervention.py` but passive (no ball teleportation). ~20 games each mode = ~20k-40k frame samples.

**Why this is #1:** It replaces interpretive inference with direct measurement. Score diversity can come from noise (L-012). Intervention retention needs dead calibration (L-001). This directly answers: "when the ball is to the left of the paddle, does the policy go LEFT more often than when the ball is to the right?"

---

## Idea 2: Intervention Gradient — Dose-Response Curve

**The gap:** `probe_107_intervention.py` tests ONE teleport magnitude (±30px) and gives a binary result. You can't track partial progress or compare models on a continuum.

**What it does:** Run the existing intervention probe at multiple teleport magnitudes: ±0 (control), ±8, ±15, ±30, ±45, ±60px. At each magnitude, measure reversal rate.

**Key metric:** **Reversal AUC** — the area under the reversal-rate vs. displacement curve. Also the **half-max displacement** — the teleport distance at which reversal rate drops to 50% of its ±8px value.

A reactive policy:
```
±8px:  65% reversal
±15px: 52%
±30px: 33%  ← current probe only sees this point
±45px: 18%
±60px:  8%
AUC ≈ 0.35
```

A memorized script:
```
All magnitudes: ~0% reversal
AUC ≈ 0.0
```

**Why this fills a gap:** You can track a single number over training ("PPO_107: AUC 0.22 at 10M → 0.35 at 30M → 0.41 at 50M") and compare it across variants. The current binary probe can't do this.

**Implementation:** Minor modification to `probe_107_intervention.py` — wrap the teleport in a loop over magnitudes, collect separate reversal counts per magnitude. Add `--sweep` flag or make the magnitude a list parameter.

---

## Idea 3: Feature-Attribution Probe — Trained CNN Check

**The gap:** The Perception POC proved a supervised NatureCNN can encode ball position to 1.9px MAE. But we've only proven this for a dedicated supervised model, not for any actual PPO-trained policy. A given PPO checkpoint might have unlearned ball features during training — we don't know whether the perception-policy gap is "sees the ball but ignores it" or "stopped seeing the ball."

**What it does:** Take a trained PPO checkpoint. Freeze the CNN. Run frames through it, extract feature maps (the output of the conv stack before the FC policy/value heads). Train a linear regressor on these frozen features to predict `(ball_x, ball_y)`. Compare MAE to the supervised POC baseline (1.9px).

**Key metric:** Ball-position prediction MAE from frozen PPO features.
- ~2px → features encode ball position perfectly (perception-policy gap confirmed)
- ~5-10px → features degraded somewhat (partial perception collapse)
- ~20+px → features lost (full perception collapse — the CNN unlearned ball tracking)

Run this across checkpoints to see if ball encoding degrades over training (does PPO actively unlearn ball features as it converges to a script?).

**Implementation:** No gameplay needed — forward passes through the CNN on recorded/random frames. The supervised regression POC code from `ball_tracker_cnn_4frame.py` can be mostly reused. Just swap the CNN weights and freeze them.

**Why this matters:** If features survive at 2px in a SINGLE_SCRIPT model, we know with certainty it's a policy optimization problem, not a representation problem. If features degrade, we might need to protect them (aux supervision, layer freezing, etc.).

---

## Implementation Notes

### General rules (from CLAUDE.md — these apply to ALL new scripts):

1. **Present design decisions before coding** — Critical Rule #11. Any new script has embedded decisions (parameters, distributions, sample sizes).
2. **Dead baseline mandatory** — Critical Rule #14. Every new metric must be calibrated against a known-dead baseline before supporting claims.
3. **`if __name__ == "__main__":` guard required** — no importing side effects.
4. **Both inference modes** — always test det=True AND det=False (Critical Rule #9).
5. **Standalone** — no imports from training scripts. Load model checkpoints, run evaluation, exit.

### Architecture decisions common to all three:

- **RAM access pattern:** Standard — `env.venv.envs[0].unwrapped.ale.getRAM()` / `setRAM()`.
- **Env pipeline:** ALE/Breakout-v5, frameskip=4, repeat_action_probability=0, NoopResetEnv(30), FireResetEnv, EpisodicLifeEnv, GrayscaleResize(84,84), AutoResetWrapper, VecFrameStack(4). Match `probe_107_intervention.py` exactly.
- **Dead baselines:** Sweep script (paddle sweeps left-right at fixed rate) and center-hold (holds center, fires on serve). Implement inline like the intervention probe, don't import.
- **Model loading:** `PPO.load(path, env=env, device="cuda")`.
- **CLI:** argparse or manual argv parsing — `--model PATH`, `--games N`, `--det/--stoch`.

### Don't overthink sample sizes:

For SCAD: 20 games × ~1500 frames × 32 actions each = fine-grained conditional probabilities. Don't benchmark to statistical saturation — 20 games is enough for a strong signal.

For intervention gradient: ~20 teleports per magnitude per model. The current probe uses 40 at one magnitude — 20 is enough for a gradient.

For feature attribution: a few thousand frames from random gameplay. Even 1000 frames should converge a linear regressor on frozen features.
