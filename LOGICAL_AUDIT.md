# Logical Audit — BreakoutBot Project

**Date:** 2026-07-19  
**Update (2026-07-19):** Phase 1 calibration results appended. L-001 and L-002 now CONFIRMED with data — PPO_34 (dead) matches or exceeds PPO_35 on all metrics. L-007 now CONFIRMED with ALE cross-evaluation data — PPO_35's GymBreakout 212-point argmax script becomes a 2-point script on ALE (99.1% score drop).
**Scope:** Reasoning patterns, interpretive frameworks, evidence standards, and documentation consistency across the entire project.
**Relationship to FLAWS.md:** FLAWS.md catalogs methodological flaws (confounded variables, missing baselines, measurement artifacts). This document catalogs logical flaws (reasoning errors, interpretive inversions, reification of speculative categories, asymmetric skepticism). Some issues span both documents; cross-references are provided where relevant.

---

## Summary

The project demonstrates genuine scientific instincts — it maintains a 21-entry flaw catalog (FLAWS.md), has a written Breakthrough Verification Protocol, cross-references data against documentation, and has correctly identified and publicly corrected three major false positives. However, **it fails to apply its own critical standards to its most recent and most consequential claims.** The same pattern that produced the three documented false positives (PPO_26, PPO_30b/31b, PPO_35-mk1) is visible in the current experimental narrative and memory files: a promising number is interpreted as confirmation before the test that would falsify it is run.

This document identifies 16 specific logical issues organized by severity, plus a cross-cutting pattern analysis.

---

## CRITICAL — Threaten Core Current Claims

### L-001: The PPO_35 intervention test result (47% retention) is logically inverted — it's evidence AGAINST reactivity, not for it

**What the project claims** (memory file `ppo35-first-non-memorized-model.md`):
> "PPO_35 is the first model in BreakoutBot history that demonstrably responds to game state. When the ball is teleported mid-game, PPO_35 adapts its actions to the new position — retaining 47% of normal score."

**Why this is wrong:** A policy that genuinely tracks the ball should be **largely invariant** to teleportation — it sees the ball in a new position on the very next frame and adjusts its paddle accordingly. Losing 53% of normal score means the perturbation is severely disrupting performance. That pattern is more consistent with a **memorized script being broken by perturbation** than with a reactive policy being tested:

- If the policy executes a timed action sequence, teleporting the ball causes the script to miss, and scores drop substantially — exactly what we see.
- If the policy genuinely tracks the ball, it observes the new position in the next frame and adjusts — score impact should be modest.
- A truly blind script would collapse to near-zero (as PPO_36-40 do at 8-26% retention). A partially-conditioned script — one that uses ball position for some decisions but timing for others — would show intermediate retention (40-60%). **That's what 47% looks like.**

The comparison to PPO_36-40 (8-26% retention) doesn't rescue the claim: it just means PPO_35's script is less brittle, not that it isn't a script. The project's own F-001 lesson applies here by analogy: "a dead policy + noise produces a score distribution." A script + teleportation noise also produces a score distribution. The 47% number is being interpreted backward.

**What would actually demonstrate reactivity:** A score correlation >0.7 between normal and intervention games (the policy's skill transfers across perturbations), combined with small mean delta (<10% of normal score). `eval_intervention.py` already computes this correlation metric — it just wasn't the headline number.

**Required to resolve:** Run a known-dead model (PPO_30a — confirmed 2 unique scores, pure argmax script) through the identical intervention test. If a dead script retains, say, 30-50% of score under teleportation (because the teleport only fires on ~30% of paddle bounces, and many bounces happen in positions where teleportation doesn't affect the script's next few actions), then PPO_35's 47% is indistinguishable from the dead-script baseline. This is exactly the calibration logic from F-001, applied to a different test. **This calibration has not been run.**

**➡️ STATUS UPDATE (2026-07-19): CONFIRMED.** Calibration run on PPO_34 (confirmed dead argmax script: det=True produces 1 unique score, std=0.0 on GymBreakout). Results:

| Model | Normal Mean | Intervention Mean | Retention | Correlation | det=False Unique | det=False Shape |
|-------|------------|-------------------|-----------|-------------|-----------------|-----------------|
| **PPO_34 (dead baseline)** | 314.0 | 155.7 | **49.6%** | 0.000 | 19 | UNCLEAR |
| PPO_35 (claimed reactive) | 751.0 | 335.8 | **44.7%** | 0.000 | 21 | UNCLEAR |

**The dead baseline retains MORE score under intervention than the claimed-reactive model (49.6% vs 44.7%).** Both have zero correlation between normal and intervention scores. det=False diversity is statistically indistinguishable (19 vs 21 unique, 44% vs 42% top-3 concentration). The intervention test and det=False diversity do not distinguish a known-dead argmax script from the model claimed to be "the first sighted policy in BreakoutBot history." The original claim is NOT SUPPORTED by currently available evidence.

### L-002: The Breakthrough Verification Protocol was not followed for the PPO_35 memory claim

CLAUDE.md specifies a 6-gate Breakthrough Verification Protocol that "MUST be completed before writing ANY claim to memory or documentation that a model 'generalizes,' 'is reactive,' 'has solved memorization,' or represents a 'first ever' or 'breakthrough.'"

The PPO_35 memory file (written 2026-07-19) claims "first sighted policy" and "first model demonstrably responsive to game state." Gate compliance:

| Gate | Requirement | Status |
|------|-------------|--------|
| 1 | Both inference modes tested | Done (det=True + det=False via eval_reactivity.py) |
| 2 | **Calibration baseline** — what does a KNOWN-MEMORIZED model produce under identical conditions? | **NOT DONE.** No dead model was run through the intervention test. |
| 3 | **Comparison baseline** — run same test on at least one other model | **PARTIAL.** PPO_36-40 compared but PPO_34 (same experiment family, most similar architecture) not shown. |
| 4 | Std column checked — verify eval stds | Done |
| 5 | **Falsification test pre-registered** — "This claim would be proven wrong if _______" | **NOT DONE.** The memory file contains no falsification statement. |
| 6 | Environment match confirmed | Done (GymBreakout, though ALE validation is pending — see L-007) |

Gates 2, 3, and 5 were skipped. Gate 2 is the most consequential: the project spent months calibrating the sticky-action noise baseline (F-001), establishing that a dead policy + p=0.25 sticky produces 8-14 unique scores. The exact same calibration logic applies to intervention tests — a dead policy + teleportation produces SOME score retention. Without running the dead-policy calibration, 47% is an uninterpretable number.

**The protocol also states:** "If ANY gate is ambiguous, write 'TENTATIVE:' prefix in the memory title." The memory file title is "PPO_35: First Sighted Policy" — no TENTATIVE prefix, despite 3 of 6 gates being incomplete.

### L-003: PPO_35's kill decision relied on wrong-environment data that the project's own documentation warns is meaningless

The PPO_35 memorization track CSV header explicitly states:
> "WARNING: memorization check uses ALE Breakout (not GymBreakout)"

CLAUDE.md session bootstrap step 1 warns:
> "WARNING: meaningless for GymBreakout-trained models (PPO_33/34/35) — the callback tests ALE, not GymBreakout."

Yet EXPERIMENTS.md cites this wrong-environment data as the basis for killing the run:
> "KILLED at 268M/400M (2026-07-16). The argmax remained SINGLE_SCRIPT on every check (268/268)."

The eval callback data — which DOES use the correct GymBreakout environment — showed the policy was still changing: "scripts cycling through 63 unique scores, peaking at 212 pts." The policy was not frozen; it was actively exploring the script space. The kill decision relied on data the project's own documentation warns is meaningless, while the correct-environment data showed ongoing change.

**This is not a small oversight.** The experiment was killed at 67% of its target step count. The kill freed GPU resources for Experiment 6 (PPO_36), which was then itself killed at 294M for a separate interpretive reason (L-004). The project killed one experiment based on wrong-environment data to make room for another experiment that was killed based on two data points.

### L-004: The "LR decay caused dissolution regression" causal chain rests on two data points 13M steps apart

The entire Experiment 10 (three concurrent training runs: PPO_41, PPO_42, PPO_43) is predicated on this causal claim:
> "PPO_36 achieved dissolution at 169M but regressed by 182M as LR decayed to ~1.4e-4"

The evidence for "regression":
- At 169M steps: top-3 concentration 41%, max gap 5 pts → labeled "most continuous distribution in project history"
- At 182M steps: top-3 concentration 58%, max gap 21 pts → labeled "regression"
- Difference: 13M steps (3.25% of the run at that point)

**Why this is insufficient:** Every model in this project's history has shown wide oscillation. PPO_26 swung from 134 to 47 eval score in adjacent checkpoints. PPO_35 cycled through 63 different scripts at different training steps. Two data points 13M steps apart cannot distinguish a structural regression from normal checkpoint-to-checkpoint oscillation. The 169M checkpoint could have been a transient favorable draw (the 20-game memorization check or 100-game reactivity eval landing on a particularly well-spread sample), with 182M being regression to the mean.

**The corroborating evidence is also weak:**
- KL divergence at 0.0016 is cited as proof the "policy is barely updating." But KL divergence measures update magnitude, not policy quality. A policy near convergence naturally has low KL — that's expected behavior, not evidence of a problem. The linear LR schedule was designed to produce low KL divergence late in training.
- The claim "LR had decayed to ~1.4e-4 (45% through schedule)" is accurate but doesn't establish causation. Many things change at 45% through a training run.

**The decision tree didn't consider the null hypothesis:** that dissolution was a transient fluctuation, not a structural trend, and that the policy would have re-dissolved with continued training at the same LR schedule.

### L-005: "Top-3 concentration of 41%" is still a highly clustered distribution — the "dissolution" framing is misleading

EXPERIMENTS.md describes PPO_36 at 169M as having "the most continuous distribution in project history" and labels its shape "UNCLEAR (near-continuous)." The facts:

- 23 unique scores in 100 games
- Top-3 concentration: 41% — meaning 41 out of 100 games landed on exactly 3 score values
- A uniform distribution across 23 values would have top-3 concentration of ~13%
- 41% is 3.2× the uniform baseline

**This is a clustered distribution by any external standard.** The project has no external reference for what "continuous" looks like — only internal comparisons to its own worst cases (70% top-3, 36-pt gaps). The "dissolution" narrative is created by comparing against a very low bar rather than against a principled baseline.

**What's actually happening:** The policy maintains multiple scripts (det=False reveals them) and the relative dominance of those scripts shifts over training. This is script portfolio rebalancing, not dissolution. A truly dissolving distribution would show top-3 concentration trending toward the uniform baseline (~13% for 23 categories), not oscillating in the 41-70% range.

### L-006: "0% zero-score" is not evidence of reactivity — PPO_26 proved this

PPO_36's 0% zero-score rate is repeatedly highlighted as a key positive signal:
> "Zero-score rate: 0% across all 95+ checks — first model in the project to achieve this" (EXPERIMENTS.md)

But PPO_26 had 0% zero-score across 10,000 games with sticky=on and turned out to be a 60-point memorized script. A script that consistently scores non-zero produces 0% zero-score by construction. "0% zero-score" means the policy never produces a score of exactly zero — it doesn't mean the policy is reactive. A dead policy that plays a 5-point script from every starting state also has 0% zero-score.

**What 0% zero-score actually means:** The policy's worst script or worst stochastic draw still scores >0. This is a floor-quality metric (the policy isn't completely broken), not a reactivity metric. It's worth noting but should not be framed as evidence of ball-tracking.

---

## HIGH — Undermine Specific Interpretive Frameworks

### L-007: The GymBreakout-to-ALE transfer gap is systematically acknowledged and systematically deferred

The project moved from ALE to a custom GymBreakout engine to enable dynamics randomization (Experiments 5+). Seven experiments (5A/B/C, 6, 7, 8, 9) plus three current runs (41/42/43) have been conducted on this custom engine. Every memory file and CLAUDE.md caveat says "ALE validation pending." But there has been zero validation:

- No baseline showing a known-ALE-memorized model behaves identically on GymBreakout
- No comparison of observational distributions between the two engines (rendering differences, physics differences, timing differences)
- No documentation of how GymBreakout's physics differ from ALE's (ball speed-up mechanics, brick scoring, collision geometry, frame timing)

The entire post-Experiment-4 research program's relevance to actual Atari Breakout depends on this transfer being valid. The `ale-return-direction.md` memory acknowledges this gap and frames ALE return as the next step, but **ten experiments have already been run on the unvalidated engine.** If GymBreakout differs from ALE in ways that affect CNN behavior (e.g., different visual rendering of the ball, different frame timing), all post-Experiment-4 conclusions may not transfer.

**This is the project's single largest unaddressed risk.** It's equivalent to developing a medical treatment in mice and writing up conclusions about humans without having run the human trial — while acknowledging at the bottom of every page that the human trial is "pending."

**RESOLUTION (2026-07-19): L-007 CONFIRMED — transfer gap is catastrophic.** PPO_35's best_model (251M steps, GymBreakout-trained, the model at the center of the "sighted policy" claim) was cross-evaluated on ALE/Breakout-v5 (frameskip=1, repeat_action_probability=0, 100 games per mode):

| Environment | det=True mean | det=True unique | det=False mean | det=False unique |
|-------------|-------------|----------------|---------------|-----------------|
| GymBreakout | **212.0** | 1 (std=0.0) | **113.0** | 21 |
| ALE/Breakout-v5 | **2.0** | 1 (std=0.0) | **6.8** | 19 |
| **Transfer (GB→ALE)** | **-99.1%** | — | **-94.0%** | — |

Both environments produce argmax scripts (unique=1, std=0.0), and both produce similar det=False score diversity (21 vs 19 unique). But the GymBreakout script scores 212 points while the ALE script scores 2. The engines are so different that the same learned policy produces a 106× score difference. The "sighted policy" claim was evaluated on an engine whose results do not transfer to authentic Atari Breakout.

All post-Experiment-4 conclusions should be treated as **custom-engine findings pending ALE replication.** The return to ALE (ale-return-direction.md) is the correct next step, and no further GymBreakout experiments should be launched without ALE validation.

### L-008: The DropoutNatureCNN mechanism claim is speculative and the only ablation was killed before reaching parity

The project claims dropout works by "preventing the network from encoding frame-precise timing information" because "single-neuron timing representations" are killed by dropout while "ball position (a large visual feature encoded across many neurons) survives."

**This mechanistic claim has no direct evidence:**
1. It assumes the CNN encodes frame timing in individual neurons — not established. CNNs learn distributed representations; timing could be encoded across many neurons, making it robust to dropout.
2. It assumes dropout at p=0.1 is sufficient to disrupt timing encoding while preserving position encoding — not tested. No ablation of dropout probabilities was run.
3. It assumes the differential effect (timing disrupted, position preserved) without measuring either.

**The only evidence is circumstantial:** PPO_37 (no dropout) experienced "entropy collapses" while PPO_36 (with dropout) didn't. But PPO_37 was killed at 100M steps while PPO_36 ran to 294M. We don't know whether PPO_37's collapses would have self-resolved with continued training. The project's own history shows that training dynamics include periodic collapses and recoveries (PPO_26's mid-run dip, PPO_35's script cycling).

**A proper ablation requires:** running both conditions to the same step count and replicating the effect (not a single pair of runs, which could differ by random seed).

### L-009: The experiment numbering implies linear, sequential investigation that didn't happen

Experiments 4, 5A/B/C, 6, 7, 8, and 9 were all launched before their predecessors completed. Specifically:

- Experiment 5C (PPO_35) was still running when Experiment 6 (PPO_36) was launched
- Experiment 6 was launched based on PPO_35's 64M-step behavior; PPO_35 later ran to 268M
- Experiment 7 (PPO_38) and Experiment 8 (PPO_39) launched before Experiment 6 produced interpretable results
- Experiment 9 (PPO_40) launched while Experiment 7 was still in its first few million steps

The numbering (4 → 5A → 5B → 5C → 6 → 7 → 8 → 9 → 10) creates an illusion of systematic, sequential investigation where each experiment informs the next. The actual process was parallel exploration with decisions made based on incomplete data from still-running experiments. This matters because it affects how the narrative is constructed: each experiment is written up as a response to the previous one's "findings," when in reality many were launched before those findings existed.

### L-010: Most training scripts have been deleted — the project is not reproducible

The git status shows nearly all `train_ppo*.py` files deleted from the working tree. Only `train_ppo36.py` remains at the project root. The documentation describes 43+ PPO runs with detailed configurations, hyperparameters, and phase transition rules, but a new contributor cannot reproduce any of them from the current state of the repository.

The deleted files still exist in git history (they can be recovered), but their absence from the working tree means:
- No one can verify that EXPERIMENTS.md accurately describes what each script did
- No one can re-run an experiment with identical configuration
- The documentation IS the project at this point — a narrative about experiments whose code no longer exists on disk

This is a portfolio project. Reproducibility is the difference between a portfolio piece and a journal entry.

### L-011: The "sticky actions conclusively don't work" conclusion cites Zhang et al. (2018) for the mechanism, not just the result

F-021 and RL_REFERENCE.md Lesson #39 correctly cite Zhang et al. (2018) for the empirical finding that sticky actions don't prevent memorization in deep ConvNet RL. The project independently confirmed this across 5 PPO models — that part is solid.

However, the project also adopts Zhang et al.'s mechanistic explanation ("CNNs are naturally noise-robust / automatically robust to sticky perturbations") and treats it as established fact rather than as one plausible interpretation. The project did not test this mechanism — it observed the same empirical result and adopted the upstream explanation. "CNNs are noise-robust" is Zhang et al.'s post-hoc interpretation of their result, not an independently verified mechanism.

This matters because the project's subsequent experimental direction (dynamics randomization rather than perceptual noise) is partly motivated by this mechanistic claim. If the mechanism is wrong, the direction might still be right (dynamics randomization could work for different reasons), but the reasoning chain has an unexamined link.

### L-012: Conceptual vocabulary creep — speculative categories are reified as established entities

The project has developed specialized terminology that treats interpretive judgments as objective categories:

| Term | What it describes | What it assumes (untested) |
|------|-------------------|---------------------------|
| **"Argmax-script + policy-entropy"** | det=True is single-score, det=False produces diverse scores | That det=False diversity comes from genuine state-conditioned action distributions rather than script-switching via stochastic sampling |
| **"Script diversification"** | Maintaining multiple scripts vs. one script | That the policy is sampling from discrete scripts rather than a continuous reactive distribution that happens to produce clusters |
| **"Dissolution"** | Top-3 concentration decreasing over time | That decreasing top-3 concentration represents scripts breaking down, rather than script portfolio rebalancing |
| **"CLUSTERED vs. CONTINUOUS"** | Binary shape classification from eval_reactivity.py | That threshold-based classification (>50% top-3 = CLUSTERED, <35% or >60% singleton = CONTINUOUS) captures a real underlying distinction |

The project's own F-001 lesson is instructive here: "a dead policy + sticky noise produces 8-14 unique scores." The score diversity was real but the interpretation ("GENERALIZING") was wrong. The same caution applies to det=False diversity: score diversity exists, but whether it represents "policy entropy" (state-conditioned action distributions) or "script-switching under stochastic sampling" is an interpretive claim that has not been independently verified.

**The only way to distinguish these is frame-level action analysis** — examining whether the policy chooses different actions when the ball is in different positions, controlling for frame count. The project acknowledges this ("a frame-level action analysis would be needed to settle this") but has never done it.

---

## MEDIUM — Methodological Weaknesses with Logical Implications

### L-013: The 20-game memorization check sample size is inadequate for the binary verdict system

The MemorizationCheckCallback uses `n_games=20`. With 20 games:

- A policy with 3 scripts sampled uniformly would produce 3 unique scores (verdict: MULTIPLE_SCRIPTS) in expectation, but would produce 2 or fewer unique scores (verdict: SINGLE_SCRIPT) with probability ~15% (coupon collector problem with small sample).
- The 95% confidence interval for unique score count from 20 games is wide — a policy with 10 true scripts could appear as 5-9 unique in any given 20-game batch.
- The verdict flips between SINGLE_SCRIPT and MULTIPLE_SCRIPTS at exactly 2 → 3 unique scores, which is the region of maximum sampling noise.

The project treats these verdicts as reliable per-check indicators (killing a run because of "268 consecutive SINGLE_SCRIPT" verdicts, tracking GENERALIZING streaks). But the sampling error at n=20 means some fraction of verdict flips are sampling artifacts, not policy changes.

### L-014: The eval_reactivity.py shape classifier uses uncalibrated, arbitrary thresholds

The `analyze_distribution()` function in `eval_reactivity.py` classifies distributions using hard thresholds:

```python
if top3_pct > 50 and singleton_ratio < 0.5:
    shape = "CLUSTERED (script-switching)"
elif top3_pct < 35 or singleton_ratio > 0.6:
    shape = "CONTINUOUS (ball-tracking)"
else:
    shape = "UNCLEAR"
```

**Problems:**
- The thresholds (50%, 35%, 0.5, 0.6) have no statistical justification. They weren't calibrated against known-reactive or known-memorized baselines.
- The decision boundary is asymmetric: >50% top-3 = CLUSTERED, <35% = CONTINUOUS, leaving a wide "UNCLEAR" band (35-50%) where no conclusion is drawn. But the documentation narratives almost always collapse UNCLEAR into the category that supports the current hypothesis (PPO_36 at 169M: 41% top-3 = "near-continuous").
- With n=100 games, the 95% CI on a 41% top-3 concentration spans roughly 31-51% — covering both the "CONTINUOUS" (<35%) and "CLUSTERED" (>50%) regions. The point estimate alone cannot support either classification.
- The "singleton_ratio" metric is mechanically correlated with unique score count — more unique scores → higher singleton ratio by construction. It adds little independent information beyond unique score count itself.

### L-015: The BrickCountingVecWrapper has a silent failure mode

`BrickRolloutCallback._on_step()` walks the env chain with `while hasattr(env, 'venv')` looking for `BrickCountingVecWrapper`. This:

1. Assumes a specific wrapping order (BrickCountingVecWrapper must wrap the VecFrameStack env)
2. Silently does nothing if the wrapper isn't found — no error, no warning, just zero brick counts logged
3. Would break silently if the env chain structure changes (e.g., a new wrapper is inserted between VecFrameStack and BrickCountingVecWrapper)

This is a minor code issue but a significant logical one: the project has a metric (bricks cleared per episode) that could stop being collected with no visible error, and conclusions that depend on it (tracking whether policies are actually clearing bricks vs. just surviving) would be silently undermined.

### L-016: EVALUATION_PROTOCOL.md and EXPERIMENTS.md have diverged

EVALUATION_PROTOCOL.md Part 4 requires:
- Bootstrap 95% CIs on all gold-standard evaluation metrics
- Clopper-Pearson binomial CIs on funnel rates
- Explicit caveats when zero-score rates differ substantially between comparison models
- Row-count verification before declaring evaluations complete

EXPERIMENTS.md reports:
- Point estimates only (no CIs on any metric)
- Funnel rates reported without binomial CIs (F-010, filed as MEDIUM and "fixable now," remains unfixed)
- Conditional stats (non-zero averages) reported without the mechanical-inflation caveat (F-017)

The protocol document and the actual reporting practice have diverged. The protocol exists as an aspiration, not as a constraint on what gets written.

---

## Cross-Cutting Pattern: Asymmetric Skepticism

The project has documented three confirmed false positives:

1. **PPO_26** (June): "both ingredients work" → nosticky revealed 60-pt memorized script
2. **PPO_30b/31b** (July 12): "33/33 GENERALIZING streak" → calibration showed dead policy = same signal
3. **PPO_35-mk1** (July 14): "first genuinely reactive model" → std=0.0 on every eval row; "21+ unique" was cross-checkpoint cycling

Each had the identical shape: **a promising number → written to memory/documentation as a finding → later falsified by a test that could have been run before the claim was written.**

The Breakthrough Verification Protocol was designed specifically to prevent this pattern. But the PPO_35-mk2 memory file (2026-07-19) — written AFTER the protocol was established — skipped Gates 2, 3, and 5. The protocol exists on paper but is not being followed for the claims that matter most.

**The underlying driver appears to be asymmetric skepticism:** the project is excellent at identifying flaws in old, abandoned hypotheses (FLAWS.md is genuinely good, and the post-hoc analyses of Experiments 1-3 are thorough and honest). But it relaxes its standards for the current leading hypothesis. This is a natural human tendency — it's harder to be skeptical of an idea you're actively invested in — but it's the exact thing the protocol was meant to prevent.

**The specific asymmetry:**
- When PPO_30b showed 14-19 unique scores with sticky=on, the project eventually asked "what does a dead policy produce?" and ran the calibration. Answer: 8-14 unique. Claim overturned.
- When PPO_35 shows 47% score retention under intervention, the project has NOT asked "what does a dead policy retain?" The calibration has not been run. The claim stands.

These are structurally identical situations. The calibration exists for one and not the other because one is the current hypothesis and the other is an abandoned one.

---

## What Holds Up Well

To be clear about what IS logically sound in this project:

1. **FLAWS.md is genuinely excellent.** The 21-entry catalog is honest, specific, and actionable. Many professional research projects never produce something this self-critical. Entries include severity ratings, whether they're fixable with existing data, and specific remedies.

2. **The nosticky verification protocol is correct and has been consistently applied.** Every sticky-trained model was tested without sticky actions. The collapses were documented. This protocol is the single most important methodological contribution of the project.

3. **The calibration methodology (F-001) is sound.** Running a known-dead policy through the same test to establish a noise baseline is exactly the right approach. It overturned the Experiment 3 conclusions. The failure is not in the methodology but in its inconsistent application (see L-001/L-002).

4. **The `if __name__ == "__main__"` guard on all training scripts** (Critical Rule #12) shows real attention to engineering hygiene.

5. **The design insight that dynamics randomization is the right lever** (RL_REFERENCE.md Lesson #40) has a clear mechanistic rationale: perturb what a script depends on (environment physics) rather than what CNNs are robust to (output noise). Whether the empirical evidence supports it yet is a separate question, but the reasoning is sound.

6. **EVALUATION_PROTOCOL.md is thorough.** If followed consistently, it would produce rigorous, statistically justified comparisons. The problem is the gap between the protocol and the practice (L-016).

7. **The documentation of failure is unusually honest.** Most projects bury their negative results. This project catalogs them, cross-references them, and uses them to constrain future interpretations. The existence of FLAWS.md, the post-hoc analyses in EXPERIMENTS.md, and the Breakthrough Verification Protocol all reflect genuine commitment to getting it right.

---

## The Central Tension

The project's explicit standards (FLAWS.md, Breakthrough Verification Protocol, EVALUATION_PROTOCOL.md) describe a level of rigor it doesn't consistently achieve. The gap is not in awareness — the project knows what good practice looks like. The gap is in **applying known standards to currently-favored hypotheses.**

The clean-slate question is: what would it take to bring the project's practices into alignment with its stated standards?

---

## References

- `FLAWS.md` — 21-entry methodological flaw catalog (complementary, not replaced by this document)
- `EXPERIMENTS.md` — Full experiment writeups (especially post-hoc analyses for Experiments 1-3)
- `CLAUDE.md` — Breakthrough Verification Protocol (gates 1-6), Critical Rules 1-12
- `EVALUATION_PROTOCOL.md` — Statistical standards that have diverged from reporting practice
- `eval_reactivity.py` — Shape classifier with uncalibrated thresholds (L-014)
- `eval_intervention.py` — Intervention test whose calibration baseline hasn't been run (L-001)
- `memorization_check_callback.py` — 20-game samples used for binary verdicts (L-013)
- Memory files: `ppo35-first-non-memorized-model.md`, `dynamics-randomization-forces-reactivity.md`, `ale-return-direction.md`
