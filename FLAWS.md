# Flaws in Experimental Process and Data Interpretation

This document catalogs known flaws in the BreakoutBot experimental program. Each entry includes severity, which conclusion it affects, whether it can be fixed with existing data or requires a rerun, and the recommended remedy.

**Severity scale:**
- **CRITICAL** — could invalidate a core conclusion of a completed experiment
- **HIGH** — undermines confidence in a specific claim or leaves a significant gap
- **MEDIUM** — methodological weakness that limits interpretability
- **LOW** — documentation error or minor overstatement

---

## CRITICAL

### F-001: MemorizationCheckCallback "GENERALIZING" verdict uncalibrated for sticky actions — **CONFIRMED with data (2026-07-14)**

**Status:** CONFIRMED. Calibration run on 2026-07-14 using PPO_30a/final_model (confirmed MEMORIZED — 2 unique scores in all non-sticky checks).

**Calibration results:**
- Non-sticky (p=0.0): 2.0 unique scores per 20-game batch (range 2-2) — confirms threshold
- Sticky (p=0.25): **11.3 unique scores** per 20-game batch (range 8-14, P95=14)
- A dead policy + sticky noise produces 8-14 unique scores — well above the old ≤2 threshold
- PPO_30b observed range: 10-19 unique. PPO_31b observed range: 10-19 unique.
- Both models' minimum unique-score counts (10) are within the noise P95 (14)

**Additional confirmation from sticky probability sweep (2026-07-14):**
- At p=0.05 (just 5% action-repeat probability), both PPO_30b and PPO_31b jump from 2 unique scores to 55-63 unique scores
- The "GENERALIZING" signal appears at the smallest possible noise level
- Unique-score count in sticky environments is purely a function of sticky probability, not policy reactivity

**Additional confirmation from nosticky verification (2026-07-14):**
- PPO_31b without sticky actions: 2 unique scores (29, 31). Games 2-500 all exactly 31.0 points in 178 frames — a single fixed script
- PPO_30b without sticky actions: 2 unique scores, 99.8% zero-score — a dead script
- Both models are confirmed MEMORIZED

**Conclusion:** The GENERALIZING verdict (≥3 unique scores) is NOT a reliable indicator of reactive behavior for sticky-action models. The ≤2 threshold is valid for non-sticky environments only. For sticky models, the calibrated threshold would need to be >14 unique scores (P95 of noise), and even then, the sticky probability sweep shows unique-score count is primarily a function of sticky probability, not policy quality.

**Calibration data:** `recordings/memorization_calibration.csv`
**Sweep data:** `recordings/sticky_sweep_results.csv`

---

### F-002: Pretraining duration and sticky-step count perfectly anti-correlated in Experiment 3

**Affected conclusion:** "Sticky steps drive right-tail performance; non-sticky pretraining suppresses catastrophic failure."

**Description:** PPO_30b = 100M pretrain + 300M sticky. PPO_31b = 300M pretrain + 100M sticky. These two variables are a perfect negative correlation at the 400M total budget. The data is equally consistent with a single-variable explanation: "more sticky training improves everything" or "less pretraining hurts the floor." You cannot distinguish "more sticky → better right tail" from "less pretraining → better right tail" because they are the same variable.

**Fixable with existing data?** Partially. The conclusion can be weakened to what the data actually supports: "At 400M total steps, allocating more budget to sticky training produced better right-tail performance than allocating more to non-sticky pretraining." Whether this reflects a general property of sticky training or is specific to the 400M budget is unknown. To fully resolve: extend PPO_31b past 400M to give it 200M+ sticky steps (Experiment 4 Option C).

**Remedy:** Reword the trade-off conclusion in EXPERIMENTS.md to state the confound explicitly. Add a "what this doesn't tell us" paragraph.

**References:** EXPERIMENTS.md "Outcome — A Trade-Off, Not a Winner", train_ppo30b.py, train_ppo31b.py

---

### F-003: PPO_26 confirmed MEMORIZED — "both ingredients" framework invalidated — **RESOLVED (2026-07-14)**

**Status:** RESOLVED. PPO_26 nosticky verification completed 2026-07-14 via `funnel_recorder_ppo_26_nosticky.py`.

**Results:** 28+ games completed (500-game run in progress). Game 1: 67 points, 303 frames. Games 2-28: **every game exactly 60.0 points, exactly 264 frames.** 1 unique score after game 1. PPO_26 plays a fixed 60-point, 264-frame script when sticky actions are removed.

**What this changes:**
- PPO_26's sticky-on performance (avg 54.3, 0% zero-score, 0.07% funnel — the best in the project) was noise-masked memorization, exactly like PPO_30b and PPO_31b.
- **No model in this project has ever genuinely generalized.** Every sticky-trained model tested without sticky actions has collapsed to a deterministic script.
- The "both ingredients" framework (deep non-sticky pretraining + sticky fine-tuning) does NOT build reactive policies. It builds better memorized scripts (PPO_26: 60 pts, PPO_31b: 31 pts, PPO_30b: 0 pts), but they're all still scripts.
- The ranking of script quality (PPO_26 > PPO_31b > PPO_30b) correlates with total non-sticky pretraining depth — which makes sense: more pretraining = more opportunities to discover a good script.
- The 1.8B-step recipe produces the best script but does not produce generalization.

**Implication for Experiment 4:** Low-sticky single-phase training (p=0.05 from scratch) is now the only untested path that might prevent memorization from forming in the first place. No post-hoc sticky fine-tuning recipe has ever cured it.

**Nosticky data:** `recordings/PPO_26_nosticky_funnel_log.csv` (in progress)

---

### F-004: PPO_31b 10k-game evaluation incomplete — results presented as final — **RESOLVED (2026-07-14)**

**Status:** RESOLVED. Evaluation completed on 2026-07-14 via `complete_ppo31b_funnel.py`.

**Completion results:** 10,000 games. Avg 22.2, median 20, 2.4% zero-score (238/10000), 0 funnels. The incomplete-data statistics (avg 22.1, 2.3% zero-score from 9,247 games) were within 0.1 points of the complete statistics — the error from reporting incomplete data was negligible in this case. However, the process failure (reporting as complete without verifying row count) stands.

**Lesson:** Always verify funnel log row count before declaring completion. `complete_ppo31b_funnel.py` now serves as a template for resuming interrupted evaluations.

---

## HIGH

### F-005: No sticky-off verification for PPO_31b — **RESOLVED (2026-07-14)**

**Status:** RESOLVED. `funnel_recorder_ppo_31b_nosticky.py` created and run on 2026-07-14.

**Results:** 500 games, sticky=off. 2 unique scores (29, 31). Games 2-500 all exactly 31.0 points, 178 frames — a single fixed script. **PPO_31b is confirmed MEMORIZED.** Matches PPO_28/29 collapse pattern exactly.

**Implication:** Combined with the calibration data (F-001), PPO_31b's apparent "generalization" in Phase 2 was sticky-action noise acting on a memorized policy, not genuine reactive behavior.

---

### F-006: LR restart and clip_range differ between Experiment 1 and Experiment 3

**Affected conclusion:** Any comparison between PPO_26 (Experiment 1) and PPO_30b/31b (Experiment 3).

**Description:** PPO_26 phase transition used LR restart at 2.5e-4→1e-5 and clip_range 0.2→0.05. PPO_30b/31b used LR restart at 1e-4→1e-5 and clip_range 0.15→0.05 (deliberately conservative, based on PPO_28/29 collapse findings). The phase-transition dynamics are not comparable — PPO_26's more aggressive restart may have given it more plasticity to adapt to sticky actions. The gap between PPO_26 and PPO_30b could be partially or fully explained by this difference rather than by pretraining duration.

**Fixable with existing data?** No — would require rerunning Experiment 3 with matching LR/clip values, or rerunning Experiment 1 with conservative values. Both are expensive.

**Remedy:** Document the LR/clip difference as a known confound in all PPO_26 vs. PPO_30b/31b comparisons. Note its directional effect: PPO_26's higher LR restart likely helped it adapt faster to sticky actions, so the PPO_26/PPO_30b gap may overstate the importance of pretraining depth.

**References:** train_ppo30b.py:109-111, train_ppo31b.py:109-111, EXPERIMENTS.md Experiment 1 PPO_26 config

---

### F-007: repeat_action_probability=0.25 never swept

**Affected conclusion:** That sticky actions are "permanently required at inference time" and that they "prevent memorization."

**Description:** The entire experimental program uses exactly one sticky probability: 0.25, adopted from Machado et al. (2018). Machado et al. chose this value for benchmark standardization across all Atari games, not because it's optimal for Breakout specifically. The central findings about sticky actions — that they're required permanently, that removing them causes collapse, that they enable generalization — may be specific to p=0.25. A lower probability (p=0.05 or p=0.10) might provide memorization resistance with less precision cost, potentially improving single-env performance. Experiment 4 Option A implicitly acknowledges this gap by proposing a low-sticky pretraining run.

**Fixable with existing data?** Partially. Existing models can be evaluated at different sticky probabilities without retraining (see `sticky_probability_sweep.py`). But the training-dynamics effects of different sticky probabilities would require new training runs.

**Remedy:** Run sticky probability sweep on existing models (B4 in the plan). For future experiments, consider sweeping p in at least one run.

**References:** Machado et al. (2018), EXPERIMENTS.md Experiment 4 Option A

---

### F-008: No non-sticky baseline at equivalent total steps (>1B)

**Affected conclusion:** That "both ingredients" (non-sticky pretraining + sticky actions) are necessary for good single-env performance.

**Description:** PPO_25 is the only long-running non-sticky model, trained to ~1B steps (20% zero-score). PPO_26 reached ~1.8B steps with both ingredients. There is no model trained for 1.8B non-sticky-only steps. It's possible that non-sticky training alone, given enough steps, eventually resolves the zero-score problem — PPO_25's trajectory was still improving when training stopped. The "both ingredients needed" conclusion compares models at different total step counts.

**Fixable with existing data?** No — would require training a non-sticky model to 1.8B steps or continuing PPO_25. Expensive (1-2 weeks GPU).

**Remedy:** Document this as a known gap. Note that PPO_25's eval curve was still rising at cutoff. Until a long-run non-sticky baseline exists, the "both ingredients" claim should be stated as "both ingredients dramatically accelerate reaching good single-env performance" rather than "both ingredients are necessary."

**References:** RL_REFERENCE.md Part 6 (PPO_25 entry), EXPERIMENTS.md Experiment 1

---

### F-009: "Both ingredients" hypothesis conflates total step count with training regime

**Affected conclusion:** Same as F-008.

**Description:** PPO_26 is the only model with large amounts of both non-sticky pretraining (~838M) and sticky training (~163M). It is also the only model with ~1.8B total steps. Every other comparison model has ≤1B total steps. The observation that PPO_26 performs best is equally consistent with "training for 1.8B steps helps" as with "both ingredients are needed." There is no 1B-step model with both ingredients, and no 1.8B-step model without both ingredients.

**Fixable with existing data?** No. Requires matched-total-step controls.

**Remedy:** Reframe the "both ingredients" claim to acknowledge the total-steps confound. The claim should be: "At matched ~1B total steps, the model with both ingredients (PPO_26, at its 1B-step mark) outperformed models with only non-sticky (PPO_25) or only sticky (PPO_27). Whether this advantage persists at matched higher step counts is untested."

**References:** EXPERIMENTS.md Experiment 1 conclusions, RL_REFERENCE.md Lesson #22

---

## MEDIUM

### F-010: Funnel rate comparisons statistically unreliable — **ADDRESSED (2026-07-27)**

**Affected conclusion:** Funnel rate rankings between models.

**Status:** ADDRESSED. Binomial 95% CI footnote added to EXPERIMENTS.md Experiment 1 funnel rate table (2026-07-27). All rates have overlapping CIs. Fisher's exact test for PPO_26 (7) vs. PPO_25 (2): p ≈ 0.18 — not significant. Note added: "Treat funnel rate differences of <5 events as directional indicators, not settled differences."

**Description (original):** Funnel rates (400+ point games) across all models: PPO_26 = 0.07% (7/10,000), PPO_25 = 0.02% (2/10,000), PPO_27 = 0.01% (1/10,000), PPO_30b = 0.00% (0/10,000), PPO_31b = 0.00% (0/10,000). With counts this low, 95% binomial confidence intervals overlap heavily. Fisher's exact test for PPO_26 (7) vs. PPO_25 (2): p ≈ 0.18 — not significant at the conventional threshold. The document treats funnel rate as a meaningful ordinal metric without acknowledging the statistical uncertainty.

**References:** EXPERIMENTS.md funnel rate comparisons

---

### F-011: PPO_27 evaluated mid-training vs. PPO_25/26 at completion

**Affected conclusion:** Three-way comparison between PPO_25, PPO_26, and PPO_27 on single-env metrics.

**Description:** PPO_27's 10k-game evaluation ran at ~880M steps while training was still ongoing. PPO_25 and PPO_26 were evaluated at their final checkpoints (~1B and ~1.8B steps respectively). If single-env performance continues to improve with training (which PPO_26's trajectory suggests), PPO_27's results may underestimate its final single-env capability. The comparison is between a mid-training snapshot and two completed runs.

**Fixable with existing data?** No — PPO_27 training ended at ~880M. A final-model evaluation at the true end of training would be needed.

**Remedy:** Add a caveat to the three-way comparison noting the timing difference.

**References:** EXPERIMENTS.md Experiment 1 single-env comparison table, RL_REFERENCE.md Part 6 (PPO_27 listed as "ongoing")

---

### F-012: MemorizationCheckCallback creates fresh ALE per check — different from persistent-env funnel evaluations — **ADDRESSED (2026-07-27)**

**Affected conclusion:** Comparability of memorization-check verdicts with 10k-game funnel results.

**Status:** ADDRESSED. Methodology note added to EXPERIMENTS.md Experiment 3 section (2026-07-27) documenting the fresh-env vs. persistent-env difference and warning that the callback may over- or under-estimate diversity relative to gold-standard funnel evaluations.

**Description (original):** `memorization_check_callback.py:84-86` creates a new ALE environment (`make_atari_env` with `seed=None`) on every check. This is "Approach B" from the Experiment 2 diagnostic — fresh env per check. The 10k-game funnel recorders use "Approach A" — persistent env with `env.reset()` between games. Experiment 2 showed these approaches produce different results for memorized models. The callback may systematically under- or over-estimate generalization relative to the gold-standard evaluation.

**References:** memorization_check_callback.py:84-86, EXPERIMENTS.md Experiment 2 seeding investigation

---

### F-013: Eval uses n_envs=1 with 50 sequential episodes — different sampling regime from training rollouts

**Affected conclusion:** Gap between eval score and rollout `ep_rew_mean`.

**Description:** Training callbacks use `EvalCallback` with `n_envs=1` and `n_eval_episodes=50` — 50 sequential episodes in one environment. Training rollouts use 32 parallel environments. The eval-vs-rollout gap documented throughout the project (e.g., PPO_30b rollout mean 60-71 vs. single-env avg 28-33) is partially a measurement artifact of different environment counts, not purely a property of the policy. Sequential episodes in one env allow ALE internal state to accumulate; parallel envs sample diverse states simultaneously.

**Fixable with existing data?** Partially. Could run side-by-side eval with n_envs=1 and n_envs=32 to quantify the gap. But changing the eval protocol would break comparability with historical results.

**Remedy:** Document the n_envs difference in the eval-vs-rollout gap discussion. Note that the gap has two components: (a) genuine policy behavior differences between parallel and sequential play, and (b) measurement artifact from different environment counts.

**References:** RL_REFERENCE.md Lesson #13, train_ppo30b.py:49-58

---

### F-014: deterministic=True in all evaluations — no stochastic-policy baseline — **PARTIALLY ADDRESSED (2026-07-27)**

**Affected conclusion:** That the observed score variance reflects genuine policy diversity.

**Status:** PARTIALLY ADDRESSED. The Experiment 3 Post-Hoc Variance Decomposition section in EXPERIMENTS.md now includes a det=True vs. det=False comparison for PPO_30b (July 2026), showing dramatically different results: det=True sticky=off → 2 unique scores (memorized collapse); det=False sticky=off → 43 unique scores (residual policy entropy). This confirms the flaw's concern was valid — deterministic sampling suppresses real policy diversity. However, the main evaluation pipeline (funnel recorder, eval callback) still uses `deterministic=True` exclusively. No det=False baseline exists in the standard evaluation protocol.

**Remaining work:** Add a `deterministic=False` option to the funnel recorder script and run a matched comparison on at least one model to quantify the gap in the standard evaluation framework.

**Description (original):** All funnel recorders and eval callbacks use `model.predict(obs, deterministic=True)`. This selects the argmax action at every step. For non-sticky models, this means zero randomness — same ALE state → same action → same outcome (variation comes only from ALE internal state drift between episodes). For sticky models, randomness comes from the environment (25% action-repeat probability), not from the policy. No evaluation has been run with `deterministic=False` to measure how much variance the policy's own action distribution would introduce. If the policy's action probabilities are nearly one-hot (entropy near zero), deterministic=True is representative. But we don't know that.

**References:** All funnel recorder scripts, memorization_check_callback.py:95

---

### F-015: n_envs varies across experiments (64 vs 32)

**Affected conclusion:** Cross-experiment comparisons, particularly PPO_26 (64 envs) vs. PPO_30b/31b (32 envs).

**Description:** Different environment counts mean different batch sizes, different ratios of exploration noise to gradient updates, and different effective horizons. The document acknowledges this as a caveat for Experiment 1 but doesn't adjust Experiment 3 interpretations when comparing against PPO_26. With 64 envs, each PPO update sees twice as many diverse trajectories as with 32 envs, which may affect exploration and convergence speed.

**Fixable with existing data?** No — would require rerunning experiments with matched n_envs.

**Remedy:** Add n_envs to all comparison tables. Note it as a confound when comparing across experiments. For future experiments, standardize on one n_envs value or make n_envs an explicit experimental variable.

**References:** train_ppo30b.py:40 (32 envs), EXPERIMENTS.md Experiment 1 (PPO_26: 64 envs)

---

### F-016: Phase 1 and Phase 2 data missing from EXPERIMENTS.md — **RESOLVED (2026-07-27)**

**Affected conclusion:** Completeness of the experimental record.

**Status:** RESOLVED. Verified 2026-07-27: PPO_31a Phase 1 data in EXPERIMENTS.md now covers all 30 unique checkpoints from 10M to 294M (CSV has 33 rows with 3 duplicates at 160M). PPO_31b Phase 2 data now covers all 10 checkpoints from 300M to 390M. Both tables match CSV ground truth. The data was transcribed at some point after this flaw was filed.

**Note on verification:** The convention established in the remedy has been applied retroactively — cross-checking EXPERIMENTS.md tables against raw CSV files. Add this to the session bootstrap checklist.

**Description (original):** (a) PPO_31a Phase 1 memorization track has 33 rows in the CSV (10M-294M) but the EXPERIMENTS.md table stops at 200M (20 rows). The final 100M steps are missing from the document. (b) PPO_31b Phase 2 memorization track has 10 rows (300M-390M) but zero appear in EXPERIMENTS.md. The CSV data exists on disk but was never transcribed.

**References:** `recordings/PPO_31a_memorization_track.csv`, `recordings/PPO_31b_memorization_track.csv`

---

## LOW

### F-017: "Non-zero average" mechanically inflated for high-zero-score models — **ADDRESSED (2026-07-27)**

**Affected conclusion:** Conditional stats comparison (PPO_30b non-zero avg 36.1 vs. PPO_31b non-zero avg 22.7).

**Status:** ADDRESSED. Caveat added to EXPERIMENTS.md Experiment 3 conditional stats table (2026-07-27) noting that PPO_30b excludes 23.2% of games vs. PPO_31b's 2.3%, mechanically inflating the conditional mean. Unconditional medians (21 vs. 20) are nearly identical.

**Description (original):** PPO_30b excludes 23.2% of its games (zeros) from the conditional average; PPO_31b excludes only 2.3%. PPO_30b's higher conditional average is partially a selection effect — it drops a much larger fraction of its worst games. The unconditional medians (21 vs. 20) show near-identical typical performance. The document presents the conditional stats without noting this mechanical inflation.

**References:** EXPERIMENTS.md Experiment 3 conditional stats table

---

### F-018: Direction-correctness conclusion from 6-game sample — **ADDRESSED (2026-07-27)**

**Affected conclusion:** That "frame-level instantaneous direction-correctness" is "not a reliable diagnostic."

**Status:** ADDRESSED. Statistical caveat added to EXPERIMENTS.md (2026-07-27) noting that the 6-game quick-death sample is too small for reliable inference — the 95% binomial CI around 61.3% could span anywhere from ~30%–85% depending on the (unrecorded) frame-level n. The conclusion was reworded from "this diagnostic doesn't work" to "this pilot data is too noisy to evaluate the diagnostic — a larger sample would be needed to settle the question."

**Description (original):** The quick-death cluster analysis used 6 games (61.3% direction-correct) vs. 40 control games (55.3%). A binomial 95% CI around 61.3% with n=6 spans roughly 22%-96% — the observed difference from 55.3% is consistent with sampling noise. The investigation's negative result could reflect insufficient data rather than a genuinely flat relationship. The document calls it "inconclusive" but also concludes the diagnostic "isn't worth refining" — a recommendation that assumes the null result is trustworthy.

**References:** EXPERIMENTS.md Experiment 1 direction-correctness investigation

---

### F-019: Claims about "GENERALIZING streak" inconsistently reported — **ADDRESSED (2026-07-27)**

**Affected conclusion:** Documentation accuracy.

**Status:** ADDRESSED. PPO_30b memorization track table in EXPERIMENTS.md now includes all 33 checkpoints (was 11). PPO_31b table includes all 10 checkpoints (was 0). The historical "12/12" for PPO_31b was a counting error — CSV confirms exactly 10 checks. The updated narrative for PPO_30b Phase 2 references the CSV as the authoritative source. All counts now standardized on CSV ground truth.

**Description (original):** Historical records (formerly in `.opencode/instructions.md`, now deleted) reported PPO_30b as "33/33 GENERALIZING" and PPO_31b as "12/12 GENERALIZING." The CSV data confirms 33 checks for PPO_30b (all GENERALIZING) and 10 checks for PPO_31b (all GENERALIZING — the "12/12" appears to include 2 extras or a counting error). EXPERIMENTS.md only shows 11 of PPO_30b's 33 checks and 0 of PPO_31b's 10. The numbers aren't reconciled across documents.

**References:** `recordings/PPO_30b_memorization_track.csv` (33 rows), `recordings/PPO_31b_memorization_track.csv` (10 rows). Note: `.opencode/instructions.md` no longer exists.

---

### F-020: Missing PPO_31a Phase 1 snapshot evaluations

**Affected conclusion:** Understanding of what non-sticky pretraining builds at different depths.

**Description:** PPO_31a trained for ~298M non-sticky steps. Checkpoints may exist at intermediate points (every 100k steps per the checkpoint callback). Evaluating these checkpoints with sticky-off funnel runs would reveal whether the pretraining is building progressively better policies or just cycling through different memorized scripts at different score levels. The memorization track data shows persistent cycling (same 2 scores), but the score levels of those scripts over time could illuminate what the pretraining is actually doing.

**Fixable with existing data?** Maybe — depends on whether intermediate checkpoints were preserved beyond the rolling latest_checkpoint.

**Remedy:** Check for preserved checkpoints. If available, run snapshot evaluations.

**References:** train_ppo31a.py, `recordings/PPO_31a_memorization_track.csv`

---

### F-021: Sticky actions independently confirmed ineffective for deep RL — Zhang et al. (2018)

**Severity:** N/A (validates project findings, not a flaw to fix)

**Description:** Zhang et al. (2018), "A Study on Overfitting in Deep Reinforcement Learning," published months after Machado et al. (2018), directly tested whether sticky actions prevent memorization in deep ConvNet RL agents. Their central finding: *"Stochasticity could neither prevent deep RL agents from serious overfitting nor detect overfitted agents effectively."* ConvNets are "automatically robust" to sticky perturbations — memorized action sequences still work through noise. This project independently confirmed Zhang et al.'s finding across 5 PPO models (PPO_26, PPO_27, PPO_28, PPO_29, PPO_30b, PPO_31b), discovering two distinct failure modes:

1. **Script memorization** (non-sticky pretrained models): Policy plays a fixed action sequence. Noise perturbs it into a score distribution, masking the memorization. Deeper pretraining → higher-scoring scripts (PPO_26: 60 pts > PPO_31b: 31 pts > PPO_30b: 0 pts).

2. **Noise coupling** (sticky-from-scratch): Policy never learns to track the ball — it learns action preferences that depend on p=0.25 noise to produce varied behavior. Without noise: PPO_27 dies in 19 frames (100% zeros).

Neither regime produces ball-tracking. The nosticky-verification protocol is the only reliable diagnostic. The field adopted sticky actions as standard evaluation; Zhang et al.'s warning appears to have been largely overlooked.

**Implication for Experiment 4:** p=0.05 from scratch may also fail — there may be no sticky probability that both prevents memorization AND avoids noise-coupling. But it remains the only untested path.

**References:** [Zhang et al. 2018](http://bengio.abracadoudou.com/cv/publications/pdf/zhang_2018_arxiv.pdf), [Machado et al. 2018](https://jair.org/index.php/jair/article/download/11182/26388), `recordings/PPO_*_nosticky_funnel_log.csv`

---

### F-022: make_check_env missing EpisodicLifeEnv — INCOMPLETE det=True false positive — **CONFIRMED and FIXED (2026-07-23)**

**Severity:** CRITICAL

**Affected conclusion:** "PPO_55b has no functional deterministic policy — 18+ consecutive INCOMPLETE det=True checks"

**Description:** The `make_check_env` function in all 14 active training scripts was missing `EpisodicLifeEnv` and `AutoResetWrapper` from the wrapper pipeline. The `MemorizationCheckCallback._run_episodes` method's custom autoreset logic requires EpisodicLifeEnv to detect intermediate life-loss events (`info[0]["lives"]`). Without it:
- `done[0]` fires only on actual game-over (all 5 lives lost)
- Callback's autoreset doesn't trigger correctly
- 0 games complete within `max_check_steps` (200,000 frames)
- Every check reports INCOMPLETE with games_played=0

This produced 18+ consecutive INCOMPLETE det=True verdicts for PPO_55b (and others), which were interpreted as evidence that the entropy-boosted model couldn't produce a deterministic FIRE action. A memory file was written claiming this was the first model to "break argmax memorization." The interpretation drove 3 days of experimental decisions.

**Secondary bugs in same code path:**
- Score accumulation (`_run_episodes`) read `info[0]["episode"]["r"]` at game-over — with EpisodicLifeEnv, this is only the last life's score, not the full game. Fixed to accumulate per-frame rewards.
- Entropy variant scripts (55a/b/c/d/e, 57b) had no `get_latest_checkpoint()` or resume logic — every restart reloaded from 9.6M source.

**Fix:** Added `EpisodicLifeEnv` and `AutoResetWrapper` to `make_check_env` in all scripts. Fixed score accumulation to sum per-frame rewards. Added resume logic to entropy variants. All fixes applied 2026-07-23.

**Lesson:** The env pipeline is part of the experimental apparatus. Wrapper ordering and presence matters. Every check env should be verified by inspecting at least one completed check with non-zero games_played before trusting anomalous signals.

**Status:** CONFIRMED and FIXED. All training scripts updated. See EXPERIMENTS.md 4b-ERRATA for full documentation.

---

### F-023: Entropy variant scripts lacked resume logic — silently restarted from source checkpoint

**Severity:** HIGH

**Affected conclusion:** Training progress tracking for all six entropy variants (PPO_55a/b/c/d/e, 57b)

**Description:** The entropy variant scripts were generated from `train_ppo_55.py` but the `get_latest_checkpoint()` function and resume-logic block were not included. On every restart (system reboot, manual stop/start, crash), the scripts reloaded from `SOURCE_CHECKPOINT_PATH/latest_checkpoint_9600000_steps.zip` instead of their own latest checkpoint. All training progress since the last save was silently discarded.

With checkpoint save_freq=100,000 (model steps) and 32 envs, each save is every 3.2M timesteps. A restart at step 38.4M that reloaded from 9.6M lost 28.8M steps of training.

**Fix:** Added `get_latest_checkpoint()` function and resume logic (check own CHECKPOINT_PATH first, fall back to source if none found, set fresh_start flag) to all six scripts.

**Detection difficulty:** HIGH. The scripts printed the checkpoint path on startup but didn't clearly indicate whether it was their own or the source. The step counter in TensorBoard would show the correct absolute step (since `reset_num_timesteps=False` preserves the counter from the loaded checkpoint), making it appear as if training progressed normally. Only by comparing checkpoint file modification times against expected training rate could the issue be detected.

**Lesson:** All training scripts should include resume logic as a standard component. A template or base class would prevent this class of bug. The `fresh_start` flag should be printed prominently on startup.

**Status:** FIXED. All entropy variant scripts updated 2026-07-23.

---

### F-024: MULTIPLE_SCRIPTS memcheck verdicts can be false positives from timing variance — **CONFIRMED (2026-07-30)**

**Severity:** CRITICAL

**Affected conclusion:** PPO_114 and PPO_115 have genuine argmax diversity (MULTIPLE_SCRIPTS on det=True)

**Description:** PPO_114 (31M) and PPO_115 (50M) both showed MULTIPLE_SCRIPTS on the memorization check callback — the first models in project history to do so without sticky actions. However, the split-watcher (independent-prediction variant) revealed both models produce identical paddle movement on different brick layouts (perfect transfer: px_corr > 0.99, ALT score ≈ FULL score). The MULTIPLE_SCRIPTS verdicts were caused by score variance from life-loss timing and ball-bounce stochasticity, not from the argmax adapting to different game states.

**Confirmation data:** `verify_split_watcher.py` batch run on 2026-07-30. PPO_114: 1/9 perfect transfer, FULL=436 script. PPO_115: 2/9 perfect transfer, FULL=420 script. All 7 cursor models confirmed memorized.

**Lesson:** The memorization check callback's 20-game sample and binary threshold (≤2 = SINGLE_SCRIPT, ≥3 = MULTIPLE_SCRIPTS) cannot distinguish "one script with timing variance" from "multiple scripts from genuine adaptation." A memorized sequence that scores differently depending on exactly when lives are lost produces different scores from a single deterministic argmax. The split-watcher is the definitive behavioral test.

**Status:** CONFIRMED. `verify_split_watcher.py` and `watch_model_split.py` now available as standard verification tools.

---

### F-025: The intervention probe and gradient measure distribution shifts, not argmax changes — **CONFIRMED (2026-07-30)**

**Severity:** CRITICAL

**Affected conclusion:** PPO_107 has "verifiable ball-tracking behavior" (33% reversal rate); cursor wrapper "taught the policy to attend to ball position"

**Description:** The intervention probe (`probe_107_intervention.py`) measures whether the action probability distribution shifts after ball teleportation. It does NOT measure whether the argmax action changes. A model with 90% confidence on its top action can show real probability shifts (the other 10% redistributes) without ever changing which action it takes — exactly what happened with PPO_107-117. The 33-50% reversal rates and AUC=0.329 were real distribution shifts, but the models are all confirmed memorized by split-watcher.

The dead baseline (center-hold = 0%) was correct — a hardcoded center-hold has no distribution to shift. The problem is the metric's ceiling for "memorized argmax + reactive distribution" was never established. PPO_116 (0/9 perfect transfer, 57% retention) fits this profile and should be used as a calibration baseline.

**Confirmation data:** All 7 cursor models showed positive intervention signals while confirmed memorized by split-watcher. PPO_115: intervention gradient AUC=0.329, "STRONG dose-response curve" — MEMORIZED (2/9 perfect transfer).

**Lesson:** The intervention probe should be renamed/reframed as a "distribution sensitivity" probe. Its results should be reported as "distribution shifted X% of the time" not "paddle reversed toward ball X% of the time." Positive intervention signals without split-watcher confirmation are NOT evidence of argmax reactivity.

**Status:** CONFIRMED. Critical Rules 17-18 added to CLAUDE.md. Intervention probe documentation updated.

---

### F-026: Perfect transfer (px_corr > 0.99 on altered layout) is definitive memorization — **CONFIRMED (2026-07-30)**

**Severity:** CRITICAL

**Affected conclusion:** Any future claim of argmax reactivity in Breakout

**Description:** The split-watcher with independent predictions runs the same model on two different brick layouts. If the paddle positions are perfectly correlated (px_corr > 0.99) AND ALT scores match FULL scores, the policy played identically on a different layout. This is physically impossible for a reactive policy: different bricks → different ball bounces → different tracking responses → different paddle positions. The only way to produce identical paddle movement on an altered layout is with a memorized sequence that ignores ball position entirely.

Every cursor model tested (PPO_111-117) showed at least one perfect-transfer game. The pattern is bimodal: either the script accidentally transfers (ALT ≈ FULL, px_corr ≈ 1.0) or it fails (ALT << FULL, px_corr low). This is the signature of memorization.

**Confirmation data:** PPO_111: 5/9 perfect transfer. PPO_114: 1/9. PPO_115: 2/9. PPO_116: 0/9 (scrambled cues variant — visual triggers never preserved). PPO_117: 2/9.

**Lesson:** Perfect transfer is a sufficient condition for memorization. A single game with px_corr > 0.99 and ALT ≈ FULL is definitive — no further testing needed. The absence of perfect transfer (PPO_116) does NOT mean reactive — it means the visual cues are never preserved, which is itself evidence of memorization (scrambled cues).

**Status:** CONFIRMED. Split-watcher is now the standard verification gate (CLAUDE.md Critical Rule #17).

---

### F-027: BeamRider is the first verified reactive PPO argmax — **FALSIFIED (2026-07-30)**

**Severity:** CRITICAL

**Affected conclusion:** The project's central claim that BeamRider's MULTIPLE_SCRIPTS represented genuine argmax reactivity, and that the "adversarial entities targeting agent position" mechanism forces reactive policies.

**Description:** BEAMRIDER_BASELINE (10M, SEED=206) was verified on July 28 as MULTIPLE_SCRIPTS (6 unique scores at noop=0, det=True, std=33.7) — the first model in project history to pass the definitive no-noop test. This was treated as verified reactive argmax and motivated the entire cursor-wrapper experiment series (PPO_107-117) as a port of the BeamRider mechanism to Breakout.

The independent-prediction split-watcher (`verify_beamrider_split.py`, July 30) revealed both BeamRider models are SINGLE_SCRIPT:
- BEAMRIDER_baseline: exactly 4200 points every game, std=0.0, unique=1 across 10 games with different noop offsets. Action divergence 78.6% (scrambled visual cues) but identical score outcomes.
- BEAMRIDER_MULTILIFE: exactly 2160 points every game, std=0.0, unique=1. 2/10 perfect transfer (ship_x_corr>0.99 on different enemy patterns).

The previous MULTIPLE_SCRIPTS verdicts were the same confound as Breakout (F-024): score variance from environment stochasticity (noop offset, enemy timing) was mistaken for argmax diversity. The ClipRewardEnv (reward clipping to [-1,1]) in the original verification pipeline may have amplified apparent score diversity by counting clipped-reward events rather than raw game points.

**Confirmation data:** `verify_beamrider_split.py` output. Two independent BeamRider instances, different noop RNG seeds, independent model predictions per side, det=True, 10 games each.

**Lesson:** The distribution-vs-argmax confound is universal — it affects Breakout AND BeamRider. The "adversarial threat forces reactivity" hypothesis is falsified: PPO learns to hedge (reactive distribution + script argmax) in BeamRider the same way it does in Breakout. PPO's objective maximizes expected return; when a fixed sequence is viable, the argmax converges to it regardless of game mechanics.

After 117 Breakout + 2 BeamRider experiments, no PPO model in this project's history has ever produced a verified reactive argmax.

**Status:** FALSIFIED. Both BeamRider models are memorized SINGLE_SCRIPT. The split-watcher is now the only trusted reactivity diagnostic for any Atari game.

---

## Summary Table

| ID | Severity | Affected Conclusion | Fixable Now? |
|----|----------|-------------------|--------------|
| F-001 | CRITICAL | Phase 2 models genuinely generalize | **CONFIRMED** — calibration data, nosticky collapse, and sticky sweep all confirm models are memorized |
| F-002 | CRITICAL | Trade-off framework (sticky vs pretrain) | Partially — weaken conclusion. Both models memorized; the "trade-off" is between which memorized script each learned |
| F-003 | CRITICAL | Sticky steps as performance driver | **RESOLVED (2026-07-14)** — PPO_26 nosticky: 60-pt script, 264 frames, confirmed MEMORIZED |
| F-004 | CRITICAL | PPO_31b final statistics | **RESOLVED** — eval completed 2026-07-14, stats unchanged |
| F-005 | HIGH | PPO_31b not memorized | **RESOLVED** — nosticky confirms collapse to 2 unique scores |
| F-006 | HIGH | PPO_26 vs PPO_30b/31b comparability | No — requires rerun |
| F-007 | HIGH | Sticky p=0.25 is the right value | Partially — sweep existing models |
| F-008 | HIGH | Both ingredients necessary | No — requires long training run |
| F-009 | HIGH | Both ingredients hypothesis | No — requires matched-step controls |
| F-010 | MEDIUM | Funnel rate rankings | **ADDRESSED 2026-07-27** — CIs and caveat added to EXPERIMENTS.md |
| F-011 | MEDIUM | Three-way PPO_25/26/27 comparison | No — PPO_27 training ended |
| F-012 | MEDIUM | Callback vs funnel comparability | **ADDRESSED 2026-07-27** — methodology note added to EXPERIMENTS.md |
| F-013 | MEDIUM | Eval-vs-rollout gap attribution | Partially — document artifact |
| F-014 | MEDIUM | Observed variance = policy diversity | **PARTIALLY ADDRESSED 2026-07-27** — det vs stoch data exists for PPO_30b; main eval pipeline still hardcodes det=True |
| F-015 | MEDIUM | Cross-experiment comparisons | No — requires rerun |
| F-016 | MEDIUM | Documentation completeness | **RESOLVED 2026-07-27** — data was already transcribed; verified against CSV |
| F-017 | LOW | Conditional stats comparison | **ADDRESSED 2026-07-27** — caveat added to EXPERIMENTS.md |
| F-018 | LOW | Direction-correctness diagnostic | **ADDRESSED 2026-07-27** — CIs and reworded conclusion added to EXPERIMENTS.md |
| F-019 | LOW | Streak counts consistency | **ADDRESSED 2026-07-27** — PPO_30b 33/33 entries added; PPO_31b count corrected to 10 |
| F-020 | LOW | Pretraining progression | Maybe — check checkpoints |
| F-021 | CONFIRMATORY | Sticky actions don't prevent memorization in deep RL | N/A — validates project findings. Independently confirms Zhang et al. (2018) |
| **F-022** | **CRITICAL** | **PPO_55b has no functional deterministic policy** | **CONFIRMED and FIXED — env pipeline bug. All scripts updated 2026-07-23.** |
| **F-023** | **HIGH** | **Entropy variant training progress was tracked correctly** | **FIXED — resume logic added to all six scripts 2026-07-23.** |
| **F-024** | **CRITICAL** | **PPO_114/115 have genuine argmax diversity (MULTIPLE_SCRIPTS)** | **CONFIRMED 2026-07-30 — split-watcher shows perfect transfer. Timing variance, not argmax diversity.** |
| **F-025** | **CRITICAL** | **PPO_107 has verifiable ball-tracking (33% reversal)** | **CONFIRMED 2026-07-30 — intervention probe measures distribution shifts, not argmax changes. All cursor models memorized.** |
| **F-026** | **CRITICAL** | **Any future claim of argmax reactivity in Breakout** | **CONFIRMED 2026-07-30 — perfect transfer on altered layout = definitive memorization. Split-watcher is standard gate.** |
| **F-027** | **CRITICAL** | **BeamRider is the first verified reactive PPO argmax** | **FALSIFIED 2026-07-30 — both BeamRider models SINGLE_SCRIPT (std=0.0, unique=1) under split-watcher. Same distribution-vs-argmax confound.** |
