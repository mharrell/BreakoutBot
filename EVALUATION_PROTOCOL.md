# Evaluation Protocol — BreakoutBot

Standardized procedures for running and interpreting model evaluations. Every evaluation in this project should follow this protocol. Deviations must be documented with rationale.

---

## Part 1: Evaluation Types

### 1A. Training Eval (EvalCallback)

- **Purpose:** Monitor training progress, save best_model
- **Config:** `n_envs=1`, `n_eval_episodes=50`, `deterministic=True`, `eval_freq=50_000`
- **What it measures:** Parallel-sampled clipped-reward performance
- **Caveat:** Different sampling regime than single-env sequential play. Scores are NOT comparable to funnel recorder results.
- **Do NOT use for:** Model comparisons, final performance claims

### 1B. Single-Env 10k-Game Funnel Evaluation (GOLD STANDARD)

- **Purpose:** Definitive model comparison
- **Config:** `n_envs=1`, persistent env, `seed=None`, `deterministic=True`, 10,000 games
- **Sticky setting:** Must match the model's training config at inference time
- **Script pattern:** `funnel_recorder_{RUN_NAME}.py`
- **Metrics collected:** Average, median, std dev, min, max, zero-score rate, funnel rate (400+), score distribution (percentiles)
- **Use for:** All model comparisons, final conclusions

### 1C. Sticky-Off Verification

- **Purpose:** Confirm that sticky-trained models collapse without sticky actions (Experiment 2 pattern)
- **Config:** 500 games, `sticky=False`, persistent env, `seed=None`, `deterministic=True`
- **Script pattern:** `funnel_recorder_{RUN_NAME}_nosticky.py`
- **Verdict:** ≤2 unique scores = collapse confirmed (memorized); ≥3 = unexpected (investigate)
- **Requirement:** MUST be run for EVERY Phase 2 (sticky-trained) model

### 1D. Memorization Check (MemorizationCheckCallback)

- **Purpose:** In-training behavioral monitoring (every 10M steps)
- **Config:** 20 games, fresh ALE per check, `deterministic=True`
- **Verdict:** ≤2 unique scores = MEMORIZED; ≥3 = GENERALIZING
- **⚠️ CALIBRATION GAP:** The ≤2 threshold was calibrated for non-sticky environments. For sticky-action models, see Part 2 below and FLAWS.md F-001.

### 1E. Deterministic vs. Stochastic Comparison

- **Purpose:** Measure how much variance comes from policy stochasticity vs. environment
- **Config:** 500 games deterministic + 500 games stochastic, same model, matched sticky setting
- **Script:** `eval_variance_test.py`
- **Use for:** Validating that `deterministic=True` is representative

### 1F. Sticky Probability Sweep

- **Purpose:** Test whether findings are specific to p=0.25
- **Config:** 500 games each at p ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25}
- **Script:** `sticky_probability_sweep.py`
- **Use for:** Testing generalizability of sticky-action conclusions

---

## Part 2: Pre-Evaluation Checklist

Before starting any evaluation:

- [ ] **Model path verified.** `os.path.exists(MODEL_PATH + ".zip")` returns True
- [ ] **Sticky setting confirmed.** `repeat_action_probability` matches the model's training config (check the training script if unsure)
- [ ] **Script uses correct evaluation type.** Gold standard = 1B (10k games), verification = 1C (500 games), sweep = 1F
- [ ] **Output paths don't collide.** If resuming, the script opens in append mode. If starting fresh, the log path doesn't already have data.
- [ ] **For sticky-model memorization checks:** Note that the GENERALIZING verdict is uncalibrated (FLAWS.md F-001). If `recordings/memorization_calibration.csv` exists, check the noise baseline before interpreting.

---

## Part 3: During Evaluation

- **Monitor progress.** The funnel recorder prints per-game output. Check that scores are varying and the game count is incrementing.
- **Log interruptions.** If the evaluation is stopped (crash, power loss, manual interrupt), note the game number reached.
- **Don't compute "final" stats mid-run.** Wait until 10,000 games complete, or explicitly label partial stats as interim.

---

## Part 4: Post-Evaluation Verification

### Required for ALL evaluations:

- [ ] **Verify row count.** The funnel log must have exactly N+1 rows (N data rows + 1 header):
  ```powershell
  (Get-Content recordings/RUN_NAME_funnel_log.csv | Measure-Object -Line).Lines
  ```
  Expected: 10,001 for 10k games, 501 for 500 games, etc. If not, the evaluation is INCOMPLETE.

### Required for Gold Standard (10k) evaluations:

- [ ] **Row count = 10,001** (10,000 data + 1 header)
- [ ] **Compute all standard metrics:** average, median, std dev, min, max, zero-score count/rate, funnel count/rate
- [ ] **Compute bootstrap 95% CIs** on: mean, median, zero-score rate, funnel rate
- [ ] **For funnel rates:** Report binomial (Clopper-Pearson) 95% CI. If count < 50, add caveat: "funnel rate comparisons are directional, not statistically significant at N=10,000"
- [ ] **Compute score distribution:** percentiles at P5, P10, P25, P50, P75, P90, P95, P99
- [ ] **If zero-score rates differ substantially between comparison models:** Caveat the "non-zero average" metric — it's mechanically inflated for the model with more zeros
- [ ] **Sticky-off verification completed** for this model (if sticky-trained)
- [ ] **Results transcribed into EXPERIMENTS.md** with all caveats

### Required for Sticky-Off Verification:

- [ ] **Row count = 501** (500 data + 1 header)
- [ ] **Unique score count computed**
- [ ] **Verdict recorded:** ≤2 = collapse confirmed, ≥3 = unexpected (document and investigate)
- [ ] **Results noted in EXPERIMENTS.md** Experiment 4 measurement protocol section

---

## Part 5: Comparison Rules

When comparing two models:

1. **Match evaluation types.** Only compare gold-standard (10k) against gold-standard. Do not compare training eval scores against funnel recorder scores.
2. **Check total step counts.** Models at different total steps are not directly comparable. Note the step count difference explicitly.
3. **Check n_envs.** Models trained with different n_envs (32 vs. 64) may have different training dynamics. Note the difference.
4. **Check LR/clip at phase transitions.** If these differ, the phase transition dynamics are not comparable (FLAWS.md F-006).
5. **Check sticky probability.** Only compare models at the same sticky probability unless doing a deliberate cross-probability analysis.
6. **Check evaluation completion.** Do not compare an incomplete evaluation (e.g., PPO_31b at 9,247 games) against a complete one without noting the shortfall.
7. **Report CIs, not just point estimates.** Especially for rare events (funnel rate) and rates (zero-score %).
8. **List all confounds.** Before writing "Model A beat Model B because of X," list every variable that differs between them beyond X. If the list has >1 item, the attribution is underdetermined (FLAWS.md F-002).

---

## Part 6: Statistical Reference

### Bootstrap Confidence Intervals (Recommended for means and medians)

```python
import numpy as np

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, alpha=0.05):
    """Compute bootstrap confidence interval for a statistic."""
    n = len(data)
    bootstrapped = np.array([
        statistic(np.random.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(bootstrapped, 100 * alpha / 2)
    upper = np.percentile(bootstrapped, 100 * (1 - alpha / 2))
    return lower, upper
```

### Binomial CI for Rates (e.g., zero-score rate, funnel rate)

Use Clopper-Pearson (exact) intervals:
```python
from scipy.stats import beta

def binomial_ci(count, n, alpha=0.05):
    """Clopper-Pearson binomial confidence interval."""
    lower = beta.ppf(alpha / 2, count, n - count + 1) if count > 0 else 0.0
    upper = beta.ppf(1 - alpha / 2, count + 1, n - count) if count < n else 1.0
    return lower, upper
```

### Fisher's Exact Test for 2×2 comparisons

```python
from scipy.stats import fisher_exact

# Example: zero-score rates for two models
# Model A: 2000 zeros, 8000 non-zeros
# Model B: 230 zeros, 9770 non-zeros
table = [[2000, 8000], [230, 9770]]
odds_ratio, p_value = fisher_exact(table)
```

### Minimum Detectable Effect at N=10,000

At 80% power, α=0.05, N=10,000 per group (two-sided):
- Zero-score rate: can detect ~1.5 percentage point difference when baseline is 5-25%
- Funnel rate: would need ~50,000 games per group to detect a 2× difference
- Mean difference: can detect ~1.5 score points (depends on variance, typically σ≈35-45 for these models)

---

## Part 7: Script Template

New funnel recorder scripts should follow this template:

```python
# Required header comment with:
# - Purpose of this evaluation
# - Model being evaluated
# - Sticky setting
# - Number of games
# - Which experiment/protocol item this satisfies

# Standard imports (copy from existing funnel recorder)

# Configuration block:
RUN_NAME = "..."
STICKY_ACTIONS = True/False
MODEL_PATH = f"models/{RUN_NAME}/final_model"
FUNNEL_THRESHOLD = 400
NUM_GAMES = 10000  # or 500 for verification
OUTPUT_DIR = "recordings"
LOG_PATH = os.path.join(OUTPUT_DIR, f"{RUN_NAME}_funnel_log.csv")

# Rest of the script follows the standard pattern:
# - Load model
# - Create env with correct sticky setting
# - Main loop (standard funnel recorder logic)
# - Final summary with all metrics
```

---

## References

- `FLAWS.md` — Full audit of known methodological limitations
- `EXPERIMENTS.md` — Experiment writeups and results
- `RL_REFERENCE.md` — Lessons learned, especially #23 (eval vs. single-env), #28-31 (memorization collapse), #32-35 (new methodology lessons)
- `calibrate_memorization_check.py` — Sticky-action noise baseline calibration
- `statistical_comparison.py` — Bootstrap CI and significance test utility
