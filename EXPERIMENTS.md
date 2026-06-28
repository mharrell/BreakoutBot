# Experiments

## Experiment 1: Sticky Actions and Training Regime — PPO_26 vs PPO_27

> **Status update:** Both runs have now **completed**. PPO_26 finished at 1,001,828,352 total steps (eval peak 134.16 @ 905.6M). PPO_27 finished at ~867M steps with eval peak **147.02 @ 867.2M** — the new all-time eval record. Both runs' full 10,000-game single-env funnel baselines are **complete** — see the updated Results and Conclusions below, including a critical correction to the earlier "sticky actions eliminate zero-score" finding.

### Background

After PPO_25 reached 1 billion steps and an eval record of 140.94, two behavioral problems remained unresolved:

- **Dead balls:** After losing a life, the agent would frequently let subsequent balls pass through untouched, suggesting it had memorized paddle positions rather than learned to track the ball in real time.
- **Inconsistent tunnel exploitation:** The agent discovered the tunnel strategy (drilling through the brick wall to trap the ball behind it) emergently, but only completed it in roughly 3-5% of single-env games. It consistently carved multiple partial tunnels simultaneously rather than committing to one.

Both problems pointed to the same root cause: the agent was memorizing sequences of positions rather than developing a reactive ball-tracking policy. This is a known failure mode in Atari RL documented by Machado et al. (2018), who proposed **sticky actions** (`repeat_action_probability=0.25`) as the standard mitigation. With sticky actions enabled, 25% of steps repeat the previous action regardless of the agent's choice, making pure memorization unreliable and forcing more adaptive behavior.

The question was whether to apply sticky actions to the existing trained weights or start fresh. To answer this, two runs were launched simultaneously:

- **PPO_26:** Started from PPO_25's best_model (the 838M step checkpoint that achieved 140.94 eval), with sticky actions added. 64 parallel environments, same hyperparameters as PPO_25.
- **PPO_27:** Started from random initialization with sticky actions enabled from step one. 32 parallel environments (reduced to share hardware with PPO_26), otherwise same hyperparameters.

Both used `repeat_action_probability=0.25` in both training and eval environments.

---

### Configurations

**PPO_26 (train.py)**
```python
env = make_atari_env("ALE/Breakout-v5", n_envs=64, seed=None,
                     env_kwargs={"repeat_action_probability": 0.25})
# Started from PPO_25 best_model with reset_num_timesteps=True
model = PPO.load(PPO25_BEST_MODEL_PATH, env=env, device="cuda",
                 custom_objects={"n_envs": 64})
model.learning_rate = linear_schedule(2.5e-4, 1e-5)
model.clip_range = linear_schedule(0.2, 0.05)
model.ent_coef = 0.006
```

**PPO_27 (train_ppo27.py)**
```python
env = make_atari_env("ALE/Breakout-v5", n_envs=32, seed=None,
                     env_kwargs={"repeat_action_probability": 0.25})
# Fresh agent — no inherited weights
model = PPO(
    "CnnPolicy", env,
    n_steps=128, batch_size=1024, n_epochs=4, gamma=0.99,
    learning_rate=linear_schedule(2.5e-4, 1e-5),
    clip_range=linear_schedule(0.2, 0.05),
    ent_coef=0.006, vf_coef=0.5,
)
```

---

### Results

#### Eval Score Trajectory

| Milestone | PPO_26 (inherited weights) | PPO_27 (fresh agent) |
|-----------|---------------------------|----------------------|
| First eval above 80 | ~22M steps | ~129M steps |
| First eval above 90 | ~131M steps | ~169M steps |
| First eval above 100 | 304M steps | 299M steps |
| All-time eval peak | 134.16 @ 905.6M steps (run complete) | **147.02 @ 867.2M steps** 🏆 — new all-time record, surpasses PPO_25's 140.94 (run ongoing) |

PPO_26's inherited weights gave it a significant early advantage — it hit 80+ within 22M steps while PPO_27 took 129M. PPO_27 caught up and both crossed 100 at nearly identical step counts (~300M). **Since then, PPO_27 has pulled ahead decisively on eval score** — its 147.02 peak beats both PPO_26's final 134.16 and PPO_25's original 140.94. Note, however, that eval score and single-env score tell different stories here — see the updated Single-Env Watch Performance section below, where PPO_26 currently leads.

#### Convergence Speed

At equivalent wall-clock training time (~319,000 seconds elapsed for both), the runs stood at:
- PPO_26: 317M steps, rollout mean ~81
- PPO_27: 236M steps, rollout mean ~85

PPO_27 was producing higher rollout means despite fewer steps, because 32 envs run faster per step than 64. Normalizing by steps rather than time, PPO_26 held a small advantage in peak scores at that point — but this has since reversed (see above).

#### Training Stability

PPO_26 showed significantly wider oscillation throughout, with eval scores swinging between 47 and 113 in the 250-400M step range. In the most recent logged batch (889.6M–947.2M), PPO_26 continued this pattern, swinging from 71.50 to 134.16 — a wide range that includes both its new personal record and its batch low arriving just a few checkpoints apart. Notably, this batch **ends on a downward note** (71.50 at the final logged step), which is worth monitoring for a real regression vs. ordinary exploration noise.

PPO_27's oscillation has remained tighter and more consistent by comparison. In its most recent logged batch (798.4M–856.0M), PPO_27 averaged 112.3 against PPO_26's 104.2 over a comparable batch — and PPO_27's lowest score in that batch (88.34) is still well above PPO_26's batch low (71.50). This is visible in the approx_kl values from earlier in the runs — PPO_27 consistently showed lower KL divergence (0.007-0.020) vs PPO_26 (0.025-0.090).

#### Single-Env Watch Performance — Matched 10,000-Game Funnel Runs (Completed)

All three runs now have matched 10,000-game single-env evaluations using identical scripts (`funnel_recorder_ppo_25.py` / `funnel_recorder_ppo_26.py` / `funnel_recorder_ppo_27.py`), identical funnel threshold (400+), and sticky-action config per run.

| Metric | PPO_25 (no sticky, 10,000 games) | PPO_26 (inherited + sticky, 10,000 games) | PPO_27 (fresh + sticky, 10,000 games) |
|--------|----------------------------------|--------------------------------------------|---------------------------------------|
| Average score | 34.6 | **54.3** | 27.9 |
| Median score | 30.0 | **46.0** | 23.0 |
| Std dev | 39.5 | 46.4 | 32.2 |
| Min score | 0 | **5** | 0 |
| Zero-score games | 1,998 (20.0%) | **0 (0.0%)** | 2,127 (21.3%) |
| Best score | 406 | **415** | 406 |
| Funnel rate (400+) | 2/10,000 (0.02%) | **7/10,000 (0.07%)** | 1/10,000 (0.01%) |

**PPO_26 still beats PPO_25 on every single-env metric** — that finding holds. But the PPO_27 column tells a different and important story.

**Critical correction — PPO_27 did NOT inherit PPO_26's zero-score elimination.** PPO_27's zero-score rate (21.3%) is actually *worse* than PPO_25's (20.0%) — a higher failure rate than the no-sticky baseline. This definitively disproves the earlier hypothesis that "sticky actions eliminate the zero-score failure mode." Sticky actions alone do not fix it. The zero-score elimination in PPO_26 required a combination of factors: a deep non-sticky foundation (838M steps from PPO_25) *followed by* sticky-action refinement (~1B steps). PPO_27, trained entirely with sticky from scratch, lacks that foundation and performs the worst in single-env play.

**PPO_27's single-env story: highest eval (147.02), worst actual gameplay (avg 27.9, 21.3% zero-score).** The eval/single-env gap has never been wider. This strongly suggests fresh sticky-from-scratch training produces a policy that performs well under parallel eval sampling but collapses on sequential single-env execution — the opposite of what was hoped for.

**PPO_26 remains the clear single-env leader across all three runs** — highest average, highest median, highest floor, highest funnel rate, and the only model with zero zero-score games. Its recipe (~838M non-sticky + ~1B sticky) is the proven formula for actual gameplay quality.

---

### Investigation: The "Quick Death" Cluster and Direction-Correctness (Side Investigation)

Following up on PPO_26's much milder version of the zero-score pattern (no literal zeros, but a real cluster of very-low-score games with short, similar frame counts), a hunting script (`watch_quickdeath_ppo26.py`) was built to specifically capture games matching that signature, with per-frame RAM tracing (paddle x, ball x, ball y) and the raw action chosen each frame.

**Methodology note — RAM coordinate convention correction.** Initial analysis assumed `paddle_x`/`ball_x` of 0 and 191 meant "left wall" and "right wall" respectively. Empirical testing across the captured traces showed this was backwards relative to the action labels: `LEFT` consistently increases `paddle_x` (139 increases vs. 11 decreases across the sample) and `RIGHT` consistently decreases it. This doesn't indicate a bug — it just means the RAM convention runs opposite to naive intuition, and all directional analysis was corrected to account for it.

**Hypothesis tested:** that quick-death games show the paddle choosing the geometrically wrong direction (relative to the ball) more often than ordinary play — i.e., a "panic" effect.

**Result: not supported.** A direction-correctness metric (does the chosen LEFT/RIGHT action move the paddle toward the ball, given its pre-action position) was computed for the 6 captured quick-death games and for 40 unfiltered control games (`watch_controltrace_ppo26.py`):

| Sample | Direction-correct rate |
|--------|------------------------|
| Quick-death cluster (6 games) | 61.3% |
| Control — unfiltered (40 games) | 55.3% |
| Control — low score (<20 pts) | 57.1% |
| Control — mid score (20-49 pts) | 55.0% |
| Control — high score (50+ pts) | 55.2% |

Direction-correctness sits flat around 55% regardless of how well the game went, and the quick-death cluster actually scored *higher* on this metric than the control baseline — the opposite of the panic hypothesis. A velocity-aware version of the metric (predicting where the ball is heading rather than just its current position) was also tested and produced essentially the same result (54.6%), ruling out "the metric ignores velocity" as the explanation.

**Conclusion:** this specific diagnostic is inconclusive and not worth refining further. Whatever causes the milder quick-death cluster in PPO_26, it isn't captured by frame-level instantaneous direction-correctness. Given that PPO_26's actual headline result (zero zero-score games, better-than-PPO_25 on every other axis) is already a strong, settled finding independent of this investigation, this thread is closed for now rather than pursued further.

**Open question raised during this investigation, not yet tested:** would disabling sticky actions (`repeat_action_probability=0.0`) after training with them on for a substantial period improve precision without reintroducing the original positional-memorization problem? This is genuinely untestable from existing data — it requires running the experiment. It maps directly onto **Option B** in the Planned Next Steps below, and this investigation is a (weak, inconclusive) point in favor of trying it, since execution noise from sticky actions remains a plausible contributor to imprecision even though we couldn't isolate it cleanly here.

---

### Conclusions

**1. Inherited weights with sticky actions show diminishing returns on eval score, but the single-env story is more positive than that.**
PPO_26 recovered quickly early on (reaching 80+ in 22M steps) but this eval-score advantage eroded over training, and PPO_27 went on to set the all-time eval record. However, on single-env matched-script testing, PPO_26 is unambiguously the stronger model of the two trained so far — higher average, higher median, higher floor, more funnels, and a fully eliminated zero-score blind spot. Eval score and single-env score are not telling the same story here, which is itself a useful reminder (see RL_REFERENCE.md Lesson #13).

**2. Fresh training with sticky actions is more sample efficient for high performance — and now holds the outright eval record.**
PPO_27 broke 100 at 299M steps — nearly the same as PPO_26's 304M — despite starting from random weights. As of the latest evals, PPO_27 has gone on to set the new all-time eval record (147.02), surpassing both PPO_26 and the original PPO_25 baseline. **This resolves the open question from earlier in this experiment:** for reaching the highest performance tier, starting fresh with the correct environment configuration is now confirmed to outperform inheriting a billion steps of prior (mismatched) training, at least on eval score.

**3. Sticky actions alone do not eliminate the zero-score failure mode.**
PPO_27 (fresh + sticky, 867M steps) has a 21.3% zero-score rate — *worse* than PPO_25's 20.0% no-sticky baseline. The earlier conclusion that "sticky actions eliminate zero-score games" was based on PPO_26's result, which combined a deep non-sticky foundation (838M steps from PPO_25) with sticky refinement (~1B steps). PPO_27 definitively rules out the "sticky-only" explanation.

**4. SUPERSEDED — Sticky actions do not trade funnel rate for general consistency; PPO_26 wins on both.**
*(Original conclusion, kept for the record: "Sticky-action training improves general play consistency but reduces tunnel exploitation rate." This was based on a small, inconsistently-thresholded snapshot and is now contradicted by the full matched-script data above — PPO_26 has both a higher single-env average AND a higher funnel rate than PPO_25. There is no funnel-rate cost to sticky actions in this comparison.)*

**5. Peak eval scores favor PPO_27; single-env quality favors PPO_26 — resolved, PPO_27 does NOT catch up.**
PPO_27 holds the eval-score record outright (147.02). But on single-env sequential play, PPO_27 is the **worst** of the three models tested — avg 27.9 (vs PPO_26's 54.3, PPO_25's 34.6), zero-score rate 21.3% (vs PPO_26's 0%, PPO_25's 20.0%). The eval leader is not the best model for actual gameplay. This gap is now measured and confirmed at full 10,000-game sample size.

**6. The proven recipe for single-env quality: deep non-sticky foundation → sticky refinement.**
PPO_26's ~838M non-sticky + ~1B sticky steps produced better results than both PPO_25 (all non-sticky) and PPO_27 (all sticky) across every single-env metric. This strongly suggests sticky actions work best as a refinement layer applied to an already-competent policy, not as a primary training regime.

---

### Status

**Done:** PPO_25, PPO_26, and PPO_27 full 10,000-game matched-script funnel runs. Quick-death investigation (inconclusive, closed). All three-way single-env comparison data collected and analyzed.
**Pending:** PPO_28 and PPO_29 — the two-phase "remove sticky" experiments, testing whether disabling sticky actions after deep sticky training recovers funnel precision while preserving zero-score immunity.

---

### Caveats

This is not a perfectly controlled experiment. Confounding variables include:

- **Environment count:** PPO_26 used 64 envs, PPO_27 used 32. This affects both training dynamics and the effective batch size despite batch_size being scaled proportionally.
- **Starting point:** PPO_26 inherited not just PPO_25's weights but also its optimizer state and learning rate schedule, which were reset. The schedule restart may have affected early training dynamics.
- **Hardware sharing:** Both runs trained simultaneously on the same GPU, with each receiving roughly half the available compute. Neither was running at full speed.
- **System restarts:** Multiple unplanned interruptions throughout both runs may have affected continuity, including a recurring bug (now fixed) where `total_timesteps` was being treated as "train N more steps" on every restart rather than an absolute target.
- **Small quick-death sample:** The direction-correctness investigation above used only 6 quick-death games against 40 control games — both are small samples, and the inconclusive result could simply reflect insufficient data rather than a genuinely flat relationship.

Despite these limitations, the directional conclusions on training stability and single-env quality are consistent across multiple metrics and observation methods.

---



## Option D (Considered and Rejected): Process-Based Reward via Paddle Bounces + Episode Length

Raised as an alternative to score-based reward, on the reasoning that human skill training often benefits from rewarding correct *process* over a graded *outcome* metric (e.g. math pedagogy rewarding correct steps over just the final answer). The proposed implementation: reward successful paddle bounces and episode length instead of score, on the theory that these are "fundamental skills" that can't be metagamed.

**Rejected for two independent reasons:**

1. **Already tried, twice, and failed both times.** PPO_15 rewarded ball-tracking (a process proxy similar in spirit) and the agent learned to mirror the ball without scoring. PPO_18 rewarded paddle hits directly and collapsed to an eval score of 19.0. This is the same failure mode recurring, not new territory.
2. **Structurally unsound, not just under-tuned.** Both proposed metrics (bounces, episode length) can be maximized by a degenerate strategy that never engages with the actual task: bounce the ball straight up at 0°, park the paddle beneath it, and let it loop against the ceiling indefinitely. This clears a column or two of bricks incidentally on the way to settling into the loop, then accumulates large reward from survival time alone while doing nothing further toward the actual goal. There's no equivalent shortcut available for "bricks removed" — see commentary below.

**Resolved clarification — the project has actually been running a process-style reward all along.** `ClipRewardEnv` flattens every brick hit to exactly 1.0 regardless of point value, so every run (PPO_5–PPO_27) has trained on "bricks removed," not raw score — raw score has only ever been an eval/comparison readout, never the training signal. This is a useful contrast case: "bricks removed" is a process metric that has no cheap degenerate maximum (you cannot rack up brick-clear reward without clearing bricks), while "bounces + survival time" does. The lesson isn't "process rewards don't work" — it's that a process proxy is only safe if there's no way to satisfy it without doing the real task. Full writeup added to RL_REFERENCE.md Part 3 and Key Lessons Learned #22.

---

## Planned Next Steps — PPO_28 and PPO_29

**Trigger:** All three prior runs (PPO_25, PPO_26, PPO_27) have complete 10,000-game single-env funnel baselines. The three-way comparison is settled: PPO_26 (non-sticky foundation → sticky refinement) is the clear single-env leader; PPO_27 (all-sticky from scratch) is the worst.

**Open question:** Would disabling sticky actions after deep sticky training improve precision (funnel rate) without sacrificing the zero-score immunity or average score gains? Two experiments answer this in parallel:

| Run | Base Model | Total Steps Before | Intervention | Target |
|-----|-----------|-------------------|--------------|--------|
| **PPO_28** | PPO_26 best_model | 838M non-sticky + ~1B sticky | Remove sticky (`repeat_action_probability=0.0`) | +500M steps |
| **PPO_29** | PPO_27 best_model | ~867M sticky (from scratch) | Remove sticky (`repeat_action_probability=0.0`) | +500M steps |

Both use the corrected continuation pattern (`remaining = TARGET - model.num_timesteps`), dedicated training scripts (`train_ppo28.py` / `train_ppo29.py`), 32 envs each (sharing GPU), and eval at matched intervals.

### What this pairing reveals

| Outcome | Meaning |
|---------|---------|
| PPO_28 recovers funnel rate, keeps zero-score elimination | Two-phase (non-sticky → sticky → no-sticky) works; sticky can be safely removed after refinement |
| PPO_29 also recovers zero-score, improves average | Sticky-from-scratch was masking underlying competence — removal reveals it |
| PPO_29 stays high zero-score, low average | The damage from all-sticky training is structural, not just execution noise |
| Neither recovers funnel | Sticky permanently suppresses funnel precision regardless of training history |

**Deferred:** sticky-action intensity sweep, `net=[512,512]` revisit, and funnel-commitment reward shaping — any of these may become relevant depending on PPO_28/29 outcomes.

---

### Reference

- Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., & Bowling, M. (2018). *Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents.* Journal of Artificial Intelligence Research, 61, 523-562.
