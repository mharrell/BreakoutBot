# .opencode/instructions.md — Required Manual Updates

The file `.opencode/instructions.md` is gitignored and cannot be committed.
Apply these changes manually to the original file at:
`C:\Users\Silver Pangolin\PycharmProjects\breakoutBot\.opencode\instructions.md`

---

## Change 1: Current Direction (line 7)

OLD:
> **We're returning to ALE.** After 43 PPO runs on a custom Breakout engine proved that dynamics randomization forces policy entropy (and the intervention test confirmed PPO_35 is genuinely sighted), we're validating against authentic Atari Breakout. The tool that makes this possible: `ALE.setRAM(addr, value)` ...

NEW:
> **We're returning to ALE.** After 43 PPO runs on a custom Breakout engine, the 2026-07-19 logical audit confirmed that the custom engine does not transfer to authentic Atari Breakout. PPO_35 scores 212 points (GymBreakout) vs 2 points (ALE) — a 99.1% drop. The intervention test does not distinguish reactive from dead (dead PPO_34 retains 49.6% vs PPO_35's 44.7%). The "sighted policy" claim is unsupported. The return to ALE is urgent: all post-Experiment-4 conclusions require ALE replication. The tool that makes this possible: `ALE.setRAM(addr, value)` ...

---

## Change 2: New Session Bootstrap (lines 17-20)

OLD:
> 1. Read `EXPERIMENTS.md` "Project Direction" section — understand current state
> 2. Read `FLAWS.md` — refresh awareness of active limitations
> 3. Check `models/*/checkpoint/` — newest checkpoint filenames give actual step counts
> 4. If evaluating a model, run the **Diagnostic Checklist** in `CLAUDE.md` before writing claims

NEW:
> 1. Read `CURRENT_STATE.md` — **start here.** Claim status board, model roster, what to trust/distrust. 5 minutes.
> 2. Read `LOGICAL_AUDIT.md` — 16 logical flaws, 3 confirmed with data. Understand the reasoning patterns to avoid.
> 3. Read `FLAWS.md` — 21 methodological flaws. Understand the measurement pitfalls.
> 4. Check `models/*/checkpoint/` — newest checkpoint filenames give actual step counts
> 5. If evaluating a model, run the **Diagnostic Checklist** in `CLAUDE.md` before writing claims

---

## Change 3: PPO_35 row in Live Run Status (line 33)

OLD:
> | PPO_35 | Continuous mid-game physics | 268M | Killed | det=False: 26 unique. Intervention: 47% retention — sighted |

NEW:
> | PPO_35 | Continuous mid-game physics | 268M | Killed | det=False: 21 unique (dead baseline: 19). Intervention: 44.7% retention (dead baseline: 49.6%). **NOT sighted — argmax script.** ALE cross-eval: 2 pts. See L-001, L-007. |

---

## Change 4: Key References section

ADD at the top:
> - `CURRENT_STATE.md` — **start here.** Claim status board, model roster, lessons learned, next steps.
> - `LOGICAL_AUDIT.md` — 16 logical flaws, 3 confirmed with data. Complements FLAWS.md.

---

## Change 5: File Map section

ADD after `CURRENT_STATE.md` entry and update existing entries:

| `CURRENT_STATE.md` | **Start here.** Definitive project status. |
| `LOGICAL_AUDIT.md` | Logical flaw catalog (complements FLAWS.md) |
| `calibration_phase1.py` | Dead-model calibration runner (intervention + reactivity) |
| `cross_eval_ale.py` | GymBreakout-to-ALE cross-evaluation |
| `eval_reactivity_bootstrap.py` | eval_reactivity.py + bootstrap CIs (use this version) |

UPDATE:
| `eval_intervention.py` | Teleportation-based reactivity test (uncalibrated — see L-001) |
| `gym_breakout.py` | Custom Breakout engine (historical — findings do not transfer to ALE) |
| `memorization_check_callback.py` | In-training behavioral monitoring (verdicts invalid for sticky models) |
| `recordings/PPO_*_memorization_track.csv` | Callback output (meaningless for custom-engine models) |

---

## Change 6: Known Misinterpretation Traps

REPLACE:
> - **Unique score count ≠ reactivity** — in a deterministic environment, a perfectly reactive policy produces 1 unique score with det=True. The intervention test is the correct diagnostic.

WITH:
> - **Unique score count ≠ reactivity** — in a deterministic environment, a perfectly reactive policy produces 1 unique score with det=True. Dead scripts also produce 1 unique score with det=True. Score diversity alone does not distinguish.
> - **Intervention test retention ≠ reactivity** — dead PPO_34 retains 49.6% under teleportation, indistinguishable from PPO_35's 44.7% (L-001). The test is uncalibrated. The only reliable behavioral test is nosticky verification (≤2 unique scores = memorized).

REPLACE:
> - **det=True single-script + det=False diverse** = expected for a policy with entropy in a deterministic MDP. Not a contradiction. The intervention test tells you whether the script is blind or sighted.

WITH:
> - **det=True single-script + det=False diverse** = ALSO the signature of a dead argmax script with residual policy entropy (PPO_34: 1 unique det=True, 19 unique det=False). This pattern is NOT evidence of reactivity. Dead baselines produce it.
