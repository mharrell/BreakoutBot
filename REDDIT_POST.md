**Title:** [R] Three Lines of Code Fixed 123 Failed PPO Experiments on Atari Breakout

**Body:**

After 124 controlled PPO experiments on Atari Breakout, I found that every single model — across sticky actions, cursor wrappers, entropy tuning, dynamics randomization, adversarial bumpers, and everything else — converged to a memorized action sequence, not a reactive ball-tracking policy. The argmax was always a script.

The fix wasn't more environment engineering. It was three lines of reward shaping:

```python
distance = abs(paddle_x - ball_x)
bonus = 0.05 * max(0.0, 1.0 - distance / 80.0)
reward += bonus
```

Directly rewarding the paddle for being horizontally close to the ball during descent. A tiny bonus (0.05 per frame vs 1.0-7.0 per brick) that fires every frame the ball is descending. Applied during training only — eval is clean Breakout with no bonus. The behavior transfers.

**How I verified it (the split-watcher):** Run the same model on two different brick layouts side-by-side with independent predictions per side. Different bricks → different ball bounces → a reactive policy MUST move differently on each side. Compute Pearson correlation of paddle positions: px_corr > 0.99 = definitive memorization (physically impossible for reactive behavior). PPO_124 scored 0/240 perfect transfers and cleared every layout every game in the no-timing variant.

**Key finding:** Every prior approach tried to penalize scripts by making the environment harder to memorize. PPO always found a way around it — timing-robust scripts, layout-conditioned scripts, noise-tolerant scripts. The optimum was always a script; only the shape changed. Proximity reward changes what the optimum *is*. A center-hold script gets incidental bonus when the ball passes near center. A reactive tracker gets the maximum bonus on every descent frame. The optimization pressure is unambiguous: track the ball, get more reward.

**Results (clean eval, no proximity reward):**
- Split-watcher: 0/240 perfect transfers (120 no-timing, 120 with timing)
- No-timing ALT retention: 100% (clears every layout, every game)
- Intervention AUC: 0.421 (clean dose-response, peaks 60% reversal at 15px)
- Stoch best: 216 (highest on clean Breakout in the project)
- First model to sustain MULTIPLE_SCRIPTS on det=True without sticky masking

**Full writeup with code, reproducible training script, and verification tools:**
https://github.com/mharrell/breakout-reactive-ppo

**The messy history of all 123 failures:**
https://github.com/mharrell/BreakoutBot

---

*P.S. — I'm an independent researcher trying to get this posted on arXiv so it's citable. If you're an active arXiv author in cs.AI or cs.LG and found this useful, you can endorse me here (one click): https://arxiv.org/auth/endorse?x=MUM8BP — or email mikey.harrell@gmail.com*
