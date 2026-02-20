# PPO Training Reference Guide
### Breakout RL Agent — Levers, Outputs & Experiment History

**Current Best: 36.6 rollout / 33.4 eval | Best Individual Game: 43**

---

## Part 1: Training Levers

These are the parameters you control in `train.py`. Each one affects a different aspect of how the agent learns. Key principle: **change one thing at a time** so you can interpret the results.

| Parameter | What It Does | Effect on Training |
|-----------|-------------|-------------------|
| `ent_coef` | Entropy coefficient — adds bonus reward for taking varied actions | Higher = more exploration, prevents early convergence. Too high = unstable training |
| `learning_rate` | How big each update step is | Lower = more stable but slower. Larger networks need lower rates. Too low = entropy collapse |
| `clip_range` | Limits how much the policy can change in one update | Lower = more conservative updates. Watch approx_kl for signs it's too high |
| `n_steps` | How many steps to collect before each update | Higher = more data per update, smoother gradients but slower iteration |
| `batch_size` | How much data to use per gradient update | Scale up proportionally when increasing n_envs |
| `n_epochs` | How many times to reuse collected data per update | Higher = more efficient data use but risk of overfitting |
| `gamma` | Discount factor — how much to value future vs immediate rewards | Higher = agent thinks further ahead. Lower = more focused on immediate rewards |
| `net_arch` | Size of fully connected layers after the CNN | Larger = more capacity but needs lower learning rate and more careful tuning |
| `vf_coef` | How much to weight value function loss vs policy loss | Higher = more focus on accurate reward prediction |
| `n_envs` | Number of parallel game environments | More = faster experience collection. Scale batch_size proportionally |
| `total_timesteps` | Total training duration | More = more time to learn. Can extend by restarting with checkpoint |

---

## Part 2: Training Outputs

| Output | What It Means | Healthy Range / What to Watch |
|--------|--------------|-------------------------------|
| `ep_rew_mean` | Average score per episode during training (noisy) | Should climb over time. Dips normal during exploration phases |
| `ep_len_mean` | Average game length in frames | Longer = agent keeping ball alive longer. Sudden drops signal a problem |
| `eval/mean_reward` | Average score during deterministic evaluation — most reliable metric | Your ground truth. Should trend upward. Plateau or decline = time to intervene |
| `fps` | Frames processed per second across all environments | Higher is better. Drops suggest GPU throttling or background processes |
| `entropy_loss` | How much the agent is exploring vs exploiting | Early: -1.0 to -1.5. Mid: -0.3 to -0.5. Late: -0.1 to -0.3. Near 0 = collapsed |
| `explained_variance` | How accurately the agent predicts future rewards | Closer to 1.0 is better. Below 0.5 = value function is struggling |
| `approx_kl` | How much the policy changed this update | Should stay below 0.05. Spikes above 0.1 = watch closely. Above 0.15 = concern |
| `clip_fraction` | How often PPO's safety clamp triggered | 0.05-0.15 is healthy. Consistently high = learning rate may be too aggressive |
| `loss` | Combined training loss | Should generally decrease over time. Spikes can indicate instability |
| `value_loss` | How wrong the reward predictions were | Should decrease over time. Climbing = agent seeing unfamiliar situations |
| `policy_gradient_loss` | How much the policy is being pushed to change | Should be small and stable. Large values indicate aggressive updates |
| `n_updates` | Total number of gradient updates performed | Useful for tracking how much training has happened. Confirms checkpoint loaded correctly |

### Healthy Entropy Loss by Training Stage

| Stage | Steps | Healthy Range | What It Means |
|-------|-------|--------------|---------------|
| Early | 0 - 500k | -1.0 to -1.5 | Agent exploring freely, not committed to any strategy |
| Mid | 500k - 2M | -0.3 to -0.5 | Narrowing down good strategies while still exploring |
| Late | 2M+ | -0.1 to -0.3 | Some confidence is fine as long as reward is still climbing |
| 🚨 Red flag | Any | Near 0.0 before 1M | Premature convergence — agent got confident too early |

---

## Part 3: Experiment History

All runs used ALE/Breakout-v5 with VecFrameStack (n_stack=4). Parameters not listed were kept at defaults.

| Run | TensorBoard | Key Parameters | Peak Reward | Notes |
|-----|-------------|----------------|-------------|-------|
| 1 | PPO_5 | Default: ent_coef=0.0, lr=2.5e-4, net=[64,64], n_envs=8 | ~30 | Entropy collapsed ~2M steps |
| 2 | PPO_6 | ent_coef=0.01, clip_range=0.1 | ~26 | Too much entropy, unstable, tanked at 2M |
| 3 | PPO_7 | ent_coef=0.003, clip_range=0.2 | ~31 | Best small-network run, declined after 2M |
| 4 | PPO_8 | Added net=[512,512] | ~22 | Large network underperformed — needs lower lr |
| 5 | PPO_9 | net=[512,512], lr=1.25e-4 | ~21 | Entropy still collapsed early |
| 6 | PPO_10 | net=[512,512], lr=1.25e-4, ent_coef=0.006 | ~25 | Better entropy but network still limiting |
| 7 | PPO_11 | Back to net=[64,64], ent_coef=0.006, lr=1.25e-4 | ~20 | lr too low for small network, entropy collapsed |
| 8 | PPO_12 | net=[64,64], n_envs=32, batch=1024, lr=1.25e-4 | ~6 | 32 envs + low lr = extreme entropy collapse |
| 9 | PPO_13 | net=[64,64], n_envs=32, batch=1024, lr=2.5e-4, ent_coef=0.006 | 36.6+ | Best run. Stable entropy, still training at 25M steps |

### Key Lessons Learned

1. **Larger networks are not always better** — net=[512,512] consistently underperformed net=[64,64] for Breakout with these hyperparameters.
2. **Learning rate must match network size** — large networks need lower learning rates, small networks need higher ones.
3. **More parallel environments requires proportionally larger batch size** — n_envs=32 with batch_size=256 caused extreme entropy collapse.
4. **Winning combination so far**: ent_coef=0.006, n_envs=32, batch_size=1024, lr=2.5e-4, net=[64,64].
5. **approx_kl spikes are normal** with this config — the run has been resilient and recovers each time.
6. **Extending total_timesteps by restarting from checkpoint is effective** — the model continues improving past the original target.

---

## Decision Framework

| What You See | What It Means | What To Do |
|-------------|--------------|-----------|
| Reward climbing steadily | Healthy training | Keep going, don't touch anything |
| Reward plateau, healthy entropy | May need more time | Wait it out, check again in 500k steps |
| Reward plateau, collapsed entropy | Premature convergence | Increase `ent_coef` or restart with new params |
| Reward declining after peak | Instability or collapse | Check if both rollout AND eval are declining. If yes, stop and adjust |
| `approx_kl` consistently above 0.1 | Updates too aggressive | Monitor closely. Above 0.15 = consider reducing learning rate |
| fps suddenly drops | Hardware throttling | Close background apps, check GPU temp with `nvidia-smi` |
| best_model.zip not updating | Eval hasn't beaten previous best | Check eval logs with `get_eval_logs.py` — may be normal variance |
| Both rollout and eval declining | Real regression | Stop training, diagnose before continuing |
| Rollout dips but eval holds | Exploration phase | Normal, wait it out |

---

## Useful Commands

```bash
# Check GPU status and temperature
nvidia-smi

# Monitor training in TensorBoard
tensorboard --logdir ./tensorboard/

# Parse evaluation log history
python get_eval_logs.py

# Watch trained agent play
python watch.py
```
