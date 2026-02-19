# Breakout RL Agent

A deep reinforcement learning agent trained to play Atari Breakout using PPO.

## Technologies
- Python
- Stable-Baselines3
- Gymnasium
- PyTorch
- CUDA

## How to Run
1. Install dependencies: `pip install stable-baselines3[extra] gymnasium[atari] ale-py autorom`
2. Accept ROM licenses: `autorom --accept-license`
3. Train: `python train.py`
4. Watch: `python watch.py`






## Experiments
- Run 1: Default params, peaked at ~30, entropy collapsed around 2.5M steps
- Run 2: ent_coef=0.01, clip_range=0.1, peaked at 26.5, unstable exploration
- Run 3: ent_coef=0.003, clip_range=0.2, in progress

