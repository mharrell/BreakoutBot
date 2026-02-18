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