import numpy as np
import os

RUN_NAME = "PPO_30b"

project_root = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(project_root, "logs", RUN_NAME, "evaluations.npz")

print("Looking for:", log_path)
if os.path.exists(os.path.join(project_root, "logs")):
    print("Logs folder contents:", os.listdir(os.path.join(project_root, "logs")))
else:
    print("logs folder missing")

data = np.load(log_path)
timesteps = data["timesteps"]
results = data["results"]

for step, scores in zip(timesteps, results):
    mean_score = np.mean(scores)
    print(f"Step {step:>12,} | Mean Reward: {mean_score:.2f}")
