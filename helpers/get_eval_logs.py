import numpy as np
import os

RUN_NAME = "PPO_15"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(project_root, "logs", "evaluations.npz")

print("Looking for:", log_path)
print("Logs folder contents:", os.listdir(os.path.join(project_root, "logs")) if os.path.exists(os.path.join(project_root, "logs")) else "logs folder missing")

data = np.load(log_path)

timesteps = data["timesteps"]
results = data["results"]

for i, (step, scores) in enumerate(zip(timesteps, results)):
    mean_score = np.mean(scores)
    print(f"Step {step:>10,} | Mean Reward: {mean_score:.2f}")