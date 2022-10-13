"""Trains a PPO agent for CartPole-v0 environment"""

import sys
import os
import pickle
import numpy as np

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path = list(set(sys.path))  # remove doublets

from ruletree.agents import CartPolePPO
from ruletree import utils

SEED = 41
DEVICE = 'cpu'
MODELFILE = "cp_ppo"

oracle = CartPolePPO(seed=SEED, device=DEVICE)
hist = oracle.learn(learning_timesteps=int(2e5))

# Print summary of training
n_eps = len(hist['episode_rewards'])
perf = np.mean(hist['episode_rewards'][-int(0.1 * n_eps):])
perf_std = np.std(hist['episode_rewards'][-int(0.1 * n_eps):])
print(f"Training of PPO agent CartPole completed after {hist['total_steps']} "
      f"total steps in {n_eps} episodes.\n"
      f"Average return in the last 10% of training episodes was {perf} +- "
      f"{perf_std}.")

# Save training history, model and plot of learning curve
with open(MODELFILE + "_learning_hist.pkl", "wb") as f:
    pickle.dump(hist, f)
oracle.save(MODELFILE)
utils.plot_learning(hist, outfile=MODELFILE + '_learning_plot.pdf')
