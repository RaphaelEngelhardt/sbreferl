"""Trains a TD3 agent for Pendulum-v0 environment"""

import sys
import os
import pickle
import gym
import numpy as np

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path = list(set(sys.path))  # remove doublets

from ruletree.agents import PendulumTD3
from stable_baselines3.common.noise import NormalActionNoise
from ruletree import utils

SEED = 41
DEVICE = 'cpu'
MODELFILE = "pend_td3"

env = gym.make("Pendulum-v0")
n_actions = env.action_space.shape[-1]
del env
noise = NormalActionNoise(mean=np.zeros(n_actions),
                          sigma=0.1*np.ones(n_actions))

oracle = PendulumTD3(seed=SEED, action_noise=noise, device=DEVICE,
                     verbose=1)
hist = oracle.learn(learning_timesteps=int(4e4))

# Print summary of training
n_eps = len(hist['episode_rewards'])
perf = np.mean(hist['episode_rewards'][-int(0.1 * n_eps):])
perf_std = np.std(hist['episode_rewards'][-int(0.1 * n_eps):])
print(f"Training of TD3 agent Pendulum completed after {hist['total_steps']} "
      f"total steps in {n_eps} episodes.\n"
      f"Average return in the last 10% of training episodes was {perf} +- "
      f"{perf_std}.")

# Save training history, model and plot of learning curve
with open(MODELFILE + "_learning_hist.pkl", "wb") as f:
    pickle.dump(hist, f)
oracle.save(MODELFILE)
utils.plot_learning(hist, outfile=MODELFILE + '_learning_plot.pdf')
