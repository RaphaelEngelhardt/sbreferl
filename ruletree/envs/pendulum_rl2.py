""" Create the Environment class for Pendulum-v0 RL2 (Reinforcement
learning  rule learning) experiments

Classes:
    Pendulum
"""

from .envrl2 import EnvRL2
import gym
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import TimeLimit


class PendulumRL2(EnvRL2):
    def __init__(self):

        self.env_id = 'Pendulum-v0'
        self.goal = 90.
        self.obs_names = ["cos", "sin", "angular velocity"]

        self.env = Monitor(TimeLimit(gym.make(self.env_id)))
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)

        self.is_solved = None

    def make_dict(self, s, a):
        return {"cos_theta": s[0],
                "sin_theta": s[1],
                "angular_vel": s[2],
                "Decision": a[0]}

