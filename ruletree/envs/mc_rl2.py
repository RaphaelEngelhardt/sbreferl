""" Create the Environment class for MountainCar RL2 (Reinforcement
learning  rule learning) experiments
Classes:
    MountainCarRL2
"""
from .envrl2 import EnvRL2
import gym
from stable_baselines3.common.monitor import Monitor


class MountainCarRL2(EnvRL2):
    def __init__(self):

        self.env_id = 'MountainCar-v0'
        self.goal = -110.
        self.obs_names = ["Car_position", "Car_velocity"]

        self.env = Monitor(gym.make(self.env_id))
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)

        self.is_solved = None

    def make_dict(self, s, a):
        return {"car_pos": s[0],
                "car_vel": s[1],
                "Decision": a}
