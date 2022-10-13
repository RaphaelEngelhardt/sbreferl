""" Create the Environment class for CartPole RL2 (Reinforcement
learning rule learning) experiments
Classes:
    CartPoleRL2
"""
from .envrl2 import EnvRL2
import gym
from stable_baselines3.common.monitor import Monitor


class CartPoleRL2(EnvRL2):
    def __init__(self, version=0):

        if version != 1 and version != 0:
            raise ValueError("CartPole version unknown")
        else:
            self.version = version

        self.env_id = 'CartPole-v' + str(self.version)

        if self.version == 0:
            self.goal = 195.
        elif self.version == 1:
            self.goal = 475.
        else:
            raise NotImplementedError("No version of CartPole with this "
                                      "version number implemented")

        self.obs_names = ["Cart_position", "Cart_velocity", "Pole_angle",
                          "Pole_angular_velocity"]

        self.env = Monitor(gym.make(self.env_id))
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)

        self.is_solved = None

    def make_dict(self, s, a):
        return {"cart_pos": s[0],
                "cart_vel": s[1],
                "pole_angle": s[2],
                "pole_angular_vel": s[3],
                "Decision": a}
