"""
This module contains the classes of all available agents. Each class is
generally very thin and simply combines the constructors of the
environment and the algorithm
"""

import numpy as np
from .algorithms.dqnrl2 import DQNRL2
from .algorithms.pporl2 import PPORL2
from .algorithms.td3rl2 import TD3RL2

from .algorithms.cart import CART
from .algorithms.opct import OPCT

from .algorithms.cart_reg import CARTReg
from .algorithms.opct_reg import OPCTReg

from .algorithms.handcrafted import HandCraftedRL2

from .envs.mc_rl2 import MountainCarRL2
from .envs.mc_cont_rl2 import MountainCarContRL2
from .envs.cp_rl2 import CartPoleRL2
from .envs.pendulum_rl2 import PendulumRL2


# __________MOUNTAINCAR_________________________________________________
class MountainCarDQN(MountainCarRL2, DQNRL2):
    def __init__(self, seed=None, basename='MountainCarDQN', **model_kwargs):
        self.seed = seed
        self.basename = basename
        MountainCarRL2.__init__(self)
        DQNRL2.__init__(self, **model_kwargs)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_algo_seed(seed)


class MountainCarHandCrafted(MountainCarRL2, HandCraftedRL2):
    def __init__(self, seed=None, ruleset_id="MountainCar_advanced",
                 basename='MountainCarHandCrafted'):
        self.seed = seed
        self.basename = basename
        MountainCarRL2.__init__(self)
        HandCraftedRL2.__init__(self, ruleset_id)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)


class MountainCarCART(MountainCarRL2, CART):
    def __init__(self, seed=None, basename='MountainCarCART'):
        self.seed = seed
        self.basename = basename
        MountainCarRL2.__init__(self)
        CART.__init__(self)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)


class MountainCarOPCT(MountainCarRL2, OPCT):
    def __init__(self, seed=None, basename='MountainCarOPCT'):
        self.seed = seed
        self.basename = basename
        MountainCarRL2.__init__(self)
        OPCT.__init__(self)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)


# __________MOUNTAINCAR_CONT____________________________________________
class MountainCarContTD3(MountainCarContRL2, TD3RL2):
    def __init__(self, seed=None, basename='MountainCarContTD3', **model_kwargs):
        self.seed = seed
        self.basename = basename
        MountainCarContRL2.__init__(self)
        TD3RL2.__init__(self, **model_kwargs)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_algo_seed(seed)


class MountainCarContCARTReg(MountainCarContRL2, CARTReg):
    def __init__(self, seed=None, basename='MountainCarContCARTReg'):
        self.seed = seed
        self.basename = basename
        MountainCarContRL2.__init__(self)
        CARTReg.__init__(self)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)


class MountainCarContOPCTReg(MountainCarContRL2, OPCTReg):
    def __init__(self, seed=None, basename='MountainCarContOPCTReg'):
        self.seed = seed
        self.basename = basename
        MountainCarContRL2.__init__(self)
        OPCTReg.__init__(self)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)


class MountainCarContHC(MountainCarContRL2, HandCraftedRL2):
    def __init__(self, seed=None, ruleset_id="MountainCarContinuous",
                 basename='MountainCarContHandCrafted'):
        self.seed = seed
        self.basename = basename
        MountainCarContRL2.__init__(self)
        HandCraftedRL2.__init__(self, ruleset_id)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)


# __________CARTPOLE____________________________________________________
class CartPolePPO(CartPoleRL2, PPORL2):
    def __init__(self, seed=None, basename='CartPolePPO', **model_kwargs):
        self.seed = seed
        self.basename = basename
        CartPoleRL2.__init__(self)
        PPORL2.__init__(self, **model_kwargs)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_algo_seed(seed)


class CartPoleCART(CartPoleRL2, CART):
    def __init__(self, seed=None, basename='CartPoleCART'):
        self.seed = seed
        self.basename = basename
        CartPoleRL2.__init__(self)
        CART.__init__(self)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)


class CartPoleOPCT(CartPoleRL2, OPCT):
    def __init__(self, seed=None, basename='CartPoleOPCT'):
        self.seed = seed
        self.basename = basename
        CartPoleRL2.__init__(self)
        OPCT.__init__(self)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)


# __________CARTPOLE_NONMDP_____________________________________________
class CPEval:
    """This very shallow class just provides a modified version of
    run_eval_episode to account for the previous action, that all
    derived agent classes inherit
    """
    def run_eval_episode(self, sample, render=False, **kwargs):
        done = False
        s = self.env.reset()
        prev_a = self.env.action_space.sample()
        finite_horizon_return = 0.
        episode_samples_list = []

        while not done:

            # Include the previous action in the observation
            s = np.hstack((s, prev_a))
            a = self.select_eval_action(s)
            if sample:
                episode_samples_list.append(self.make_dict(s, a))
            if render:
                self.env.render()
            s, r, done, _ = self.env.step(a)

            finite_horizon_return += r
            prev_a = a

        return finite_horizon_return, episode_samples_list


class CartPoleNonMDP(CPEval, CartPoleRL2, HandCraftedRL2):
    def __init__(self, seed=None,  ruleset_id='CartPole_nonMDP',
                 basename='CartPoleNonMDPRL2'):
        self.seed = seed
        self.basename = basename
        CartPoleRL2.__init__(self)
        HandCraftedRL2.__init__(self, ruleset_id)

    def make_dict(self, s, a):
        return {"cart_pos": s[0],
                "cart_vel": s[1],
                "pole_angle": s[2],
                "pole_angular_vel": s[3],
                "Previous_action": s[4],
                "Decision": a}

    def set_seed(self, seed=None):
        self.set_env_seed(seed)


class CPNonMdpCART(CPEval, CartPoleRL2, CART):
    def __init__(self, seed=None, basename='CartPoleCART'):
        self.seed = seed
        self.basename = basename
        CartPoleRL2.__init__(self)
        CART.__init__(self)
        self.obs_names = ["Cart_position", "Cart_velocity", "Pole_angle",
                          "Pole_angular_velocity", "Previous_action"]

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)


class CPNonMdpOPCT(CPEval, CartPoleRL2, OPCT):
    def __init__(self, seed=None, basename='CartPoleOPCT'):
        self.seed = seed
        self.basename = basename
        CartPoleRL2.__init__(self)
        OPCT.__init__(self)
        self.obs_names = ["Cart_position", "Cart_velocity", "Pole_angle",
                          "Pole_angular_velocity", "Previous_action"]

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)
# ______END_CARTPOLE_NONMDP_____________________________________________


# ______Pendulum________________________________________________________
class PendulumTD3(PendulumRL2, TD3RL2):
    def __init__(self, seed=None, basename='PendulumTD3', **model_kwargs):
        self.seed = seed
        self.basename = basename
        PendulumRL2.__init__(self)
        TD3RL2.__init__(self, **model_kwargs)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_algo_seed(seed)


class PendulumCARTReg(PendulumRL2, CARTReg):
    def __init__(self, seed=None, basename='PendulumCARTReg'):
        self.seed = seed
        self.basename = basename
        PendulumRL2.__init__(self)
        CARTReg.__init__(self)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)


class PendulumOPCTReg(PendulumRL2, OPCTReg):
    def __init__(self, seed=None, basename='PendulumOPCTReg'):
        self.seed = seed
        self.basename = basename
        PendulumRL2.__init__(self)
        OPCTReg.__init__(self)

    def set_seed(self, seed=None):
        self.set_env_seed(seed)
        self.set_tree_seed(seed)


AGENTS = {"MountainCar-v0_HC": MountainCarHandCrafted(),
          "MountainCar-v0_DQN": MountainCarDQN(),
          "MountainCar-v0_CART": MountainCarCART(),
          "MountainCar-v0_OPCT": MountainCarOPCT(),
          "CartPole-v0_NonMdpHC": CartPoleNonMDP(),
          "CartPole-v0_NonMdpCART": CPNonMdpCART(),
          "CartPole-v0_NonMdpOPCT": CPNonMdpOPCT(),
          "CartPole-v0_PPO": CartPolePPO(),
          "CartPole-v0_CART": CartPoleCART(),
          "CartPole-v0_OPCT": CartPoleOPCT(),
          "MountainCarContinuous-v0_HC": MountainCarContHC(),
          "MountainCarContinuous-v0_TD3": MountainCarContTD3(),
          "MountainCarContinuous-v0_CART": MountainCarContCARTReg(),
          "MountainCarContinuous-v0_OPCT": MountainCarContOPCTReg(),
          "Pendulum-v0_TD3": PendulumTD3(),
          "Pendulum-v0_CART": PendulumCARTReg(),
          "Pendulum-v0_OPCT": PendulumOPCTReg(),
          }
