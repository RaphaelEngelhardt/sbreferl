"""Implements handcrafted rules for RL agents"""

class HandCraftedRL2():

    def __init__(self, ruleset_id):
        self.ruleset_id = ruleset_id
        self.is_solved = None


    def learn(self):
        print("This is a set of hand crafted fixed rule. There is nothing to "
              "learn here.")

    def select_eval_action(self, obs):
        if self.ruleset_id == 'MountainCar_simple':
            if obs[1] >= 0.:
                action = 2
            else:
                action = 0

        elif self.ruleset_id == 'MountainCar_advanced':
            if obs[1] == 0:  # only the initial state has v==0
                if obs[0] > -0.4887:
                    action = 0
                else:
                    action = 2  # push according to position
            else:  # in all other cases:
                if obs[1] > 0:
                    action = 2
                else:
                    action = 0  # push in direction of velocity v

        elif self.ruleset_id == 'MountainCarContinuous':
            if obs[1] == 0:  # only the initial state has v==0
                if obs[0] > -0.4887:
                    action = [-1.]
                else:
                    action = [1.]  # push according to position
            else:  # in all other cases:
                if obs[1] > 0:
                    action = [1.]
                else:
                    action = [-1.]  # push in direction of velocity v

        elif self.ruleset_id == 'CartPole_nonMDP':
            if abs(obs[2]) < 0.08:  # if angle small...
                if abs(obs[3]) < 0.4:  # ...and pole slow...
                    action = int(not obs[4])  # do the opposite of previous action (imitate "do nothing")
                else:  # ...if pole fast countersteer
                    if obs[3] > 0:
                        action = 1
                    else:
                        action = 0
            else:  # if angle large countersteer
                if obs[2] < 0.:
                    action = 0
                else:
                    action = 1

        elif self.ruleset_id == 'CartPole_nonMDP_simple':
            action = int(not obs[4])  # just do the opposite of previous action

        else:
            print("No such set of rules available (yet).")
            raise NotImplementedError

        return action

    def save(self, filename):
        print("This is a set of hand crafted fixed rule. There is no model to save.")

    def load(self, filename, device='auto'):
        print("This is a set of hand crafted fixed rule. There is no model to load.")
