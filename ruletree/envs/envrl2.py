""" Create a high-level Environment class for RL2 (Reinforcement
learning rule learning) experiments
Classes:
    EnvRL2
"""
import numpy as np
import pandas as pd


class EnvRL2:
    """Class for general RL2 Environment

    ...
    Methods
    _______
    check_solved():
        Checks whether environment's solved condition is fulfilled, sets
        is_solved attribute and returns it accordingly

    evaluate(eval_episodes, sample, save_samples, render, **kwargs):
        Evaluates the trained agent

    run_eval_episode(sample, render **kwargs):
        Runs one single evaluation episode
    """

    def check_solved(self):
        """
        Checks whether the agent passes the solved-condition by
        performing 100 evaluation episodes and comparing to self.goal.
        Sets self.is_solved accordingly
        :return: bool True if agent passed the "solved-condition"
        """
        m, s, _ = self.evaluate(eval_episodes=100, sample=False, render=False)

        if m >= self.goal:
            neg = ''
            self.is_solved = True
        else:
            neg = ' not'
            self.is_solved = False

        print(f"{self.env_id} is{neg} considered solved with average reward of"
              f" {m} +- {s} over 100 consecutive trials (At least {self.goal}"
              f" required).")

        return self.is_solved

    def evaluate(self, eval_episodes=100, sample=False, save_samples=True,
                 render=False, **kwargs):
        """
        Evaluates the agent for a given number of episodes
        :param int eval_episodes: how many episodes should be performed
                                  (default: 100)
        :param bool sample: if samples should be taken (default: False)
        :param bool save_samples: if samples should be saved to a file
                                  in feather-format or just be returned
                                  (ignored if sample=False)
                                  (default: True)
        :param bool render: if it should be rendered (for visual
                            inspection) (default: False)
        :param kwargs: additional arguments passed to run_eval_episode
        :return: tuple np.mean(returns), np.std(returns), samples
                 Where
                 float np.mean(returns) average return of evaluation
                                        episodes
                 float np.std(returns) evaluation episodes' return's
                                       standard deviation
                 pandas DataFrame samples containing the collected
                                          samples
        """
        # Will contain a list of samples, each sample is its own dict
        # Reason for creating DataFrame from list of dicts:
        # https://stackoverflow.com/questions/57000903/what-is-the-fastest-and-most-efficient-way-to-append-rows-to-a-dataframe
        returns = np.zeros(eval_episodes)
        samples_list = []

        eps_idx = []  # will contain column of episodes' indices
        eps_len = []  # will contain column of episodes' lengths
        eps_fhr = []  # will contain column of episodes' returns

        for eval_episode in range(eval_episodes):

            fhr, episode_samples_list = self.run_eval_episode(sample, render,
                                                              **kwargs)

            # Append return and samples of last episode to list
            returns[eval_episode] = fhr
            samples_list.extend(episode_samples_list)

            # Extend the index, lengths and return columns of samples dataset
            n = len(episode_samples_list)
            eps_idx.append(np.full(shape=n, fill_value=eval_episode))
            eps_len.append(np.full(shape=n, fill_value=n))
            eps_fhr.append(np.full(shape=n, fill_value=fhr))



        self.env.close()

        if len(samples_list) != 0:  # if samples have been produced

            # make pandas DataFrame out of list of dicts
            samples = pd.DataFrame.from_dict(samples_list)
            del samples_list

            # insert the index, lengths and return columns
            eps_idx = np.concatenate(eps_idx)
            eps_len = np.concatenate(eps_len)
            eps_fhr = np.concatenate(eps_fhr)

            samples.insert(loc=0, column='eps_return', value=eps_fhr)
            samples.insert(loc=0, column='eps_len', value=eps_len)
            samples.insert(loc=0, column='eps_idx', value=eps_idx)

            if save_samples:
                samples.to_feather(self.basename + '.samples')
        else:
            samples = None

        print(f"Evaluation completed: <R> = {np.mean(returns)} +- {np.std(returns)}")
        return np.mean(returns), np.std(returns), samples

    def run_eval_episode(self, sample, render=False, **kwargs):
        """
        Performs exactly one evaluation episode
        :param bool sample: if True, samples state and action at each
                            timestep
        :param bool render: if True, episode will be rendered
        :param kwargs: kwargs passed to select_eval_action
        :returns: tuple finite_horizon_return, episode_samples_list
                  WHERE
                  float finite_horizon_return return of the episode
                  episode_samples_list list of samples (each a dict),
                                       empty if sample=False
        """
        done = False
        s = self.env.reset()
        finite_horizon_return = 0.

        episode_samples_list = []

        while not done:

            a = self.select_eval_action(s, **kwargs)
            if sample:
                episode_samples_list.append(self.make_dict(s, a))
            if render:
                self.env.render()
            s, r, done, _ = self.env.step(a)
            finite_horizon_return += r

        return finite_horizon_return, episode_samples_list

    def set_env_seed(self, seed):
        self.env.seed(seed)
        self.env.action_space.seed(seed)
