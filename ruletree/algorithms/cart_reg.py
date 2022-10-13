"""Provides an RL agent using CART (regression)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import cross_val_score


class CARTReg:
    def __init__(self):
        np.random.seed(self.seed)
        self.tree = None
        self.is_trained = False
        self.is_solved = None

    def select_eval_action(self, obs):
        """Returns the predicted action based on the observation"""
        action = self.tree.predict(obs.reshape(1, -1))
        return action

    def learn(self, samplesfile, threshold=None, nsamples=None,
              print_tree=True, **dtc_kwargs):
        """
        Trains the Decision tree DecisionTreeRegressor as implemented
        by scikit-learn
        :param samplesfile: if string it will be interpreted as filename
                            containing the samples in feather format
                            otherwise it is assumed to be the pandas
                            DataFrame itself
        :param threshold: if not None only samples from episodes with
                          return of at least `threshold` will be
                          considered (default: None)
        :param nsamples: if not None limits the number of considered
                         samples to `nsamples` (not that this is applied
                         after thresholding) (default: None)
        :param print_tree: if True the induced tree is saved to
                           self.basename + '_tree.pdf' and printed in
                           text form (default: True)
        :param dtc_kwargs: all other kwargs are passed to
                           DecisionTreeRegressor (max_depth etc.)
                           random_state is explicitly set to self.seed
        :return: dict {'cvs': 5-fold cross validation scores,
                       'used_samples': number of samples effectively
                                       used after thresholding and
                                       limiting the number of samples
                                       (int),
                       'tree': tree in text form (str)}
        """

        if isinstance(samplesfile, str):
            dataset = pd.read_feather(samplesfile)
            print(f"Loaded dataset with {len(dataset)} samples")
        else:
            dataset = samplesfile
            print(f"Dataset with {len(dataset)} samples passed to method")

        print("Threshold and number of samples to chose: ",
              threshold, nsamples)
        if threshold is not None:
            dataset = dataset.loc[dataset['eps_return'] >= threshold]

        print(f"Dataset contains {len(dataset)} entries after applying "
              f"threshold.")

        if nsamples is not None:
            if nsamples > len(dataset):
                nsamples = len(dataset)
                print("More samples required than in dataset, clipping")
            dataset = dataset.sample(n=nsamples, random_state=self.seed)
        print(f"Dataset contains {len(dataset)} entries after sampling.")

        X = dataset.drop(columns=["eps_idx", "eps_len", "eps_return",
                                  "Decision"]).to_numpy()
        y = dataset["Decision"].to_numpy().reshape(-1, 1)

        # Fit the model
        self.tree = tree.DecisionTreeRegressor(**dtc_kwargs,
                                                random_state=self.seed)
        self.tree.fit(X, y)
        self.is_trained = True

        text_tree = tree.export_text(self.tree,
                                     feature_names=self.obs_names,
                                     max_depth=self.tree.get_depth(),
                                     decimals=10,
                                     show_weights=True)

        if print_tree:
            tree.plot_tree(self.tree, class_names=True)
            plt.savefig(self.basename + '_tree.pdf', bbox_inches='tight')
            plt.close()
            print(text_tree)

        cvs = cross_val_score(self.tree, X, y, cv=5)
        return {'cvs': cvs,
                'used_samples': len(dataset),
                'tree': text_tree}

    def set_tree_seed(self, seed):
        """Sets the seed of the CART agent"""
        self.seed = seed
        np.random.seed(self.seed)
