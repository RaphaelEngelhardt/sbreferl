"""
Provides an RL agent using OPCT (classification)
Needs package spyct, https://gitlab.com/TStepi/spyct

On first-time use
    pip install git+https://gitlab.com/TStepi/spyct.git
"""
import copy
import numpy as np
import pandas as pd
import spyct
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score


class OPCT:
    def __init__(self):
        np.random.seed(self.seed)
        self.tree = None
        self.is_trained = False
        self.is_solved = None
        self.enc = OneHotEncoder(categories=[np.arange(self.env.action_space.n)])
        self.enc_categories = None

    def select_eval_action(self, obs):
        """Returns the predicted action based on the observation"""
        action = self.tree.predict(obs.reshape(1, -1))
        action = int(self.enc_categories[np.argmax(action, axis=1)])

        return action

    def learn(self, samplesfile, threshold=None, nsamples=None,
              print_tree=True, **opct_kwargs):
        """
        Trains an OPCT as implemented by spyct
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
        :param opct_kwargs: all other kwargs are passed to spyct.Model
                            (max_depth, C etc.) random_state is
                            explicitly set to self.seed
        :return: dict {'cvs': 5-fold cross validation scores
                       'used_samples': number of samples effectively
                                       used after thresholding and
                                       limiting the number of samples
                                       (int),
                       'tree': a deepcopy of the decision tree object}
        """

        if isinstance(samplesfile, str):
            dataset = pd.read_feather(samplesfile)
            print(f"Loaded dataset with {len(dataset)} samples")
        else:
            dataset = samplesfile

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

        y_train_OH = self.enc.fit_transform(y).toarray()
        self.enc_categories = self.enc.categories_[0]

        # Fit the model
        # As default value for num_trees is 100 (which takes
        # significantly more time and results in decisions trees, which
        # are much harder to interpret), we set it to 1 if not otherwise
        # specified
        if "num_trees" not in opct_kwargs:
            print("num_trees not spefified, overriding the default value of 100 with 1")
            opct_kwargs["num_trees"] = 1
        self.tree = spyct.Model(**opct_kwargs, random_state=self.seed)
        self.tree.fit(X, y_train_OH)
        self.is_trained = True

        if print_tree:
            self.print_tree_pythonlike()

        cvs = cross_val_score(self.tree, X, y_train_OH, cv=5, scoring="r2")
        return {'cvs': cvs,
                'used_samples': len(dataset),
                'tree': copy.deepcopy(self)}

    def print_tree(self, inode=0):
        """
        print the sub-tree starting at node inode (default: 0 = root
        node)
        """
        tree = self.tree.trees[0]
        node = tree[inode]
        if node.left != -1:
            self.print_node_rule(inode, node, " < ")
            self.print_tree(node.left)
            space = (node.depth - 1) * "  |       "
            if node.depth != 0:
                space += "  |-------"
            print(f"{space}node={inode}, else: ")
            self.print_tree(node.right)
        else:
            pnum = np.argmax(np.asarray(node.prototype))
            space = (node.depth - 1) * "  |       "
            if node.depth != 0:
                space += "  |-------"
            print("{space}node={inode}, class: {proto} ({cls_name})".format(
                space=space,
                inode=inode,
                proto=pnum,
                cls_name=self.enc_categories[pnum])
            )

    def print_node_rule(self, inode, node, comp):
        space = (node.depth - 1) * "  |       "
        if node.depth != 0:
            space += "  |-------"
        print("{space}node={inode}, rule: {weights} * X {comp} {thresh:.4f}".format(
            space=space,
            inode=inode,
            weights=node.split_weights.to_ndarray()[0],
            comp=comp,
            thresh=node.threshold)
        )

    def print_tree_pythonlike(self, inode=0):
        """
        print the sub-tree starting at node inode (default: 0 = root
        node)
        """
        tree = self.tree.trees[0]
        node = tree[inode]
        if node.left != -1:
            self.print_node_rule_pythonlike(node, " < ")
            self.print_tree_pythonlike(node.left)
            space = (node.depth - 1) * "    "
            if node.depth != 0:
                space += "    "
            print(f"{space}else: ")
            self.print_tree_pythonlike(node.right)
        else:
            pnum = np.argmax(np.asarray(node.prototype))
            space = (node.depth - 1) * "    "
            if node.depth != 0:
                space += "    "
            print(f"{space}action = {pnum}")

    def print_node_rule_pythonlike(self, node, comp):
        space = (node.depth - 1) * "    "
        if node.depth != 0:
            space += "    "
        print("{space}if np.dot(np.array({weights}), obs) {comp} {thresh:.4f}:".format(
            space=space,
            weights=node.split_weights.to_ndarray()[0],
            comp=comp,
            thresh=node.threshold)
        )

    def set_tree_seed(self, seed):
        self.seed = seed
        np.random.seed(self.seed)
