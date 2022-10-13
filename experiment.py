"""Parametersweep for combination of environment, oracle and decision tree
induction
"""

import sys
import os
import argparse
import numpy as np
import psweep as ps
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path = list(set(sys.path))
from ruletree.agents import AGENTS


def do_exp(pset):

    LEN = len(oracle_samples[int(pset["seed"])])

    if (pset['nsamples'] is None) or (int(pset['nsamples']) <= LEN):
        n = pset["nsamples"]
    else:
        print(f"The number of samples was None or exceeds the available "
              f"number of samples in the samplefile, using all available"
              f" {LEN} samples.")
        n = LEN

    rule_extractor.set_seed(pset["seed"]+1)

    try:
        hist = rule_extractor.learn(oracle_samples[int(pset["seed"])],
                                    nsamples=n,
                                    threshold=pset["threshold"],
                                    print_tree=True,
                                    max_depth=pset["max_depth"])
    except ValueError:
        return {'oracle_mean': oracle_means[pset["seed"]],
                'oracle_std': oracle_stds[pset["seed"]],
                'oracle_samples': oracle_samples[pset["seed"]],
                'mean': None,
                'std': None,
                'tree_samples': None,
                'cvs': None,
                'used_samples': None,
                'tree': None}
    else:
        mean, std, tree_samples = rule_extractor.evaluate(sample=True)
        return {'oracle_mean': oracle_means[pset["seed"]],
                'oracle_std': oracle_stds[pset["seed"]],
                'oracle_samples': oracle_samples[pset["seed"]],
                'mean': mean,
                'std': std,
                'tree_samples': tree_samples,
                'cvs': hist['cvs'],
                'used_samples': hist['used_samples'],
                'tree': hist['tree']}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perform experiments for '
                                                 'extraction of decision '
                                                 'trees from RL oracles')
    parser.add_argument('--env', type=str,
                        help='OpenAI Gym environment id')
    parser.add_argument('--oracle', type=str,
                        help='Algorithm of the oracle')
    parser.add_argument('--dt', type=str,
                        help='Algorithm for the induction of the decision tree')
    parser.add_argument('--model', type=str, default=None,
                        help='File containing a pre-trained DRL agent')
    args = parser.parse_args()

    ENV = args.env
    print(ENV, type(ENV))
    ORACLE_ALGO = args.oracle
    TREE_ALGO = args.dt

    rule_extractor = AGENTS[ENV + "_" + TREE_ALGO]

    oracle_samples = {}
    oracle_means = {}
    oracle_stds = {}

    oracle = AGENTS[ENV + "_" + ORACLE_ALGO]
    if args.model is not None:
        oracle.load(args.model, device="cpu")

    SEEDS = [16, 17, 18, 19, 20]

    for i in SEEDS:
        oracle.set_seed(i)
        m, s, samples = oracle.evaluate(100, sample=True, render=False)
        oracle_means[i] = m
        oracle_stds[i] = s
        oracle_samples[i] = samples

    seed_range = ps.plist('seed', SEEDS)
    threshold_range = ps.plist('threshold', [None])
    nsamples_range = ps.plist('nsamples', [None])
    max_depth_range = ps.plist('max_depth',
                               np.arange(1, 10).astype(int).tolist())

    params = ps.pgrid(seed_range, threshold_range, nsamples_range,
                      max_depth_range)

    print(params)
    print(len(params))

    df = ps.run_local(do_exp, params, calc_dir=ENV + "_" + ORACLE_ALGO + "_" + TREE_ALGO)
    print(df)
