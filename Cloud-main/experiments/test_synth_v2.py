"""
This script is for experiment A (Reichenbach Problem) in our paper.
"""

from __future__ import division
import pprint
import math
import random
import sys
import time
import numpy as np
import pandas as pd
import json
from collections import defaultdict

sys.path.append("..")
from cloud import Cloud

random.seed(0)
np.random.seed(0)

def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f


def generate_sequence(dom_size: int, size: int):
    # generate X from multinomial disctibution
    p_nums = [np.random.random() for _ in range(dom_size)]
    p_vals = [v / sum(p_nums) for v in p_nums]
    X = np.random.choice(a=range(dom_size), p=p_vals, size=size)
    return X
    

def map_randomly(dom_f, img_f):
    f = dict((x, random.choice(img_f)) for x in dom_f)
    # ensure that f is not a constant function
    if len(set(f.values())) == 1:
        f = map_randomly(dom_f, img_f)
    assert len(set(f.values())) != 1
    return f

def test_accuracy_4types(args):
    nsim = 1000
    sample_sizes = [100, 1000, 10000, 100000]
    causal_types = ["to", "gets", "indep", "confounder"]

    img_f = range(args.img)
    dom_f = range(args.dom)

    print("%18s%18s%10s" % ("causal type", "sample_size", "Cloud"))
    print("-" * 80)
    sys.stdout.flush()
    fp = open("results/fixed-dim-acc-nsize.dat", "w")
    fp.write("%s\t%s\t%s\n" % ("causal_type", "size", "Cloud"))

    dd = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for sample_size in sample_sizes:
        for causal_type in causal_types:
            nc_Cloud = 0
            for i in range(nsim):
                if causal_type == "to":
                    X = generate_sequence(args.dom, sample_size)
                    f = map_randomly(dom_f, img_f)
                    N = generate_sequence(args.img, sample_size)
                    Y = [(f[x] + n) % args.img for x, n in zip(X, N)]

                elif causal_type == "gets":
                    Y = generate_sequence(args.img, sample_size)
                    g = map_randomly(img_f, dom_f)
                    N = generate_sequence(args.dom, sample_size)
                    X = [(g[y] + n) % args.dom for y, n in zip(Y, N)]

                elif causal_type == "indep":
                    X = generate_sequence(args.dom, sample_size)
                    Y = generate_sequence(args.img, sample_size)

                elif causal_type == "confounder":
                    C = generate_sequence(args.confounder, sample_size)
                    X = C // args.dom
                    Y = C % args.img

                assert len(X) == len(Y)

                sys.stdout.write("\r{}/{}".format(i+1, nsim))
                sys.stdout.flush()
                Cloud_score = Cloud(X, Y, n_candidates=4)
                Cloud_score.sort(key=lambda x: x[0])
                Cloud_score = Cloud_score[0][1]
                dd[causal_type][Cloud_score][sample_size] += 1
                if Cloud_score == causal_type:
                    nc_Cloud += 1

            pprint.pprint(dd)
            acc_Cloud = nc_Cloud * 100 / nsim
            print("%18s%18s%10.2f" % (causal_type, sample_size, acc_Cloud))
            sys.stdout.flush()
            fp.write(
                "%s\t%s\t%.2f\n" % (causal_type, sample_size, acc_Cloud)
            )
    with open("results/test_acc_4types.json", "w") as f:
        json.dump(dd, f)

    print("-"*80)
    sys.stdout.flush()
    fp.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="systhetic experiment")
    parser.add_argument("--dom", type=int, default=10, help="domain size")
    parser.add_argument("--img", type=int, default=10, help="image size")
    parser.add_argument("--confounder", type=int, default=100, help="domain size of confounder")
    args = parser.parse_args()

    test_accuracy_4types(args)


