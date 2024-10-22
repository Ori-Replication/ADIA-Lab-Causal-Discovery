"""
This script is for experiment A (decision rate) in our paper.
"""

from __future__ import division
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

def test_decision_rate(args):
    nsim = 100
    sample_sizes = [200, 100, 1000]
    causal_types = ["to", "indep", "confounder"]

    img_f = range(args.img)
    dom_f = range(args.dom)

    print("%18s%18s%10s" % ("causal type", "sample_size", "Cloud"))
    print("-" * 80)
    sys.stdout.flush()

    for causal_type, sample_size in zip(causal_types, sample_sizes):
        df = {"flag": [],
              "conf": [],
             }
        for i in range(nsim):
            if causal_type == "to":
                X = generate_sequence(args.dom, sample_size)
                f = map_randomly(dom_f, img_f)
                N = generate_sequence(args.img, sample_size)
                Y = [(f[x] + n) % args.img for x, n in zip(X, N)]

            elif causal_type == "indep":
                X = generate_sequence(args.dom, sample_size)
                Y = generate_sequence(args.img, sample_size)


            elif causal_type == "confounder":
                C = generate_sequence(args.dom * args.img, sample_size)
                X = C // args.dom
                Y = C % args.img

            assert len(X) == len(Y)

            sys.stdout.write("\r{}/{}".format(i+1, nsim))
            sys.stdout.flush()
            Cloud_score = Cloud(X, Y)
            Cloud_score.sort(key=lambda x: x[0])
            pred = Cloud_score[0][1]
            conf = abs(Cloud_score[0][0] - Cloud_score[1][0])
            df["flag"].append(pred==causal_type)
            df["conf"].append(conf)

        sys.stdout.flush()

        df = pd.DataFrame(df)
        df.to_csv(f"results/test_decision_rate_{causal_type}.csv")

    print("-"*80)
    sys.stdout.flush()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="systhetic experiment")
    parser.add_argument("--dom", type=int, default=10, help="domain size")
    parser.add_argument("--img", type=int, default=10, help="image size")
    args = parser.parse_args()

    test_decision_rate(args)


