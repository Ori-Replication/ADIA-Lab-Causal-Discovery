#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module assess the performance of various discrete causal inference
methods on real-world discrete cause-effect pairs.
"""
from __future__ import division
import enum
import os
import sys
sys.path.append("..")

import numpy as np
import pandas as pd

from cloud import Cloud

def Cloud_print(method_name, score, llabel, rlabel):
    score.sort(key=lambda x: x[0])
    pred = score[0][1]
    if pred == "to":
        arrow = "⇒"
    elif pred == "gets":
        arrow = "⇐"
    elif pred == "indep":
        arrow = "⫫"
    elif pred == "confounder":
        arrow = "⇐  C ⇒"
    conf = abs(score[0][0] - score[1][0])
    out_str = "%s:: %s %s %s\t Δ=%.2f" % (method_name,
                                          llabel, arrow, rlabel, conf)
    print(out_str)
        

def test_abalone():
    # We do not discretise the variables just as what DR paper does
    print("evaluating on abalone dataset")
    abalone_dir = os.path.join(os.path.dirname(__file__), "data", "abalone")
    abalone_dat_path = os.path.join(abalone_dir, "abalone.dat")
    data = np.loadtxt(abalone_dat_path)
    ncols = data.shape[1]
    X = data[:, 0]

    colnames = ["sex", "length", "diameter", "height"]
    for i in range(1, ncols):
        Y = data[:, i]
        Cloud_score = Cloud(X, Y, n_candidates=4)
        Cloud_print("  Cloud", Cloud_score, colnames[0], colnames[i])


def test_nlschools():
    print("evaluating on nlschools dataset")
    abalone_dir = os.path.join(os.path.dirname(__file__), "data", "nlschools")
    abalone_dat_path = os.path.join(abalone_dir, "nlschools.dat")
    data = np.loadtxt(abalone_dat_path)
    X = data[:, 0]
    Y = data[:, 1]

    Cloud_score = Cloud(X, Y, n_candidates=4)
    Cloud_print("  Cloud", Cloud_score, "score", "status")


def test_horse_colic():
    print("evaluating on horse colic dataset")
    horse_dir = os.path.join(os.path.dirname(__file__), "data", "horse")

    def get_data(fname):
        horse_dat_path = os.path.join(horse_dir, fname)
        data = np.genfromtxt(horse_dat_path)
        abdomen = data[:, 17]
        surgical = data[:, 23]
        missing_indices = np.argwhere(np.isnan(abdomen))
        abdomen = np.delete(abdomen, missing_indices)
        surgical = np.delete(surgical, missing_indices)
        return abdomen, surgical

    abdomen_train, surgical_train = get_data("horse_train.dat")
    abdomen_test, surgical_test = get_data("horse_test.dat")
    abdomen = np.concatenate((abdomen_train, abdomen_test))
    surgical = np.concatenate((surgical_train, surgical_test))
    assert len(abdomen) == len(surgical)

    Cloud_score = Cloud(abdomen, surgical, n_candidates=4)
    Cloud_print("  Cloud", Cloud_score, "abdomen", "surgical")


if __name__ == "__main__":
    test_horse_colic()
    test_abalone()
    test_nlschools()
