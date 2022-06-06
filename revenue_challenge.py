#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluates a submission on the challenge server,
computing the expected revenue.
"""

import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str)
    parser.add_argument("--target", type=str, default=None)
    args = parser.parse_args()
    # read files
    submission = pd.read_csv(args.submission)
    target = pd.read_csv(args.target)
    # extract data
    submission = pd.merge(target[['pianist_id', 'segment_id', 'snippet_id']], submission, 'left')
    # calculate confusion matrix
    confusions = confusion_matrix(target.quadrant, submission.quadrant)
    # calculate revenue
    gain_matrix = np.array([
        [5, -5, -5, 2],
        [-5, 10, 2, -5],
        [-5, 2, 10, -5],
        [2, -5, -2, 5]])
    revenue = (confusions * gain_matrix).sum()
    print(revenue)
