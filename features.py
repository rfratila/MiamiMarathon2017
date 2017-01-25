import pandas as pd
import numpy as np

class BernoulliNB(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        count_sample = X.shape[0]
        # group by class
        separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
        # class prior
        self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
        # count of each word
        count = np.array([np.array(i).sum(axis=0) for i in separated]) + self.alpha

        smoothing = 2 * self.alpha
        # number of documents in each class + smoothing
        n_doc = np.array([len(i) + smoothing for i in separated])
        print(n_doc)