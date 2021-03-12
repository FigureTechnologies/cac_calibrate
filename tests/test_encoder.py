"""
Ensure that scikit-learn encoders work the way I'd expect.
"""
import unittest

import numpy as np
import sklearn.preprocessing as sp


class EncoderTest(unittest.TestCase):

    def test_one_hot(self):
        x_tr = np.array([1, 2, 3]).reshape(-1, 1)
        ohe = sp.OneHotEncoder(categories="auto", drop="first", sparse=False)
        ohe.fit(x_tr)
        x_te = np.array([3, 2, 1]).reshape(-1, 1)
        output = ohe.transform(x_te)
        expected = np.array([[0, 1], [1, 0], [0, 0]])
        correct = (output == expected).all()
        self.assertTrue(correct)

    def test_quantile(self):
        x_tr = np.array([1, 2, 3]).reshape(-1, 1)
        binner = sp.KBinsDiscretizer(n_bins=2, encode="ordinal")
        binner.fit(x_tr)
        x_val = np.array([-100, 100]).reshape(-1, 1)
        binned = binner.transform(x_val).reshape(-1,)
        expected = np.array([0, 1])
        correct = (binned == expected).all()
        self.assertTrue(correct)
