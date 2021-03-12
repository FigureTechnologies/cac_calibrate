import unittest

import numpy as np

import cac_calibrate.binner as cb


class QbTester(unittest.TestCase):

    def test_transform(self):
        xs_tr = np.arange(-5, 6)
        xs_val = 10*xs_tr
        num_bins = 50
        qb = cb.QuantileBinner(num_bins=num_bins)
        qb.fit(xs_tr)
        xf = qb.fit_transform(xs_val)
        monotone = ((xf[1:] - xf[:-1]) >= 0).all()
        self.assertTrue(monotone)

    def test_transform_unsorted(self):
        num_bins = 5
        qb = cb.QuantileBinner(num_bins=num_bins)
        xs = np.array([9, 8, 7, 6, 5])
        qb.fit(xs)
        target = np.array([4, 3, 2, 1, 0])
        xf = qb.transform(xs)
        correct = (xf == target).all()
        self.assertTrue(correct)
