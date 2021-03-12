from functools import cached_property

import numpy as np
import pandas as pd


class QuantileBinner:
    """
    Do a quantile-based fit/transform using qcut from pandas. This class is designed to mimic
    the functionality found in the legacy calibrator.

    Parameters
    ----------
    num_bins: int
        Number of bins to use in qcut.
    """
    def __init__(self, num_bins: int):
        self.num_bins = num_bins

        self.buckets = None
        self._idx_col = "__bucket_idx__"
        self._value_col = "__bucket_value__"

    def fit(self, x: np.array):
        x = np.squeeze(x)
        buckets = pd.qcut(x, q=self.num_bins).unique()
        buckets = pd.DataFrame({
            self._idx_col: np.arange(len(buckets)),
            self._value_col: sorted([i.left for i in buckets])
        })
        buckets[self._value_col] = buckets[self._value_col].astype(np.float64)
        self.buckets = buckets

    @cached_property
    def _min_bucket_val(self) -> float:
        return self.buckets[self._value_col].min()

    @cached_property
    def _max_bucket_val(self) -> float:
        return self.buckets[self._value_col].max()

    def transform(self, x: np.array) -> np.array:
        x = np.squeeze(x).astype(np.float64)
        idx_sort = np.argsort(x)
        x = x[idx_sort]
        score_col = "score"
        df = pd.DataFrame(x, columns=[score_col])

        xf = pd.merge_asof(
            left=df,
            right=self.buckets,
            left_on=score_col,
            right_on=self._value_col,
            allow_exact_matches=False
        )
        xf.loc[xf[score_col] <= self._min_bucket_val, self._idx_col] = 0
        xf.loc[xf[score_col] >= self._max_bucket_val, self._idx_col] = self.num_bins - 1

        buckets_sorted = xf[self._idx_col].values.astype(int)
        idx_unsort = np.zeros_like(buckets_sorted)
        idx_unsort[idx_sort] = np.arange(len(idx_unsort))
        buckets_unsorted = buckets_sorted[idx_unsort]
        return buckets_unsorted

    def fit_transform(self, x: np.array) -> np.array:
        self.fit(x)
        return self.transform(x)
