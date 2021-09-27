from functools import partial
from typing import Tuple

import numpy as np
import pandas as pd

import cac_calibrate.core as cc

NUM_ROW = int(1e4)
SCORE_NAME = "score"
FEATURE_NAME = "feature"
TARGET_NAME = "y"
NUM_BINS = 2  # Should only need two bins, because the model for these tests is perfect
MAIL_COST = 1.0
CONV_RATE = 1.0
QUANTILES = [0.5]
FEATURE_CARD = 5


def make_data() -> Tuple[pd.DataFrame]:
    """
    Make data with scores from a perfect but poorly calibrated model.
    """
    np.random.seed(0)
    target = np.random.choice([0, 1], size=NUM_ROW)
    score = np.random.uniform(0, 1, size=NUM_ROW)
    score[target == 0] = 0
    feature = np.random.choice(range(FEATURE_CARD), size=NUM_ROW)
    data = pd.DataFrame({SCORE_NAME: score, FEATURE_NAME: feature, TARGET_NAME: target})
    cutoff = NUM_ROW // 2
    tr = data.loc[:cutoff]
    test = data.loc[cutoff:]
    return tr, test


def _test_cac(tr: pd.DataFrame, test: pd.DataFrame, use_feature: bool) -> None:
    """
    Given the data, the calibrator should assign lower cacs to positive cases and
    higher cacs to negative cases.
    """
    init_rc = partial(
        cc.RegressionCalibrator,
        score_name=SCORE_NAME,
        target_name=TARGET_NAME,
        num_bins=NUM_BINS
    )

    if use_feature:
        rc = init_rc(feature_cols=[FEATURE_NAME])
    else:
        rc = init_rc()

    rc.fit(tr)
    xf = rc.transform(test, mail_cost=MAIL_COST, conv_rate=CONV_RATE, quantiles=QUANTILES)
    by_actual = xf.groupby(TARGET_NAME)["cac"].median()
    negative_case_cac = by_actual.at[0]
    positive_case_cac = by_actual.at[1]
    assert negative_case_cac > positive_case_cac


def test_with_feature():
    tr, test = make_data()
    _test_cac(tr, test, True)


def test_without_feature():
    tr, test = make_data()
    _test_cac(tr, test, False)
