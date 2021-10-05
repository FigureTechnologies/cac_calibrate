import numpy as np
import pandas as pd
import ray

import cac_calibrate as cc


QUANTILES = [0.025, 0.5, 0.975]


def compute_transform() -> pd.DataFrame:
    ray.shutdown()
    ray.init()
    np.random.seed(42)
    N = 500_000
    y = np.random.choice((0, 1), size=N, p=(0.95, 0.05))
    score = np.random.uniform(0, 1, size=N)
    score[y == 0] = np.random.uniform(1e-6, 1e-1, size=int(len(y) - sum(y)))
    score_name = "score"
    target_name = "y"
    df = pd.DataFrame({score_name: score, target_name: y})

    ic = cc.IsotonicCalibrator(score_name=score_name, target_name=target_name, num_bootstrap_samples=100)
    ic.fit(df)
    xf = ic.transform(df, mail_cost=1.0, conv_rate=1.0, quantiles=QUANTILES, num_rows_per_chunk=10_000)
    ray.shutdown()
    return xf


def test_monotone(xf: pd.DataFrame):
    df = xf[["score", "mean"]].copy()
    df = df.sort_values(by="score")
    monotone = (df["mean"].diff().iloc[1:] >= 0).all()
    assert monotone


def test_quantile(xf: pd.DataFrame):
    quantile_cols = [f"cac_{q:.3}" for q in QUANTILES]
    for lower, upper in zip(quantile_cols[:-1], quantile_cols[1:]):
        differences_exist = (xf[lower] != xf[upper]).any()
        assert differences_exist
        order_correct = (xf[lower] <= xf[upper]).all()
        assert order_correct
