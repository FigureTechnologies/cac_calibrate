from typing import List, Union

import numpy as np
import pandas as pd
import ray
import sklearn.isotonic as si

import cac_calibrate.config as cfg


class IsotonicCalibrator:

    def __init__(self, score_name: str, target_name: str, num_bootstrap_samples: int):
        self.score_name = score_name
        self.target_name = target_name
        self.num_bootstrap_samples = num_bootstrap_samples
        self.models: List[si.IsotonicRegression] = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit multiple IsotonicRegressions in parallel.
        """
        models = ray.get([
            _ic_fit.remote(ic=self, df=df, seed=i)
            for i in range(self.num_bootstrap_samples)
        ])
        self.models = [m for m in models if m is not None]

    def transform(
        self,
        df: pd.DataFrame,
        mail_cost: float,
        conv_rate: float,
        quantiles: List[float] = None,
        num_rows_per_chunk: int = 1_000_000
    ) -> pd.DataFrame:
        """
        Compute cac statistics.

        Parameters
        ----------
        df: DataFrame
        mail_cost: float
        conv_rate: float
        quantiles (optional): List[float]
        num_rows_per_chunk: int
            Split df into len(df) // num_rows_per_chunk chunks, and transform in parallel.

        Returns
        -------
        DataFrame
        """
        if quantiles is None:
            quantiles = [100*q for q in cfg.quantile_default]

        df_split = np.array_split(df, len(df) // num_rows_per_chunk, axis=0)
        output = ray.get([
            _ic_transform.remote(
                ic=self,
                df=df_chunk,
                mail_cost=mail_cost,
                conv_rate=conv_rate,
                quantiles=quantiles
            )
            for df_chunk in df_split
        ])
        output = pd.concat(output, axis=0).reset_index()
        idx_cols = df.columns.difference(output.columns).to_list()
        output = pd.concat([df[idx_cols].reset_index(drop=True), output], axis=1)
        return output


@ray.remote
def _ic_fit(ic: IsotonicCalibrator, df: pd.DataFrame, seed: int) -> Union[si.IsotonicRegression, None]:
    """
    Factored out of IsotonicCalibrator for ray.

    Fit a single isotonic regression to df.
    """
    calibrator = si.IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    df = df[[ic.score_name, ic.target_name]].sample(frac=1.0, random_state=seed)
    X = df[ic.score_name].values.reshape(-1, 1)
    y = df[ic.target_name].values
    if sum(y) == 0:
        return None
    calibrator.fit(X, y)
    return calibrator


@ray.remote(num_cpus=4)
def _ic_transform(
    ic: IsotonicCalibrator,
    df: pd.DataFrame,
    mail_cost: float,
    conv_rate: float,
    quantiles: List[float]
) -> pd.DataFrame:
    """
    Factored out of IsotonicCalibrator for ray.

    For each IsotonicRegression in IsotonicCalibrator, compute calibrated
    calibrated probabilities. Then use the sample of calibrated probabilities
    to calculate cac statistics.
    """
    X = df[ic.score_name].values.reshape(-1, 1)
    prob_samples = np.concatenate(
        [model.transform(X).reshape(-1, 1) for model in ic.models],
        axis=1
    )
    output = _ic_compute_stats(
        prob_samples=prob_samples,
        mail_cost=mail_cost,
        conv_rate=conv_rate,
        quantiles=quantiles
    )
    return output


def _ic_compute_stats(
    prob_samples: np.array,
    mail_cost: float,
    conv_rate: float,
    quantiles: List[float]
) -> pd.DataFrame:
    """
    Factored out of IsotonicCalibrator for ray.

    Given an array of probabilities from each IsotonicRegression, calculate cac statistics.
    """
    def prob2cac(prob):
        return mail_cost/(prob*conv_rate)

    mean = prob_samples.mean(axis=1)
    se = prob_samples.std(axis=1)

    cac_samples = prob2cac(prob_samples)
    cac_mean = cac_samples.mean(axis=1)
    cac_se = cac_samples.std(axis=1)

    output = pd.DataFrame({"mean": mean, "se": se, "cac_mean": cac_mean, "cac_se": cac_se})

    quantiles = pd.DataFrame(
        np.percentile(cac_samples, q=quantiles, axis=1).T,
        columns=[f"cac_{q/100:4}" for q in quantiles]
    )

    output = pd.concat([output, quantiles], axis=1)
    output[output < 0] = np.inf
    return output
