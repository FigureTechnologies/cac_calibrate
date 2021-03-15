from typing import List

import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import statsmodels.api as stm

import cac_calibrate.config as cfg

BUCKET_NAME = "score_bucket"


class RegressionCalibrator:  # pylint: disable=R0902

    def __init__(
        self,
        score_name: str,
        target_name: str,
        num_bins: int,
        feature_cols: List[str]
    ):
        """
        Convert scores to cac using binning/regression.

        Parameters
        ----------
        score_name: str
            Name of score to be calibrated.
        target_name: str
            Name of target.
        num_bins: int
            Number of bins to use for calibration.
        feature_cols: [str]
            Names of features for the calibration regression.
            NOTE: Currently, we expect all features to be categorical.
        """
        self.score_name = score_name
        self.target_name = target_name
        self.num_bins = num_bins
        self.feature_cols = feature_cols

        self.regr_cols = feature_cols + [BUCKET_NAME]

        self.fit_cols_required = self.feature_cols + [score_name, target_name]
        self.xf_cols_required = self.feature_cols + [score_name]

        self.binner = sp.KBinsDiscretizer(n_bins=num_bins, encode="ordinal")
        self.ohe = sp.OneHotEncoder(categories="auto", drop="first", sparse=False)
        self.calibrator = None

    @staticmethod
    def _validate_columns(df: pd.DataFrame, expected: List) -> None:
        diff = set(expected) - set(df.columns)
        if diff:
            raise ValueError(f"Columns {diff} missing from input DataFrame.")

    def fit(self, df: pd.DataFrame) -> None:
        """
        Parameters
        ----------
        df: DataFrame(
            columns=[self.feature_cols] + [self.score_name, self.target_name]
        )
        """
        self._validate_columns(df, self.fit_cols_required)
        df = df.copy()
        df = df.sort_values(by=self.score_name)

        y_bin_tr = df[self.score_name].values.reshape(-1, 1)
        self.binner.fit(y_bin_tr)
        df[BUCKET_NAME] = self.binner.transform(y_bin_tr)
        df_regr = (
            df.
            groupby(self.regr_cols, as_index=False, observed=True)[self.target_name].
            mean()
        )
        X = df_regr[self.regr_cols]
        X_hot = self.ohe.fit_transform(X)
        y = df_regr[self.target_name].values
        self.calibrator = stm.OLS(y, X_hot).fit()

    def transform(
        self,
        df: pd.DataFrame,
        mail_cost: float,
        conv_rate: float,
        quantiles: List[float] = None
    ) -> pd.DataFrame:
        """
        Provide cac estimates.

        Parameters
        ----------
        df: DataFrame(
            columns=self.feature_cols + [self.score_name] + ...
        )
            Note: This method will overwrite df's index.
        mail_cost: float
        conv_rate: float
        quantiles: [float]
            cac quantiles to return.

        Returns
        -------
        DataFrame(
            columns=df.columns.difference(self.feature_cols) + [cac columns]
        )
        """
        self._validate_columns(df, self.xf_cols_required)

        def prob2cac(prob):
            return mail_cost/(prob*conv_rate)

        if quantiles is None:
            quantiles = cfg.quantile_default

        df = df.sort_values(by=self.score_name).reset_index(drop=True)

        idx = df.drop(self.feature_cols, axis=1)

        df[BUCKET_NAME] = self.binner.transform(
            df[self.score_name].values.reshape(-1, 1)
        )
        X = df[self.regr_cols]
        X_hot = self.ohe.transform(X)
        preds = self.calibrator.get_prediction(X_hot)

        # mean and std error of cac
        moment_cols = ["mean", "mean_se"]
        cac_moment = pd.DataFrame(
            preds.summary_frame()[moment_cols].values,
            columns=moment_cols
        )
        moment_map = dict(zip(moment_cols, ["cac", "cac_se"]))
        for prob_label, cac_label in moment_map.items():
            cac_moment[cac_label] = prob2cac(cac_moment[prob_label])

        # cac quantiles
        cac_quantile = dict()
        calc_quantiles = [q for q in quantiles if q <= 0.5]
        for quantile in calc_quantiles:
            p_hat = preds.summary_frame(alpha=2*quantile)[["obs_ci_lower", "obs_ci_upper"]]
            # note: inentionally swap labels for upper/lower because we'll translate probs
            # to cacs later
            cac_quantile[f"cac_{quantile:.4}"] = p_hat["obs_ci_upper"]
            cac_quantile[f"cac_{1 - quantile:.4}"] = p_hat["obs_ci_lower"]
        cac_quantile = pd.DataFrame(cac_quantile)
        cac_quantile = cac_quantile[sorted(cac_quantile.columns)]
        cac_quantile = prob2cac(cac_quantile)

        cac = pd.concat([cac_moment, cac_quantile], axis=1)
        cac[cac < 0] = np.inf

        res = pd.concat((idx, cac), axis=1)
        return res
