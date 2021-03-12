import os
from os.path import join
from pathlib import Path
from typing import Tuple
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame
import statsmodels.api as stm

from data_science_util.utils.python_ml import (RobustHot,
                                               results_summary_to_dataframe)

import cac_calibrate.calibrate as cc

DATA_DIR = join(Path(__file__).parent.absolute(), "data")
MAIL_COST = 0.415
CONV_RATE = 0.1338
NUM_BINS = 20
CAC_COMPARE_THRESHOLD = 5000
MAX_DIFF_THRESHOLD_PCT = 0.025
QUANTILES = [
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975
]


class DataManager:

    def __init__(self):
        self.train_path_remote = "gs://figure-ml-pipeline/heloc/32/net_objects/train_pred.h5"
        self.serve_path_remote = "gs://figure-ml-pipeline/heloc/32/net_objects/serve_pred.h5"
        self.train_path_local = join(DATA_DIR, "train_pred.h5")
        self.serve_path_local = join(DATA_DIR, "serve_pred.h5")

    def copy_data_to_local(self):
        if not os.path.isdir(DATA_DIR):
            os.mkdir(DATA_DIR)
        os.system(f"gsutil cp {self.train_path_remote} {self.train_path_local}")
        os.system(f"gsutil cp {self.serve_path_remote} {self.serve_path_local}")

    @staticmethod
    def _check_path(path):
        if not os.path.exists(path):
            print(path)
            raise ValueError("Data not instantiated. Run DataManager().copy_data_to_local().")

    def load_train(self):
        self._check_path(self.train_path_local)
        res = pd.read_hdf(self.train_path_local)
        res = res.rename(columns={"y": "applied"})
        return res

    def load_serve(self):
        self._check_path(self.serve_path_local)
        res = pd.read_hdf(self.serve_path_local, start=0, stop=int(5e6)).iloc[:, :3]
        res = res.rename(columns={"score_raw": "score"})
        return res


class TestRegression(unittest.TestCase):
    setup_done = False

    def setUp(self):
        if not self.setup_done:
            dm = DataManager()
            self.df_train = dm.load_train()
            self.df_serve = dm.load_serve()
            self.cal_new = self.compute_calibration_new()
            self.cal_old = self.compute_calibration_old()
            self.setup_done = True
        self.cal = None

    def compute_calibration_old(self):
        cf = CacForecaster()
        res = cf.main(
            champion_scores=self.df_train,
            score_payload=self.df_serve,
            mail_cost=MAIL_COST,
            conv_rate=CONV_RATE,
            cac_quantile=QUANTILES,
            score_name="score"
        )
        return res

    def compute_calibration_new(self):
        self.cal = cc.Regression(
            score_name="score",
            target_name="applied",
            num_bins=NUM_BINS
        )
        self.cal.fit(self.df_train)
        res = self.cal.transform(
            df=self.df_serve,
            mail_cost=MAIL_COST,
            conv_rate=CONV_RATE,
            quantiles=QUANTILES
        )
        return res

    def test_calibration_match(self):
        cal_new = self.cal_new.loc[self.cal_new["cac"] <= CAC_COMPARE_THRESHOLD]
        cal_old = self.cal_old.loc[self.cal_old["cac"] <= CAC_COMPARE_THRESHOLD]
        merged = cal_new.merge(cal_old, on=["RECORD_NB", "ENCRYPTED_NB"], suffixes=("_new", "_old"))
        test_cols = ["cac", "cac_0.2", "cac_0.4", "cac_0.5", "cac_0.7"]
        for c in test_cols:
            abs_diff = np.abs(merged[f"{c}_new"] - merged[f"{c}_old"]).mean()
            mean = merged[[f"{c}_new", f"{c}_old"]].mean(axis=1).mean()
            pct_diff = abs_diff/mean
            close_enough = pct_diff < MAX_DIFF_THRESHOLD_PCT
            self.assertTrue(close_enough)


class CacForecaster:  # pylint: disable=R0903
    """
    Stripped down version of original cac class we want to replace.
    """
    # pylint: disable=R0915
    @staticmethod
    def main(
        champion_scores,
        score_payload,
        mail_cost,
        conv_rate,
        cac_quantile: Tuple[float] = (
            0.025,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.975,
        ),
        score_name: str = None,
    ):

        champion_scores = champion_scores.copy()
        score_payload = score_payload.copy()

        df = champion_scores

        df = df.loc[df["flag"] != "TRAIN"]

        df["campaign"] = df.campaign.str[0:4]

        min_campaign: str = df.campaign.min()

        max_campaign: str = df.campaign.max()

        df_scores = score_payload

        df_scores["campaign"] = max_campaign

        df_scores = df_scores.loc[:, ["RECORD_NB", "ENCRYPTED_NB", "campaign", score_name]]

        df_scores["campaign"] = df_scores.campaign.str[0:4]

        df_scores = df_scores.sort_values(by=score_name)

        df["score_bucket"] = pd.qcut(df[score_name], q=NUM_BINS)

        precision_lookup = df.groupby(["score_bucket", "campaign"]).mean()["applied"].reset_index()

        df_train_cac = precision_lookup.copy()

        precision_lookup["lower_bucket"] = (
            precision_lookup["score_bucket"]
            .apply(lambda x: x.left)
            .astype("str")
            .astype(np.float32)
        )
        precision_lookup["upper_bucket"] = (
            precision_lookup["score_bucket"]
            .apply(lambda x: x.right)
            .astype("str")
            .astype(np.float32)
        )
        precision_lookup["mid_bucket"] = \
            precision_lookup[["lower_bucket", "upper_bucket"]].mean(axis=1)

        lower_bound = precision_lookup.lower_bucket.min()
        upper_bound = precision_lookup.upper_bucket.max()

        df_scores[score_name] = np.where(
            df_scores[score_name] < lower_bound, lower_bound, df_scores[score_name]
        )
        df_scores[score_name] = np.where(
            df_scores[score_name] > upper_bound, upper_bound, df_scores[score_name]
        )

        df_scores = pd.merge_asof(
            df_scores,
            precision_lookup.loc[precision_lookup["campaign"] == max_campaign],
            left_on=score_name,
            right_on="lower_bucket",
        )

        df_scores = df_scores.drop(
            columns=["campaign_x", "campaign_y", "lower_bucket", "upper_bucket", "applied"]
        )

        hot = RobustHot()

        X = hot.fit_transform(df_train_cac, ["score_bucket", "campaign"], drop_first=False)

        X = X.drop(columns=[i for i in X.columns if ("nan" in i) or (min_campaign in i)])

        y = df_train_cac["applied"]

        y_hat = stm.OLS(y, X).fit()

        results = results_summary_to_dataframe(y_hat)

        fixed_effect = float(results.at["campaign__2101", "coeff"])

        cm_cols = []
        bin_cols = []

        for col in X.columns:
            if col.startswith("campaign"):
                cm_cols.append(col)
            elif col.startswith("score_bucket"):
                bin_cols.append(col)

        cols_order = bin_cols + cm_cols

        x_cpg = np.zeros((len(bin_cols), len(cm_cols)))

        X_p = pd.DataFrame(np.c_[np.eye(len(bin_cols)), x_cpg], columns=cols_order)

        y_p = y_hat.get_prediction(X_p)

        ref_table: DataFrame = pd.DataFrame(
            y_p.summary_frame()[["mean", "mean_se"]].values,
            columns=["mean", "mean_se"],
            index=bin_cols,
        )

        ref_table["cac"] = mail_cost / ((ref_table["mean"] + fixed_effect) * conv_rate)

        for quantile in cac_quantile:
            if quantile <= 0.5:
                p = y_p.summary_frame(alpha=2 * quantile)[["obs_ci_lower", "obs_ci_upper"]]
                ref_table[f"cac_{quantile:.4}"] = mail_cost / (
                    (p["obs_ci_upper"].values + fixed_effect) * conv_rate
                )
                ref_table[f"cac_{1 - quantile:.4}"] = mail_cost / (
                    (p["obs_ci_lower"].values + fixed_effect) * conv_rate
                )

        ref_table = ref_table[
            ref_table.columns.to_list()[:3] + sorted(ref_table.columns.to_list()[3:])
        ]

        ref_table[ref_table.iloc[:, 2:] < 0] = np.inf

        ref_table["score_bucket"] = [f.split("__")[-1] for f in bin_cols]

        df_scores["score_bucket"] = df_scores.score_bucket.astype(str)

        ds = pd.merge(
            df_scores, ref_table, left_on=["score_bucket"], right_on=["score_bucket"]
        ).drop("score_bucket", axis=1)

        return ds
