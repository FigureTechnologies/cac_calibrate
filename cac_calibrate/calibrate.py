import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import statsmodels.api as stm

BUCKET_NAME = "score_bucket"
DEFAULT_Q = [
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


class QuantileBinner:

    def __init__(self, num_bins):
        self.num_bins = num_bins

        self.buckets = None
        self._idx_col = "__bucket_idx__"
        self._value_col = "__bucket_value__"

    def fit(self, x):
        x = np.squeeze(x)
        buckets = pd.qcut(x, q=self.num_bins).unique()
        buckets = pd.DataFrame({
            self._idx_col: np.arange(len(buckets)),
            self._value_col: sorted([i.left for i in buckets])
        })
        buckets[self._value_col] = buckets[self._value_col].astype(np.float64)
        self.buckets = buckets
        print(self.buckets)

    @property
    def min_bucket_val(self):
        return self.buckets[self._value_col].min()

    @property
    def max_bucket_val(self):
        return self.buckets[self._value_col].max()

    def transform(self, x):
        x = np.squeeze(x).astype(np.float64)
        # idx_sort = np.argsort(x)
        # x = x[idx_sort]
        df = pd.DataFrame(x, columns=["score"])

        merged = pd.merge_asof(
            left=df,
            right=self.buckets,
            left_on="score",
            right_on=self._value_col,
            allow_exact_matches=False
        )
        merged.loc[merged["score"] <= self.min_bucket_val, self._idx_col] = 0
        merged.loc[merged["score"] >= self.max_bucket_val, self._idx_col] = self.num_bins - 1
        buckets_sorted = merged[self._idx_col].values.astype(int)
        # idx_unsort = np.zeros_like(buckets_sorted)
        # idx_unsort[idx_sort] = np.arange(len(idx_unsort))
        # buckets_unsorted = buckets_sorted[idx_unsort]
        return buckets_sorted


class Regression:  # pylint: disable=R0902

    def __init__(self, score_name, target_name, num_bins):
        self.score_name = score_name
        self.target_name = target_name
        self.num_bins = num_bins
        self.expected_cols = ["RECORD_NB", "ENCRYPTED_NB", "campaign", self.score_name]

        self.max_campaign = None
        # self.binner_tr = sp.KBinsDiscretizer(n_bins=num_bins, encode="ordinal")
        # self.binner_serve = sp.KBinsDiscretizer(n_bins=num_bins, encode="ordinal")
        self.binner_tr = QuantileBinner(num_bins=self.num_bins)
        self.binner_serve = QuantileBinner(num_bins=self.num_bins)
        self.ohe = sp.OneHotEncoder(categories="auto", drop="first", sparse=False)
        self.reg_cols = ["campaign", BUCKET_NAME]
        self.calibrator = None

    def fit(self, df):
        df = df.copy()
        df["campaign"] = df.campaign.str[0:4]
        df = df.sort_values(by=self.score_name)
        self.max_campaign = df["campaign"].max()

        y_bin_serve = (
            df.loc[df["campaign"] == self.max_campaign, self.score_name].
            values.
            reshape(-1, 1)
        )
        self.binner_serve.fit(y_bin_serve)

        y_bin_tr = df[self.score_name].values.reshape(-1, 1)
        self.binner_tr.fit(y_bin_tr)
        df[BUCKET_NAME] = self.binner_tr.transform(y_bin_tr)
        tmp = df.copy()
        tmp["bin"] = pd.qcut(tmp[self.score_name], q=self.num_bins)
        tmp = tmp.groupby(["campaign", "bin"])[self.target_name].mean()
        df_regr = df.copy()
        df_regr[BUCKET_NAME] = pd.qcut(
            df_regr[self.score_name],
            q=self.num_bins,
            labels=np.arange(self.num_bins)
        )
        df_regr = (
            df_regr.
            groupby(["campaign", BUCKET_NAME], as_index=False, observed=True)[self.target_name].
            mean().
            sort_values(by=["campaign", BUCKET_NAME])
        )
        print(df_regr)
        X = df_regr[self.reg_cols]
        X_hot = self.ohe.fit_transform(X)
        y = df_regr[self.target_name].values
        self.calibrator = stm.OLS(y, X_hot).fit()

    def transform(self, df, mail_cost, conv_rate, quantiles=None):

        def prob2cac(prob):
            return mail_cost/(prob*conv_rate)

        quantiles = quantiles or DEFAULT_Q

        df = df.sort_values(by=self.score_name).reset_index(drop=True)
        idx = df.copy()

        # NOTE: use binner_serve or binner_tr?
        df[BUCKET_NAME] = self.binner_tr.transform(
            df[self.score_name].values.reshape(-1, 1)
        )
        X = df[[BUCKET_NAME]].copy()
        X["campaign"] = self.max_campaign
        X = X[self.reg_cols]
        X_hot = self.ohe.transform(X)
        preds = self.calibrator.get_prediction(X_hot)

        # mean and std error of cac
        cac_moment = pd.DataFrame(
            preds.summary_frame()[["mean", "mean_se"]].values,
            columns=["mean", "mean_se"]
        )
        cac_moment[["cac", "cac_se"]] = prob2cac(cac_moment)

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
