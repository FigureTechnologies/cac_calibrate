import numpy as np
import pandas as pd
import sklearn.preprocessing as sp
import statsmodels.api as stm

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


class Regression:  # pylint: disable=R0902

    def __init__(self, score_name, target_name, num_bins):
        self.score_name = score_name
        self.target_name = target_name
        self.num_bins = num_bins
        self.expected_cols = ["RECORD_NB", "ENCRYPTED_NB", "campaign", self.score_name]

        self.max_campaign = None
        self.bucket_name = "score_bucket"
        self.binner_tr = sp.KBinsDiscretizer(n_bins=num_bins, encode="ordinal")
        self.binner_serve = sp.KBinsDiscretizer(n_bins=num_bins, encode="ordinal")
        self.ohe = sp.OneHotEncoder(categories="auto", drop="first", sparse=False)
        self.reg_cols = ["campaign", self.bucket_name]
        self.calibrator = None

    def fit(self, df):
        df = df.copy()
        df["campaign"] = df.campaign.str[0:4]
        self.max_campaign = df["campaign"].max()

        y_bin_serve = (
            df.loc[df["campaign"] == self.max_campaign, self.score_name].
            values.
            reshape(-1, 1)
        )
        self.binner_serve.fit(y_bin_serve)

        y_bin_tr = df[self.score_name].values.reshape(-1, 1)
        self.binner_tr.fit(y_bin_tr)
        df[self.bucket_name] = self.binner_tr.transform(y_bin_tr)
        df_regr = (
            df.
            groupby(["campaign", "score_bucket"], as_index=False, observed=True)[self.target_name].
            mean().
            sort_values(by=["campaign", "score_bucket"])
        )
        X = df_regr[self.reg_cols]
        X_hot = self.ohe.fit_transform(X)
        y = df_regr[self.target_name].values
        self.calibrator = stm.OLS(y, X_hot).fit()

    def transform(self, df, mail_cost, conv_rate, quantiles=None):

        def prob2cac(prob):
            return mail_cost/(prob*conv_rate)

        quantiles = quantiles or DEFAULT_Q

        df = df.copy()
        idx = df.copy()

        df[self.bucket_name] = self.binner_serve.transform(
            df[self.score_name].values.reshape(-1, 1)
        )
        X = df[[self.bucket_name]].copy()
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


