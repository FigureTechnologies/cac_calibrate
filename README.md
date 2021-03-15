# cac-calibrate
Calibrate model scores, and translate them to cac.

NOTE: At this point, we assume that all features passed to the calibrator are categorical.

## Example
Givent data frames `df_train` and `df_serve`, to calculate cac outputs, we'd do something like the following.
```python
from cac_calibrate import RegressionCalibrator

rc = RegressionCalibrator(
    score_name="xgbt_score",
    target_name="applied",
    num_bins=30,
    feature_cols=["campaign"]
)

# df_train.columns = [record_nb, campaign, xgbt_score, applied, ...]
df_train = df_train.loc[df_train["flag"].isin(("VALIDATE", "TEST"))].copy()
df_train["campaign"] = df_train["campaign"].str.slice(0, 4)  # get rid of A, B, ... suffixes
rc.fit(df_train)

# df_serve.columns = [record_nb, encrypted_nb, xgbt_score, ...]
df_serve["campaign"] = df_train["campaign"].astype(str).max()  # use latest campaign as feature

# get cac output
cac_output = rc.transform(
    df=df_serve,
    mail_cost=0.415,
    conv_rate=0.1338
)
```

## Notes on Tests
* Tests require `data-science-util`, which is not listed in `setup.py`.
* Tests use actual serving data, so they require 50+ GB of RAM and 20+ GB of disk space.
