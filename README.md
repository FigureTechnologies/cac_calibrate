# cac-calibrate
Calibrate model scores, and translate them to cac.

NOTE: Tests require packages not listed in `setup.py`.

# Example
Givent data frames `df_train` and `df_serve`, to calculate cac outputs, we'd do something like the following.
```python
from cac_calibrate import RegressionCalibrator

# instantiate and fit calibrator
rc = RegressionCalibrator(
    score_name="xgbt_score",
    target_name="applied",
    num_bins=30,
    feature_cols=["campaign"]
)

# df_train.columns = [record_nb, campaign, xgbt_score, applied]
df_train["campaign"] = df_train["campaign"].str.slice(0, 4)
rc.fit(df_train)

# df_serve.columns = [record_nb, encrypted_nb, xgbt_score]
# use latest campaign from training as feature for serving
df_serve["campaign"] = df_train["campaign"].astype(str).max()

# get cac output
cac_output = rc.transform(
    df=df_serve,
    mail_cost=0.415,
    conv_rate=0.1338
)
```
