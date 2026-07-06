"""
SAMRIDH Startup Dataset — Geographic Valuation Discount Analysis
==================================================================
Hypothesis: Startups outside Tier-I cities are valued LOWER relative to
their fundamentals (revenue, tech readiness, IP, team) than Tier-I startups,
despite showing comparable or better technical maturity.

Approach: train a model to predict valuation from fundamentals ONLY
(location excluded), then check whether the model's errors (residuals)
are systematically biased by location tier. If yes, that's evidence of
a valuation gap that fundamentals don't explain.
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# STEP 1: Load & clean
# ---------------------------------------------------------------------
df = pd.read_excel("SAMRIDH_c2.xlsx", sheet_name="Main Sheet")


def clean_num(x):
    """Pull the first numeric value out of a messy text cell.
    Many cells contain currency text, commas, or (bad data) a Drive
    link instead of a number -- those become NaN rather than a bogus 0."""
    if pd.isna(x):
        return np.nan
    s = str(x)
    if "http" in s.lower():
        return np.nan
    s = s.replace(",", "")
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else np.nan


df["Valuation"] = df["Current Valuation （in INR Cr.）"].apply(clean_num)
df["Revenue"] = df["Current Revenue （in INR Cr.）"].apply(clean_num)
df["TeamSize"] = df["Current team size"].apply(clean_num)
df["Employment"] = df["Number of employment generated"].apply(clean_num)
df["IPs"] = df["No. of IPs generated (Patents, Trademarks, Copyrights etc.)"].apply(clean_num)
df["TRL"] = df["Technology Readiness Level （TRL）"].apply(clean_num)
df["HasWomenCofounder"] = df["Women Co-Founder name (if any)"].notna().astype(int)
df["LocationType"] = df["Location type"].str.strip()

# Data-entry error fixes found during EDA:
df["IPs"] = df["IPs"].clip(lower=0)                     # one row had -2 patents
df = df[(df["Valuation"].isna()) | (df["Valuation"] < 2000)]  # drop ~2 rows entered
                                                                # in raw INR, not INR Cr.
                                                                # (450,000,000 "Cr" is not real)

# Standardize the free-text "stage" column (60+ inconsistent variants)
def bucket_stage(s):
    if pd.isna(s):
        return "Unknown"
    s = str(s).lower()
    if any(k in s for k in ["ideation", "prototyp", "mvp", "poc", "validation", "concept"]):
        return "Ideation/Prototype"
    if any(k in s for k in ["pmf", "fit", "early traction", "early revenue", "pilot", "early adopt"]):
        return "PMF/Early Traction"
    if "launch" in s:
        return "Product Launched"
    if any(k in s for k in ["scale", "growth", "commercial"]):
        return "Scale Up"
    return "Other"


df["StageBucket"] = df["Stage of startup (Ideation, PMF, Product launched, Scale up)"].apply(bucket_stage)

# ---------------------------------------------------------------------
# STEP 2: Build the merit-only feature set (location deliberately excluded)
# ---------------------------------------------------------------------
model_df = df[df["Valuation"].notna() & (df["Valuation"] > 0)].copy()

model_df["Revenue"] = model_df["Revenue"].fillna(0)
for c in ["TeamSize", "Employment", "IPs"]:
    model_df[c] = model_df[c].fillna(model_df[c].median() if c != "IPs" else 0)

# Log-transform: valuation, revenue, team size etc. are heavily right-skewed
# (a handful of startups are 100x bigger than the median). Modeling on the raw
# scale lets those few large startups dominate the loss function; log1p
# compresses that skew so the model learns from the whole distribution,
# not just the outliers. It also matches how these quantities are usually
# reasoned about -- proportionally, not in absolute rupees.
model_df["log_Valuation"] = np.log1p(model_df["Valuation"])
model_df["log_Revenue"] = np.log1p(model_df["Revenue"])
model_df["log_TeamSize"] = np.log1p(model_df["TeamSize"])
model_df["log_Employment"] = np.log1p(model_df["Employment"])
model_df["log_IPs"] = np.log1p(model_df["IPs"])

stage_dummies = pd.get_dummies(model_df["StageBucket"], prefix="Stage")

CORE_FEATURES = ["log_Revenue", "TRL", "log_IPs", "log_TeamSize",
                  "log_Employment", "HasWomenCofounder"]
STAGE_COLS = ["Stage_Ideation/Prototype", "Stage_PMF/Early Traction",
              "Stage_Product Launched", "Stage_Scale Up"]

X_merit = pd.concat(
    [model_df[CORE_FEATURES].reset_index(drop=True), stage_dummies[STAGE_COLS].reset_index(drop=True)],
    axis=1,
)
y = model_df["log_Valuation"].reset_index(drop=True)

# ---------------------------------------------------------------------
# STEP 3: Train the merit-only model with cross-validated predictions
# ---------------------------------------------------------------------
# Why Random Forest:
#   - n=180 rows, ~10 features -> too small/noisy for a high-capacity model
#     (gradient boosting overfit badly here: CV R^2 went NEGATIVE with default
#     settings). A shallow, heavily-regularized Random Forest was the most
#     stable performer across several models tried (Ridge, XGBoost, RF).
#   - Non-linear + handles mixed continuous/binary features without needing
#     interaction terms specified by hand.
#   - Only 3 hyperparameters to tune (depth, leaf size, n_estimators) --
#     less overfitting risk than tuning XGBoost's larger hyperparameter space
#     on a dataset this small.
#
# Why cross-validated predictions, not in-sample predictions:
#   - If we fit the model on all 180 rows and score it on those same rows,
#     the residuals are partly just "how well did the model memorize this
#     row" rather than "how surprising is this row's valuation." That would
#     make the Tier-I vs Tier-II/III comparison unreliable. Using
#     out-of-fold predictions (5-fold CV) means every residual is computed
#     on data the model did NOT see during that fold's training -- an honest
#     estimate of prediction error.

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=3,          # shallow trees -- deliberately limits overfitting
    min_samples_leaf=5,   # each leaf needs 5+ samples -- smooths noisy splits
    random_state=RANDOM_STATE,
)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_preds = cross_val_predict(model, X_merit, y, cv=kf)

r2 = r2_score(y, cv_preds)
mae = mean_absolute_error(y, cv_preds)
print(f"Merit-only model -- CV R^2: {r2:.4f}, CV MAE (log scale): {mae:.4f}")

model_df["pred_log_val"] = cv_preds
model_df["residual"] = model_df["log_Valuation"] - model_df["pred_log_val"]
# residual > 0  => market values the startup MORE than fundamentals justify
# residual < 0  => market values the startup LESS than fundamentals justify

# ---------------------------------------------------------------------
# STEP 4: Test the hypothesis -- do residuals differ systematically by tier?
# ---------------------------------------------------------------------
summary = model_df.groupby("LocationType")["residual"].agg(["count", "mean", "median", "std"])
print("\nResidual summary by location tier:")
print(summary)

tier1 = model_df.loc[model_df["LocationType"] == "Tier-I", "residual"]
non_tier1 = model_df.loc[model_df["LocationType"] != "Tier-I", "residual"]

t_stat, p_ttest = stats.ttest_ind(tier1, non_tier1, equal_var=False)
u_stat, p_mw = stats.mannwhitneyu(tier1, non_tier1, alternative="two-sided")
print(f"\nWelch t-test:      t={t_stat:.3f}, p={p_ttest:.4f}")
print(f"Mann-Whitney U:    p={p_mw:.4f}")

# ---------------------------------------------------------------------
# STEP 5: Add location back in -- does it earn its own predictive power?
# ---------------------------------------------------------------------
loc_dummies = pd.get_dummies(model_df["LocationType"], prefix="Loc")
X_with_loc = pd.concat([X_merit.reset_index(drop=True), loc_dummies.reset_index(drop=True)], axis=1)

model_with_loc = RandomForestRegressor(
    n_estimators=300, max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
)
cv_preds_loc = cross_val_predict(model_with_loc, X_with_loc, y, cv=kf)
r2_with_loc = r2_score(y, cv_preds_loc)
print(f"\nWith location added -- CV R^2: {r2_with_loc:.4f} (was {r2:.4f} without)")

# Fit once on full data to inspect which features matter, using permutation
# importance rather than the model's built-in feature_importances_.
# Built-in importance is biased toward high-cardinality / high-variance
# features on small samples; permutation importance measures the actual
# drop in model performance when a feature is shuffled -- a more honest
# read on a dataset this size.
model_with_loc.fit(X_with_loc, y)
perm = permutation_importance(model_with_loc, X_with_loc, y, n_repeats=50, random_state=RANDOM_STATE)
perm_imp = pd.Series(perm.importances_mean, index=X_with_loc.columns).sort_values(ascending=False)
print("\nPermutation importance (with location included):")
print(perm_imp)
