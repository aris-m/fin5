import pandas as pd
import numpy as np

# ── 0. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("/home/aris/fin5/csv_data/company_year.csv", sep="|")

# ── 1. Per-executive normalisation ───────────────────────────────────────────
# All _bt columns are board totals; divide by n_executives for per-person figures.
comp_cols = [
    "salary_bt", "one_year_bonus_bt", "multi_year_bonus_bt",
    "total_equity_grants_bt", "stock_grants_bt", "option_grants_bt",
    "other_annual_bt", "total_comp_bt", "one_time_payment_bt",
    "pension_bt", "total_comp_pens_and_one_time_bt",
]
for col in comp_cols:
    df[col + "_pe"] = df[col] / df["n_executives"]

# ── 2. Pay mix ratios (row-level) ─────────────────────────────────────────────
# Replace 0 denominators with NaN so divisions produce NaN rather than inf.
tc  = df["total_comp_bt"].replace(0, np.nan)   # total comp denominator
sal = df["salary_bt"].replace(0, np.nan)        # salary denominator (for vpi)
eq  = df["total_equity_grants_bt"].replace(0, np.nan)  # equity denominator

df["ratio_salary"]           = df["salary_bt"]            / tc
df["ratio_sti"]              = df["one_year_bonus_bt"]     / tc  # short-term incentive share
df["ratio_lti"]              = df["multi_year_bonus_bt"]   / tc  # long-term incentive share
df["ratio_equity"]           = df["total_equity_grants_bt"]/ tc  # equity grant share
df["ratio_variable"]         = (                                  # total variable pay share
    df["one_year_bonus_bt"] + df["multi_year_bonus_bt"] + df["total_equity_grants_bt"]
) / tc
df["ratio_other"]            = df["other_annual_bt"]       / tc
df["ratio_pension"]          = df["pension_bt"]            / tc

# Variable pay intensity: how many × base salary is variable pay
# (separate dimension from mix — captures aggressiveness of incentive design)
df["vpi"] = (
    df["one_year_bonus_bt"] + df["multi_year_bonus_bt"] + df["total_equity_grants_bt"]
) / sal

# Within equity grants: stock vs options split
df["ratio_stock_in_equity"]  = df["stock_grants_bt"]  / eq
df["ratio_option_in_equity"] = df["option_grants_bt"] / eq

# Log-transformed per-executive total comp (reduces right-skew for clustering)
df["log_tc_pe"] = np.log1p(df["total_comp_bt_pe"])

# ── 3. Index tier encoding ────────────────────────────────────────────────────
index_map = {"DAX": 1.0, "MDAX": 0.0, "DAX and MDAX": 0.5}
df["index_score"] = df["index_listing"].map(index_map)

# ── 4. Aggregate per company (group by ISIN — the true entity key) ────────────
# Using median rather than mean: robust to single-year outliers (bonus cliffs,
# restructuring payments, Wirecard-style anomalies).
ratio_cols   = [
    "ratio_salary", "ratio_sti", "ratio_lti", "ratio_equity",
    "ratio_variable", "ratio_other", "ratio_pension",
    "vpi", "ratio_stock_in_equity", "ratio_option_in_equity",
    "log_tc_pe",
]
level_pe_cols = [c + "_pe" for c in comp_cols]

feats = df.groupby("isin").agg(
    **{c: (c, "median") for c in ratio_cols},
    **{c: (c, "median") for c in level_pe_cols},
    n_executives_median = ("n_executives",  "median"),
    index_score         = ("index_score",   "median"),  # median handles DAX↔MDAX migrations
    ever_opted_out      = ("opting_out",    "max"),     # 1 if company ever opted out of disclosure
    year_count          = ("year",          "count"),
    year_min            = ("year",          "min"),
    year_max            = ("year",          "max"),
).reset_index()

# Attach most-recent name and index label (last year's value per ISIN)
last_meta = (
    df.sort_values("year")
      .groupby("isin")[["company_shortname", "company_name", "index_listing"]]
      .last()
      .reset_index()
      .rename(columns={"index_listing": "index_listing_last"})
)
feats = feats.merge(last_meta, on="isin", how="left")

# ── 5. Fill NaNs ──────────────────────────────────────────────────────────────
# Companies with zero equity grants have no stock/option split → set to 0
feats["ratio_stock_in_equity"]  = feats["ratio_stock_in_equity"].fillna(0)
feats["ratio_option_in_equity"] = feats["ratio_option_in_equity"].fillna(0)

# 3 companies with unknown index → treat conservatively as MDAX-equivalent (0.0)
feats["index_score"] = feats["index_score"].fillna(0.0)

# ── 6. Data confidence flag ───────────────────────────────────────────────────
feats["data_confidence"] = pd.cut(
    feats["year_count"],
    bins=[0, 2, 5, 10, 16],
    labels=["very_low", "low", "medium", "high"],
)

# ── 7. Filter to DAX-listed companies ────────────────────────────────────────
# Keep only companies whose most-recent index listing is DAX (pure DAX, not MDAX).
# "DAX and MDAX" are dual-listed companies and are excluded here.
dax_feats = feats[feats["index_listing_last"] == "DAX"].copy()

# ── 8. Clustering feature columns (for downstream use) ────────────────────────
CLUSTERING_FEATURES = [
    # Pay mix (shares of total comp) — pick sub-components OR ratio_variable, not both
    "ratio_salary",
    "ratio_sti",
    "ratio_lti",
    "ratio_equity",
    "ratio_other",
    "ratio_pension",
    # Incentive aggressiveness
    "vpi",
    # Equity instrument preference
    "ratio_stock_in_equity",
    "ratio_option_in_equity",
    # Pay level and firm scale
    "log_tc_pe",
    "n_executives_median",
    # Index tier (1 = DAX, 0.5 = dual, 0 = MDAX)
    "index_score",
]

# ── 9. Save ───────────────────────────────────────────────────────────────────
feats.to_csv("company_aggregated_features_all.csv", index=False)
dax_feats.to_csv("company_aggregated_features_dax.csv", index=False)

print(f"All companies : {len(feats):>3} rows, {len(feats.columns)} columns")
print(f"DAX companies : {len(dax_feats):>3} rows")
print()
print("DAX company list:")
print(
    dax_feats[["company_shortname", "year_count", "data_confidence"]]
    .sort_values("company_shortname")
    .to_string(index=False)
)
print()
print("NaN check on clustering features (DAX subset):")
print(dax_feats[CLUSTERING_FEATURES].isnull().sum().to_string())