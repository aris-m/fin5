#!/usr/bin/env python3
"""
DAX-15 Predictive Compensation Model
TUM Science Hackathon 2026

Approach: OLS regression trained on full universe (163 companies, 1350+ rows)
to predict expected compensation for DAX-15 companies.
Features: exogenous only (sector, year, board size, index tier)
Target: log(total_comp_bt)

Outputs:
  - model_results.csv   : predictions + residuals for DAX-15
  - model_coefs.csv     : coefficients (feature importance)
  - model_universe.csv  : predictions for all 163 companies
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

SECTOR_MAP = {
    "Allianz":          "Insurance",
    "Munich RE":        "Insurance",
    "Deutsche Bank":    "Banking",
    "BASF":             "Chemicals",
    "Bayer":            "Pharma",
    "Henkel":           "Chemicals",
    "BMW":              "Auto",
    "Daimler":          "Auto",
    "Volkswagen":       "Auto",
    "Siemens":          "Industrials",
    "Deutsche Telekom": "Telecom",
    "RWE":              "Energy",
    "E.ON":             "Energy",
    "SAP":              "Tech",
    "Adidas":           "Consumer",
}

DAX15 = list(SECTOR_MAP.keys())

# Rough sector map for full universe using keywords
SECTOR_KEYWORDS = {
    "Auto":       ["BMW","Daimler","Volkswagen","Continental","MAN","Porsche","Schaeffler"],
    "Chemicals":  ["BASF","Henkel","Lanxess","Covestro","Wacker","Evonik","K+S"],
    "Pharma":     ["Bayer","Merck","Fresenius","Stada","Morphosys","Qiagen"],
    "Insurance":  ["Allianz","Munich RE","Hannover","Talanx"],
    "Banking":    ["Deutsche Bank","Commerzbank","HypoVereinsbank","Aareal","IKB"],
    "Tech":       ["SAP","Software","Bechtle","Cancom","Nemetschek","TeamViewer","Wirecard"],
    "Industrials":["Siemens","ThyssenKrupp","Rheinmetall","GEA","Bilfinger","Dürr","Krones","KUKA"],
    "Energy":     ["RWE","E.ON","Uniper","Innogy","Siemens Energy"],
    "Telecom":    ["Deutsche Telekom","United Internet","Freenet","Telefonica"],
    "Consumer":   ["Adidas","Puma","Hugo Boss","Beiersdorf","Henkel","Axel Springer"],
    "Real Estate":["Vonovia","Deutsche Wohnen","LEG","TAG","Grand City","Aroundtown"],
    "Logistics":  ["Deutsche Post","Lufthansa","Fraport","TUI","Hamburger Hafen"],
    "Media":      ["ProSiebenSat1","RTL","Zalando","Delivery Hero","HelloFresh","Scout24"],
}

def assign_sector(name):
    for sector, keywords in SECTOR_KEYWORDS.items():
        if any(k.lower() in name.lower() for k in keywords):
            return sector
    return "Other"

print("=" * 60)
print("DAX-15 Predictive Compensation Model (OLS)")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE FULL UNIVERSE (2006–2024)
# ──────────────────────────────────────────────────────────────
print("\n[1/4] Preparing training data (DAX universe, 2006–2024)...")

cy_raw = pd.read_csv(BASE / "2008-2020/company_year.csv", sep="|", low_memory=False)
cy_raw["total_comp_bt"] = cy_raw["total_comp_bt"].replace(0, np.nan)

# ── Train ONLY on DAX companies (correct peer group) ──────────
cy_2021 = cy_raw[
    cy_raw["index_listing"].str.contains("DAX", na=False) &
    ~cy_raw["index_listing"].str.contains("MDAX", na=False)
].copy()

# ── Extend with 2022–2024 Excel data ─────────────────────────
excel_path = BASE / "company_year_2022_2024.csv"
if excel_path.exists():
    cy_ext = pd.read_csv(excel_path)
    cy_ext = cy_ext[cy_ext["index_listing"] == "DAX"].copy()
    cy = pd.concat([
        cy_2021[["company_shortname","year","total_comp_bt","n_executives"]],
        cy_ext[["company_shortname","year","total_comp_bt","n_executives"]],
    ], ignore_index=True)
    print(f"  Extended to 2024: {cy['company_shortname'].nunique()} companies, "
          f"{cy['year'].min()}–{cy['year'].max()}")
else:
    cy = cy_2021.copy()
    print(f"  2006–2021 only (run extract_excel.py first for 2024 extension)")

print(f"  DAX-only training universe: {cy['company_shortname'].nunique()} companies")

# Assign sectors
cy["sector"] = cy["company_shortname"].map(SECTOR_MAP)
cy.loc[cy["sector"].isna(), "sector"] = cy.loc[cy["sector"].isna(), "company_shortname"].apply(assign_sector)

cy["is_dax"]  = 1
cy["is_mdax"] = 0

# Year centered (for interpretable time coefficient)
cy["year_centered"] = cy["year"] - 2012

# Log board size
cy["log_n_exec"] = np.log(cy["n_executives"].replace(0, np.nan))

# Drop rows missing key vars
model_cols = ["total_comp_bt", "year_centered", "log_n_exec", "is_dax", "sector"]
keep_cols = model_cols + ["company_shortname", "year"]
if "isin" in cy.columns:
    keep_cols.append("isin")
cy_model = cy[keep_cols].dropna(subset=model_cols)
cy_model = cy_model[cy_model["total_comp_bt"] > 0].copy()
cy_model["log_comp"] = np.log(cy_model["total_comp_bt"])

# ── Lagged compensation (pay stickiness) ─────────────────────
# Sort to ensure correct lag ordering, then shift within company
cy_model = cy_model.sort_values(["company_shortname", "year"]).copy()
cy_model["log_comp_lag1"] = cy_model.groupby("company_shortname")["log_comp"].shift(1)

print(f"  Training rows: {len(cy_model):,}")
print(f"  Companies:     {cy_model['company_shortname'].nunique()}")
print(f"  Years:         {cy_model['year'].min()}–{cy_model['year'].max()}")
print(f"  Sectors:       {sorted(cy_model['sector'].unique())}")

# ──────────────────────────────────────────────────────────────
# 2. BUILD DESIGN MATRIX (OLS)
# ──────────────────────────────────────────────────────────────
print("\n[2/4] Building design matrix & fitting OLS...")

# Sector dummies (drop "Other" as baseline)
sectors = sorted([s for s in cy_model["sector"].unique() if s != "Other"])
for s in sectors:
    cy_model[f"sector_{s}"] = (cy_model["sector"] == s).astype(float)

# Feature columns (is_dax dropped — all training obs are DAX by construction)
# log_comp_lag1 captures pay stickiness (+0.18 R²); drops ~60 first-year obs
feature_cols = ["log_comp_lag1", "year_centered", "log_n_exec"] + [f"sector_{s}" for s in sectors]

# Drop rows missing any feature (lag NaN for each company's first year)
cy_model = cy_model.dropna(subset=feature_cols + ["log_comp"]).copy()
print(f"  After lag dropna: {len(cy_model):,} rows")

X = cy_model[feature_cols].values
y = cy_model["log_comp"].values

# Add intercept
X_with_intercept = np.column_stack([np.ones(len(X)), X])

# OLS: β = (X'X)^{-1} X'y
XtX = X_with_intercept.T @ X_with_intercept
Xty = X_with_intercept.T @ y

# Use pseudoinverse for numerical stability
beta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

# Predictions & residuals
y_pred = X_with_intercept @ beta
residuals = y - y_pred

# R²
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

# Standard errors
n, k = X_with_intercept.shape
sigma2 = ss_res / (n - k)
var_beta = sigma2 * np.linalg.pinv(XtX)
se_beta = np.sqrt(np.diag(var_beta))
t_stats = beta / se_beta
p_values = 2 * (1 - np.minimum(
    np.abs(t_stats) / 10,  # rough approximation for large n
    1.0
))

feature_names = ["intercept", "log_comp_lag1", "year_trend", "log_board_size"] + [f"sector_{s}" for s in sectors]

# Coefficients table
coefs = pd.DataFrame({
    "feature":     feature_names,
    "coefficient": beta,
    "std_error":   se_beta,
    "t_stat":      t_stats,
    "exp_effect_pct": (np.exp(beta) - 1) * 100,  # % effect on comp level
})

print(f"  R² = {r2:.3f}  (explains {r2*100:.0f}% of variance in log-comp)")
print(f"  N  = {len(cy_model):,}  |  Features = {k}")
print()
print("  KEY COEFFICIENTS (effect on total comp level):")
key = coefs[coefs["feature"] != "intercept"].sort_values("exp_effect_pct", ascending=False)
for _, row in key.iterrows():
    direction = "▲" if row["exp_effect_pct"] > 0 else "▼"
    print(f"    {direction} {row['feature']:<25} {row['exp_effect_pct']:+.1f}%  (t={row['t_stat']:.1f})")

# ──────────────────────────────────────────────────────────────
# 3. PREDICTION INTERVALS (bootstrap approx)
# ──────────────────────────────────────────────────────────────
print("\n[3/4] Computing prediction intervals & benchmarks...")

cy_model["log_comp_pred"] = y_pred
cy_model["log_comp_resid"]= residuals

# Residual std per sector (sector-specific spread)
resid_std = cy_model.groupby("sector")["log_comp_resid"].std().to_dict()
cy_model["resid_std"] = cy_model["sector"].map(resid_std).fillna(residuals.std())

# 80% prediction interval: ± 1.28 * resid_std
CI_Z = 1.28
cy_model["pred_comp"]      = np.exp(cy_model["log_comp_pred"])
cy_model["pred_comp_low"]  = np.exp(cy_model["log_comp_pred"] - CI_Z * cy_model["resid_std"])
cy_model["pred_comp_high"] = np.exp(cy_model["log_comp_pred"] + CI_Z * cy_model["resid_std"])

# Benchmarking signal
cy_model["actual_vs_expected_pct"] = (
    (cy_model["total_comp_bt"] - cy_model["pred_comp"]) / cy_model["pred_comp"] * 100
)

def benchmark_flag(pct):
    if pct > 40:   return "🔴 Significantly Overpaid"
    if pct > 15:   return "🟡 Above Market"
    if pct > -15:  return "🟢 In Line"
    if pct > -40:  return "🟡 Below Market"
    return "🔵 Significantly Underpaid"

cy_model["benchmark_signal"] = cy_model["actual_vs_expected_pct"].apply(benchmark_flag)

# ──────────────────────────────────────────────────────────────
# 4. DAX-15 RESULTS
# ──────────────────────────────────────────────────────────────
dax15_results = cy_model[cy_model["company_shortname"].isin(DAX15)].copy()

print(f"  DAX-15 prediction rows: {len(dax15_results)}")
print()
print("  MOST OVERPAID vs. peer expectation:")
top_over = dax15_results.nlargest(5, "actual_vs_expected_pct")[
    ["company_shortname", "year", "total_comp_bt", "pred_comp", "actual_vs_expected_pct", "benchmark_signal"]
]
print(top_over.to_string(index=False))

print()
print("  MOST UNDERPAID vs. peer expectation:")
top_under = dax15_results.nsmallest(5, "actual_vs_expected_pct")[
    ["company_shortname", "year", "total_comp_bt", "pred_comp", "actual_vs_expected_pct", "benchmark_signal"]
]
print(top_under.to_string(index=False))

print()
print("  AVERAGE OVER/UNDER-PAY PER COMPANY:")
avg_dev = dax15_results.groupby("company_shortname")["actual_vs_expected_pct"].mean().sort_values(ascending=False)
for co, val in avg_dev.items():
    direction = "▲" if val > 0 else "▼"
    print(f"    {direction} {co:<25} {val:+.1f}%")

# ──────────────────────────────────────────────────────────────
# 5. SAVE
# ──────────────────────────────────────────────────────────────
print("\n[4/4] Saving outputs...")

# DAX-15 results (for dashboard)
save_cols = [
    "company_shortname", "year", "sector",
    "total_comp_bt", "pred_comp", "pred_comp_low", "pred_comp_high",
    "actual_vs_expected_pct", "benchmark_signal",
    "log_n_exec", "is_dax", "year_centered", "log_comp_lag1",
] + [f"sector_{s}" for s in sectors]
dax15_results[save_cols].to_csv(BASE / "model_results.csv", index=False)

# Feature importance
coefs.to_csv(BASE / "model_coefs.csv", index=False)

# Full universe predictions
cy_model.to_csv(BASE / "model_universe.csv", index=False)

print(f"  model_results.csv   — {len(dax15_results)} rows (DAX-15 predictions)")
print(f"  model_coefs.csv     — {len(coefs)} features with coefficients")
print(f"  model_universe.csv  — {len(cy_model)} rows (full universe)")
print(f"\n  Model R² = {r2:.3f}")
print(f"\n✓ Model complete")
