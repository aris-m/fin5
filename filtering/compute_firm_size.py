import pandas as pd
import numpy as np

# ── 1. Load datasets ─────────────────────────────────────────────────────
exec_df = pd.read_csv("/home/aris/fin5/filtering/exec_output/exec_features_all_imputed.csv")
orbis_df = pd.read_csv("/home/aris/Downloads/ORBIS_Abzug_DE_2005_2024.csv", low_memory=False)  # <-- replace with your ORBIS filename

# ── 2. Prepare ORBIS: keep relevant columns & deduplicate ────────────────
# Normalize column names to lowercase (ORBIS export uses uppercase)
orbis_df.columns = orbis_df.columns.str.lower()

orbis_cols = ["sd_isin", "closdate_year", "toas", "opre", "empl"]
orbis = orbis_df[orbis_cols].copy()

# Rename for merging
orbis = orbis.rename(columns={"sd_isin": "isin", "closdate_year": "year"})

# Convert year to int (closdate_year may be float like 2012.0)
orbis["year"] = orbis["year"].astype("Int64")

# Drop rows where all three size vars are missing
orbis = orbis.dropna(subset=["toas", "opre", "empl"], how="all")

# If multiple ORBIS entries per isin-year (e.g. different consolidation codes),
# keep the one with the most complete data, then take the first
orbis["n_present"] = orbis[["toas", "opre", "empl"]].notna().sum(axis=1)
orbis = (
    orbis.sort_values("n_present", ascending=False)
         .drop_duplicates(subset=["isin", "year"], keep="first")
         .drop(columns="n_present")
)

# ── 3. Log-transform ─────────────────────────────────────────────────────
# Use log(1 + x) to handle zeros; clip negatives to NaN
for col in ["toas", "opre", "empl"]:
    orbis[f"log_{col}"] = np.log1p(orbis[col].clip(lower=0))
    orbis.loc[orbis[col] <= 0, f"log_{col}"] = np.nan

# ── 4. Standardize (z-score) across the full ORBIS panel ─────────────────
log_cols = ["log_toas", "log_opre", "log_empl"]
for col in log_cols:
    mean = orbis[col].mean()
    std = orbis[col].std()
    orbis[f"z_{col}"] = (orbis[col] - mean) / std

# ── 5. Composite firm_size = mean of available z-scores ───────────────────
z_cols = ["z_log_toas", "z_log_opre", "z_log_empl"]
orbis["firm_size"] = orbis[z_cols].mean(axis=1)  # nanmean behavior by default

# ── 6. Merge into exec compensation data ──────────────────────────────────
# Keep only the merge key + result
orbis_merge = orbis[["isin", "year", "toas", "opre", "empl", "firm_size"]].copy()

result = exec_df.merge(orbis_merge, on=["isin", "year"], how="left")

# ── 7. Report & save ──────────────────────────────────────────────────────
matched = result["firm_size"].notna().sum()
total = len(result)
print(f"Matched {matched}/{total} rows ({matched/total:.1%})")
print(f"\nfirm_size distribution:")
print(result["firm_size"].describe())

# Show a few examples
print("\nSample rows:")
print(
    result[["isin", "company_shortname", "year", "role", "toas", "opre", "empl", "firm_size"]]
    .drop_duplicates(subset=["isin", "year"])
    .head(10)
    .to_string(index=False)
)

result.to_csv("exec_features_with_firm_size.csv", index=False)
print(f"\nSaved to exec_features_with_firm_size.csv")