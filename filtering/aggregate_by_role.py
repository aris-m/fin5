import pandas as pd
import numpy as np

# ── 1. Load ──────────────────────────────────────────────────────────────
df = pd.read_csv("/home/aris/Downloads/fin_data/2008-2021/person_year.csv", sep="|")

# ── 2. Assign role ───────────────────────────────────────────────────────
def assign_role(row):
    if row["ceo_flag_eoy"] == 1:
        return "CEO"
    elif row["cfo_flag_eoy"] == 1:
        return "CFO"
    else:
        return "Other"

df["role"] = df.apply(assign_role, axis=1)

# ── 3. Define which columns to sum ───────────────────────────────────────
comp_cols = [
    "salary",
    "one_year_bonus",
    "multi_year_bonus",
    "multi_year_bonus_grants",
    "multi_year_bonus_payout",
    "total_equity_grants",
    "stock_grants",
    "option_grants",
    "other_annual_comp",
    "total_comp",
    "pension",
    "one_time_payment",
    "total_comp_pens_and_one_time",
]

group_keys = ["isin", "company_shortname", "year", "role"]

# ── 4. Aggregate ─────────────────────────────────────────────────────────
agg_dict = {col: "sum" for col in comp_cols}
agg_dict["company_person_id"] = "count"  # counts executives in each group

grouped = (
    df.groupby(group_keys, as_index=False)
      .agg(agg_dict)
      .rename(columns={"company_person_id": "n_seat_holders"})
)

# ── 5. Per-seat metrics ──────────────────────────────────────────────────
grouped["comp_per_seat"] = grouped["total_comp"] / grouped["n_seat_holders"]
grouped["log_tc_seat"] = np.log(grouped["comp_per_seat"].clip(lower=1))

# ── 6. Compensation mix ratios (share of total_comp) ─────────────────────
grouped["ratio_salary"]  = grouped["salary"]              / grouped["total_comp"]
grouped["ratio_sti"]     = grouped["one_year_bonus"]       / grouped["total_comp"]
grouped["ratio_lti"]     = grouped["multi_year_bonus"]     / grouped["total_comp"]
grouped["ratio_equity"]  = grouped["total_equity_grants"]  / grouped["total_comp"]
grouped["ratio_stock"]   = grouped["stock_grants"]         / grouped["total_comp"]
grouped["ratio_option"]  = grouped["option_grants"]        / grouped["total_comp"]
grouped["ratio_other"]   = grouped["other_annual_comp"]    / grouped["total_comp"]
grouped["ratio_pension"] = grouped["pension"]              / grouped["total_comp"]

# Replace inf / NaN from zero-total-comp rows
ratio_cols = [c for c in grouped.columns if c.startswith("ratio_")]
grouped[ratio_cols] = grouped[ratio_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# ── 7. Save ──────────────────────────────────────────────────────────────
grouped.to_csv("comp_by_role.csv", index=False)
print(f"Done — {len(grouped)} rows written to comp_by_role.csv")
print(grouped.head(10))