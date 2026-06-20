"""
prepare_exec_features.py
=========================
Prepares the individual executive dataset for peer group clustering.
Outputs one row per company × year × role (the "seat" level), not per person.

Why seat-level?
---------------
Multiple executives can hold the same role in one year (e.g. two CFOs due to
a mid-year handover). Aggregating to the seat level gives one comparable unit
per role per firm-year, which is what peer benchmarking requires.

Aggregation logic per isin × year × role group
-----------------------------------------------
1. Annualise each individual's flow compensation (× 365 / days_served).
2. Sum annualised components across all seat-holders.
3. Divide by headcount (n_seat_holders) → per-seat figures.
4. Compute pay mix ratios from the summed components (not from individual ratios)
   to avoid partial-year distortion.
5. Carry firm-level features (sector, firm_size, index_score, ever_opted_out)
   from the enriched file — these are the same for every exec at the same firm-year.

Pipeline
--------
1. Load executive file + enriched company-year file (sector & firm_size)
2. Join sector and firm_size via isin × year
3. Drop unusable rows (zero comp, missing role, zero days)
4. Annualise individual compensation for partial-year executives
5. Aggregate to isin × year × role seat level
6. Compute seat-level pay features:
     log_tc_seat       — log(comp_per_seat), the primary pay level feature
     ratio_salary      — salary share of total comp
     ratio_sti         — short-term bonus share
     ratio_lti         — long-term cash bonus share
     ratio_equity      — equity grants share
     ratio_other       — other annual benefits share
     ratio_pension     — pension share (winsorised at 1.0)
     vpi               — variable pay / fixed salary (incentive aggressiveness)
     ratio_stock       — stock share of equity grants (0 if no equity)
     ratio_option      — option share of equity grants (0 if no equity)
7. Add structural features (index_score, ever_opted_out)
8. Save combined + role-stratified files

Output columns (one row per isin × year × role)
------------------------------------------------
Identifiers : isin, company_shortname, year, role, n_seat_holders
Pay level   : comp_per_seat, log_tc_seat
Pay mix     : ratio_salary, ratio_sti, ratio_lti, ratio_equity,
              ratio_other, ratio_pension
Incentive   : vpi
Equity      : ratio_stock, ratio_option
Structural  : firm_size, index_score, ever_opted_out
Grouping    : sector

Usage
-----
    python prepare_exec_features.py \\
        --exec     individual_exec_data.csv \\
        --enriched company_year_enriched.csv \\
        --outdir   ./exec_output

Notes
-----
- Monetary values assumed in EUR thousands (€k).
- Pension and one_time_payment are NOT annualised (event-driven lump sums).
- ratio_pension winsorised at 1.0 (actuarial spikes in low-rate years).
- sector and firm_size joined from enriched file; NaN where no ORBIS match.
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_MAP = {"DAX": 1.0, "MDAX": 0.0, "DAX and MDAX": 0.5}

# Flow-based columns to annualise before aggregation
ANNUALISE_COLS = [
    "salary", "one_year_bonus", "multi_year_bonus",
    "total_equity_grants", "stock_grants", "option_grants",
    "other_annual_comp", "total_comp",
]

# Grouping key for seat aggregation
SEAT_KEY = ["isin", "year", "role"]

# Final output column order
IDENTIFIER_COLS = ["isin", "company_shortname", "year", "role", "n_seat_holders"]

FEATURE_COLS = [
    "comp_per_seat", "log_tc_seat",
    "ratio_salary", "ratio_sti", "ratio_lti",
    "ratio_equity", "ratio_other", "ratio_pension",
    "vpi",
    "ratio_stock", "ratio_option",
    "firm_size", "index_score", "ever_opted_out",
    "sector",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    """Divide; return NaN where denom is zero or NaN."""
    return np.where((denom.notna()) & (denom != 0), num / denom, np.nan)


def normalise_isin(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_and_validate(exec_path: str, enriched_path: str):
    print(f"\n[1] Loading files …")

    # Auto-detect separator: try pipe first, fall back to comma
    exec_df = pd.read_csv(exec_path, sep="|", low_memory=False)
    if exec_df.shape[1] <= 2:
        exec_df = pd.read_csv(exec_path, sep=",", low_memory=False)
    print(f"    Executive file : {exec_df.shape[0]:,} rows × {exec_df.shape[1]} cols")

    enriched = pd.read_csv(enriched_path, sep="|", low_memory=False)
    print(f"    Enriched file  : {enriched.shape[0]:,} rows × {enriched.shape[1]} cols")

    required_exec = [
        "isin", "year", "role", "days",
        "salary", "one_year_bonus", "multi_year_bonus",
        "total_equity_grants", "stock_grants", "option_grants",
        "other_annual_comp", "total_comp", "pension",
        "opting_out", "index_listing",
    ]
    missing = [c for c in required_exec if c not in exec_df.columns]
    if missing:
        sys.exit(f"ERROR: Missing columns in exec file: {missing}")

    required_enriched = ["isin", "year", "sector", "firm_size"]
    missing_e = [c for c in required_enriched if c not in enriched.columns]
    if missing_e:
        sys.exit(
            f"ERROR: Missing columns in enriched file: {missing_e}\n"
            f"  Run merge_orbis_firm_size.py first."
        )

    return exec_df, enriched


def join_firm_context(exec_df: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[2] Joining sector and firm_size …")

    exec_df["isin"] = normalise_isin(exec_df["isin"])
    exec_df["year"] = pd.to_numeric(exec_df["year"], errors="coerce").astype("Int64")
    enriched["isin"] = normalise_isin(enriched["isin"])
    enriched["year"] = pd.to_numeric(enriched["year"], errors="coerce").astype("Int64")

    context = (
        enriched[["isin", "year", "sector", "firm_size"]]
        .drop_duplicates(subset=["isin", "year"])
    )

    df = exec_df.merge(context, on=["isin", "year"], how="left")
    n = len(df)
    print(f"    sector matched    : {df['sector'].notna().sum():,} / {n:,} rows "
          f"({df['sector'].notna().mean()*100:.1f}%)")
    print(f"    firm_size matched : {df['firm_size'].notna().sum():,} / {n:,} rows "
          f"({df['firm_size'].notna().mean()*100:.1f}%)")

    if df["sector"].notna().sum() == 0:
        warnings.warn("Zero sector matches — check ISIN formats between files.")

    return df


def drop_unusable_rows(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[3] Dropping unusable rows …")
    before = len(df)

    df = df[df["total_comp"] > 0]
    print(f"    Dropped {before - len(df):,} rows with total_comp = 0")
    before = len(df)

    df = df[df["role"].notna() & (df["role"].str.strip() != "")]
    print(f"    Dropped {before - len(df):,} rows with missing role")
    before = len(df)

    df = df[df["days"] > 0]
    print(f"    Dropped {before - len(df):,} rows with days = 0")

    # Normalise role labels → CEO, CFO, Other
    df = df.copy()
    df["role"] = df["role"].str.strip().str.title()
    df["role"] = df["role"].replace({"Cfo": "CFO", "Ceo": "CEO"})

    print(f"    Remaining: {len(df):,} individual exec rows")
    print("    Role distribution:")
    for role, cnt in df["role"].value_counts().items():
        print(f"      {role}: {cnt:,}")
    return df


def annualise_compensation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale flow-based pay to a full-year equivalent per individual.
    Pension and one_time_payment are excluded (event-driven, not time-scaled).
    """
    print(f"\n[4] Annualising partial-year compensation …")
    partial = (df["days"] < 365).sum()
    print(f"    Partial-year rows (days < 365): {partial:,}")

    df = df.copy()
    scale = 365.0 / df["days"]
    for col in ANNUALISE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0) * scale

    print(f"    Annualised: {[c for c in ANNUALISE_COLS if c in df.columns]}")
    return df


def aggregate_to_seat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse individual exec rows to one row per isin × year × role (the seat).

    For each seat group:
      - Sum annualised comp components across all seat-holders
      - Divide by headcount → per-seat figures
      - Carry firm-level features (mode/first — same for all in group)
    """
    print(f"\n[5] Aggregating to seat level (isin × year × role) …")

    comp_cols = [
        "salary", "one_year_bonus", "multi_year_bonus",
        "total_equity_grants", "stock_grants", "option_grants",
        "other_annual_comp", "total_comp", "pension",
    ]
    comp_cols = [c for c in comp_cols if c in df.columns]

    # Firm-level columns: same for every exec at the same isin-year
    firm_cols = ["company_shortname", "sector", "firm_size",
                 "index_listing", "opting_out"]
    firm_cols = [c for c in firm_cols if c in df.columns]

    # --- Sum compensation per seat group ---
    comp_agg = (
        df.groupby(SEAT_KEY)[comp_cols]
        .sum(min_count=1)   # NaN if all values were NaN
        .reset_index()
    )

    # --- Headcount per seat group ---
    headcount = (
        df.groupby(SEAT_KEY)
        .size()
        .reset_index(name="n_seat_holders")
    )

    # --- Firm-level features: take first value (identical within group) ---
    firm_agg = (
        df.groupby(SEAT_KEY)[firm_cols]
        .first()
        .reset_index()
    )

    # --- Merge all together ---
    seat = comp_agg.merge(headcount, on=SEAT_KEY)
    seat = seat.merge(firm_agg, on=SEAT_KEY)

    # --- Per-seat compensation (sum / headcount) ---
    for col in comp_cols:
        seat[f"{col}_ps"] = safe_div(seat[col], seat["n_seat_holders"])

    before_rows = len(df)
    print(f"    Collapsed {before_rows:,} individual rows → {len(seat):,} seat rows")
    print("    Seat-level role distribution:")
    for role, cnt in seat["role"].value_counts().items():
        print(f"      {role}: {cnt:,}")
    print(f"    n_seat_holders distribution:")
    print(f"      {seat['n_seat_holders'].value_counts().to_dict()}")

    return seat


def compute_seat_features(seat: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[6] Computing seat-level pay features …")

    # Use per-seat total comp as the base for all ratios
    tc = seat["total_comp_ps"]

    # Pay level
    seat["comp_per_seat"] = tc
    seat["log_tc_seat"] = np.where(tc > 0, np.log(tc), np.nan)

    # Pay mix ratios — computed from summed components, not averaged individual ratios
    seat["ratio_salary"]  = safe_div(seat["salary_ps"],              tc)
    seat["ratio_sti"]     = safe_div(seat["one_year_bonus_ps"],      tc)
    seat["ratio_lti"]     = safe_div(seat["multi_year_bonus_ps"],    tc)
    seat["ratio_equity"]  = safe_div(seat["total_equity_grants_ps"], tc)
    seat["ratio_other"]   = safe_div(seat["other_annual_comp_ps"],   tc)
    seat["ratio_pension"] = safe_div(seat["pension_ps"].fillna(0),   tc)

    # Winsorise pension ratio — actuarial spikes can push it above 1
    n_over = (seat["ratio_pension"] > 1).sum()
    if n_over > 0:
        print(f"    WARNING: {n_over} seat-rows have ratio_pension > 1 "
              f"(actuarial spike) — winsorised to 1.0")
        seat["ratio_pension"] = seat["ratio_pension"].clip(upper=1.0)

    # VPI — variable pay intensity (unbounded, relative to fixed salary)
    variable_ps = (
        seat["one_year_bonus_ps"]
        + seat["multi_year_bonus_ps"].fillna(0)
        + seat["total_equity_grants_ps"]
    )
    seat["vpi"] = safe_div(variable_ps, seat["salary_ps"])

    # Equity instrument split (0 for seats with no equity grants)
    eq = seat["total_equity_grants_ps"]
    has_equity = eq > 0
    seat["ratio_stock"]  = 0.0
    seat["ratio_option"] = 0.0
    seat.loc[has_equity, "ratio_stock"] = safe_div(
        seat.loc[has_equity, "stock_grants_ps"].fillna(0), eq[has_equity]
    )
    seat.loc[has_equity, "ratio_option"] = safe_div(
        seat.loc[has_equity, "option_grants_ps"].fillna(0), eq[has_equity]
    )

    # Report
    feat_cols = [
        "log_tc_seat", "ratio_salary", "ratio_sti", "ratio_lti",
        "ratio_equity", "ratio_other", "ratio_pension",
        "vpi", "ratio_stock", "ratio_option",
    ]
    for col in feat_cols:
        null_pct = seat[col].isna().mean() * 100
        med = seat[col].median()
        print(f"    {col:<22}  {null_pct:.1f}% null  |  median={med:.3f}")

    return seat


def compute_structural_features(seat: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[7] Computing structural features …")

    # index_score: map index_listing, fill gaps within company via ffill/bfill
    seat = seat.sort_values(["isin", "year"]).copy()
    seat["index_score"] = seat["index_listing"].map(INDEX_MAP)
    seat["index_score"] = (
        seat.groupby("isin")["index_score"]
        .transform(lambda g: g.ffill().bfill())
    )
    n_null = seat["index_score"].isna().sum()
    if n_null > 0:
        global_med = seat["index_score"].median()
        seat["index_score"] = seat["index_score"].fillna(global_med)
        print(f"    index_score: {n_null} unfilled → global median ({global_med})")
    print(f"    index_score distribution: "
          + str(seat["index_score"].value_counts().to_dict()))

    # ever_opted_out: 1 if company ever had opting_out = 1 in any year
    seat["ever_opted_out"] = (
        seat.groupby("isin")["opting_out"]
        .transform(lambda g: g.fillna(0).max())
        .fillna(0)
        .astype(int)
    )
    print(f"    ever_opted_out=1: {(seat['ever_opted_out'] == 1).sum():,} seat-rows")

    return seat


def save_outputs(seat: pd.DataFrame, outdir: str) -> None:
    print(f"\n[8] Saving outputs to {outdir} …")
    os.makedirs(outdir, exist_ok=True)

    output_cols = [c for c in IDENTIFIER_COLS + FEATURE_COLS if c in seat.columns]
    out = seat[output_cols].copy()

    # Combined
    path_all = os.path.join(outdir, "exec_features_all.csv")
    out.to_csv(path_all, index=False)
    print(f"    exec_features_all.csv   — {len(out):,} rows × {len(out.columns)} cols")

    # Role-stratified
    for role in ["CEO", "CFO", "Other"]:
        role_df = out[out["role"] == role]
        path = os.path.join(outdir, f"exec_features_{role}.csv")
        role_df.to_csv(path, index=False)
        print(f"    exec_features_{role}.csv   — {len(role_df):,} seat-rows")

    # Feature completeness summary
    print(f"\n    Feature completeness:")
    for col in FEATURE_COLS:
        if col in out.columns:
            pct = (1 - out[col].isna().mean()) * 100
            bar = "█" * int(pct / 5)
            print(f"      {col:<28}  {pct:5.1f}%  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exec",     required=True,
                   help="Path to individual executive CSV")
    p.add_argument("--enriched", required=True,
                   help="Path to company_year_enriched.csv (pipe-delimited)")
    p.add_argument("--outdir",   default="./exec_output",
                   help="Output directory (default: ./exec_output)")
    return p.parse_args()


def main():
    args = parse_args()

    exec_df, enriched = load_and_validate(args.exec, args.enriched)
    df = join_firm_context(exec_df, enriched)
    df = drop_unusable_rows(df)
    df = annualise_compensation(df)
    seat = aggregate_to_seat(df)
    seat = compute_seat_features(seat)
    seat = compute_structural_features(seat)
    save_outputs(seat, args.outdir)

    print(f"\nDone.")
    print(f"  Output unit  : one row per isin × year × role (seat level)")
    print(f"  Cluster on   : sector, year, firm_size, index_score")
    print(f"  Pay level    : log_tc_seat  (log of comp_per_seat)")
    print(f"  Pay mix      : ratio_salary, ratio_sti, ratio_lti,")
    print(f"                 ratio_equity, ratio_other, ratio_pension")
    print(f"  Incentive    : vpi")
    print(f"  Equity split : ratio_stock, ratio_option")
    print(f"  Structural   : ever_opted_out, index_score")


if __name__ == "__main__":
    main()