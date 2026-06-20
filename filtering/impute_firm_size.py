"""
impute_firm_size.py
====================
Fills missing firm_size values in exec_features_all.csv using a
three-step imputation cascade:

  1. Sector × year median   — primary: same industry, same year
  2. Sector median          — fallback: same industry, any year
                              (used when an entire sector-year group
                               has no firm_size at all)
  3. Global median          — last resort: dataset-wide median
                              (used when an entire sector has no data)

Also saves an audit file showing the imputation source for every row.

Usage
-----
    python impute_firm_size.py --inp exec_features_all.csv
    python impute_firm_size.py --inp exec_features_all.csv --out exec_features_imputed.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp", required=True,
                   help="Path to exec_features_all.csv")
    p.add_argument("--out", default=None,
                   help="Output path (default: <inp>_imputed.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = args.out or os.path.splitext(args.inp)[0] + "_imputed.csv"
    audit_path = os.path.splitext(out_path)[0] + "_firm_size_audit.csv"

    # --- Load ---
    print(f"\nLoading: {args.inp}")
    df = pd.read_csv(args.inp, low_memory=False)
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} cols")

    if "firm_size" not in df.columns:
        raise ValueError("'firm_size' column not found in input file.")
    if "sector" not in df.columns or "year" not in df.columns:
        raise ValueError("'sector' and 'year' columns required for imputation.")

    n_null_before = df["firm_size"].isna().sum()
    print(f"\nfirm_size before imputation: {n_null_before:,} null / {len(df):,} rows "
          f"({n_null_before / len(df) * 100:.1f}%)")

    # --- Track imputation source ---
    df["firm_size_source"] = np.where(df["firm_size"].notna(), "original", pd.NA)

    # --- Step 1: sector × year median ---
    sy_medians = df.groupby(["sector", "year"])["firm_size"]\
        .transform(lambda g: g.fillna(g.median()))
    mask1 = df["firm_size"].isna() & sy_medians.notna()
    df.loc[mask1, "firm_size_source"] = "sector_year_median"
    df["firm_size"] = df["firm_size"].fillna(sy_medians)
    print(f"\nStep 1 — sector × year median:  filled {mask1.sum():,} rows")

    # --- Step 2: sector median ---
    s_medians = df.groupby("sector")["firm_size"]\
        .transform(lambda g: g.fillna(g.median()))
    mask2 = df["firm_size"].isna() & s_medians.notna()
    df.loc[mask2, "firm_size_source"] = "sector_median"
    df["firm_size"] = df["firm_size"].fillna(s_medians)
    print(f"Step 2 — sector median:         filled {mask2.sum():,} rows")

    # --- Step 3: global median ---
    global_median = df["firm_size"].median()
    mask3 = df["firm_size"].isna()
    df.loc[mask3, "firm_size_source"] = "global_median"
    df["firm_size"] = df["firm_size"].fillna(global_median)
    print(f"Step 3 — global median ({global_median:.4f}): "
          f"filled {mask3.sum():,} rows")

    # --- Summary ---
    print(f"\nImputation source breakdown:")
    for src, cnt in df["firm_size_source"].value_counts().items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"  {src:<22}  {cnt:>5,}  ({pct:5.1f}%)  {bar}")

    print(f"\nfirm_size after imputation: {df['firm_size'].isna().sum()} null")

    # --- Save main output (without source column) ---
    df_out = df.drop(columns=["firm_size_source"])
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # --- Save audit file ---
    audit = df[["isin", "company_shortname", "year", "role",
                "sector", "firm_size", "firm_size_source"]].copy()
    audit.to_csv(audit_path, index=False)
    print(f"Audit  → {audit_path}")


if __name__ == "__main__":
    main()