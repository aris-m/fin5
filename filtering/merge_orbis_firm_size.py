"""
merge_orbis_firm_size.py
========================
Merges ORBIS firm-size variables into company_year_with_sector.csv
by matching SD_ISIN × CLOSDATE_year, then reduces log_toas, log_opre,
and log_empl into a single composite firm_size score via PCA (PC1).

Key facts about this ORBIS export
----------------------------------
- ISIN column  : SD_ISIN  (many rows are NaN — only listed companies have it)
- Year column  : CLOSDATE_year  (already an integer, e.g. 2018)
- Units        : full EUR (not thousands). Monetary cols are divided by 1000
                 on output so they match the compensation file's €k scale.
- Financial col names are UPPERCASE in the export.

Output
------
Adds exactly two columns to the compensation file:
  firm_size   — PC1 score from PCA on [log_toas, log_opre, log_empl].
                Higher = larger firm. Standardised (mean≈0, std≈1).
  opre_match  — 1 if ORBIS data was found for that isin-year, else 0.

Usage
-----
    python merge_orbis_firm_size.py \
        --comp  company_year_with_sector.csv \
        --orbis ORBIS_Abzug_DE_2005_2024.csv \
        --out   company_year_enriched.csv

Optional
--------
    --fill-method  none | ffill | nearest   (default: none)
                   ffill  = carry last known value forward within each company
                   nearest = fill from nearest year in either direction
"""

import argparse
import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Only the three columns needed to compute firm_size
# ---------------------------------------------------------------------------

MONETARY_COLS = {
    "OPRE": "opre",   # Operating Revenue  (÷1000 → €k)
    "TOAS": "toas",   # Total Assets       (÷1000 → €k)
}

NON_MONETARY_COLS = {
    "EMPL": "empl",   # Number of Employees (headcount, no rescaling)
}

ALL_TARGET = {**MONETARY_COLS, **NON_MONETARY_COLS}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise_isin(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def read_compensation(path: str) -> pd.DataFrame:
    print(f"\n[1] Loading compensation file: {path}")
    df = pd.read_csv(path, sep="|", low_memory=False)
    print(f"    {df.shape[0]:,} rows × {df.shape[1]} cols")
    df["isin"] = normalise_isin(df["isin"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    print(f"    {df['isin'].nunique()} unique ISINs | years {df['year'].min()}–{df['year'].max()}")
    return df


def read_orbis(path: str) -> pd.DataFrame:
    print(f"\n[2] Loading ORBIS file: {path}")
    print(f"    (large file — this may take a moment…)")

    needed = ["SD_ISIN", "CLOSDATE_YEAR"] + list(ALL_TARGET.keys())

    header = pd.read_csv(path, nrows=0).columns.tolist()
    header_upper = {c.upper(): c for c in header}

    usecols = []
    missing = []
    for col in needed:
        actual = header_upper.get(col.upper())
        if actual:
            usecols.append(actual)
        else:
            missing.append(col)

    if "SD_ISIN" in missing or "CLOSDATE_YEAR" in missing:
        sys.exit(
            f"ERROR: Essential columns missing from ORBIS file.\n"
            f"  Missing: {missing}\n"
            f"  Available (first 20): {header[:20]}"
        )
    if missing:
        warnings.warn(f"These target columns not found in ORBIS, skipping: {missing}")

    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    print(f"    Loaded {df.shape[0]:,} rows × {df.shape[1]} cols (selected columns only)")

    df["SD_ISIN"] = normalise_isin(df["SD_ISIN"])
    df["CLOSDATE_YEAR"] = pd.to_numeric(df["CLOSDATE_YEAR"], errors="coerce").astype("Int64")

    before = len(df)
    df = df[df["SD_ISIN"].notna() & (df["SD_ISIN"] != "NAN") & (df["SD_ISIN"] != "")]
    print(f"    Rows with valid SD_ISIN: {len(df):,} (dropped {before - len(df):,} without ISIN)")

    before = len(df)
    df = df.dropna(subset=["CLOSDATE_YEAR"])
    if len(df) < before:
        print(f"    Dropped {before - len(df):,} rows with null year")

    df = df.rename(columns={"SD_ISIN": "isin", "CLOSDATE_YEAR": "year"})

    for col in [c for c in df.columns if c not in ("isin", "year")]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dupes = df.duplicated(subset=["isin", "year"]).sum()
    if dupes:
        print(f"    WARNING: {dupes:,} duplicate isin-year rows — keeping first occurrence.")
        df = df.drop_duplicates(subset=["isin", "year"], keep="first")

    rename_fin = {k: v for k, v in ALL_TARGET.items() if k in df.columns}
    df = df.rename(columns=rename_fin)

    print(f"    Usable rows: {len(df):,} | unique ISINs: {df['isin'].nunique():,} | "
          f"years {df['year'].min()}–{df['year'].max()}")
    return df


def rescale_to_thousands(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[3] Rescaling monetary columns from EUR → €k (÷ 1000) …")
    for orbis_col, out_col in MONETARY_COLS.items():
        if out_col in df.columns:
            df[out_col] = df[out_col] / 1000
    print(f"    Scaled: {list(MONETARY_COLS.values())}")
    return df


def merge_data(comp: pd.DataFrame, orbis: pd.DataFrame, fill_method: str) -> pd.DataFrame:
    print(f"\n[4] Merging on isin × year …")

    comp_isins  = set(comp["isin"].unique())
    orbis_isins = set(orbis["isin"].unique())
    isin_overlap = comp_isins & orbis_isins
    year_overlap = set(comp["year"].dropna().unique()) & set(orbis["year"].dropna().unique())

    print(f"    ISIN overlap : {len(isin_overlap)} / {len(comp_isins)} comp companies found in ORBIS")
    print(f"    Year overlap : {sorted(year_overlap)}")
    if len(isin_overlap) == 0:
        print(f"\n    *** ZERO ISIN OVERLAP — sample comp ISINs  : {sorted(comp_isins)[:5]}")
        print(f"    *** ZERO ISIN OVERLAP — sample ORBIS ISINs : {sorted(orbis_isins)[:5]}")
        sys.exit("ERROR: No ISIN overlap between comp and ORBIS. Check ISIN formats.")

    merged = comp.merge(orbis, on=["isin", "year"], how="left")

    # Coverage flag — 1 if any ORBIS data was matched for this row
    size_cols = [v for v in ALL_TARGET.values() if v in merged.columns]
    merged["opre_match"] = merged[size_cols].notna().any(axis=1).astype(int)

    matched = merged["opre_match"].sum()
    total   = len(merged)
    print(f"    Matched {matched:,} / {total:,} compensation rows ({matched/total*100:.1f}%)")

    if fill_method != "none" and size_cols:
        print(f"    Applying fill_method='{fill_method}' within each ISIN …")
        merged = merged.sort_values(["isin", "year"])
        if fill_method == "ffill":
            merged[size_cols] = merged.groupby("isin")[size_cols].transform(lambda g: g.ffill())
        elif fill_method == "nearest":
            merged[size_cols] = merged.groupby("isin")[size_cols].transform(
                lambda g: g.ffill().bfill()
            )
        print(f"    Rows with data after fill: "
              f"{merged[size_cols[0]].notna().sum():,} / {total:,}")

    return merged


def compute_firm_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log_toas, log_opre, log_empl then reduce to a single
    firm_size composite score using PCA (first principal component).

    Only rows where all three logs are non-null contribute to fitting
    the PCA. All other rows get firm_size = NaN.

    The intermediate log columns are dropped from the output — only
    firm_size is kept.
    """
    print(f"\n[5] Computing firm_size composite score …")

    # Log-transform (only valid for positive values)
    for raw, log in [("toas", "log_toas"), ("opre", "log_opre"), ("empl", "log_empl")]:
        if raw in df.columns:
            df[log] = np.where(df[raw] > 0, np.log(df[raw]), np.nan)
        else:
            df[log] = np.nan
            warnings.warn(f"Column '{raw}' not found in merged data — log_{raw} will be all NaN.")

    log_cols = ["log_toas", "log_opre", "log_empl"]
    complete_mask = df[log_cols].notna().all(axis=1)
    n_complete = complete_mask.sum()
    n_total    = len(df)
    print(f"    Rows with all three log values present: {n_complete:,} / {n_total:,}")

    df["firm_size"] = np.nan

    if n_complete >= 2:
        X = df.loc[complete_mask, log_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=1)
        scores = pca.fit_transform(X_scaled).flatten()

        var_explained = pca.explained_variance_ratio_[0] * 100
        loadings = dict(zip(log_cols, pca.components_[0]))
        print(f"    PC1 variance explained : {var_explained:.1f}%")
        print(f"    PC1 loadings           : { {k: round(v,3) for k,v in loadings.items()} }")
        if var_explained < 60:
            warnings.warn(
                f"PC1 explains only {var_explained:.1f}% of variance across the three size proxies. "
                f"Consider keeping log_toas, log_opre, log_empl as separate features instead."
            )

        df.loc[complete_mask, "firm_size"] = scores
    else:
        warnings.warn(
            f"Only {n_complete} rows have complete size data — skipping PCA. "
            f"firm_size will be all NaN."
        )

    # Drop the intermediate raw and log columns — firm_size is the only output
    cols_to_drop = [c for c in ["toas", "opre", "empl", "log_toas", "log_opre", "log_empl"]
                    if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"    Dropped intermediate columns: {cols_to_drop}")
    print(f"    firm_size range: [{df['firm_size'].min():.3f}, {df['firm_size'].max():.3f}] "
          f"| null: {df['firm_size'].isna().sum():,}")

    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n[6] Output summary")
    print(f"    Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    if "firm_size" in df.columns and df["firm_size"].notna().any():
        d = df["firm_size"].describe()
        print(f"\n    firm_size (PC1 composite of log_toas, log_opre, log_empl):")
        print(f"      mean={d['mean']:.3f}  std={d['std']:.3f}  "
              f"min={d['min']:.3f}  median={d['50%']:.3f}  max={d['max']:.3f}")
        print(f"      null: {df['firm_size'].isna().sum():,} rows")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--comp",        required=True,  help="Path to company_year_with_sector.csv")
    p.add_argument("--orbis",       required=True,  help="Path to ORBIS CSV export")
    p.add_argument("--out",         default=None,   help="Output path (default: <comp>_enriched.csv)")
    p.add_argument("--fill-method", default="none", choices=["none", "ffill", "nearest"],
                   help="How to fill missing years per company (default: none)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.out is None:
        base     = os.path.splitext(args.comp)[0]
        args.out = base + "_enriched.csv"

    comp   = read_compensation(args.comp)
    orbis  = read_orbis(args.orbis)
    orbis  = rescale_to_thousands(orbis)
    merged = merge_data(comp, orbis, args.fill_method)
    merged = compute_firm_size(merged)
    print_summary(merged)

    merged.to_csv(args.out, sep="|", index=False)
    new_cols = [c for c in ["firm_size", "opre_match"] if c in merged.columns]
    print(f"\n[7] Saved → {args.out}")
    print(f"    New columns added ({len(new_cols)}): {new_cols}")


if __name__ == "__main__":
    main()