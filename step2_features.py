#!/usr/bin/env python3
"""Step 2 — feature build for the CEO-year panel.

Reads:  Data/processed/ceo_year_clean.csv
Writes: Data/processed/ceo_year_features.csv

Adds:
- Pay-mix %s (pct_fixed, pct_sti, pct_lti, pct_other) — feed Lens 2 too
- Peer key: index_listing × year (forward-filled to plug the ~4% missing)
- ORBIS financials: revenue (OPRE), EBIT proxy (OPPL), ROA, ROE, TOAS, EMPL, STAF
- log_revenue, ebit_margin

Design rules (locked in by reviewer):
- ONE peer scheme: index_listing × year. Sector is kept on the row but is NOT
  a model feature (20% coverage = too thin). Use sector for showcase narrative only.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PANEL_IN = Path("Data/processed/ceo_year_clean.csv")
ORBIS_FULL = Path("Data/ORBIS_Abzug_DE_2005_2024.csv")
OUT_FILE = Path("Data/processed/ceo_year_features.csv")

# ORBIS columns we need. EBIT in ORBIS is OPPL (operating profit/loss).
ORBIS_COLS = ["SD_ISIN", "CLOSDATE", "CLOSDATE_year", "CONSCODE", "NR_MONTHS",
              "OPRE", "TURN", "OPPL", "ROA", "ROE", "TOAS", "EMPL", "STAF"]


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_IN)
    return df


def add_pay_mix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    salary = df["salary"].fillna(0)
    sti = df["one_year_bonus"].fillna(0)
    lti = df["multi_year_bonus"].fillna(0) + df["total_equity_grants"].fillna(0)
    total = df["total_comp"]
    df["pct_fixed"] = salary / total
    df["pct_sti"] = sti / total
    df["pct_lti"] = lti / total
    df["pct_other"] = 1.0 - (df["pct_fixed"] + df["pct_sti"] + df["pct_lti"])
    return df


def ffill_index_listing(df: pd.DataFrame) -> pd.DataFrame:
    """Index membership is sticky across years — forward-fill within company,
    then back-fill to cover leading gaps."""
    df = df.sort_values(["isin", "year"]).copy()
    df["index_listing"] = (df.groupby("isin")["index_listing"]
                             .transform(lambda s: s.ffill().bfill()))
    return df


def load_orbis_financials(panel_isins: set[str]) -> pd.DataFrame:
    csv.field_size_limit(sys.maxsize)
    # Read in chunks; full file is ~860MB. Only keep rows whose SD_ISIN is in panel.
    chunks = []
    reader = pd.read_csv(ORBIS_FULL, usecols=ORBIS_COLS, chunksize=100_000,
                         low_memory=False)
    for chunk in reader:
        chunks.append(chunk[chunk["SD_ISIN"].isin(panel_isins)])
    orbis = pd.concat(chunks, ignore_index=True)
    return orbis


def dedup_orbis(orbis: pd.DataFrame) -> pd.DataFrame:
    """Pick one row per (SD_ISIN, CLOSDATE_year).
    Prefer: consolidated (CONSCODE starts with 'C') > unconsolidated,
            12-month fiscal year, latest CLOSDATE.
    """
    orbis = orbis.copy()
    orbis["_conscode_rank"] = orbis["CONSCODE"].fillna("Z").str.startswith("C").map({True: 0, False: 1})
    orbis["_months_rank"] = (orbis["NR_MONTHS"] != 12).astype(int)
    orbis["CLOSDATE_dt"] = pd.to_datetime(orbis["CLOSDATE"], format="%d/%m/%Y", errors="coerce")
    orbis = orbis.sort_values(
        ["SD_ISIN", "CLOSDATE_year", "_conscode_rank", "_months_rank", "CLOSDATE_dt"],
        ascending=[True, True, True, True, False],
    )
    orbis = orbis.drop_duplicates(["SD_ISIN", "CLOSDATE_year"], keep="first")
    return orbis.drop(columns=["_conscode_rank", "_months_rank", "CLOSDATE_dt", "CLOSDATE", "NR_MONTHS", "CONSCODE"])


def merge_financials(panel: pd.DataFrame, orbis: pd.DataFrame) -> pd.DataFrame:
    orbis = orbis.rename(columns={
        "SD_ISIN": "isin", "CLOSDATE_year": "year",
        "OPRE": "revenue", "TURN": "turnover", "OPPL": "ebit",
        "ROA": "roa", "ROE": "roe", "TOAS": "total_assets",
        "EMPL": "employees", "STAF": "staff_costs",
    })
    orbis["year"] = orbis["year"].astype("Int64")
    panel["year"] = panel["year"].astype("Int64")
    merged = panel.merge(orbis, on=["isin", "year"], how="left")
    # OPRE missing? fall back to TURN.
    merged["revenue"] = merged["revenue"].fillna(merged["turnover"])
    merged = merged.drop(columns=["turnover"])
    return merged


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_revenue"] = np.log1p(df["revenue"])
    df["ebit_margin"] = df["ebit"] / df["revenue"]
    df.loc[df["revenue"] <= 0, "ebit_margin"] = np.nan
    return df


def report(df: pd.DataFrame) -> None:
    print(f"rows: {len(df)}   columns: {df.shape[1]}")
    print(f"\npay-mix sanity (CEO median):")
    for c in ["pct_fixed", "pct_sti", "pct_lti", "pct_other"]:
        print(f"  {c:<10} median={df[c].median():.2%}  mean={df[c].mean():.2%}")

    print(f"\npeer-key coverage:")
    print(f"  index_listing not-null: {df['index_listing'].notna().sum()} / {len(df)} "
          f"({df['index_listing'].notna().mean():.1%})")
    print(f"\nindex_listing distribution:")
    print(df["index_listing"].value_counts(dropna=False).to_string())

    print(f"\nORBIS financial coverage on CEO-year:")
    for c in ["revenue", "ebit", "roa", "roe", "total_assets", "employees", "staff_costs"]:
        n = df[c].notna().sum()
        print(f"  {c:<13} {n:>4} / {len(df)}  ({n/len(df):.0%})")

    print(f"\nORBIS coverage by sector (DAX15 subset, sanity):")
    dax15 = df[df["sector"].notna()]
    if len(dax15):
        cov = dax15.groupby("sector")["revenue"].apply(lambda s: s.notna().mean())
        print(cov.to_string())


def main() -> None:
    if not PANEL_IN.exists():
        sys.exit(f"missing input — run step0_clean.py first: {PANEL_IN}")
    if not ORBIS_FULL.exists():
        sys.exit(f"missing ORBIS file: {ORBIS_FULL}")

    print("loading panel...")
    panel = load_panel()
    print(f"  {len(panel)} CEO-year rows")

    print("computing pay-mix %s...")
    panel = add_pay_mix(panel)

    print("forward-filling index_listing within company...")
    before = panel["index_listing"].notna().sum()
    panel = ffill_index_listing(panel)
    after = panel["index_listing"].notna().sum()
    print(f"  index_listing not-null: {before} -> {after}")

    print("extracting ORBIS financials for panel companies...")
    panel_isins = set(panel["isin"].unique())
    orbis = load_orbis_financials(panel_isins)
    print(f"  ORBIS rows for panel companies: {len(orbis)}")
    orbis = dedup_orbis(orbis)
    print(f"  after dedup to one row per (isin, year): {len(orbis)}")

    print("merging financials into panel...")
    panel = merge_financials(panel, orbis)
    panel = add_derived(panel)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_FILE, index=False)
    print()
    report(panel)
    print(f"\nwrote: {OUT_FILE}")


if __name__ == "__main__":
    main()
