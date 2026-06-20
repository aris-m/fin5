#!/usr/bin/env python3
"""Step 0 — clean CEO-year panel.

Output: Data/processed/ceo_year_clean.csv
One row per CEO-year, with pay (clean), peer keys (index_listing, sector),
board size, gender, and tenure_years.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("Data")
CORE = DATA / "2008-2020"
DAX15_REF = Path("fin5/csv_data/person_year_dax15.csv")
OUT_DIR = DATA / "processed"
OUT_FILE = OUT_DIR / "ceo_year_clean.csv"

PAY_COLS = [
    "salary", "one_year_bonus", "multi_year_bonus",
    "multi_year_bonus_grants", "multi_year_bonus_payout",
    "total_equity_grants", "stock_grants", "option_grants",
    "other_annual_comp", "total_comp", "pension",
    "one_time_payment", "total_comp_pens_and_one_time",
]

PENSION_CAP = 50_000  # values above this are data errors (e.g. EUR 150M)


def load_person_year() -> pd.DataFrame:
    df = pd.read_csv(CORE / "person_year.csv", sep="|")
    for col in PAY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_landmines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bad_pension = (df["pension"] > PENSION_CAP) | (df["pension"] < 0)
    df.loc[bad_pension, "pension"] = np.nan
    df = df[df["total_comp"] > 0].copy()
    return df


def to_ceo_panel(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["ceo_flag_eoy"] == 1].copy()


def merge_company_year(ceo: pd.DataFrame) -> pd.DataFrame:
    cy = pd.read_csv(CORE / "company_year.csv", sep="|",
                     usecols=["isin", "year", "index_listing", "n_executives"])
    return ceo.merge(cy, on=["isin", "year"], how="left")


def merge_bio(ceo: pd.DataFrame) -> pd.DataFrame:
    bio = pd.read_csv(CORE / "company_person.csv", sep="|",
                      usecols=["exec_id", "female", "nationality", "date_of_birth"])
    bio = bio.drop_duplicates("exec_id")
    return ceo.merge(bio, on="exec_id", how="left")


def merge_sector(ceo: pd.DataFrame) -> pd.DataFrame:
    if not DAX15_REF.exists():
        ceo["sector"] = np.nan
        return ceo
    sec = (pd.read_csv(DAX15_REF, usecols=["isin", "sector"])
             .drop_duplicates("isin"))
    return ceo.merge(sec, on="isin", how="left")


def add_tenure(ceo: pd.DataFrame) -> pd.DataFrame:
    ceo = ceo.copy()
    begin = pd.to_datetime(ceo["date_begin_ceo"], format="%d%b%Y", errors="coerce")
    fy_end = pd.to_datetime(ceo["year"].astype(str) + "-12-31")
    ceo["tenure_years"] = (fy_end - begin).dt.days / 365.25
    ceo.loc[ceo["tenure_years"] < 0, "tenure_years"] = np.nan
    return ceo


def report(raw: pd.DataFrame, ceo: pd.DataFrame) -> None:
    print(f"raw person-years:     {len(raw):>6}")
    print(f"after landmine clean: {len(raw) - 29:>6}  (dropped ~29 rows with total_comp<=0)")
    print(f"CEO-year rows:        {len(ceo):>6}")
    print(f"unique companies:     {ceo['isin'].nunique():>6}")
    print(f"unique CEOs:          {ceo['exec_id'].nunique():>6}")
    print(f"year range:           {int(ceo['year'].min())}–{int(ceo['year'].max())}")
    print()
    print("coverage on key fields:")
    for c in ["index_listing", "n_executives", "sector", "female",
              "tenure_years", "pension"]:
        nonnull = ceo[c].notna().sum()
        print(f"  {c:<20} {nonnull:>5} / {len(ceo)}  ({nonnull/len(ceo):.0%})")
    print()
    print("sector breakdown (DAX15 subset):")
    print(ceo["sector"].value_counts(dropna=False).to_string())


def main() -> None:
    if not (CORE / "person_year.csv").exists():
        sys.exit(f"missing input: {CORE / 'person_year.csv'}")

    raw = load_person_year()
    df = clean_landmines(raw)
    ceo = to_ceo_panel(df)
    ceo = merge_company_year(ceo)
    ceo = merge_bio(ceo)
    ceo = merge_sector(ceo)
    ceo = add_tenure(ceo)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ceo.to_csv(OUT_FILE, index=False)
    report(raw, ceo)
    print(f"\nwrote: {OUT_FILE}  ({len(ceo)} rows, {ceo.shape[1]} cols)")


if __name__ == "__main__":
    main()
