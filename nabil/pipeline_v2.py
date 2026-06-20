"""
pipeline_v2.py  —  ExComp DAX Feature Engineering
Reads:  2008-2024_longitudinal_orbis.csv  (pipe-delimited, person-year)
Output: features_dax_all.csv             (company-year, all DAX companies 2006-2024)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

# ── DAX universe (companies that appear in the longitudinal file as DAX members) ──
DAX_COMPANIES = [
    "Adidas", "Airbus", "Allianz", "BASF", "Bayer", "Beiersdorf", "BMW",
    "Brenntag", "Commerzbank", "Continental", "Covestro", "Daimler",
    "Daimler Truck", "Delivery Hero", "Deutsche Bank", "Deutsche Börse",
    "Deutsche Lufthansa", "Deutsche Post", "Deutsche Postbank", "Deutsche Telekom",
    "E.ON", "Fresenius", "Fresenius Medical Care", "Hannover Rück",
    "Heidelberg Cement", "Heidelberg Materials", "Henkel", "Hypo Real Estate",
    "Infineon", "K+S", "Lanxess", "Linde", "MAN", "Mercedes-Benz", "Merck",
    "Metro", "MTU", "Munich RE", "Porsche", "Porsche AG", "Porsche SE",
    "Porsche Automobil Holding", "ProSiebenSat1", "Qiagen", "Rheinmetall",
    "RWE", "SAP", "Salzgitter", "Sartorius", "Siemens", "Siemens Energy",
    "Siemens Healthineers", "Symrise", "ThyssenKrupp", "TUI", "Volkswagen",
    "Vonovia", "Wirecard", "Zalando",
]

SECTOR_MAP = {
    "Adidas":                    "Consumer",
    "Airbus":                    "Industrials",
    "Allianz":                   "Financials",
    "BASF":                      "Materials",
    "Bayer":                     "Healthcare",
    "Beiersdorf":                "Consumer",
    "BMW":                       "Automotive",
    "Brenntag":                  "Materials",
    "Commerzbank":               "Financials",
    "Continental":               "Automotive",
    "Covestro":                  "Materials",
    "Daimler":                   "Automotive",
    "Daimler Truck":             "Automotive",
    "Delivery Hero":             "Consumer",
    "Deutsche Bank":             "Financials",
    "Deutsche Börse":            "Financials",
    "Deutsche Lufthansa":        "Industrials",
    "Deutsche Post":             "Industrials",
    "Deutsche Postbank":         "Financials",
    "Deutsche Telekom":          "Telecoms",
    "E.ON":                      "Energy",
    "Fresenius":                 "Healthcare",
    "Fresenius Medical Care":    "Healthcare",
    "Hannover Rück":             "Financials",
    "Heidelberg Cement":         "Materials",
    "Heidelberg Materials":      "Materials",
    "Henkel":                    "Consumer",
    "Hypo Real Estate":          "Financials",
    "Infineon":                  "Technology",
    "K+S":                       "Materials",
    "Lanxess":                   "Materials",
    "Linde":                     "Materials",
    "MAN":                       "Automotive",
    "Mercedes-Benz":             "Automotive",
    "Merck":                     "Healthcare",
    "Metro":                     "Consumer",
    "MTU":                       "Industrials",
    "Munich RE":                 "Financials",
    "Porsche":                   "Automotive",
    "Porsche AG":                "Automotive",
    "Porsche SE":                "Automotive",
    "Porsche Automobil Holding": "Automotive",
    "ProSiebenSat1":             "Consumer",
    "Qiagen":                    "Healthcare",
    "Rheinmetall":               "Industrials",
    "RWE":                       "Energy",
    "SAP":                       "Technology",
    "Salzgitter":                "Materials",
    "Sartorius":                 "Healthcare",
    "Siemens":                   "Industrials",
    "Siemens Energy":            "Energy",
    "Siemens Healthineers":      "Healthcare",
    "Symrise":                   "Materials",
    "ThyssenKrupp":              "Materials",
    "TUI":                       "Consumer",
    "Volkswagen":                "Automotive",
    "Vonovia":                   "Real Estate",
    "Wirecard":                  "Financials",
    "Zalando":                   "Consumer",
}

ESG_KEYWORDS = [
    "esg", "nachhaltig", "co2", "klima", "sozial", "diversity", "diversit",
    "gender", "scope", "carbon", "ghg", "environment", "umwelt", "csrd",
    "sustainability", "sdg", "emission", "frauen", "arbeitnehmer",
    "sicherheit", "mitarbeiter", "gesundheit", "co₂",
]


def load_raw():
    print("Loading longitudinal ORBIS file …")
    df = pd.read_csv(
        BASE / "2008-2024_longitudinal_orbis.csv",
        sep="|",
        on_bad_lines="skip",
        low_memory=False,
        quotechar='"',
    )
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


def flag_dax(df):
    """Keep only rows belonging to known DAX universe companies.

    Use company names from model_universe.csv as the authoritative list so that
    features_dax_all.csv stays aligned with the prediction universe.
    Fall back to the hard-coded DAX_COMPANIES list if the file is absent.
    """
    mu_path = BASE / "model_universe.csv"
    if mu_path.exists():
        mu_cos = set(pd.read_csv(mu_path)["company_shortname"].unique())
        print(f"  Universe from model_universe.csv: {len(mu_cos)} companies")
    else:
        mu_cos = set(DAX_COMPANIES)
        print("  Universe from hard-coded DAX_COMPANIES list")

    return df[df["company_shortname"].isin(mu_cos)].copy()


def detect_esg_per_exec(df):
    """Detect ESG KPIs and estimate ESG-linked pay fraction per exec-year."""
    kpi_cols    = ["stipi1",  "stipi2",  "stipi3",  "mtipi1",  "mtipi2",  "mtipi3"]
    weight_cols = ["stipi1weight", "stipi2weight", "stipi3weight",
                   "mtipi1weight", "mtipi2weight", "mtipi3weight"]

    def _is_esg(val):
        if pd.isna(val):
            return False
        return any(k in str(val).lower() for k in ESG_KEYWORDS)

    df["has_esg_kpi"] = df[kpi_cols].apply(lambda row: row.map(_is_esg).any(), axis=1)

    # ESG weight sum (0-100 %)
    esg_wt = pd.Series(0.0, index=df.index)
    for kpi_c, wgt_c in zip(kpi_cols, weight_cols):
        mask = df[kpi_c].map(_is_esg)
        raw  = pd.to_numeric(df[wgt_c], errors="coerce").fillna(0)
        esg_wt += mask * raw
    df["esg_kpi_weight"] = esg_wt

    return df


def aggregate_company_year(df):
    """Aggregate person-year rows to company-year level."""

    # ── Compensation structure from bt (before-tax) columns ──────────────────
    comp_agg = df.groupby(["company_shortname", "year"]).agg(
        total_comp_bt          = ("total_comp_bt",           "sum"),
        salary_bt              = ("salary_bt",               "sum"),
        one_year_bonus_bt      = ("one_year_bonus_bt",       "sum"),
        multi_year_bonus_bt    = ("multi_year_bonus_grants_bt", "sum"),
        total_equity_grants_bt = ("total_equity_grants_bt",  "sum"),
        other_annual_bt        = ("other_annual_bt",         "sum"),
        pension_bt             = ("pension_bt",              "sum"),
        n_executives           = ("exec_id",                 "nunique"),
        n_female               = ("female",                  "sum"),
        # ESG
        has_esg_kpi            = ("has_esg_kpi",             "any"),
        esg_kpi_weight_sum     = ("esg_kpi_weight",          "sum"),
        esg_kpi_weight_cnt     = ("esg_kpi_weight",          "count"),
        # Financial (same for all execs in a company-year)
        EMPL                   = ("EMPL",                    "first"),
        TURN                   = ("TURN",                    "first"),
        OPPL                   = ("OPPL",                    "first"),
        PLBT                   = ("PLBT",                    "first"),
        STAF                   = ("STAF",                    "first"),
        TOAS                   = ("TOAS",                    "first"),
        ROE                    = ("ROE",                     "first"),
        ROA                    = ("ROA",                     "first"),
        index_listing          = ("index_listing",           "first"),
    ).reset_index()

    # ── Comp structure percentages ───────────────────────────────────────────
    tot = comp_agg["total_comp_bt"].replace(0, np.nan)
    comp_agg["fixed_pct"]     = (comp_agg["salary_bt"] / tot * 100).round(1)
    comp_agg["sti_pct"]       = (comp_agg["one_year_bonus_bt"] / tot * 100).round(1)
    comp_agg["lti_pct"]       = (
        (comp_agg["multi_year_bonus_bt"] + comp_agg["total_equity_grants_bt"]) / tot * 100
    ).round(1)
    comp_agg["variable_pct"]  = (
        (comp_agg["one_year_bonus_bt"] + comp_agg["multi_year_bonus_bt"]
         + comp_agg["total_equity_grants_bt"]) / tot * 100
    ).round(1)
    comp_agg["pension_pct"]   = (comp_agg["pension_bt"] / tot * 100).round(1)

    # ── ESG pay fraction ─────────────────────────────────────────────────────
    # esg_kpi_weight_sum is the sum of individual exec weights → average it
    cnt = comp_agg["esg_kpi_weight_cnt"].replace(0, np.nan)
    comp_agg["esg_sti_weight_avg"] = (comp_agg["esg_kpi_weight_sum"] / cnt).round(1)

    # ── CEO compensation ─────────────────────────────────────────────────────
    ceo = df[df["ceo_flag_eoy"] == 1].groupby(["company_shortname", "year"]).agg(
        ceo_comp   = ("total_comp_bt", "sum"),
        ceo_salary = ("salary_bt",     "sum"),
        ceo_sti    = ("one_year_bonus_bt", "sum"),
        n_ceos     = ("exec_id",        "nunique"),
    ).reset_index()
    comp_agg = comp_agg.merge(ceo, on=["company_shortname", "year"], how="left")

    return comp_agg


def add_derived_features(cy):
    """Add YoY changes, rolling stats, ratios, flags."""
    cy = cy.sort_values(["company_shortname", "year"]).copy()

    # ── Financial renames ────────────────────────────────────────────────────
    cy.rename(columns={
        "EMPL": "employees",
        "TURN": "revenue",
        "OPPL": "ebit",
        "PLBT": "profit_before_tax",
        "STAF": "staff_costs",
        "TOAS": "total_assets",
    }, inplace=True)

    # ── Per-exec comp ────────────────────────────────────────────────────────
    cy["comp_per_exec"] = (cy["total_comp_bt"] / cy["n_executives"].replace(0, np.nan)).round(2)

    # ── YoY changes ─────────────────────────────────────────────────────────
    for co, grp in cy.groupby("company_shortname"):
        idx = grp.index
        cy.loc[idx, "comp_yoy_pct"]   = grp["total_comp_bt"].pct_change() * 100
        cy.loc[idx, "salary_yoy_pct"] = grp["salary_bt"].pct_change() * 100
        cy.loc[idx, "sti_yoy_pct"]    = grp["one_year_bonus_bt"].pct_change() * 100
        cy.loc[idx, "lti_yoy_pct"]    = grp["multi_year_bonus_bt"].pct_change() * 100
        cy.loc[idx, "ebit_yoy_pct"]   = grp["ebit"].pct_change() * 100
        cy.loc[idx, "empl_yoy_pct"]   = grp["employees"].pct_change() * 100

    # ── 3-year rolling stats ─────────────────────────────────────────────────
    for co, grp in cy.groupby("company_shortname"):
        idx = grp.index
        cy.loc[idx, "comp_3yr_mean"]       = grp["total_comp_bt"].rolling(3, min_periods=1).mean()
        cy.loc[idx, "comp_volatility_3yr"] = grp["comp_yoy_pct"].rolling(3, min_periods=2).std()
        cy.loc[idx, "lti_trend_3yr"]       = grp["lti_pct"].rolling(3, min_periods=2).mean()

    # ── "Schlechte Zeiten" flag: comp up ≥5% while ebit down ≥5% ─────────────
    cy["schlechte_zeiten"] = (
        (cy["comp_yoy_pct"] >= 5) & (cy["ebit_yoy_pct"] <= -5)
    ).astype(int)
    cy["schlechte_zeiten_score"] = (
        cy["comp_yoy_pct"].clip(0, 30) - cy["ebit_yoy_pct"].clip(-30, 0)
    ).round(1)

    # ── P4P metrics ──────────────────────────────────────────────────────────
    cy["p4p_gap_vs_ebit"] = (cy["comp_yoy_pct"] - cy["ebit_yoy_pct"]).round(1)
    cy["p4p_score_vs_ebit"] = np.where(
        cy["ebit_yoy_pct"].notna() & cy["comp_yoy_pct"].notna(),
        np.sign(cy["ebit_yoy_pct"]) * np.sign(cy["comp_yoy_pct"]),
        np.nan
    )

    # ── Female exec % ────────────────────────────────────────────────────────
    cy["female_exec_pct"] = (cy["n_female"] / cy["n_executives"].replace(0, np.nan) * 100).round(1)

    # ── CEO board premium ────────────────────────────────────────────────────
    cy["board_avg_comp"] = cy["comp_per_exec"]
    cy["ceo_board_premium_ratio"] = (cy["ceo_comp"] / cy["board_avg_comp"].replace(0, np.nan)).round(2)
    cy["ceo_board_premium_pct"]   = ((cy["ceo_board_premium_ratio"] - 1) * 100).round(1)

    # ── Executive/worker pay ratio ───────────────────────────────────────────
    worker_avg = cy["staff_costs"] / cy["employees"].replace(0, np.nan)
    cy["exec_worker_pay_ratio"] = (cy["comp_per_exec"] / worker_avg).round(1)

    # ── Sector ───────────────────────────────────────────────────────────────
    cy["sector"] = cy["company_shortname"].map(SECTOR_MAP).fillna("Other")

    # ── Peer rank within year (all companies) ────────────────────────────────
    cy["peer_rank_total_comp"] = cy.groupby("year")["total_comp_bt"].rank(ascending=False, method="min")
    cy["peer_pct_total_comp"]  = cy.groupby("year")["total_comp_bt"].rank(pct=True, ascending=True) * 100
    cy["peer_rank_per_exec"]   = cy.groupby("year")["comp_per_exec"].rank(ascending=False, method="min")

    # ── Sector rank ──────────────────────────────────────────────────────────
    cy["sector_rank"] = cy.groupby(["year", "sector"])["total_comp_bt"].rank(ascending=False, method="min")
    cy["sector_pct"]  = cy.groupby(["year", "sector"])["total_comp_bt"].rank(pct=True, ascending=True) * 100

    # ── Median comp by year (all companies) ─────────────────────────────────
    med = cy.groupby("year")["total_comp_bt"].median().rename("year_median_comp")
    cy  = cy.merge(med, on="year", how="left")
    cy["comp_vs_median_pct"] = ((cy["total_comp_bt"] / cy["year_median_comp"]) - 1) * 100

    # ── Comp vs own 3yr mean ─────────────────────────────────────────────────
    cy["comp_above_own_3yr_mean_pct"] = ((cy["total_comp_bt"] / cy["comp_3yr_mean"]) - 1) * 100

    # ── Crisis dummies ───────────────────────────────────────────────────────
    cy["crisis_GFC_2009"]        = (cy["year"] == 2009).astype(int)
    cy["crisis_EuroDebt_2012"]   = (cy["year"] == 2012).astype(int)
    cy["crisis_Dieselgate_2015"] = (cy["year"] == 2015).astype(int)
    cy["crisis_COVID_2020"]      = (cy["year"] == 2020).astype(int)
    cy["dieselgate_era"]         = cy["year"].between(2015, 2018).astype(int)
    cy["energiewende_era"]       = cy["year"].between(2011, 2016).astype(int)
    cy["is_energy_company"]      = (cy["sector"] == "Energy").astype(int)
    cy["is_auto_company"]        = (cy["sector"] == "Automotive").astype(int)

    return cy


def add_anomaly_scores(cy):
    """Simple rolling z-score anomaly detection on comp_yoy_pct."""
    cy = cy.sort_values(["company_shortname", "year"]).copy()
    for co, grp in cy.groupby("company_shortname"):
        idx = grp.index
        roll_mean = grp["comp_yoy_pct"].rolling(5, min_periods=2).mean()
        roll_std  = grp["comp_yoy_pct"].rolling(5, min_periods=2).std()
        z = (grp["comp_yoy_pct"] - roll_mean) / roll_std.replace(0, np.nan)
        cy.loc[idx, "z_score"] = z

    # Percentile within year
    cy["anomaly_score_pct"] = cy.groupby("year")["z_score"].rank(pct=True) * 100
    cy["is_anomaly"]        = (cy["anomaly_score_pct"] >= 90).astype(int)

    cy.drop(columns=["z_score"], inplace=True, errors="ignore")
    return cy


def main():
    raw = load_raw()
    dax = flag_dax(raw)
    print(f"  DAX rows: {len(dax):,} | companies: {dax['company_shortname'].nunique()} | years: {sorted(dax['year'].unique())}")

    dax = detect_esg_per_exec(dax)

    print("Aggregating to company-year …")
    cy  = aggregate_company_year(dax)
    print(f"  Company-year rows: {len(cy)}")

    print("Deriving features …")
    cy  = add_derived_features(cy)
    cy  = add_anomaly_scores(cy)

    # Round floats
    float_cols = cy.select_dtypes(include="float").columns
    cy[float_cols] = cy[float_cols].round(3)

    # Final column order — match features_dax15.csv where possible
    key_cols = [
        "company_shortname", "year", "sector", "index_listing",
        "n_executives", "n_female", "female_exec_pct",
        "total_comp_bt", "salary_bt", "one_year_bonus_bt", "multi_year_bonus_bt",
        "total_equity_grants_bt", "other_annual_bt", "pension_bt",
        "fixed_pct", "sti_pct", "lti_pct", "variable_pct", "pension_pct",
        "comp_per_exec", "comp_yoy_pct", "salary_yoy_pct", "sti_yoy_pct", "lti_yoy_pct",
        "comp_3yr_mean", "comp_volatility_3yr", "lti_trend_3yr",
        "ceo_comp", "ceo_salary", "ceo_sti", "n_ceos",
        "board_avg_comp", "ceo_board_premium_ratio", "ceo_board_premium_pct",
        "peer_rank_total_comp", "peer_pct_total_comp", "peer_rank_per_exec",
        "sector_rank", "sector_pct", "year_median_comp", "comp_vs_median_pct",
        "comp_above_own_3yr_mean_pct",
        "employees", "revenue", "ebit", "profit_before_tax", "staff_costs",
        "total_assets", "ROE", "ROA",
        "ebit_yoy_pct", "empl_yoy_pct",
        "p4p_gap_vs_ebit", "p4p_score_vs_ebit",
        "exec_worker_pay_ratio",
        "has_esg_kpi", "esg_sti_weight_avg",
        "anomaly_score_pct", "is_anomaly",
        "schlechte_zeiten", "schlechte_zeiten_score",
        "crisis_GFC_2009", "crisis_EuroDebt_2012", "crisis_Dieselgate_2015", "crisis_COVID_2020",
        "is_energy_company", "is_auto_company", "dieselgate_era", "energiewende_era",
    ]
    available = [c for c in key_cols if c in cy.columns]
    rest = [c for c in cy.columns if c not in available]
    cy = cy[available + rest]

    out = BASE / "features_dax_all.csv"
    cy.to_csv(out, index=False)
    print(f"\nSaved {out.name}  ({len(cy)} rows × {cy.shape[1]} cols)")
    print(f"Companies: {cy['company_shortname'].nunique()}")
    print(f"Years:     {sorted(cy['year'].unique())}")
    print("\nColumn preview:")
    print(cy.head(3).to_string())


if __name__ == "__main__":
    main()
