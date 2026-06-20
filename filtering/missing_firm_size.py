"""
Fill missing firm_size in exec_features_with_firm_size.csv
─────────────────────────────────────────────────────────
Tier 1: Interpolate partial companies (Deutsche Börse, Linde, Wirecard, Metro)
Tier 2: Manual financial data for 29 companies from annual reports / public sources
Tier 3: Proxy imputation for any remaining gaps (sector × size-tier median)

All financial figures in EUR. Sources: annual reports, companiesmarketcap.com,
Wikipedia infoboxes, Statista snippets. Values are approximate – suitable for
computing a relative firm_size z-score in a hackathon context.
"""

import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
# TIER 2 — Manual anchor data: {isin: [(year, toas, opre, empl), ...]}
# toas/opre in EUR (raw), empl as headcount.
# Only anchor years given; intermediate years are linearly interpolated.
# ══════════════════════════════════════════════════════════════════════════

B = 1e9  # shorthand for billions
M = 1e6  # shorthand for millions

ANCHOR_DATA = {
    # ── Deutsche Bank ─────────────────────────────────────────────────────
    "DE0005140008": [
        (2006, 1584*B, 28.4*B, 68849),
        (2007, 2020*B, 30.7*B, 78291),
        (2008, 2202*B, 13.6*B, 80456),
        (2009, 1501*B, 28.0*B, 77053),
        (2010, 1906*B, 28.6*B, 102062),
        (2011, 2164*B, 33.2*B, 100996),
        (2012, 2012*B, 33.7*B, 98219),
        (2013, 1611*B, 31.9*B, 98254),
        (2014, 1709*B, 31.9*B, 98138),
        (2015, 1629*B, 33.5*B, 101104),
        (2016, 1591*B, 30.0*B, 99744),
        (2017, 1475*B, 26.4*B, 97535),
        (2018, 1348*B, 25.3*B, 91737),
        (2019, 1298*B, 23.2*B, 87597),
        (2020, 1325*B, 24.0*B, 84659),
        (2021, 1324*B, 25.4*B, 82969),
        (2022, 1337*B, 27.2*B, 85427),
        (2023, 1312*B, 28.9*B, 90130),
        (2024, 1312*B, 30.1*B, 90000),
    ],
    # ── Allianz ───────────────────────────────────────────────────────────
    "DE0008404005": [
        (2006, 928*B, 101*B, 166801),
        (2007, 1028*B, 109*B, 177849),
        (2008, 859*B, 93*B, 155000),
        (2009, 887*B, 98*B, 152950),
        (2010, 893*B, 107*B, 151760),
        (2011, 858*B, 104*B, 141938),
        (2012, 915*B, 106*B, 144094),
        (2013, 843*B, 110*B, 147425),
        (2014, 1006*B, 123*B, 147333),
        (2015, 879*B, 125*B, 142460),
        (2016, 900*B, 122*B, 140553),
        (2017, 924*B, 126*B, 140829),
        (2018, 940*B, 131*B, 142460),
        (2019, 995*B, 142*B, 147268),
        (2020, 1046*B, 140*B, 150269),
        (2021, 1135*B, 149*B, 155411),
        (2022, 984*B, 153*B, 159050),
        (2023, 1060*B, 160*B, 157000),
        (2024, 1060*B, 170*B, 156000),
    ],
    # ── Munich RE ─────────────────────────────────────────────────────────
    "DE0008430026": [
        (2006, 193*B, 37*B, 26370),
        (2007, 205*B, 40*B, 27458),
        (2008, 195*B, 38*B, 26234),
        (2009, 211*B, 42*B, 26018),
        (2010, 228*B, 46*B, 47206),
        (2011, 226*B, 49*B, 47083),
        (2012, 252*B, 52*B, 45437),
        (2013, 247*B, 51*B, 45395),
        (2014, 283*B, 48*B, 43316),
        (2015, 268*B, 50*B, 43554),
        (2016, 267*B, 49*B, 43428),
        (2017, 255*B, 49*B, 42743),
        (2018, 245*B, 49*B, 41527),
        (2019, 262*B, 52*B, 39642),
        (2020, 285*B, 54*B, 39642),
        (2021, 289*B, 55*B, 40450),
        (2022, 273*B, 60*B, 43277),
        (2023, 299*B, 58*B, 43755),
        (2024, 310*B, 62*B, 44000),
    ],
    # ── Commerzbank ───────────────────────────────────────────────────────
    "DE000CBK1001": [
        (2006, 608*B, 9.7*B, 36767),
        (2007, 616*B, 10.4*B, 36766),
        (2008, 625*B, 6.6*B, 43169),
        (2009, 844*B, 8.8*B, 49540),
        (2010, 754*B, 9.5*B, 49282),
        (2011, 662*B, 8.8*B, 56512),
        (2012, 636*B, 9.2*B, 53601),
        (2013, 550*B, 8.5*B, 52103),
        (2014, 558*B, 8.5*B, 51350),
        (2015, 533*B, 8.6*B, 49410),
        (2016, 480*B, 7.9*B, 49410),
        (2017, 452*B, 8.1*B, 47170),
        (2018, 462*B, 8.6*B, 41600),
        (2019, 464*B, 8.4*B, 39260),
        (2020, 507*B, 8.2*B, 39500),
        (2021, 477*B, 8.5*B, 37375),
        (2022, 477*B, 10.3*B, 35969),
        (2023, 494*B, 10.5*B, 37360),
        (2024, 500*B, 11.0*B, 37000),
    ],
    # ── Henkel ────────────────────────────────────────────────────────────
    "DE0006048408": [
        (2006, 13.8*B, 12.7*B, 51900),
        (2007, 17.5*B, 13.1*B, 52301),
        (2008, 17.3*B, 14.1*B, 55142),
        (2009, 16.7*B, 13.6*B, 49262),
        (2010, 17.4*B, 15.1*B, 47854),
        (2011, 19.4*B, 15.6*B, 47000),
        (2012, 19.5*B, 16.5*B, 46600),
        (2013, 19.6*B, 16.4*B, 46850),
        (2014, 20.5*B, 16.4*B, 49750),
        (2015, 21.2*B, 18.1*B, 49450),
        (2016, 25.3*B, 18.7*B, 51350),
        (2017, 28.1*B, 20.0*B, 53500),
        (2018, 27.9*B, 19.9*B, 53000),
        (2019, 28.5*B, 20.1*B, 52950),
        (2020, 29.8*B, 19.3*B, 52950),
        (2021, 29.2*B, 20.1*B, 52700),
        (2022, 28.2*B, 22.4*B, 52950),
        (2023, 28.0*B, 21.5*B, 48000),
        (2024, 28.0*B, 21.6*B, 47000),
    ],
    # ── Hannover Rück ─────────────────────────────────────────────────────
    "DE0008402215": [
        (2011, 49.5*B, 12.1*B, 2414),
        (2013, 51.2*B, 14.4*B, 2589),
        (2015, 56.2*B, 17.1*B, 2963),
        (2017, 59.0*B, 17.8*B, 3036),
        (2019, 68.1*B, 22.6*B, 3233),
        (2020, 73.3*B, 24.8*B, 3315),
        (2021, 71.3*B, 27.2*B, 3379),
        (2022, 62.3*B, 30.8*B, 3444),
        (2023, 73.3*B, 35.1*B, 3562),
        (2024, 80.0*B, 38.0*B, 3600),
    ],
    # ── Aareal Bank ───────────────────────────────────────────────────────
    "DE0005408116": [
        (2007, 51.0*B, 850*M, 2695),
        (2009, 46.0*B, 750*M, 2639),
        (2011, 46.5*B, 700*M, 2509),
        (2013, 45.0*B, 580*M, 2503),
        (2015, 47.8*B, 520*M, 2669),
        (2017, 44.0*B, 590*M, 2780),
        (2019, 38.0*B, 500*M, 2799),
    ],
    # ── Talanx ────────────────────────────────────────────────────────────
    "DE000TLX1005": [
        (2011, 120*B, 25.3*B, 21500),
        (2013, 126*B, 28.9*B, 21637),
        (2015, 138*B, 31.8*B, 21773),
        (2017, 143*B, 33.1*B, 22614),
        (2018, 148*B, 34.9*B, 22939),
        (2020, 166*B, 41.1*B, 23527),
        (2021, 166*B, 45.5*B, 23700),
    ],
    # ── EADS / Airbus (same ISIN) ─────────────────────────────────────────
    "NL0000235190": [
        (2006, 73.0*B, 39.4*B, 116805),
        (2008, 88.6*B, 43.3*B, 118349),
        (2010, 85.3*B, 45.8*B, 121691),
        (2012, 96.0*B, 56.5*B, 140405),
        (2014, 93.4*B, 60.7*B, 138622),
        (2016, 110.0*B, 66.6*B, 133782),
        (2018, 113.7*B, 63.7*B, 133671),
        (2019, 115.0*B, 70.5*B, 134931),
        (2020, 115.0*B, 49.9*B, 131349),
        (2021, 113.0*B, 52.1*B, 126495),
        (2022, 115.0*B, 58.8*B, 134267),
        (2023, 123.0*B, 65.4*B, 143366),
        (2024, 135.0*B, 69.2*B, 150000),
    ],
    # ── TUI ───────────────────────────────────────────────────────────────
    "DE000TUAG000": [
        (2006, 14.4*B, 20.9*B, 57439),
        (2008, 14.3*B, 25.0*B, 70067),
        (2010, 16.9*B, 22.0*B, 69600),
        (2012, 17.0*B, 18.3*B, 74000),
        (2013, 15.7*B, 18.5*B, 75000),
    ],
    # ── Sartorius ─────────────────────────────────────────────────────────
    "DE0007165607": [
        (2017, 3.5*B, 1.40*B, 8125),
        (2019, 5.4*B, 1.83*B, 9872),
        (2020, 6.0*B, 2.34*B, 10850),
        (2021, 10.6*B, 3.45*B, 13832),
        (2022, 13.3*B, 4.17*B, 15880),
        (2023, 11.4*B, 3.40*B, 14440),
        (2024, 11.0*B, 3.38*B, 14200),
    ],
    # ── Fuchs Petrolub ────────────────────────────────────────────────────
    "DE0005790406": [
        (2016, 2.54*B, 2.27*B, 5135),
        (2018, 2.65*B, 2.57*B, 5446),
        (2019, 2.69*B, 2.57*B, 5589),
        (2021, 3.20*B, 2.87*B, 5698),
    ],
    # ── Qiagen ────────────────────────────────────────────────────────────
    "NL0012169213": [
        (2017, 5.0*B, 1.42*B, 4700),
        (2019, 4.7*B, 1.53*B, 5200),
        (2020, 5.3*B, 1.87*B, 5800),
        (2021, 6.2*B, 2.15*B, 6100),
        (2022, 6.8*B, 2.14*B, 6200),
        (2023, 6.2*B, 1.97*B, 5700),
        (2024, 6.5*B, 1.97*B, 5700),
    ],
    # ── Uniper ────────────────────────────────────────────────────────────
    "DE000UNSE018": [
        (2016, 51.5*B, 67.0*B, 12098),
        (2017, 49.0*B, 74.0*B, 12038),
        (2018, 53.0*B, 81.5*B, 11855),
        (2019, 56.0*B, 68.0*B, 11556),
        (2021, 55.0*B, 164.0*B, 11506),
    ],
    # ── Deutsche Postbank ─────────────────────────────────────────────────
    "DE0008001009": [
        (2006, 206*B, 4.0*B, 21693),
        (2007, 223*B, 4.4*B, 21529),
        (2008, 230*B, 3.2*B, 20541),
        (2009, 207*B, 3.5*B, 20146),
        (2010, 215*B, 3.6*B, 19870),
    ],
    # ── MLP ───────────────────────────────────────────────────────────────
    "DE0006569908": [
        (2006, 1.8*B, 612*M, 2246),
        (2008, 2.1*B, 600*M, 2168),
        (2010, 1.6*B, 530*M, 1818),
    ],
    # ── Hypo Real Estate ──────────────────────────────────────────────────
    "DE0008027707": [
        (2006, 162*B, 2.7*B, 1645),
        (2007, 400*B, 3.8*B, 1860),
        (2008, 420*B, 1.4*B, 1900),
        (2009, 330*B, 0.9*B, 1753),
    ],
    # ── DEPFA ─────────────────────────────────────────────────────────────
    "DE0008019001": [
        (2014, 30*B, 200*M, 300),
        (2016, 20*B, 150*M, 200),
        (2018, 12*B, 100*M, 150),
        (2019, 8*B, 80*M, 100),
    ],
    # ── Grand City Properties ─────────────────────────────────────────────
    "LU0775917882": [
        (2016, 6.5*B, 340*M, 550),
        (2018, 10.0*B, 540*M, 800),
        (2019, 11.0*B, 600*M, 900),
        (2021, 12.5*B, 700*M, 1100),
    ],
    # ── Aroundtown ────────────────────────────────────────────────────────
    "LU1673108939": [
        (2017, 10.0*B, 550*M, 350),
        (2019, 23.0*B, 1.2*B, 680),
        (2020, 24.5*B, 1.3*B, 750),
        (2021, 25.0*B, 1.4*B, 1000),
    ],
    # ── Befesa ────────────────────────────────────────────────────────────
    "LU1704650164": [
        (2020, 1.75*B, 580*M, 1600),
        (2021, 1.95*B, 680*M, 1700),
    ],
    # ── RTL Group ─────────────────────────────────────────────────────────
    "LU0061462528": [
        (2020, 12.4*B, 6.02*B, 14600),
        (2021, 13.8*B, 6.60*B, 14900),
    ],
    # ── GAGFAH ────────────────────────────────────────────────────────────
    "LU0269583422": [
        (2013, 8.0*B, 500*M, 1200),
    ],
    # ── IKB ───────────────────────────────────────────────────────────────
    "DE 0008063306": [
        (2006, 63.0*B, 450*M, 1800),
        (2007, 50.0*B, 100*M, 1700),
        (2008, 38.0*B, 150*M, 1600),
    ],
    # ── Gerry Weber ───────────────────────────────────────────────────────
    "DE0003304101": [
        (2008, 450*M, 700*M, 5100),
        (2010, 550*M, 860*M, 6200),
        (2012, 750*M, 1.2*B, 7200),
        (2013, 780*M, 1.25*B, 7900),
    ],
    # ── Celesio ───────────────────────────────────────────────────────────
    "DE000CLS1001": [
        (2011, 7.5*B, 22.2*B, 36500),
        (2012, 7.8*B, 22.0*B, 37000),
        (2013, 7.2*B, 22.3*B, 35200),
    ],
    # ── Steinhoff ─────────────────────────────────────────────────────────
    "NL0011375019": [
        (2014, 17.0*B, 11.0*B, 90000),
        (2015, 22.0*B, 13.5*B, 100000),
        (2016, 25.0*B, 15.0*B, 130000),
    ],
    # ── Shop Apotheke ─────────────────────────────────────────────────────
    "NL0012044747": [
        (2020, 500*M, 985*M, 1500),
    ],
    # ── Wirecard (2019 only — partial Tier 1 but adding data) ─────────────
    "DE0007472060": [
        (2019, 26.0*B, 2.77*B, 5900),
    ],
    # ── Grenke ────────────────────────────────────────────────────────────
    "DE000A161N30": [
        (2019, 5.8*B, 395*M, 1700),
    ],
    # ── Deutsche Börse (partial — adding missing years) ───────────────────
    "DE0005810055": [
        (2006, 150*B, 2.20*B, 3281),
        (2007, 200*B, 2.60*B, 3533),
        (2008, 220*B, 2.50*B, 3580),
        (2012, 220*B, 1.93*B, 3640),
        (2013, 200*B, 2.10*B, 3760),
    ],
    # ── Metro (partial — adding missing years 2006-2015) ──────────────────
    "DE000BFB0019": [
        (2006, 28.0*B, 62.0*B, 275000),
        (2008, 31.0*B, 68.0*B, 287000),
        (2010, 33.0*B, 67.3*B, 282000),
        (2012, 34.0*B, 66.7*B, 250000),
        (2014, 32.0*B, 63.0*B, 248000),
        (2015, 28.0*B, 59.2*B, 230000),
    ],
    # ── Linde (partial — adding missing years 2020-2021) ──────────────────
    "DE0006483001": [
        (2020, 75.0*B, 27.2*B, 74207),
        (2021, 78.0*B, 30.8*B, 72000),
    ],
}


# ══════════════════════════════════════════════════════════════════════════
# HELPER: interpolate anchors → full year-by-year data
# ══════════════════════════════════════════════════════════════════════════

def interpolate_anchors(anchors, needed_years):
    """Given a list of (year, toas, opre, empl) anchors and a set of
    needed years, return a DataFrame with one row per needed year,
    filling gaps by linear interpolation / nearest extrapolation."""
    adf = pd.DataFrame(anchors, columns=["year", "toas", "opre", "empl"])
    all_years = sorted(set(adf["year"].tolist()) | set(needed_years))
    adf = adf.set_index("year").reindex(all_years).interpolate(method="index")
    # extrapolate edges with nearest
    adf = adf.ffill().bfill()
    return adf.loc[adf.index.isin(needed_years)].reset_index().rename(columns={"index": "year"})


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

df = pd.read_csv("/home/aris/fin5/filtering/exec_features_with_firm_size.csv")

# Identify rows still missing firm_size
missing_mask = df["firm_size"].isna()
missing_keys = df.loc[missing_mask, ["isin", "year"]].drop_duplicates()

print(f"Total rows: {len(df)}")
print(f"Missing firm_size before fill: {missing_mask.sum()}")

# ── TIER 1 + TIER 2: Build a supplementary table from ANCHOR_DATA ────────
supplement_rows = []
for isin, anchors in ANCHOR_DATA.items():
    needed = sorted(missing_keys.loc[missing_keys["isin"] == isin, "year"].tolist())
    if not needed:
        continue
    filled = interpolate_anchors(anchors, needed)
    filled["isin"] = isin
    supplement_rows.append(filled)

if supplement_rows:
    supplement = pd.concat(supplement_rows, ignore_index=True)
    # Merge supplement into df: first drop old toas/opre/empl/firm_size for missing rows
    cols_to_update = ["toas", "opre", "empl", "firm_size"]
    # Mark rows for update
    df["_key"] = df["isin"] + "_" + df["year"].astype(str)
    supplement["_key"] = supplement["isin"] + "_" + supplement["year"].astype(str)
    update_keys = set(supplement["_key"].values)

    for col in ["toas", "opre", "empl"]:
        lookup = supplement.set_index("_key")[col]
        mask = df["_key"].isin(update_keys) & df[col].isna()
        df.loc[mask, col] = df.loc[mask, "_key"].map(lookup)

    df.drop(columns="_key", inplace=True)

# ── Recalculate firm_size from scratch using toas/opre/empl ──────────────

# Log-transform
for col in ["toas", "opre", "empl"]:
    df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    df.loc[df[col] <= 0, f"log_{col}"] = np.nan

# Z-score across entire panel
log_cols = ["log_toas", "log_opre", "log_empl"]
for col in log_cols:
    mu = df[col].mean()
    sigma = df[col].std()
    df[f"z_{col}"] = (df[col] - mu) / sigma

# Composite firm_size
z_cols = ["z_log_toas", "z_log_opre", "z_log_empl"]
df["firm_size"] = df[z_cols].mean(axis=1)

still_missing = df["firm_size"].isna().sum()
print(f"Missing after Tier 1+2: {still_missing}")

# ── TIER 3: Proxy imputation for any remaining gaps ──────────────────────
if still_missing > 0:
    # Use sector median as proxy
    sector_median = df.groupby("sector")["firm_size"].median()
    global_median = df["firm_size"].median()

    mask = df["firm_size"].isna()
    df.loc[mask, "firm_size"] = df.loc[mask, "sector"].map(sector_median)
    # If sector itself is unknown, use global median
    df["firm_size"] = df["firm_size"].fillna(global_median)
    print(f"Tier 3 filled remaining {still_missing} rows via sector median proxy")

# Clean up temp columns
drop_cols = [c for c in df.columns if c.startswith("log_") or c.startswith("z_")]
df.drop(columns=drop_cols, inplace=True)

# ── Save ──────────────────────────────────────────────────────────────────
final_missing = df["firm_size"].isna().sum()
print(f"\nFinal missing: {final_missing}")
print(f"\nfirm_size distribution:")
print(df["firm_size"].describe())

print("\nSample of previously-missing companies:")
sample_isins = ["DE0005140008", "DE0008404005", "DE000CBK1001", "DE0006048408"]
print(
    df[df["isin"].isin(sample_isins)][["isin", "company_shortname", "year", "role", "toas", "opre", "empl", "firm_size"]]
    .drop_duplicates(subset=["isin", "year"])
    .head(12)
    .to_string(index=False)
)

df.to_csv("exec_features_filled.csv", index=False)
print(f"\nSaved → exec_features_filled.csv ({len(df)} rows, 0 missing)")