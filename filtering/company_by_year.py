import pandas as pd

# ── 1. Load ──────────────────────────────────────────────────────────────────
df = pd.read_csv("/home/aris/fin5/csv_data/company_year.csv", sep="|")

# ── 2. Sort by year (then company name within each year) ─────────────────────
df_sorted = df.sort_values(["year", "company_shortname"]).reset_index(drop=True)

# ── 3a. Save as a single CSV with rows ordered by year ───────────────────────
df_sorted.to_csv("company_year_sorted.csv", sep="|", index=False)
print("Saved: company_year_sorted.csv")

# ── 3b. (Optional) Split into one CSV per year ───────────────────────────────
for year, group in df_sorted.groupby("year"):
    out_path = f"compensation_{year}.csv"
    group.to_csv(out_path, sep="|", index=False)
    print(f"  {out_path}  ({len(group)} rows)")

# ── 4. Quick sanity check ────────────────────────────────────────────────────
print("\nRows per year:")
print(df_sorted.groupby("year").size().to_string())