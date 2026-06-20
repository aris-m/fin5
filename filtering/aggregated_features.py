import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("/home/aris/fin5/csv_data/company_year.csv", sep="|")

FEATURES = [
    "ratio_salary", "ratio_sti", "ratio_lti", "ratio_equity",
    "ratio_other", "ratio_pension", "vpi",
    "ratio_stock_in_equity", "ratio_option_in_equity",
    "log_tc_pe",
]

K = 4  # adjust as needed
results = []

for year, group in df.groupby("year"):
    # Filter to DAX only
    dax = group[group["index_listing"] == "DAX"].copy()

    # Compute ratios
    tc  = dax["total_comp_bt"].replace(0, np.nan)
    sal = dax["salary_bt"].replace(0, np.nan)
    eq  = dax["total_equity_grants_bt"].replace(0, np.nan)

    dax["ratio_salary"]           = dax["salary_bt"] / tc
    dax["ratio_sti"]              = dax["one_year_bonus_bt"] / tc
    dax["ratio_lti"]              = dax["multi_year_bonus_bt"] / tc
    dax["ratio_equity"]           = dax["total_equity_grants_bt"] / tc
    dax["ratio_other"]            = dax["other_annual_bt"] / tc
    dax["ratio_pension"]          = dax["pension_bt"] / tc
    dax["vpi"]                    = (
        dax["one_year_bonus_bt"] + dax["multi_year_bonus_bt"] + dax["total_equity_grants_bt"]
    ) / sal
    dax["ratio_stock_in_equity"]  = dax["stock_grants_bt"] / eq
    dax["ratio_option_in_equity"] = dax["option_grants_bt"] / eq
    dax["log_tc_pe"]              = np.log1p(dax["total_comp_bt"] / dax["n_executives"])

    # Drop rows with NaN in any feature
    dax_clean = dax.dropna(subset=FEATURES).copy()

    # Need at least K companies to form K clusters
    if len(dax_clean) < K:
        print(f"{year}: only {len(dax_clean)} companies, skipping")
        continue

    # Scale and cluster
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(dax_clean[FEATURES].values)

    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    dax_clean["cluster"] = km.fit_predict(X_scaled)
    dax_clean["year"] = year

    results.append(dax_clean[["isin", "company_shortname", "year", "cluster"]])

# Combine all years
panel = pd.concat(results, ignore_index=True)
print(panel.to_string(index=False))
panel.to_csv("/home/aris/fin5/clustering/cluster_assignments_by_year.csv", index=False)