"""
run_pca.py
==========
Runs PCA on the seat-level executive compensation features produced by
prepare_exec_features.py. Operates on all roles combined and also
role-stratified (CEO / CFO / Other).

What this script does
---------------------
1. Load exec_features_all.csv (or a custom path)
2. Select and validate PCA feature columns
3. Drop zero-variance columns (no signal, break StandardScaler ratio)
4. Impute remaining NaNs with column median (within role group)
5. StandardScaler — zero mean, unit variance
6. Fit PCA, print explained variance per component
7. Choose n_components automatically (threshold: 85% cumulative variance)
   or manually via --n-components
8. Save:
     pca_scores.csv          — original identifier cols + PC scores
     pca_loadings.csv        — feature weights per PC
     pca_variance.csv        — explained variance table
     pca_scree.png           — scree plot

Usage
-----
    python run_pca.py --inp exec_features_all.csv --outdir ./pca_output

    # Run only on CEOs
    python run_pca.py --inp exec_features_CEO.csv --outdir ./pca_output/ceo

    # Fix number of components
    python run_pca.py --inp exec_features_all.csv --n-components 4

Feature columns used
--------------------
    log_tc_seat     — pay level
    ratio_salary    — fixed pay share
    ratio_sti       — short-term bonus share
    ratio_lti       — long-term cash bonus share
    ratio_equity    — equity grants share
    ratio_other     — other benefits share
    ratio_pension   — pension share
    vpi             — variable pay intensity
    ratio_stock     — stock share of equity
    ratio_option    — option share of equity
    firm_size       — PC1 of log_toas/log_opre/log_empl  (may have NaNs)
    index_score     — DAX=1, MDAX=0

Columns excluded from PCA (used as labels/context only)
--------------------------------------------------------
    isin, company_shortname, year, role, sector,
    n_seat_holders, comp_per_seat, ever_opted_out
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Feature columns to attempt — script will drop any that are all-NaN
# or zero-variance after imputation
# ---------------------------------------------------------------------------
CANDIDATE_FEATURES = [
    "log_tc_seat",
    "ratio_salary",
    "ratio_sti",
    "ratio_lti",
    "ratio_equity",
    "ratio_other",
    "ratio_pension",
    "vpi",
    "ratio_stock",
    "ratio_option",
    "firm_size",
    "index_score",
]

IDENTIFIER_COLS = ["isin", "company_shortname", "year", "role",
                   "sector", "n_seat_holders", "comp_per_seat"]

VARIANCE_THRESHOLD = 0.85   # auto-select components to reach this


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: str) -> pd.DataFrame:
    # Auto-detect separator
    df = pd.read_csv(path, sep="|", low_memory=False)
    if df.shape[1] <= 2:
        df = pd.read_csv(path, sep=",", low_memory=False)
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    From CANDIDATE_FEATURES, keep only columns that:
      1. Exist in the dataframe
      2. Are not entirely NaN
      3. Have non-zero variance after median imputation
    Returns the imputed feature matrix and the list of kept column names.
    """
    present = [c for c in CANDIDATE_FEATURES if c in df.columns]
    missing_cols = [c for c in CANDIDATE_FEATURES if c not in df.columns]
    if missing_cols:
        warnings.warn(f"Columns not found in file, skipping: {missing_cols}")

    X = df[present].copy()

    # Impute NaNs with column median
    null_counts = X.isnull().sum()
    for col in present:
        if null_counts[col] > 0:
            med = X[col].median()
            X[col] = X[col].fillna(med)
            print(f"    Imputed {null_counts[col]:,} NaNs in '{col}' "
                  f"with median ({med:.4f})")

    # Drop all-NaN columns (median is NaN if all values were NaN)
    all_nan = [c for c in present if X[c].isna().all()]
    if all_nan:
        print(f"    Dropped all-NaN columns: {all_nan}")
        X = X.drop(columns=all_nan)
        present = [c for c in present if c not in all_nan]

    # Drop zero-variance columns — StandardScaler divides by std,
    # so a zero-std column produces NaN/inf and corrupts all components
    zero_var = [c for c in present if X[c].std() == 0]
    if zero_var:
        print(f"    Dropped zero-variance columns (no signal): {zero_var}")
        X = X.drop(columns=zero_var)
        present = [c for c in present if c not in zero_var]

    if len(present) < 2:
        sys.exit("ERROR: Fewer than 2 usable features after filtering. "
                 "Check your data for sufficient coverage.")

    return X, present


def run_pca(X: np.ndarray, feature_names: list[str], n_components: int | None,
            variance_threshold: float) -> tuple[PCA, np.ndarray, int]:
    """Fit PCA, auto-select n_components if not specified."""
    # Fit full PCA first to see the variance curve
    pca_full = PCA().fit(X)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    if n_components is None:
        # First component count that crosses the threshold
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_components = min(n_components, len(feature_names))
        print(f"\n    Auto-selected {n_components} components "
              f"(reaches {cumvar[n_components-1]*100:.1f}% cumulative variance)")
    else:
        print(f"\n    Using {n_components} components (manual override)")

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    return pca, scores, n_components


def print_variance_table(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    n = pca.n_components_
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    print(f"\n    Explained variance per component:")
    print(f"    {'PC':<6}  {'Var %':>7}  {'Cum %':>7}  Bar")
    rows = []
    for i in range(n):
        var_pct  = pca.explained_variance_ratio_[i] * 100
        cum_pct  = cumvar[i] * 100
        bar      = "█" * int(var_pct / 2)
        print(f"    PC{i+1:<4}  {var_pct:7.2f}%  {cum_pct:7.2f}%  {bar}")
        rows.append({"component": f"PC{i+1}",
                     "variance_pct": round(var_pct, 4),
                     "cumulative_pct": round(cum_pct, 4)})

    return pd.DataFrame(rows)


def print_loadings(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """Print and return the loadings (feature weights per PC)."""
    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    ).round(4)

    print(f"\n    Loadings (feature weights per PC):")
    print(f"    {'Feature':<22}", end="")
    for i in range(pca.n_components_):
        print(f"  {'PC'+str(i+1):>8}", end="")
    print()
    for feat in feature_names:
        print(f"    {feat:<22}", end="")
        for i in range(pca.n_components_):
            val = loadings.loc[feat, f"PC{i+1}"]
            # Bold the dominant features (|loading| > 0.3)
            marker = " *" if abs(val) > 0.3 else "  "
            print(f"  {val:>7.3f}{marker}", end="")
        print()
    print(f"    (* = |loading| > 0.3 — dominant features for that PC)")

    return loadings.reset_index().rename(columns={"index": "feature"})


def save_scree(pca: PCA, outdir: str, label: str) -> None:
    """Save a scree plot showing variance explained per component."""
    n = pca.n_components_
    var_pct = pca.explained_variance_ratio_ * 100
    cumvar  = np.cumsum(var_pct)
    pcs     = [f"PC{i+1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(6, n), 4))
    bars = ax.bar(pcs, var_pct, color="#4C72B0", alpha=0.8, label="Individual")
    ax.plot(pcs, cumvar, "o-", color="#DD8452", linewidth=2, label="Cumulative")
    ax.axhline(85, linestyle="--", color="gray", linewidth=0.8, label="85% threshold")
    for bar, val in zip(bars, var_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Variance explained (%)")
    ax.set_title(f"PCA scree plot — {label}")
    ax.legend()
    ax.set_ylim(0, 110)
    plt.tight_layout()
    path = os.path.join(outdir, f"pca_scree_{label}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Scree plot → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inp",          required=True,
                   help="Path to exec_features_all.csv (or CEO/CFO/Other file)")
    p.add_argument("--outdir",       default="./pca_output",
                   help="Output directory (default: ./pca_output)")
    p.add_argument("--n-components", type=int, default=None,
                   help="Number of PCA components (default: auto at 85%% variance)")
    p.add_argument("--by-role",      action="store_true",
                   help="Also run PCA separately per role and save role files")
    return p.parse_args()


def run_for_subset(df: pd.DataFrame, label: str,
                   n_components: int | None, outdir: str) -> pd.DataFrame:
    """Run the full PCA pipeline for a given dataframe subset."""
    print(f"\n{'='*60}")
    print(f"  PCA subset: {label}  ({len(df):,} seat-rows)")
    print(f"{'='*60}")

    if len(df) < 3:
        print(f"  SKIP: fewer than 3 rows — not enough data to fit PCA.")
        return pd.DataFrame()

    print(f"\n  Feature selection …")
    X_df, features = select_features(df)

    print(f"  Using {len(features)} features: {features}")

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    # PCA
    pca, scores, n_comp = run_pca(X_scaled, features, n_components, VARIANCE_THRESHOLD)

    # Variance table
    var_df = print_variance_table(pca, features)

    # Loadings
    load_df = print_loadings(pca, features)

    # Scree plot
    save_scree(pca, outdir, label)

    # Build scores dataframe
    id_cols = [c for c in IDENTIFIER_COLS if c in df.columns]
    scores_df = df[id_cols].reset_index(drop=True).copy()
    for i in range(n_comp):
        scores_df[f"PC{i+1}"] = scores[:, i]

    # Save outputs
    slug = label.replace(" ", "_").lower()
    scores_df.to_csv(os.path.join(outdir, f"pca_scores_{slug}.csv"), index=False)
    load_df.to_csv(os.path.join(outdir, f"pca_loadings_{slug}.csv"), index=False)
    var_df.to_csv(os.path.join(outdir, f"pca_variance_{slug}.csv"), index=False)
    print(f"\n  Saved: pca_scores_{slug}.csv | pca_loadings_{slug}.csv "
          f"| pca_variance_{slug}.csv")

    return scores_df


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\nLoading: {args.inp}")
    df = load(args.inp)
    print(f"  {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Roles present: {df['role'].value_counts().to_dict()}")

    # Run on full dataset
    run_for_subset(df, label="all_roles",
                   n_components=args.n_components, outdir=args.outdir)

    # Optionally run per role
    if args.by_role:
        for role in df["role"].unique():
            role_df = df[df["role"] == role].reset_index(drop=True)
            run_for_subset(role_df, label=role,
                           n_components=args.n_components, outdir=args.outdir)

    print(f"\nAll outputs saved to: {args.outdir}")
    print(f"\nNext step — clustering:")
    print(f"  Use the PC scores (pca_scores_*.csv) as input to K-Means or GMM.")
    print(f"  Columns PC1, PC2, ... are your reduced feature space.")
    print(f"  Keep sector and year as stratification variables, not PCA inputs.")


if __name__ == "__main__":
    main()