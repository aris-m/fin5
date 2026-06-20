#!/usr/bin/env python3
"""Step 3 — Lens 1: level benchmark (regularized log-OLS).

Reads:  Data/processed/ceo_year_features.csv
Writes: Data/processed/ceo_year_lens1.csv

What it does:
1. Predicts log(total_comp) from a small interpretable feature set.
2. Uses RidgeCV inside a Pipeline (scale + one-hot + impute).
3. Evaluates honestly with GroupKFold(5) by isin so no company leaks across folds.
4. Computes residual = y - expected.
5. Computes robust-z of residual WITHIN (index_listing, year) peer groups
   (single peer scheme per the locked design rule).
6. Flags |robust_z| > 3 as a "level anomaly" and emits a short reason string.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PANEL_IN = Path("Data/processed/ceo_year_features.csv")
OUT_FILE = Path("Data/processed/ceo_year_lens1.csv")

NUM_FEATURES = ["year", "n_executives", "tenure_years", "log_revenue", "roa"]
CAT_FEATURES = ["index_listing"]
LEVEL_FLAG_THRESHOLD = 3.0


def robust_z(x: np.ndarray) -> np.ndarray:
    """Median + MAD z-score. Returns 0 if MAD is degenerate."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def build_pipeline() -> Pipeline:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])
    model = Pipeline([
        ("pre", pre),
        ("ridge", RidgeCV(alphas=np.logspace(-2, 3, 20))),
    ])
    return model


def main() -> None:
    if not PANEL_IN.exists():
        sys.exit(f"missing input — run step2_features.py first: {PANEL_IN}")

    df = pd.read_csv(PANEL_IN)
    print(f"loaded panel: {len(df)} CEO-year rows")

    # Drop rows that can't be modeled (no peer, no target)
    mask = df["total_comp"].notna() & (df["total_comp"] > 0) & df["index_listing"].notna()
    work = df[mask].copy()
    print(f"modelable rows (target + peer key present): {len(work)}")

    X = work[NUM_FEATURES + CAT_FEATURES]
    y = np.log1p(work["total_comp"])
    groups = work["isin"]

    model = build_pipeline()

    # Honest CV: never let the model memorize a company
    cv = GroupKFold(n_splits=5)
    expected_cv = cross_val_predict(model, X, y, cv=cv, groups=groups, n_jobs=-1)
    r2_cv = r2_score(y, expected_cv)

    # Also report in-sample fit for context (will be optimistic)
    model.fit(X, y)
    expected_in = model.predict(X)
    r2_in = r2_score(y, expected_in)

    print(f"\nLens 1 — RidgeCV(alpha selected={model.named_steps['ridge'].alpha_:.3g})")
    print(f"  in-sample R²:           {r2_in:.3f}  (optimistic, ignore for defense)")
    print(f"  out-of-sample R² (CV5): {r2_cv:.3f}  (THIS is the number to quote)")

    work["expected_log_comp"] = expected_cv
    work["expected_comp"] = np.expm1(expected_cv)
    work["actual_log_comp"] = y.to_numpy()
    work["residual_log"] = y.to_numpy() - expected_cv

    # Robust-z within (index_listing, year) peer group
    work["level_robust_z"] = (
        work.groupby(["index_listing", "year"], group_keys=False)["residual_log"]
            .transform(lambda s: pd.Series(robust_z(s.to_numpy()), index=s.index))
    )
    work["level_flag"] = work["level_robust_z"].abs() > LEVEL_FLAG_THRESHOLD

    # Plain-English reason string for flagged rows
    def reason(r: pd.Series) -> str:
        if not r["level_flag"] or pd.isna(r["level_robust_z"]):
            return ""
        actual = r["total_comp"]
        expected = r["expected_comp"]
        direction = "over" if r["level_robust_z"] > 0 else "under"
        pct = (actual - expected) / expected * 100.0 if expected else float("nan")
        peer = f"{r['index_listing']} {int(r['year'])}"
        return (f"Level {r['level_robust_z']:+.1f}σ — paid {direction} peers "
                f"(actual €{actual*1000:,.0f}, expected €{expected*1000:,.0f}, "
                f"{pct:+.0f}% vs {peer} peer model)")

    work["level_reason"] = work.apply(reason, axis=1)

    # Re-attach the un-modelable rows so the output is the full panel
    out = df.merge(
        work[["isin", "year", "exec_id",
              "expected_log_comp", "expected_comp", "actual_log_comp",
              "residual_log", "level_robust_z", "level_flag", "level_reason"]],
        on=["isin", "year", "exec_id"], how="left",
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print(f"\nflag summary:")
    flagged = out[out["level_flag"] == True]
    print(f"  level-flagged rows: {len(flagged)} / {len(out)} ({len(flagged)/len(out):.1%})")
    print(f"  of which OVER peers (positive z):  {(flagged['level_robust_z'] > 0).sum()}")
    print(f"  of which UNDER peers (negative z): {(flagged['level_robust_z'] < 0).sum()}")

    print(f"\ntop 5 over-paid (highest +z):")
    cols = ["isin", "year", "company_shortname", "exec_fullname", "total_comp",
            "expected_comp", "level_robust_z"]
    print(flagged.nlargest(5, "level_robust_z")[cols].to_string(index=False))

    print(f"\ntop 5 under-paid (lowest -z):")
    print(flagged.nsmallest(5, "level_robust_z")[cols].to_string(index=False))

    print(f"\nwrote: {OUT_FILE}  ({len(out)} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
