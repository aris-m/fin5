"""
Interactive Peer-Group Explorer & Compensation Predictor
=========================================================
Run:  streamlit run peer_app.py -- --input data.csv
Install:  pip install streamlit plotly scikit-learn pandas numpy scipy
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
PAY_RATIO_COLS = [
    "ratio_salary", "ratio_sti", "ratio_lti",
    "ratio_equity", "ratio_other", "ratio_pension",
    "ratio_stock", "ratio_option",
]
DISPLAY_RATIO_COLS = [
    "ratio_salary", "ratio_sti", "ratio_lti",
    "ratio_equity", "ratio_other", "ratio_pension",
]
PAY_RATIO_LABELS = {
    "ratio_salary": "Base Salary",
    "ratio_sti": "Short-Term Bonus",
    "ratio_lti": "Long-Term Incentive",
    "ratio_equity": "Equity",
    "ratio_other": "Other",
    "ratio_pension": "Pension",
    "ratio_stock": "Stock Grants",
    "ratio_option": "Stock Options",
}
SIZE_COL = "firm_size"
INDEX_COL = "index_score"
CLUSTER_FEATURES = PAY_RATIO_COLS + [SIZE_COL]

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#DA8BC3", "#CCB974", "#64B5CD",
    "#E5AE38", "#6ACC65", "#937860", "#8C8C8C",
]
GROUP_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

ROLE_COLORS = {"CEO": "#4C72B0", "CFO": "#DD8452", "Other": "#55A868"}

# ── ESG CONFIG ───────────────────────────────
ESG_FILE_CANDIDATES = [
    ["Executive_Compensation_ESG_2023.xlsx", "Executive_Compensation_ESG_2024.xlsx"],
    ["Data/2023/Executive_Compensation_ESG_2023.xlsx", "Data/2024/Executive_Compensation_ESG_2024.xlsx"],
]
MASTER_DB_CANDIDATES = [
    "2008-2024_longitudinal.csv", "Data/2008-2024_longitudinal.csv",
    "fin5/csv_data/2008-2024_longitudinal.csv",
]
WEIGHT_HIGH = 20.0
ACHIEVE_HIT = 100.0
ESG_ACCENT = "#1D9E75"
ESG_WARN   = "#E2A33B"
ESG_DANGER = "#D1493F"
ESG_SOFT   = "#9AA0A6"


def _clean_scalar(v):
    if v is None:
        return np.nan
    v = str(v).split(";")[0].strip()
    v = "".join(ch for ch in v if ch in "0123456789,.-")
    if v in ("", "-", ".", ","):
        return np.nan
    v = v.replace(",", ".") if ("," in v and "." not in v) else v.replace(",", "")
    try:
        return float(v)
    except ValueError:
        return np.nan


def _to_num(series):
    return series.map(_clean_scalar)


def _to_flag(series):
    n = _to_num(series)
    return pd.Series(np.where(n.notna(), (n > 0).astype(float), np.nan), index=series.index)


@st.cache_data
def load_esg_data():
    files = next((fs for fs in ESG_FILE_CANDIDATES
                  if all(os.path.exists(f) for f in fs)), None)
    if files is None:
        return None

    frames = []
    for f in files:
        d = pd.read_excel(f, header=2)
        d.columns = [str(c).strip() for c in d.columns]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    out = pd.DataFrame()
    out["company"] = df.get("cnameshort")
    out["index"]   = df.get("cindex")
    out["year"]    = pd.to_numeric(df.get("year"), errors="coerce")

    weight = _to_num(df.get("STI_total_ESG_Share"))
    out["esg_weight"] = weight.where((weight >= 0) & (weight <= 100))

    ach = _to_num(df.get("STI_Zielerreichung"))
    out["achievement"] = ach.where((ach >= 0) & (ach <= 300))

    em_sti = _to_flag(df.get("STI_E_KPI_Emission reduction"))
    em_lti = (_to_flag(df["LTI_E_KPI_Emission reduction"])
              if "LTI_E_KPI_Emission reduction" in df.columns
              else pd.Series(np.nan, index=df.index))
    out["emission_sti"] = em_sti
    out["emission_lti"] = em_lti
    out["emission_any"] = pd.concat([em_sti, em_lti], axis=1).max(axis=1)

    out["has_E"] = _to_flag(df.get("STI_E_KPI"))
    out["has_S"] = _to_flag(df.get("STI_S_KPI"))
    out["has_G"] = _to_flag(df.get("STI_G_KPI"))
    out["n_kpi"] = _to_num(df.get("STI_count_of_total_ESG_KPI"))

    return out.dropna(subset=["company"])


@st.cache_data
def load_master_db():
    path = next((p for p in MASTER_DB_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return None
    m = pd.read_csv(path, sep="|", low_memory=False)
    m["year"] = pd.to_numeric(m["year"], errors="coerce")
    m["tsr"]  = pd.to_numeric(m.get("tsr"), errors="coerce")
    m["one_year_bonus"] = pd.to_numeric(m.get("one_year_bonus"), errors="coerce")
    m["name"] = m["new_cnameshort"].astype(str).str.strip()
    m.loc[m["new_cnameshort"].isna(), "name"] = m["company_shortname"].astype(str).str.strip()

    tsr_lookup = (m.dropna(subset=["tsr"])
                   .groupby(["name", "year"])["tsr"].first().reset_index())
    co = m.dropna(subset=["tsr"]).drop_duplicates(["name", "year"])
    test_a = (co.groupby("year")
                .agg(median_tsr=("tsr", "median"),
                     pct_neg=("tsr", lambda s: (s < 0).mean() * 100),
                     n=("tsr", "size")).reset_index())
    d = m.dropna(subset=["tsr", "one_year_bonus"]).copy()
    d["neg"] = d["tsr"] < 0
    test_b = (d.groupby("neg")
                .agg(pct_bonus=("one_year_bonus", lambda s: (s > 0).mean() * 100),
                     median_bonus=("one_year_bonus", "median"),
                     n=("one_year_bonus", "size")).reset_index())
    return {"tsr_lookup": tsr_lookup, "test_a": test_a, "test_b": test_b}


# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    sep = "\t" if p.suffix in (".tsv", ".tab") else ","
    df = pd.read_csv(p, sep=sep, engine="python")
    if len(df.columns) < 5:
        df = pd.read_csv(p, sep=",", engine="python")
    if len(df.columns) < 5:
        df = pd.read_csv(p, sep="\t", engine="python")

    df.columns = df.columns.str.strip().str.lower()
    for c in CLUSTER_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "comp_per_seat" in df.columns:
        df["comp_per_seat"] = pd.to_numeric(df["comp_per_seat"], errors="coerce")

    if INDEX_COL in df.columns:
        raw = pd.to_numeric(df[INDEX_COL], errors="coerce")
        if raw.notna().any() and raw.dropna().isin([0, 1]).all():
            df[INDEX_COL] = raw.map({0: "MDAX", 1: "DAX"}).fillna("Unknown")
        else:
            df[INDEX_COL] = df[INDEX_COL].astype(str).str.strip()
    return df


def aggregate_profiles(df: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df[df["year"] == year].copy()
    if df.empty:
        return df
    group_keys = ["isin", "company_shortname", "role", "sector", INDEX_COL]
    features = [c for c in CLUSTER_FEATURES if c in df.columns]
    agg = df.groupby(group_keys, as_index=False)[features].mean()
    agg = agg.dropna(subset=features, how="all")
    agg[features] = agg[features].fillna(0)
    return agg


# ──────────────────────────────────────────────
# CLUSTERING (Tab 1)
# ──────────────────────────────────────────────
def best_kmeans(X, k_min=2, k_max=8):
    n = X.shape[0]
    k_max = min(k_max, n - 1)
    if k_max < k_min:
        return np.zeros(n, dtype=int), 1
    best_score, best_labels, best_k = -1, None, k_min
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        s = silhouette_score(X, labels)
        if s > best_score:
            best_score, best_k, best_labels = s, k, labels
    return (best_labels, best_k) if best_labels is not None else (np.zeros(n, dtype=int), 1)


def run_clustering(group_df: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in CLUSTER_FEATURES if c in group_df.columns]
    X = group_df[features].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    labels, k = best_kmeans(X_s)
    group_df = group_df.copy()
    group_df["cluster"] = labels
    nc = min(2, X_s.shape[1], X_s.shape[0])
    pca = PCA(n_components=nc, random_state=42)
    coords = pca.fit_transform(X_s)
    group_df["x"] = coords[:, 0] if nc >= 1 else 0
    group_df["y"] = coords[:, 1] if nc >= 2 else 0
    return group_df


# ──────────────────────────────────────────────
# PREDICTIVE MODEL (Tab 2)
# ──────────────────────────────────────────────
MODEL_FEATURES_CAT = ["role", "sector", INDEX_COL]
MODEL_FEATURES_NUM = [SIZE_COL, "year"]
PREDICT_TARGETS = DISPLAY_RATIO_COLS  # 6 ratios


def _prepare_training_data(df: pd.DataFrame):
    """Build X matrix and y targets from raw data."""
    needed = MODEL_FEATURES_NUM + MODEL_FEATURES_CAT + PREDICT_TARGETS
    if "comp_per_seat" in df.columns:
        needed.append("comp_per_seat")

    subset = df.dropna(subset=MODEL_FEATURES_NUM + PREDICT_TARGETS).copy()
    if subset.empty:
        return None, None, None, None

    X = pd.get_dummies(subset[MODEL_FEATURES_CAT + MODEL_FEATURES_NUM],
                       columns=MODEL_FEATURES_CAT, drop_first=False)

    y_ratios = subset[PREDICT_TARGETS].values
    y_comp = subset["comp_per_seat"].values if "comp_per_seat" in subset.columns else None

    return X, y_ratios, y_comp, X.columns.tolist()


@st.cache_resource
def train_all_models(_df: pd.DataFrame):
    """Train GBR models for comp level and each pay ratio."""
    X, y_ratios, y_comp, feature_names = _prepare_training_data(_df)
    if X is None:
        return None

    models = {"feature_names": feature_names}

    # Comp-per-seat model
    if y_comp is not None:
        mask = ~np.isnan(y_comp)
        if mask.sum() > 20:
            m = GradientBoostingRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            m.fit(X[mask], y_comp[mask])
            preds = m.predict(X[mask])
            models["comp_per_seat"] = {
                "model": m,
                "r2": r2_score(y_comp[mask], preds),
                "mae": mean_absolute_error(y_comp[mask], preds),
            }

    # Ratio models — one per target
    for i, col in enumerate(PREDICT_TARGETS):
        y = y_ratios[:, i]
        mask = ~np.isnan(y)
        if mask.sum() > 20:
            m = GradientBoostingRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            m.fit(X[mask], y[mask])
            preds = m.predict(X[mask])
            models[col] = {
                "model": m,
                "r2": r2_score(y[mask], preds),
                "mae": mean_absolute_error(y[mask], preds),
            }

    return models


def predict_scenario(models, sector, index, year, firm_size, role):
    """Predict comp_per_seat and ratios for one scenario."""
    feature_names = models["feature_names"]
    row = pd.DataFrame([{SIZE_COL: firm_size, "year": year}])

    # One-hot encode categoricals to match training
    for cat_col, val in [("role", role), ("sector", sector), (INDEX_COL, index)]:
        for fname in feature_names:
            if fname.startswith(f"{cat_col}_"):
                row[fname] = 1.0 if fname == f"{cat_col}_{val}" else 0.0

    # Align columns
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0.0
    row = row[feature_names]

    result = {"role": role, "sector": sector, "index": index,
              "year": year, "firm_size": firm_size}

    # Predict comp level
    if "comp_per_seat" in models:
        result["comp_per_seat"] = max(0, models["comp_per_seat"]["model"].predict(row)[0])

    # Predict ratios and normalize to sum=1
    raw_ratios = {}
    for col in PREDICT_TARGETS:
        if col in models:
            raw_ratios[col] = max(0, models[col]["model"].predict(row)[0])
        else:
            raw_ratios[col] = 0.0

    total = sum(raw_ratios.values())
    if total > 0:
        for col in raw_ratios:
            raw_ratios[col] /= total
    result["ratios"] = raw_ratios

    return result


# ──────────────────────────────────────────────
# PLOTLY VIS — Tab 1 (Peer Groups)
# ──────────────────────────────────────────────
def _hull_shape(points, color, pad=0.35):
    if len(points) < 3:
        cx, cy = points.mean(axis=0)
        r = max(np.ptp(points, axis=0).max() * 0.6, pad)
        t = np.linspace(0, 2 * np.pi, 60)
        xs = cx + r * np.cos(t)
        ys = cy + r * np.sin(t)
    else:
        try:
            hull = ConvexHull(points)
            hp = points[hull.vertices]
            centroid = hp.mean(axis=0)
            expanded = centroid + (hp - centroid) * (1 + pad)
            xs = np.append(expanded[:, 0], expanded[0, 0])
            ys = np.append(expanded[:, 1], expanded[0, 1])
        except Exception:
            cx, cy = points.mean(axis=0)
            r = max(np.ptp(points, axis=0).max() * 0.6, pad)
            t = np.linspace(0, 2 * np.pi, 60)
            xs = cx + r * np.cos(t)
            ys = cy + r * np.sin(t)
    path = "M " + " L ".join(f"{x:.4f},{y:.4f}" for x, y in zip(xs, ys)) + " Z"
    return dict(type="path", path=path, fillcolor=color, opacity=0.12,
                line=dict(color=color, width=2), layer="below")


def build_scatter(group_df, title):
    fig = go.Figure()
    clusters = sorted(group_df["cluster"].unique())
    for i, cid in enumerate(clusters):
        sub = group_df[group_df["cluster"] == cid]
        color = PALETTE[i % len(PALETTE)]
        label = f"Peer Group {GROUP_LABELS[i]}" if i < len(GROUP_LABELS) else f"Peer Group {cid+1}"
        hovers = []
        for _, row in sub.iterrows():
            parts = [f"<b>{row['company_shortname']}</b>", f"Peer Group: {label}", "─" * 20]
            for col in DISPLAY_RATIO_COLS:
                if col in row and row[col] > 0.001:
                    parts.append(f"{PAY_RATIO_LABELS.get(col, col)}: {row[col]*100:.1f}%")
            if SIZE_COL in row:
                parts.append(f"Firm Size Score: {row[SIZE_COL]:.2f}")
            hovers.append("<br>".join(parts))
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"], mode="markers+text",
            marker=dict(size=14, color=color, line=dict(width=1.5, color="white")),
            text=sub["company_shortname"], textposition="top center",
            textfont=dict(size=10, color=color, family="Arial Black"),
            hovertext=hovers, hoverinfo="text", name=label, legendgroup=label,
        ))
        pts = sub[["x", "y"]].values
        if len(pts) >= 2:
            fig.add_shape(_hull_shape(pts, color, pad=0.4))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="", showticklabels=False, showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        yaxis=dict(title="", showticklabels=False, showgrid=True,
                   gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        plot_bgcolor="white", height=500,
        margin=dict(t=60, b=40, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        annotations=[dict(
            text="📌 Companies closer together have more similar compensation structures",
            xref="paper", yref="paper", x=0.5, y=-0.06, showarrow=False,
            font=dict(size=11, color="#666"), xanchor="center",
        )],
    )
    return fig


def build_pay_breakdown_chart(group_df):
    clusters = sorted(group_df["cluster"].unique())
    fig = go.Figure()
    ratio_cols_present = [c for c in DISPLAY_RATIO_COLS
                          if c in group_df.columns and group_df[c].sum() > 0]
    bar_colors = {"ratio_salary": "#4C72B0", "ratio_sti": "#DD8452",
                  "ratio_lti": "#55A868", "ratio_equity": "#C44E52",
                  "ratio_other": "#8172B3", "ratio_pension": "#CCB974"}
    group_names = []
    for i, cid in enumerate(clusters):
        sub = group_df[group_df["cluster"] == cid]
        label = f"Group {GROUP_LABELS[i]}" if i < len(GROUP_LABELS) else f"Group {cid+1}"
        group_names.append(f"{label} ({len(sub)})")
    for col in ratio_cols_present:
        vals, hovers = [], []
        for i, cid in enumerate(clusters):
            sub = group_df[group_df["cluster"] == cid]
            mv = sub[col].mean() * 100
            vals.append(mv)
            hovers.append(f"{PAY_RATIO_LABELS.get(col, col)}: {mv:.1f}%<br>"
                          f"Members: {', '.join(sub['company_shortname'].tolist())}")
        fig.add_trace(go.Bar(y=group_names, x=vals,
                             name=PAY_RATIO_LABELS.get(col, col), orientation="h",
                             marker_color=bar_colors.get(col, "#999"),
                             hovertext=hovers, hoverinfo="text"))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Average Pay Structure by Peer Group", font=dict(size=14)),
        xaxis=dict(title="% of Total Compensation", range=[0, 105]),
        yaxis=dict(title=""), height=max(200, 80 * len(clusters)),
        margin=dict(t=50, b=40, l=120, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="white",
    )
    return fig


# ──────────────────────────────────────────────
# PLOTLY VIS — Tab 2 (Predictions)
# ──────────────────────────────────────────────
def build_predicted_comp_chart(predictions):
    """Bar chart comparing predicted comp_per_seat across roles."""
    roles = [p["role"] for p in predictions if "comp_per_seat" in p]
    vals = [p["comp_per_seat"] for p in predictions if "comp_per_seat" in p]
    colors = [ROLE_COLORS.get(r, "#999") for r in roles]

    fig = go.Figure(go.Bar(
        x=roles, y=vals,
        marker_color=colors,
        text=[f"€{v:,.0f}K" for v in vals],
        textposition="outside",
        hovertemplate="%{x}<br>Predicted: €%{y:,.0f}K<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Predicted Total Compensation per Seat (€K)", font=dict(size=14)),
        yaxis=dict(title="€ thousands"),
        xaxis=dict(title=""),
        height=400, plot_bgcolor="white",
        margin=dict(t=50, b=40, l=60, r=40),
    )
    return fig


def build_predicted_mix_chart(predictions):
    """Stacked bar showing predicted pay composition per role."""
    fig = go.Figure()
    bar_colors = {"ratio_salary": "#4C72B0", "ratio_sti": "#DD8452",
                  "ratio_lti": "#55A868", "ratio_equity": "#C44E52",
                  "ratio_other": "#8172B3", "ratio_pension": "#CCB974"}
    roles = [p["role"] for p in predictions]

    for col in DISPLAY_RATIO_COLS:
        vals = [p["ratios"].get(col, 0) * 100 for p in predictions]
        if max(vals) < 0.1:
            continue
        fig.add_trace(go.Bar(
            x=roles, y=vals,
            name=PAY_RATIO_LABELS.get(col, col),
            marker_color=bar_colors.get(col, "#999"),
            hovertemplate="%{x}<br>" + PAY_RATIO_LABELS.get(col, col) +
                          ": %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Predicted Pay Structure by Role", font=dict(size=14)),
        yaxis=dict(title="% of Total Compensation", range=[0, 105]),
        xaxis=dict(title=""),
        height=400, plot_bgcolor="white",
        margin=dict(t=50, b=40, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def build_actual_vs_predicted_chart(actuals_df):
    """Scatter plot of actual vs predicted comp_per_seat with diagonal."""
    fig = go.Figure()
    for role in actuals_df["role"].unique():
        sub = actuals_df[actuals_df["role"] == role]
        color = ROLE_COLORS.get(role, "#999")
        fig.add_trace(go.Scatter(
            x=sub["predicted"], y=sub["actual"],
            mode="markers+text",
            marker=dict(size=10, color=color, line=dict(width=1, color="white")),
            text=sub["company_shortname"],
            textposition="top center",
            textfont=dict(size=8, color=color),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Predicted: €%{x:,.0f}K<br>"
                "Actual: €%{y:,.0f}K<br>"
                "Deviation: %{customdata:.0f}%<extra></extra>"
            ),
            customdata=sub["deviation_pct"],
            name=role,
        ))

    # Diagonal reference line
    all_vals = pd.concat([actuals_df["predicted"], actuals_df["actual"]])
    lo, hi = all_vals.min() * 0.8, all_vals.max() * 1.1
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(dash="dash", color="rgba(0,0,0,0.3)", width=1),
        showlegend=False, hoverinfo="skip",
    ))

    fig.update_layout(
        title=dict(text="Actual vs Predicted Compensation", font=dict(size=14)),
        xaxis=dict(title="Predicted (€K)"),
        yaxis=dict(title="Actual (€K)"),
        height=500, plot_bgcolor="white",
        margin=dict(t=50, b=40, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


def build_tornado_chart(outliers_df, threshold):
    """Horizontal bar chart ranking companies by pay deviation."""
    df = outliers_df.sort_values("deviation_pct", ascending=True).copy()
    colors = [("#C44E52" if d > 0 else "#4C72B0") for d in df["deviation_pct"]]
    labels = df["company_shortname"] + " (" + df["role"] + ")"

    fig = go.Figure(go.Bar(
        y=labels, x=df["deviation_pct"],
        orientation="h",
        marker_color=colors,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Deviation: %{x:+.0f}%<br>"
            "Actual: €%{customdata[0]:,.0f}K<br>"
            "Predicted: €%{customdata[1]:,.0f}K<extra></extra>"
        ),
        customdata=df[["actual", "predicted"]].values,
    ))

    # Threshold lines
    fig.add_vline(x=threshold, line_dash="dash", line_color="#C44E52",
                  annotation_text=f"+{threshold}%", annotation_position="top right")
    fig.add_vline(x=-threshold, line_dash="dash", line_color="#4C72B0",
                  annotation_text=f"−{threshold}%", annotation_position="top left")
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.3)", line_width=1)

    fig.update_layout(
        title=dict(text="Compensation Deviation from Predicted", font=dict(size=14)),
        xaxis=dict(title="Deviation %", zeroline=False),
        yaxis=dict(title=""),
        height=max(400, 28 * len(df)),
        margin=dict(t=50, b=40, l=200, r=40),
        plot_bgcolor="white",
    )
    return fig


def build_outlier_composition_chart(row_data, peer_avg):
    """Side-by-side bar comparing one company's pay mix vs peer average."""
    labels = [PAY_RATIO_LABELS.get(c, c) for c in DISPLAY_RATIO_COLS]
    company_vals = [row_data.get(c, 0) * 100 for c in DISPLAY_RATIO_COLS]
    peer_vals = [peer_avg.get(c, 0) * 100 for c in DISPLAY_RATIO_COLS]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=company_vals,
        name=row_data.get("company_shortname", "Company"),
        marker_color="#C44E52",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=peer_vals,
        name="Peer Average",
        marker_color="#4C72B0", opacity=0.6,
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text=f"{row_data.get('company_shortname', '')} vs Peer Average",
                   font=dict(size=13)),
        yaxis=dict(title="%"),
        height=350, plot_bgcolor="white",
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
    )
    return fig


# ──────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Executive Compensation Peer Groups",
        page_icon="📊",
        layout="wide",
    )

    # ── Data source ──────────────────────────
    data_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--input" and i + 1 < len(sys.argv):
            data_path = sys.argv[i + 1]

    if data_path and Path(data_path).exists():
        df_raw = load_data(data_path)
        st.sidebar.success(f"Loaded: {Path(data_path).name}")
    else:
        uploaded = st.sidebar.file_uploader("Upload compensation CSV", type=["csv", "tsv"])
        if uploaded is None:
            st.info("👈 Upload your compensation data CSV to get started, "
                    "or pass `-- --input data.csv` on the command line.")
            st.stop()
        tmp = Path("/tmp/_peer_upload.csv")
        tmp.write_bytes(uploaded.read())
        df_raw = load_data(str(tmp))

    # ── Tabs ─────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Peer Group Explorer",
                                      "📈 Compensation Predictor",
                                      "⚠️ Outlier Detection",
                                      "🌱 ESG Pay Reality Check"])

    # ==========================================================
    # TAB 1 — PEER GROUP EXPLORER (existing)
    # ==========================================================
    with tab1:
        st.header("Peer Group Explorer")
        st.caption("Identify companies with similar pay structures within "
                   "the same role, sector, and index.")

        st.sidebar.header("Peer Group Filters")
        years = sorted(df_raw["year"].dropna().unique().astype(int))
        selected_year = st.sidebar.select_slider("Year", options=years, value=years[-1])

        index_options = sorted(df_raw[INDEX_COL].unique())
        selected_index = st.sidebar.radio(
            "Index Membership", options=["Both"] + index_options,
            index=0, horizontal=True,
        )

        role_options = sorted(df_raw["role"].dropna().unique())
        selected_roles = st.sidebar.multiselect(
            "Roles", options=role_options, default=role_options,
        )

        sector_options = sorted(df_raw["sector"].dropna().unique())
        selected_sectors = st.sidebar.multiselect(
            "Sectors", options=sector_options, default=sector_options,
        )

        if not selected_roles or not selected_sectors:
            st.warning("Select at least one role and one sector.")
            st.stop()

        df_filt = df_raw.copy()
        if selected_index != "Both":
            df_filt = df_filt[df_filt[INDEX_COL] == selected_index]
        df_filt = df_filt[df_filt["role"].isin(selected_roles)]
        df_filt = df_filt[df_filt["sector"].isin(selected_sectors)]

        profiles = aggregate_profiles(df_filt, year=selected_year)
        if profiles.empty:
            st.warning(f"No data for year {selected_year} with the selected filters.")
        else:
            hard = ["role", "sector", INDEX_COL]
            all_results = []
            for _, grp in profiles.groupby(hard):
                if len(grp) < 2:
                    grp = grp.copy()
                    grp["cluster"] = 0
                    grp["x"] = 0
                    grp["y"] = 0
                else:
                    grp = run_clustering(grp)
                all_results.append(grp)
            results = pd.concat(all_results, ignore_index=True)

            combos = (results.groupby(hard).size().reset_index(name="n_companies")
                      .sort_values(["role", "sector", INDEX_COL]))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Companies", results["isin"].nunique())
            c2.metric("Panels", len(combos))
            c3.metric("Sectors", results["sector"].nunique())
            c4.metric("Year", selected_year)
            st.divider()

            for _, combo in combos.iterrows():
                role, sector, idx = combo["role"], combo["sector"], combo[INDEX_COL]
                grp = results[(results["role"] == role) &
                              (results["sector"] == sector) &
                              (results[INDEX_COL] == idx)]
                n = len(grp)
                n_groups = grp["cluster"].nunique()
                title = f"{role}  ·  {sector}  ·  {idx}"

                with st.expander(f"**{title}**  ({n} companies, "
                                 f"{n_groups} peer group{'s' if n_groups != 1 else ''})",
                                 expanded=(n >= 2)):
                    if n < 2:
                        company = grp["company_shortname"].iloc[0]
                        st.info(f"**{company}** is the only company in this group "
                                "— no peers to compare.")
                        ratios = {PAY_RATIO_LABELS.get(c, c): f"{grp[c].iloc[0]*100:.1f}%"
                                  for c in DISPLAY_RATIO_COLS
                                  if c in grp.columns and grp[c].iloc[0] > 0.001}
                        if ratios:
                            st.caption("Pay breakdown:")
                            cols = st.columns(len(ratios))
                            for j, (k, v) in enumerate(ratios.items()):
                                cols[j].metric(k, v)
                        continue

                    col_chart, col_bar = st.columns([3, 2])
                    with col_chart:
                        st.plotly_chart(build_scatter(grp, title), use_container_width=True)
                    with col_bar:
                        st.plotly_chart(build_pay_breakdown_chart(grp), use_container_width=True)

                    rows_table = []
                    for cid in sorted(grp["cluster"].unique()):
                        sub = grp[grp["cluster"] == cid]
                        ci = sorted(grp["cluster"].unique()).index(cid)
                        lbl = (f"Peer Group {GROUP_LABELS[ci]}"
                               if ci < len(GROUP_LABELS) else f"Peer Group {cid+1}")
                        for _, row in sub.iterrows():
                            rd = {"Peer Group": lbl, "Company": row["company_shortname"]}
                            for c in DISPLAY_RATIO_COLS:
                                if c in row and row[c] > 0.001:
                                    rd[PAY_RATIO_LABELS.get(c, c)] = f"{row[c]*100:.1f}%"
                            if SIZE_COL in row:
                                rd["Firm Size"] = f"{row[SIZE_COL]:.3f}"
                            rows_table.append(rd)
                    st.caption("Peer group membership & compensation details")
                    st.dataframe(pd.DataFrame(rows_table),
                                 use_container_width=True, hide_index=True)

            st.divider()
            st.download_button("⬇️  Download cluster assignments (CSV)",
                               results.to_csv(index=False), "peer_groups.csv", "text/csv")

    # ==========================================================
    # TAB 2 — COMPENSATION PREDICTOR
    # ==========================================================
    with tab2:
        st.header("Compensation Predictor")
        st.caption("Estimate expected executive pay levels and composition "
                   "based on company characteristics.")

        has_comp = "comp_per_seat" in df_raw.columns and df_raw["comp_per_seat"].notna().sum() > 50

        # Train models (cached)
        with st.spinner("Training prediction models…"):
            models = train_all_models(df_raw)

        if models is None:
            st.error("Not enough data to train predictive models.")
            st.stop()

        # ── Scenario selectors ───────────────
        st.subheader("Select a scenario")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            pred_year = st.selectbox(
                "Year",
                sorted(df_raw["year"].dropna().unique().astype(int)),
                index=len(df_raw["year"].dropna().unique()) - 1,
                key="pred_year",
            )
        with sc2:
            pred_sector = st.selectbox(
                "Sector",
                sorted(df_raw["sector"].dropna().unique()),
                key="pred_sector",
            )
        with sc3:
            pred_index = st.selectbox(
                "Index",
                sorted(df_raw[INDEX_COL].unique()),
                key="pred_index",
            )

        # Firm size: default to median for this sector+index
        sector_mask = ((df_raw["sector"] == pred_sector) &
                       (df_raw[INDEX_COL] == pred_index))
        median_size = df_raw.loc[sector_mask, SIZE_COL].median()
        if pd.isna(median_size):
            median_size = df_raw[SIZE_COL].median()
        size_min = float(df_raw[SIZE_COL].min())
        size_max = float(df_raw[SIZE_COL].max())

        pred_size = st.slider(
            "Firm Size Score (adjust to see how predictions change)",
            min_value=size_min, max_value=size_max,
            value=float(median_size), step=0.05,
            key="pred_size",
        )

        st.divider()

        # ── Run predictions ──────────────────
        roles_to_predict = sorted(df_raw["role"].dropna().unique())
        predictions = []
        for role in roles_to_predict:
            pred = predict_scenario(models, pred_sector, pred_index,
                                    pred_year, pred_size, role)
            predictions.append(pred)

        # ── Display: predicted comp level ────
        st.subheader(f"Expected Compensation  ·  {pred_sector}  ·  {pred_index}  ·  {pred_year}")

        # Metric cards per role
        role_cols = st.columns(len(predictions))
        for i, pred in enumerate(predictions):
            with role_cols[i]:
                color = ROLE_COLORS.get(pred["role"], "#666")
                st.markdown(f"### {pred['role']}")
                if "comp_per_seat" in pred:
                    st.metric("Predicted Total Comp", f"€{pred['comp_per_seat']:,.0f}K")
                # Ratio breakdown
                for col in DISPLAY_RATIO_COLS:
                    val = pred["ratios"].get(col, 0)
                    if val > 0.005:
                        st.caption(f"{PAY_RATIO_LABELS.get(col, col)}: "
                                   f"**{val*100:.1f}%**")

        st.divider()

        # ── Charts ───────────────────────────
        ch1, ch2 = st.columns(2)
        with ch1:
            if has_comp:
                fig_comp = build_predicted_comp_chart(predictions)
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.info("comp_per_seat column not found — "
                        "pay level chart unavailable.")
        with ch2:
            fig_mix = build_predicted_mix_chart(predictions)
            st.plotly_chart(fig_mix, use_container_width=True)

        # ── Actual vs Predicted comparison ───
        if has_comp:
            st.divider()
            st.subheader("Actual vs Predicted — Individual Companies")
            st.caption("Companies above the dashed line are paid more than "
                       "predicted; below means less.")

            year_data = df_raw[
                (df_raw["year"] == pred_year) &
                (df_raw["sector"] == pred_sector) &
                (df_raw[INDEX_COL] == pred_index)
            ].copy()

            if not year_data.empty and models:
                rows_avp = []
                for _, row in year_data.iterrows():
                    pred = predict_scenario(
                        models, pred_sector, pred_index,
                        pred_year, row[SIZE_COL], row["role"],
                    )
                    if "comp_per_seat" in pred:
                        actual = row["comp_per_seat"]
                        predicted = pred["comp_per_seat"]
                        dev = ((actual - predicted) / predicted * 100
                               if predicted > 0 else 0)
                        rows_avp.append({
                            "company_shortname": row["company_shortname"],
                            "role": row["role"],
                            "actual": actual,
                            "predicted": predicted,
                            "deviation_pct": dev,
                        })

                if rows_avp:
                    avp_df = pd.DataFrame(rows_avp)

                    fig_avp = build_actual_vs_predicted_chart(avp_df)
                    st.plotly_chart(fig_avp, use_container_width=True)

                    # Outlier table
                    avp_display = avp_df.sort_values("deviation_pct", key=abs, ascending=False).copy()
                    avp_display["Actual (€K)"] = avp_display["actual"].map(lambda x: f"{x:,.0f}")
                    avp_display["Predicted (€K)"] = avp_display["predicted"].map(lambda x: f"{x:,.0f}")
                    avp_display["Deviation"] = avp_display["deviation_pct"].map(
                        lambda x: f"{'🔺' if x > 0 else '🔻'} {abs(x):.0f}%"
                    )
                    avp_display = (avp_display[["company_shortname", "role",
                                                "Actual (€K)", "Predicted (€K)", "Deviation"]]
                                   .rename(columns={"company_shortname": "Company",
                                                     "role": "Role"}))
                    st.dataframe(avp_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No compensation data to compare for this selection.")
            else:
                st.info("No actual data for this sector/index/year combination.")

        # ── Model stats ──────────────────────
        with st.expander("📊 Model performance metrics"):
            perf_rows = []
            for key in ["comp_per_seat"] + PREDICT_TARGETS:
                if key in models:
                    m = models[key]
                    perf_rows.append({
                        "Target": PAY_RATIO_LABELS.get(key, key.replace("_", " ").title()),
                        "R² Score": f"{m['r2']:.3f}",
                        "Mean Abs Error": f"{m['mae']:.4f}",
                    })
            if perf_rows:
                st.dataframe(pd.DataFrame(perf_rows),
                             use_container_width=True, hide_index=True)
                st.caption("R² = how much variance the model explains (1.0 = perfect). "
                           "MAE = average prediction error.")

    # ==========================================================
    # TAB 3 — OUTLIER DETECTION
    # ==========================================================
    with tab3:
        st.header("Outlier Detection")
        st.caption("Flag executives whose actual compensation deviates "
                   "significantly from what the model predicts for their "
                   "role, sector, index, and firm size.")

        has_comp_t3 = ("comp_per_seat" in df_raw.columns
                       and df_raw["comp_per_seat"].notna().sum() > 50)

        if not has_comp_t3:
            st.error("Outlier detection requires the `comp_per_seat` column.")
            st.stop()

        # Train models (reuses cache from Tab 2)
        with st.spinner("Loading models…"):
            models_t3 = train_all_models(df_raw)
        if models_t3 is None:
            st.error("Not enough data to train models.")
            st.stop()

        # ── Filters ──────────────────────────
        st.subheader("Filters")
        oc1, oc2, oc3, oc4 = st.columns(4)
        with oc1:
            ot_year = st.selectbox(
                "Year",
                sorted(df_raw["year"].dropna().unique().astype(int)),
                index=len(df_raw["year"].dropna().unique()) - 1,
                key="ot_year",
            )
        with oc2:
            ot_index = st.selectbox(
                "Index",
                ["All"] + sorted(df_raw[INDEX_COL].unique()),
                key="ot_index",
            )
        with oc3:
            ot_sector = st.selectbox(
                "Sector",
                ["All"] + sorted(df_raw["sector"].dropna().unique()),
                key="ot_sector",
            )
        with oc4:
            ot_threshold = st.slider(
                "Deviation threshold (%)",
                min_value=10, max_value=200, value=50, step=5,
                key="ot_threshold",
                help="Companies deviating more than this % from predicted are flagged.",
            )

        # ── Compute deviations ───────────────
        year_df = df_raw[df_raw["year"] == ot_year].copy()
        if ot_index != "All":
            year_df = year_df[year_df[INDEX_COL] == ot_index]
        if ot_sector != "All":
            year_df = year_df[year_df["sector"] == ot_sector]

        if year_df.empty:
            st.warning("No data for this filter combination.")
        else:
            rows_out = []
            for _, row in year_df.iterrows():
                if pd.isna(row.get(SIZE_COL)) or pd.isna(row.get("comp_per_seat")):
                    continue
                pred = predict_scenario(
                    models_t3, row["sector"], row[INDEX_COL],
                    ot_year, row[SIZE_COL], row["role"],
                )
                if "comp_per_seat" not in pred:
                    continue
                actual = row["comp_per_seat"]
                predicted = pred["comp_per_seat"]
                dev = ((actual - predicted) / predicted * 100
                       if predicted > 0 else 0)
                rows_out.append({
                    "company_shortname": row["company_shortname"],
                    "role": row["role"],
                    "sector": row["sector"],
                    INDEX_COL: row[INDEX_COL],
                    "actual": actual,
                    "predicted": predicted,
                    "deviation_pct": dev,
                    "is_outlier": abs(dev) > ot_threshold,
                    **{c: row.get(c, 0) for c in DISPLAY_RATIO_COLS},
                    SIZE_COL: row.get(SIZE_COL, 0),
                })

            if not rows_out:
                st.warning("No valid compensation data for this selection.")
            else:
                out_df = pd.DataFrame(rows_out)
                outliers = out_df[out_df["is_outlier"]]
                n_total = len(out_df)
                n_outliers = len(outliers)
                n_over = len(outliers[outliers["deviation_pct"] > 0])
                n_under = len(outliers[outliers["deviation_pct"] < 0])

                st.divider()

                # ── Summary metrics ──────────
                st.subheader(f"Results  ·  {ot_year}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Positions Analyzed", n_total)
                m2.metric("Outliers Flagged", n_outliers)
                m3.metric("🔺 Overpaid", n_over)
                m4.metric("🔻 Underpaid", n_under)

                st.divider()

                # ── Tornado chart (outliers only) ─
                if n_outliers > 0:
                    st.subheader("Flagged Outliers")
                    fig_tornado = build_tornado_chart(outliers, ot_threshold)
                    st.plotly_chart(fig_tornado, use_container_width=True)

                    # ── Outlier table ─────────
                    tbl = outliers.sort_values("deviation_pct", key=abs,
                                               ascending=False).copy()
                    tbl_display = pd.DataFrame({
                        "Company": tbl["company_shortname"],
                        "Role": tbl["role"],
                        "Sector": tbl["sector"],
                        "Index": tbl[INDEX_COL],
                        "Actual (€K)": tbl["actual"].map(lambda x: f"{x:,.0f}"),
                        "Predicted (€K)": tbl["predicted"].map(lambda x: f"{x:,.0f}"),
                        "Deviation": tbl["deviation_pct"].map(
                            lambda x: f"{'🔺' if x > 0 else '🔻'} {abs(x):.0f}%"
                        ),
                    })
                    st.dataframe(tbl_display, use_container_width=True,
                                 hide_index=True)

                    # ── Drill-down: compare vs peers ─
                    st.divider()
                    st.subheader("Drill Down — Compare Outlier vs Peers")

                    outlier_names = (outliers["company_shortname"] + " ("
                                     + outliers["role"] + ")").tolist()
                    selected_outlier = st.selectbox(
                        "Select an outlier to inspect",
                        outlier_names,
                        key="ot_drilldown",
                    )

                    # Parse selection
                    sel_company = selected_outlier.rsplit(" (", 1)[0]
                    sel_role = selected_outlier.rsplit("(", 1)[1].rstrip(")")
                    sel_row = outliers[
                        (outliers["company_shortname"] == sel_company) &
                        (outliers["role"] == sel_role)
                    ].iloc[0]

                    # Peer average: same role + sector + index
                    peers = out_df[
                        (out_df["role"] == sel_role) &
                        (out_df["sector"] == sel_row["sector"]) &
                        (out_df[INDEX_COL] == sel_row[INDEX_COL]) &
                        (out_df["company_shortname"] != sel_company)
                    ]

                    dd1, dd2 = st.columns(2)
                    with dd1:
                        st.markdown(f"**{sel_company}** — {sel_role}")
                        st.metric("Actual Comp",
                                  f"€{sel_row['actual']:,.0f}K")
                        st.metric("Predicted Comp",
                                  f"€{sel_row['predicted']:,.0f}K")
                        dev_val = sel_row["deviation_pct"]
                        st.metric("Deviation",
                                  f"{'🔺' if dev_val > 0 else '🔻'}"
                                  f" {abs(dev_val):.0f}%")

                    with dd2:
                        if not peers.empty:
                            peer_avg = {c: peers[c].mean()
                                        for c in DISPLAY_RATIO_COLS}
                            row_data = {c: sel_row[c]
                                        for c in DISPLAY_RATIO_COLS}
                            row_data["company_shortname"] = sel_company
                            fig_comp_vs = build_outlier_composition_chart(
                                row_data, peer_avg
                            )
                            st.plotly_chart(fig_comp_vs,
                                           use_container_width=True)
                        else:
                            st.info("No peers in the same group to compare.")

                else:
                    st.success(f"✅ No outliers found at the "
                               f"±{ot_threshold}% threshold. All "
                               f"compensation is within expected range.")

    # ==========================================================
    # TAB 4 — ESG PAY REALITY CHECK
    # ==========================================================
    with tab4:
        st.header("ESG Pay Reality Check")
        st.caption("ESG-linked executive pay — DSW data, 2023–2024. "
                   "A screening tool, not a verdict.")

        esg_df = load_esg_data()
        if esg_df is None:
            st.warning("ESG data files not found. Place the DSW Excel files "
                       "in one of the expected paths, or update "
                       "`ESG_FILE_CANDIDATES` in the script.")
            st.stop()

        master = load_master_db()

        # ── ESG Filters (inline, not sidebar) ─
        ef1, ef2 = st.columns(2)
        esg_years = sorted(esg_df["year"].dropna().unique().tolist())
        esg_idxs  = sorted([x for x in esg_df["index"].dropna().unique().tolist()])
        with ef1:
            sel_esg_year = st.multiselect("Year", esg_years, default=esg_years,
                                          key="esg_year")
        with ef2:
            sel_esg_idx = st.multiselect("Index", esg_idxs, default=esg_idxs,
                                         key="esg_idx")

        f = esg_df[esg_df["year"].isin(sel_esg_year)
                   & esg_df["index"].isin(sel_esg_idx)].copy()
        if f.empty:
            st.warning("No ESG data for the current filters.")
            st.stop()

        # ── Shared computations ──────────────
        a = f["achievement"].dropna()
        w = f["esg_weight"].dropna()
        e_any = f["emission_any"].dropna()
        median_ach = a.median() if len(a) else float("nan")
        hit_rate = (a >= ACHIEVE_HIT).mean() * 100 if len(a) else float("nan")

        bad = pd.DataFrame()
        if master is not None:
            j = f.merge(master["tsr_lookup"],
                        left_on=["company", "year"],
                        right_on=["name", "year"], how="left")
            bad = j[(j["achievement"] >= ACHIEVE_HIT) & (j["tsr"] < 0)].copy()

        # ── Hero metrics ─────────────────────
        st.markdown("---")
        st.markdown('### German boards almost never miss their "green" pay targets')

        h1, h2, h3 = st.columns(3)
        h1.metric("Median ESG-target achievement",
                  f"{median_ach:.0f}%" if len(a) else "n/a",
                  help="100% = on target. Above 100% = beat it.")
        h2.metric("Hit or beat their ESG target",
                  f"{hit_rate:.0f}%" if len(a) else "n/a")
        if not bad.empty:
            h3.metric("Cashed in full while shareholders LOST money",
                      f"{len(bad)} firms")
        else:
            h3.metric("Have an emission target somewhere",
                      f"{e_any.mean()*100:.0f}%" if len(e_any) else "n/a")

        if not bad.empty:
            worst = bad.sort_values("tsr").head(3)
            names = ", ".join(f"{r['company']} ({r['tsr']*100:.0f}%)"
                              for _, r in worst.iterrows())
            st.markdown(
                f"A genuine stretch goal would not be hit this reliably. "
                f"**{len(bad)} companies hit their ESG target in a year "
                f"their shareholders lost money** (worst: {names}).")
        else:
            st.markdown(
                "A genuine stretch goal would not be hit this reliably. "
                "The sections below show where the climate teeth sit.")

        st.caption("Screening signal, not a verdict. 'Pays out when "
                   "shareholders lose' means decoupled from shareholder "
                   "value, not that the target was fake.")

        st.divider()

        # ── Section 1: The Breakdown ─────────
        with st.expander("📊 **The Breakdown** — Are the targets too easy?",
                         expanded=True):
            st.markdown(
                "*If a big slice of the bonus rides on ESG but the target "
                "is hit almost every time, the ESG component looks less like "
                "a stretch goal and more like a reliable payout.*")

            scat = f.dropna(subset=["esg_weight", "achievement"])
            if not scat.empty:
                fig = px.scatter(
                    scat, x="esg_weight", y="achievement",
                    color="index", hover_name="company",
                    hover_data={"year": True, "esg_weight": ":.1f",
                                "achievement": ":.1f"},
                    opacity=0.75, template="plotly_white",
                    labels={"esg_weight": "ESG share of bonus (%)",
                            "achievement": "Target achievement (%)"},
                )
                fig.update_traces(marker=dict(size=10,
                                  line=dict(width=0.5, color="white")))
                fig.add_hline(y=ACHIEVE_HIT, line_dash="dash",
                              line_color=ESG_DANGER,
                              annotation_text="100% — target hit",
                              annotation_position="top left")
                fig.add_vrect(x0=WEIGHT_HIGH, x1=105,
                              fillcolor=ESG_WARN, opacity=0.06,
                              line_width=0)
                fig.update_layout(xaxis_range=[-3, 105],
                                  yaxis_range=[-10, 250],
                                  legend_title_text="Index",
                                  margin=dict(l=40, r=20, t=20, b=40),
                                  height=460)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(
                    f"**Takeaway:** {hit_rate:.0f}% of company-years met or "
                    f"beat their ESG target (median {median_ach:.0f}%).")
            else:
                st.info("Not enough weight/achievement data.")

            st.markdown("---")
            st.markdown("#### Where are the climate teeth?")

            sti_pct = (f["emission_sti"].dropna().mean() * 100
                       if f["emission_sti"].notna().any() else 0)
            lti_pct = (f["emission_lti"].dropna().mean() * 100
                       if f["emission_lti"].notna().any() else 0)
            place = pd.DataFrame({
                "Plan": ["Annual bonus (STI)", "Long-term plan (LTI)"],
                "Share": [round(sti_pct), round(lti_pct)],
            })
            b1 = px.bar(place, x="Share", y="Plan", orientation="h",
                        template="plotly_white", text="Share",
                        labels={"Share": "% with emission-reduction KPI"})
            b1.update_traces(marker_color=[ESG_SOFT, ESG_ACCENT],
                             texttemplate="%{text}%",
                             textposition="outside", cliponaxis=False)
            b1.update_layout(yaxis=dict(autorange="reversed"),
                             xaxis_range=[0, 100],
                             margin=dict(l=10, r=30, t=10, b=30),
                             height=170, showlegend=False)
            st.markdown("**Emission targets sit in long-term pay, "
                        "not the annual bonus**")
            st.plotly_chart(b1, use_container_width=True)

            comp_esg = pd.DataFrame({
                "ESG type": ["Environmental", "Social", "Governance"],
                "Share": [round(f["has_E"].mean()*100),
                          round(f["has_S"].mean()*100),
                          round(f["has_G"].mean()*100)],
            })
            b2 = px.bar(comp_esg, x="Share", y="ESG type", orientation="h",
                        template="plotly_white", text="Share",
                        labels={"Share": "% of companies (STI KPIs)"})
            b2.update_traces(marker_color=[ESG_ACCENT, ESG_WARN, ESG_SOFT],
                             texttemplate="%{text}%",
                             textposition="outside", cliponaxis=False)
            b2.update_layout(yaxis=dict(autorange="reversed"),
                             xaxis_range=[0, 100],
                             margin=dict(l=10, r=30, t=10, b=30),
                             height=200, showlegend=False)
            st.markdown("**Inside the annual bonus, 'ESG' is mostly "
                        "social, not environmental**")
            st.plotly_chart(b2, use_container_width=True)

        # ── Section 2: Which Companies ───────
        with st.expander("🔍 **Which Companies** — screening table"):
            st.markdown(
                "*One point each for: high ESG weight, target hit, and no "
                "emission target anywhere.*")

            s = f.copy()
            s["f_high_weight"] = (s["esg_weight"] >= WEIGHT_HIGH).astype("Int64")
            s["f_target_hit"]  = (s["achievement"] >= ACHIEVE_HIT).astype("Int64")
            s["f_no_emission"] = (s["emission_any"] == 0).astype("Int64")
            s["Screening score (0–3)"] = s[["f_high_weight", "f_target_hit",
                                             "f_no_emission"]].sum(axis=1, min_count=1)
            table = (s.sort_values(["Screening score (0–3)", "esg_weight"],
                                   ascending=[False, False])
                       .assign(**{
                           "ESG weight %": s["esg_weight"].round(1),
                           "Achievement %": s["achievement"].round(0),
                           "Emission KPI": s["emission_any"].map(
                               {1: "Yes", 0: "No"}),
                       })
                       [["company", "year", "index", "ESG weight %",
                         "Achievement %", "Emission KPI",
                         "Screening score (0–3)"]]
                       .rename(columns={"company": "Company",
                                        "year": "Year",
                                        "index": "Index"}))
            st.dataframe(table, use_container_width=True, hide_index=True)

            if not bad.empty:
                bt = bad.copy()
                bt["Achievement %"] = bt["achievement"].round(0)
                bt["Shareholder return %"] = (bt["tsr"] * 100).round(0)
                bt = (bt.sort_values("tsr")
                        [["company", "year", "Achievement %",
                          "Shareholder return %"]]
                        .rename(columns={"company": "Company",
                                         "year": "Year"}))
                st.markdown("---")
                st.markdown(f"**{len(bt)} companies hit their ESG bonus "
                            f"target in a year their shareholders lost "
                            f"money:**")
                st.dataframe(bt, use_container_width=True, hide_index=True)

        # ── Section 3: Robustness ────────────
        with st.expander("🧪 **Is it a fluke?** — robustness checks"):
            if master is None:
                st.info("Master database not found — robustness checks "
                        "need 2008-2024_longitudinal.csv.")
            else:
                st.markdown(
                    "*Two checks against the master database. "
                    "Shareholder-return data covers 2022–2024.*")
                r1, r2 = st.columns(2)
                with r1:
                    ta = master["test_a"]
                    ta = ta[(ta["year"] >= 2008)
                            & (ta["year"] <= 2024)].copy()
                    ta["median_tsr_pct"] = (ta["median_tsr"]*100).round(0)
                    figA = px.bar(ta, x="year", y="median_tsr_pct",
                                 template="plotly_white",
                                 labels={"median_tsr_pct":
                                         "Median shareholder return (%)",
                                         "year": "Year"})
                    figA.update_traces(marker_color=[
                        ESG_DANGER if v < 0 else ESG_ACCENT
                        for v in ta["median_tsr_pct"]])
                    figA.add_hline(y=0, line_color="#888")
                    figA.update_layout(margin=dict(l=10, r=10, t=10, b=30),
                                       height=260, showlegend=False)
                    st.markdown("**Test A — market context**")
                    st.plotly_chart(figA, use_container_width=True)
                with r2:
                    tb = master["test_b"].set_index("neg")
                    pct_neg = (tb.loc[True, "pct_bonus"]
                               if True in tb.index else float("nan"))
                    mb_pos = (tb.loc[False, "median_bonus"]
                              if False in tb.index else float("nan"))
                    mb_neg = (tb.loc[True, "median_bonus"]
                              if True in tb.index else float("nan"))
                    drop = ((1 - mb_neg / mb_pos) * 100
                            if mb_pos else float("nan"))
                    st.markdown("**Test B — bonus resilience**")
                    bm1, bm2 = st.columns(2)
                    bm1.metric("Still got a bonus when shareholders lost",
                               f"{pct_neg:.0f}%")
                    bm2.metric("How much the bonus dropped",
                               f"−{drop:.0f}%")


if __name__ == "__main__":
    main()