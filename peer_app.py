"""
Interactive Peer-Group Explorer  ·  Streamlit App
==================================================
Run:  streamlit run peer_app.py -- --input data.csv
Install:  pip install streamlit plotly scikit-learn pandas numpy
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
# CLUSTERING
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
# PLOTLY VISUALISATION
# ──────────────────────────────────────────────
def _hull_shape(points, color, pad=0.35):
    """Return a plotly Shape dict for a convex-hull bubble."""
    if len(points) < 3:
        cx, cy = points.mean(axis=0)
        r = max(np.ptp(points, axis=0).max() * 0.6, pad)
        # Approximate circle with many points
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
    return dict(
        type="path", path=path,
        fillcolor=color, opacity=0.12,
        line=dict(color=color, width=2),
        layer="below",
    )


def build_scatter(group_df: pd.DataFrame, title: str) -> go.Figure:
    """Build an interactive plotly scatter for one (role, sector, index) group."""
    fig = go.Figure()
    clusters = sorted(group_df["cluster"].unique())

    for i, cid in enumerate(clusters):
        sub = group_df[group_df["cluster"] == cid]
        color = PALETTE[i % len(PALETTE)]
        label = f"Peer Group {GROUP_LABELS[i]}" if i < len(GROUP_LABELS) else f"Peer Group {cid+1}"

        # Build hover text with pay breakdown
        hovers = []
        for _, row in sub.iterrows():
            parts = [f"<b>{row['company_shortname']}</b>"]
            parts.append(f"Peer Group: {label}")
            parts.append("─" * 20)
            for col in PAY_RATIO_COLS:
                if col in row and row[col] > 0.001:
                    pct = row[col] * 100
                    parts.append(f"{PAY_RATIO_LABELS.get(col, col)}: {pct:.1f}%")
            if SIZE_COL in row:
                parts.append(f"Firm Size Score: {row[SIZE_COL]:.2f}")
            hovers.append("<br>".join(parts))

        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"],
            mode="markers+text",
            marker=dict(size=14, color=color, line=dict(width=1.5, color="white")),
            text=sub["company_shortname"],
            textposition="top center",
            textfont=dict(size=10, color=color, family="Arial Black"),
            hovertext=hovers,
            hoverinfo="text",
            name=label,
            legendgroup=label,
        ))

        # Draw bubble around cluster (only if ≥2 points)
        pts = sub[["x", "y"]].values
        if len(pts) >= 2:
            fig.add_shape(_hull_shape(pts, color, pad=0.4))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(
            title="",
            showticklabels=False,
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title="",
            showticklabels=False,
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        plot_bgcolor="white",
        height=500,
        margin=dict(t=60, b=40, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        annotations=[
            dict(
                text="📌 Companies closer together have more similar compensation structures",
                xref="paper", yref="paper",
                x=0.5, y=-0.06, showarrow=False,
                font=dict(size=11, color="#666"),
                xanchor="center",
            )
        ],
    )
    return fig


def build_pay_breakdown_chart(group_df: pd.DataFrame) -> go.Figure:
    """Stacked horizontal bar showing average pay structure per peer group."""
    clusters = sorted(group_df["cluster"].unique())
    fig = go.Figure()

    ratio_cols_present = [c for c in PAY_RATIO_COLS if c in group_df.columns and group_df[c].sum() > 0]
    bar_colors = {
        "ratio_salary": "#4C72B0",
        "ratio_sti": "#DD8452",
        "ratio_lti": "#55A868",
        "ratio_equity": "#C44E52",
        "ratio_other": "#8172B3",
        "ratio_pension": "#CCB974",
        "ratio_stock": "#64B5CD",
        "ratio_option": "#E5AE38",
    }

    group_names = []
    for i, cid in enumerate(clusters):
        sub = group_df[group_df["cluster"] == cid]
        label = f"Group {GROUP_LABELS[i]}" if i < len(GROUP_LABELS) else f"Group {cid+1}"
        members = ", ".join(sub["company_shortname"].tolist())
        group_names.append(f"{label} ({len(sub)})")

    for col in ratio_cols_present:
        vals = []
        hovers = []
        for i, cid in enumerate(clusters):
            sub = group_df[group_df["cluster"] == cid]
            mean_val = sub[col].mean() * 100
            vals.append(mean_val)
            members = ", ".join(sub["company_shortname"].tolist())
            hovers.append(f"{PAY_RATIO_LABELS.get(col, col)}: {mean_val:.1f}%<br>Members: {members}")

        fig.add_trace(go.Bar(
            y=group_names, x=vals,
            name=PAY_RATIO_LABELS.get(col, col),
            orientation="h",
            marker_color=bar_colors.get(col, "#999"),
            hovertext=hovers,
            hoverinfo="text",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Average Pay Structure by Peer Group", font=dict(size=14)),
        xaxis=dict(title="% of Total Compensation", range=[0, 105]),
        yaxis=dict(title=""),
        height=max(200, 80 * len(clusters)),
        margin=dict(t=50, b=40, l=120, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        plot_bgcolor="white",
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

    st.title("📊 Executive Compensation Peer Groups")
    st.caption("Identify companies with similar pay structures within the same role, sector, and index.")

    # ── Data source ──────────────────────────
    # Check CLI args for --input, otherwise show file uploader
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
            st.info("👈 Upload your compensation data CSV to get started, or pass `-- --input data.csv` on the command line.")
            st.stop()
        # Save to temp and load
        tmp = Path("/tmp/_peer_upload.csv")
        tmp.write_bytes(uploaded.read())
        df_raw = load_data(str(tmp))

    # ── Sidebar filters ─────────────────────
    st.sidebar.header("Filters")

    years = sorted(df_raw["year"].dropna().unique().astype(int))
    selected_year = st.sidebar.select_slider(
        "Year", options=years, value=years[-1]
    )

    index_options = sorted(df_raw[INDEX_COL].unique())
    selected_index = st.sidebar.radio(
        "Index Membership",
        options=["Both"] + index_options,
        index=0,
        horizontal=True,
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

    # ── Filter & aggregate ───────────────────
    df_filt = df_raw.copy()
    if selected_index != "Both":
        df_filt = df_filt[df_filt[INDEX_COL] == selected_index]
    df_filt = df_filt[df_filt["role"].isin(selected_roles)]
    df_filt = df_filt[df_filt["sector"].isin(selected_sectors)]

    profiles = aggregate_profiles(df_filt, year=selected_year)
    if profiles.empty:
        st.warning(f"No data for year {selected_year} with the selected filters.")
        st.stop()

    # ── Cluster each group ───────────────────
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

    # ── Display ──────────────────────────────
    combos = (
        results.groupby(hard)
        .size()
        .reset_index(name="n_companies")
        .sort_values(["role", "sector", INDEX_COL])
    )

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", results["isin"].nunique())
    c2.metric("Panels", len(combos))
    c3.metric("Sectors", results["sector"].nunique())
    c4.metric("Year", selected_year)

    st.divider()

    # Panels
    for _, combo in combos.iterrows():
        role, sector, idx = combo["role"], combo["sector"], combo[INDEX_COL]
        grp = results[
            (results["role"] == role) &
            (results["sector"] == sector) &
            (results[INDEX_COL] == idx)
        ]
        n = len(grp)
        n_groups = grp["cluster"].nunique()
        title = f"{role}  ·  {sector}  ·  {idx}"

        with st.expander(f"**{title}**  ({n} companies, {n_groups} peer group{'s' if n_groups != 1 else ''})", expanded=(n >= 2)):
            if n < 2:
                company = grp["company_shortname"].iloc[0]
                st.info(f"**{company}** is the only company in this group — no peers to compare.")

                # Still show pay breakdown for the single company
                ratios = {PAY_RATIO_LABELS.get(c, c): f"{grp[c].iloc[0]*100:.1f}%"
                          for c in PAY_RATIO_COLS if c in grp.columns and grp[c].iloc[0] > 0.001}
                if ratios:
                    st.caption("Pay breakdown:")
                    cols = st.columns(len(ratios))
                    for j, (k, v) in enumerate(ratios.items()):
                        cols[j].metric(k, v)
                continue

            # Two-column layout: scatter + pay breakdown
            col_chart, col_bar = st.columns([3, 2])

            with col_chart:
                fig = build_scatter(grp, title)
                st.plotly_chart(fig, use_container_width=True)

            with col_bar:
                fig_bar = build_pay_breakdown_chart(grp)
                st.plotly_chart(fig_bar, use_container_width=True)

            # Peer group membership table
            rows_table = []
            for cid in sorted(grp["cluster"].unique()):
                sub = grp[grp["cluster"] == cid]
                i = sorted(grp["cluster"].unique()).index(cid)
                label = f"Peer Group {GROUP_LABELS[i]}" if i < len(GROUP_LABELS) else f"Peer Group {cid+1}"
                for _, row in sub.iterrows():
                    row_data = {"Peer Group": label, "Company": row["company_shortname"]}
                    for c in PAY_RATIO_COLS:
                        if c in row and row[c] > 0.001:
                            row_data[PAY_RATIO_LABELS.get(c, c)] = f"{row[c]*100:.1f}%"
                    if SIZE_COL in row:
                        row_data["Firm Size"] = f"{row[SIZE_COL]:.3f}"
                    rows_table.append(row_data)

            st.caption("Peer group membership & compensation details")
            st.dataframe(
                pd.DataFrame(rows_table),
                use_container_width=True,
                hide_index=True,
            )

    # ── Download ─────────────────────────────
    st.divider()
    csv = results.to_csv(index=False)
    st.download_button(
        "⬇️  Download cluster assignments (CSV)",
        csv, "peer_groups.csv", "text/csv",
    )


if __name__ == "__main__":
    main()