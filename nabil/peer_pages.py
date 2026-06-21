"""
peer_pages.py — Peer Grouping, GBR Predictor, Outlier Detection
================================================================
Styled to match app.py's dark-navy theme. Uses HTML headers, sidebar nav,
and explicit text colors so nothing disappears.
"""

import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st

warnings.filterwarnings("ignore")

# ── THEME (match app.py) ─────────────────────
NAVY    = "#0a1628"
ORANGE  = "#f97316"
AMBER   = "#d97706"
GREEN   = "#16a34a"
RED     = "#dc2626"
GRAY    = "#64748b"
GRAYLT  = "#f8fafc"

# ── CONFIG ────────────────────────────────────
PAY_RATIO_COLS = [
    "ratio_salary","ratio_sti","ratio_lti","ratio_equity",
    "ratio_other","ratio_pension","ratio_stock","ratio_option",
]
DISPLAY_RATIO_COLS = [
    "ratio_salary","ratio_sti","ratio_lti",
    "ratio_equity","ratio_other","ratio_pension",
]
PAY_RATIO_LABELS = {
    "ratio_salary":"Base Salary","ratio_sti":"Short-Term Bonus",
    "ratio_lti":"Long-Term Incentive","ratio_equity":"Equity",
    "ratio_other":"Other","ratio_pension":"Pension",
}
SIZE_COL = "firm_size"
INDEX_COL = "index_score"
CLUSTER_FEATURES = PAY_RATIO_COLS + [SIZE_COL]
PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3",
           "#DA8BC3","#CCB974","#64B5CD","#E5AE38","#6ACC65"]
GROUP_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ROLE_COLORS = {"CEO":"#4C72B0","CFO":"#DD8452","Other":"#55A868"}
PEER_DATA_CANDIDATES = [
    "exec_features_filled.csv",
    "filtering/exec_features_filled.csv",
    "Data/exec_features_filled.csv",
]

# ── STYLED HELPERS (match app.py theme) ───────
def _nav_home():
    st.session_state.screen = "landing"
    st.rerun()

def _sidebar():
    with st.sidebar:
        st.markdown(f"""<div style="padding:16px 0 8px 0;">
            <div style="font-size:1.5rem;font-weight:800;color:white;">ExComp</div>
            <div style="font-size:0.72rem;color:#64748b;">AI-Powered Pay Intelligence</div>
        </div>""", unsafe_allow_html=True)
        if st.button("← Back to Home", use_container_width=True, key="peer_back"):
            _nav_home()
        st.divider()
        st.markdown('<p style="color:#475569;font-size:0.73rem;line-height:1.7;">'
                    '43 DAX Companies<br>7,500+ Exec Observations<br>2006–2024</p>',
                    unsafe_allow_html=True)

def _hero(icon, title, subtitle):
    st.markdown(f"""<div style="background:linear-gradient(135deg,{NAVY} 0%,#0f2744 55%,#162d52 100%);
        border-radius:20px;padding:28px 36px;margin-bottom:18px;position:relative;overflow:hidden;
        box-shadow:0 8px 40px rgba(0,0,0,.22);">
        <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">{icon} {title}</div>
        <div style="font-size:1.8rem;font-weight:900;color:white;letter-spacing:-.5px;margin:6px 0;">{title}</div>
        <div style="font-size:.84rem;color:#94a3b8;">{subtitle}</div>
    </div>""", unsafe_allow_html=True)

def _section(title, sub=""):
    sub_html = f'<span style="font-size:.78rem;color:{GRAY};display:block;margin-top:3px;">{sub}</span>' if sub else ""
    st.markdown(
        f'<div style="border-left:4px solid {ORANGE};padding-left:12px;margin:26px 0 12px 0;">'
        f'<span style="font-size:1.05rem;font-weight:700;color:{NAVY};display:block;">{title}</span>'
        f'{sub_html}</div>',
        unsafe_allow_html=True)

def _kpi_card(value, label, delta="", color=NAVY):
    st.markdown(f"""<div style="background:white;border-radius:14px;padding:16px;text-align:center;
        border:1px solid rgba(15,39,68,.08);box-shadow:0 2px 12px rgba(0,0,0,.04);
        border-top:3px solid {ORANGE};">
        <div style="font-size:1.8rem;font-weight:800;color:{color};line-height:1;">{value}</div>
        <div style="font-size:.68rem;color:{GRAY};text-transform:uppercase;letter-spacing:.06em;margin-top:6px;font-weight:600;">{label}</div>
        <div style="font-size:.75rem;color:{GRAY};margin-top:3px;">{delta}</div>
    </div>""", unsafe_allow_html=True)

def _text(s):
    st.markdown(f'<div style="font-size:.88rem;color:#1e293b;line-height:1.55;margin-bottom:12px;">{s}</div>',
                unsafe_allow_html=True)

def _insight(s):
    st.markdown(f"""<div style="background:linear-gradient(135deg,#fff7ed,{GRAYLT});
        border:1px solid #fed7aa;border-left:4px solid {ORANGE};
        border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:14px;
        font-size:0.85rem;color:#431407;line-height:1.55;">{s}</div>""", unsafe_allow_html=True)

# ── DATA LOADING ──────────────────────────────
@st.cache_data
def load_peer_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = next((p for p in PEER_DATA_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        return None
    p = Path(path)
    sep = "\t" if p.suffix in (".tsv",".tab") else ","
    df = pd.read_csv(p, sep=sep, engine="python")
    if len(df.columns) < 5:
        df = pd.read_csv(p, sep=",", engine="python")
    df.columns = df.columns.str.strip().str.lower()
    for c in CLUSTER_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "comp_per_seat" in df.columns:
        df["comp_per_seat"] = pd.to_numeric(df["comp_per_seat"], errors="coerce")
    if INDEX_COL in df.columns:
        raw = pd.to_numeric(df[INDEX_COL], errors="coerce")
        if raw.notna().any() and raw.dropna().isin([0,1]).all():
            df[INDEX_COL] = raw.map({0:"MDAX",1:"DAX"}).fillna("Unknown")
        else:
            df[INDEX_COL] = df[INDEX_COL].astype(str).str.strip()
    return df

def _aggregate_profiles(df, year):
    df = df[df["year"]==year].copy()
    if df.empty: return df
    keys = ["isin","company_shortname","role","sector",INDEX_COL]
    feats = [c for c in CLUSTER_FEATURES if c in df.columns]
    agg = df.groupby(keys, as_index=False)[feats].mean()
    agg = agg.dropna(subset=feats, how="all")
    agg[feats] = agg[feats].fillna(0)
    return agg

# ── CLUSTERING ────────────────────────────────
def _best_kmeans(X, k_min=2, k_max=8):
    n = X.shape[0]; k_max = min(k_max, n-1)
    if k_max < k_min: return np.zeros(n, dtype=int), 1
    best_s, best_l, best_k = -1, None, k_min
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2: continue
        s = silhouette_score(X, labels)
        if s > best_s: best_s, best_k, best_l = s, k, labels
    return (best_l, best_k) if best_l is not None else (np.zeros(n, dtype=int), 1)

def _run_clustering(group_df):
    feats = [c for c in CLUSTER_FEATURES if c in group_df.columns]
    X = group_df[feats].values
    Xs = StandardScaler().fit_transform(X)
    labels, k = _best_kmeans(Xs)
    group_df = group_df.copy(); group_df["cluster"] = labels
    nc = min(2, Xs.shape[1], Xs.shape[0])
    coords = PCA(n_components=nc, random_state=42).fit_transform(Xs)
    group_df["x"] = coords[:,0] if nc>=1 else 0
    group_df["y"] = coords[:,1] if nc>=2 else 0
    return group_df

# ── GBR MODEL ─────────────────────────────────
_MODEL_CAT = ["role","sector",INDEX_COL]
_MODEL_NUM = [SIZE_COL,"year"]
_TARGETS = DISPLAY_RATIO_COLS

@st.cache_resource
def _train_models(_df):
    subset = _df.dropna(subset=_MODEL_NUM+_TARGETS).copy()
    if subset.empty: return None
    X = pd.get_dummies(subset[_MODEL_CAT+_MODEL_NUM], columns=_MODEL_CAT, drop_first=False)
    y_ratios = subset[_TARGETS].values
    y_comp = subset["comp_per_seat"].values if "comp_per_seat" in subset.columns else None
    fnames = X.columns.tolist(); models = {"feature_names": fnames}
    if y_comp is not None:
        mask = ~np.isnan(y_comp)
        if mask.sum()>20:
            m = GradientBoostingRegressor(n_estimators=200,max_depth=4,learning_rate=0.1,subsample=0.8,random_state=42)
            m.fit(X[mask],y_comp[mask]); p=m.predict(X[mask])
            models["comp_per_seat"]={"model":m,"r2":r2_score(y_comp[mask],p),"mae":mean_absolute_error(y_comp[mask],p)}
    for i,col in enumerate(_TARGETS):
        y=y_ratios[:,i]; mask=~np.isnan(y)
        if mask.sum()>20:
            m=GradientBoostingRegressor(n_estimators=150,max_depth=3,learning_rate=0.1,subsample=0.8,random_state=42)
            m.fit(X[mask],y[mask]); p=m.predict(X[mask])
            models[col]={"model":m,"r2":r2_score(y[mask],p),"mae":mean_absolute_error(y[mask],p)}
    return models

def _predict(models, sector, index, year, firm_size, role):
    fnames = models["feature_names"]
    row = pd.DataFrame([{SIZE_COL:firm_size,"year":year}])
    for cat,val in [("role",role),("sector",sector),(INDEX_COL,index)]:
        for fn in fnames:
            if fn.startswith(f"{cat}_"): row[fn]=1.0 if fn==f"{cat}_{val}" else 0.0
    for c in fnames:
        if c not in row.columns: row[c]=0.0
    row=row[fnames]
    result={"role":role,"sector":sector,"index":index,"year":year,"firm_size":firm_size}
    if "comp_per_seat" in models:
        result["comp_per_seat"]=max(0,models["comp_per_seat"]["model"].predict(row)[0])
    raw={}
    for col in _TARGETS:
        raw[col]=max(0,models[col]["model"].predict(row)[0]) if col in models else 0.0
    total=sum(raw.values())
    if total>0: raw={k:v/total for k,v in raw.items()}
    result["ratios"]=raw; return result

# ── PLOTLY HELPERS ────────────────────────────
def _hull_shape(points, color, pad=0.35):
    from scipy.spatial import ConvexHull
    if len(points)<3:
        cx,cy=points.mean(axis=0); r=max(np.ptp(points,axis=0).max()*0.6,pad)
        t=np.linspace(0,2*np.pi,60); xs,ys=cx+r*np.cos(t),cy+r*np.sin(t)
    else:
        try:
            hull=ConvexHull(points); hp=points[hull.vertices]; c=hp.mean(axis=0)
            exp=c+(hp-c)*(1+pad); xs=np.append(exp[:,0],exp[0,0]); ys=np.append(exp[:,1],exp[0,1])
        except Exception:
            cx,cy=points.mean(axis=0); r=max(np.ptp(points,axis=0).max()*0.6,pad)
            t=np.linspace(0,2*np.pi,60); xs,ys=cx+r*np.cos(t),cy+r*np.sin(t)
    path="M "+" L ".join(f"{x:.4f},{y:.4f}" for x,y in zip(xs,ys))+" Z"
    return dict(type="path",path=path,fillcolor=color,opacity=0.12,line=dict(color=color,width=2),layer="below")

def _scatter_fig(grp, title):
    fig=go.Figure()
    for i,cid in enumerate(sorted(grp["cluster"].unique())):
        sub=grp[grp["cluster"]==cid]; color=PALETTE[i%len(PALETTE)]; lbl=f"Peer Group {GROUP_LABELS[i]}"
        hovers=[]
        for _,r in sub.iterrows():
            parts=[f"<b>{r['company_shortname']}</b>",f"Peer Group: {lbl}","─"*18]
            for c in DISPLAY_RATIO_COLS:
                if c in r and r[c]>0.001: parts.append(f"{PAY_RATIO_LABELS.get(c,c)}: {r[c]*100:.1f}%")
            if SIZE_COL in r: parts.append(f"Firm Size: {r[SIZE_COL]:.2f}")
            hovers.append("<br>".join(parts))
        fig.add_trace(go.Scatter(x=sub["x"],y=sub["y"],mode="markers+text",
            marker=dict(size=14,color=color,line=dict(width=1.5,color="white")),
            text=sub["company_shortname"],textposition="top center",
            textfont=dict(size=10,color=color,family="Arial Black"),
            hovertext=hovers,hoverinfo="text",name=lbl))
        pts=sub[["x","y"]].values
        if len(pts)>=2: fig.add_shape(_hull_shape(pts,color,pad=0.4))
    fig.update_layout(title=dict(text=title,font=dict(size=15)),
        xaxis=dict(title="",showticklabels=False,showgrid=True,gridcolor="rgba(0,0,0,0.06)",zeroline=False),
        yaxis=dict(title="",showticklabels=False,showgrid=True,gridcolor="rgba(0,0,0,0.06)",zeroline=False),
        plot_bgcolor="white",height=500,margin=dict(t=50,b=40,l=40,r=40),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
    return fig

def _pay_bar(grp):
    clusters=sorted(grp["cluster"].unique()); fig=go.Figure()
    cols_present=[c for c in DISPLAY_RATIO_COLS if c in grp.columns and grp[c].sum()>0]
    bar_colors={"ratio_salary":"#4C72B0","ratio_sti":"#DD8452","ratio_lti":"#55A868",
                "ratio_equity":"#C44E52","ratio_other":"#8172B3","ratio_pension":"#CCB974"}
    names=[f"Group {GROUP_LABELS[i]} ({len(grp[grp['cluster']==cid])})" for i,cid in enumerate(clusters)]
    for col in cols_present:
        vals=[grp[grp["cluster"]==cid][col].mean()*100 for cid in clusters]
        fig.add_trace(go.Bar(y=names,x=vals,name=PAY_RATIO_LABELS.get(col,col),
                             orientation="h",marker_color=bar_colors.get(col,"#999")))
    fig.update_layout(barmode="stack",title=dict(text="Avg Pay Structure by Peer Group",font=dict(size=13)),
        xaxis=dict(title="% of Total Compensation",range=[0,105]),yaxis=dict(title=""),
        height=max(200,80*len(clusters)),margin=dict(t=50,b=40,l=120,r=40),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5),plot_bgcolor="white")
    return fig


# ══════════════════════════════════════════════
# PAGE 1: PEER GROUP EXPLORER
# ══════════════════════════════════════════════
def show_peer_grouping(nav_fn=None):
    _sidebar()

    peer_df = load_peer_data()
    if peer_df is None:
        uploaded = st.file_uploader("Upload exec_features_filled.csv", type=["csv","tsv"], key="peer_upload")
        if uploaded is None:
            st.info("Place `exec_features_filled.csv` in the project folder, or upload it.")
            return
        tmp = Path("/tmp/_peer_upload.csv"); tmp.write_bytes(uploaded.read())
        peer_df = load_peer_data(str(tmp))

    _section("Filters")
    fc1,fc2,fc3,fc4 = st.columns(4)
    years = sorted(peer_df["year"].dropna().unique().astype(int))
    with fc1: sel_year = st.selectbox("Year", years, index=len(years)-1, key="pg_year")
    idx_opts = sorted(peer_df[INDEX_COL].unique())
    with fc2: sel_idx = st.selectbox("Index", ["Both"]+idx_opts, key="pg_idx")
    role_opts = sorted(peer_df["role"].dropna().unique())
    with fc3: sel_roles = st.multiselect("Roles", role_opts, default=role_opts, key="pg_roles")
    sect_opts = sorted(peer_df["sector"].dropna().unique())
    with fc4: sel_sects = st.multiselect("Sectors", sect_opts, default=sect_opts, key="pg_sects")

    if not sel_roles or not sel_sects:
        st.warning("Select at least one role and one sector."); return

    df_f = peer_df.copy()
    if sel_idx != "Both": df_f = df_f[df_f[INDEX_COL]==sel_idx]
    df_f = df_f[df_f["role"].isin(sel_roles) & df_f["sector"].isin(sel_sects)]
    profiles = _aggregate_profiles(df_f, sel_year)
    if profiles.empty:
        st.warning(f"No data for {sel_year} with current filters."); return

    hard = ["role","sector",INDEX_COL]; all_res = []
    for _,grp in profiles.groupby(hard):
        if len(grp)<2:
            grp=grp.copy(); grp["cluster"],grp["x"],grp["y"]=0,0,0
        else: grp=_run_clustering(grp)
        all_res.append(grp)
    results = pd.concat(all_res, ignore_index=True)
    combos = results.groupby(hard).size().reset_index(name="n").sort_values(hard)

    _section("Results")
    c1,c2,c3,c4 = st.columns(4)
    with c1: _kpi_card(results["isin"].nunique(), "Companies")
    with c2: _kpi_card(len(combos), "Panels")
    with c3: _kpi_card(results["sector"].nunique(), "Sectors")
    with c4: _kpi_card(sel_year, "Year")
    st.divider()

    for _,combo in combos.iterrows():
        role,sector,idx = combo["role"],combo["sector"],combo[INDEX_COL]
        grp = results[(results["role"]==role)&(results["sector"]==sector)&(results[INDEX_COL]==idx)]
        n=len(grp); n_g=grp["cluster"].nunique(); title=f"{role}  ·  {sector}  ·  {idx}"
        with st.expander(f"**{title}** ({n} companies, {n_g} peer group{'s' if n_g!=1 else ''})", expanded=(n>=2)):
            if n<2:
                st.info(f"**{grp['company_shortname'].iloc[0]}** — no peers in this group."); continue
            cl,cr = st.columns([3,2])
            with cl: st.plotly_chart(_scatter_fig(grp, title), use_container_width=True)
            with cr: st.plotly_chart(_pay_bar(grp), use_container_width=True)
            rows=[]
            for cid in sorted(grp["cluster"].unique()):
                sub=grp[grp["cluster"]==cid]; ci=sorted(grp["cluster"].unique()).index(cid)
                lbl=f"Peer Group {GROUP_LABELS[ci]}"
                for _,r in sub.iterrows():
                    rd={"Peer Group":lbl,"Company":r["company_shortname"]}
                    for c in DISPLAY_RATIO_COLS:
                        if c in r and r[c]>0.001: rd[PAY_RATIO_LABELS.get(c,c)]=f"{r[c]*100:.1f}%"
                    if SIZE_COL in r: rd["Firm Size"]=f"{r[SIZE_COL]:.3f}"
                    rows.append(rd)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.download_button("⬇️ Download cluster assignments", results.to_csv(index=False), "peer_groups.csv","text/csv")


# ══════════════════════════════════════════════
# PAGE 2: GBR COMPENSATION PREDICTOR
# ══════════════════════════════════════════════
def show_gbr_predictor(nav_fn=None):
    _sidebar()

    peer_df = load_peer_data()
    if peer_df is None:
        st.warning("Peer data not found. Place `exec_features_filled.csv` in the project folder."); return

    has_comp = "comp_per_seat" in peer_df.columns and peer_df["comp_per_seat"].notna().sum()>50
    with st.spinner("Training GBR models…"):
        models = _train_models(peer_df)
    if models is None:
        st.error("Not enough data to train models."); return

    _section("Select a Scenario")
    sc1,sc2,sc3 = st.columns(3)
    with sc1:
        pred_year = st.selectbox("Year", sorted(peer_df["year"].dropna().unique().astype(int)),
                                  index=len(peer_df["year"].dropna().unique())-1, key="gbr_year")
    with sc2:
        pred_sector = st.selectbox("Sector", sorted(peer_df["sector"].dropna().unique()), key="gbr_sector")
    with sc3:
        pred_index = st.selectbox("Index", sorted(peer_df[INDEX_COL].unique()), key="gbr_index")

    # Auto-compute firm size as sector/index median — no user-facing slider
    mask=(peer_df["sector"]==pred_sector)&(peer_df[INDEX_COL]==pred_index)
    pred_size=peer_df.loc[mask,SIZE_COL].median()
    if pd.isna(pred_size): pred_size=peer_df[SIZE_COL].median()
    pred_size=float(pred_size)
    st.divider()

    roles = sorted(peer_df["role"].dropna().unique())
    preds = [_predict(models, pred_sector, pred_index, pred_year, pred_size, r) for r in roles]

    _section(f"Expected Compensation  ·  {pred_sector}  ·  {pred_index}  ·  {pred_year}")
    cols = st.columns(len(preds))
    for i,p in enumerate(preds):
        with cols[i]:
            st.markdown(f'<div style="font-size:1.1rem;font-weight:700;color:{NAVY};margin-bottom:6px;">{p["role"]}</div>', unsafe_allow_html=True)
            if "comp_per_seat" in p:
                _kpi_card(f"€{p['comp_per_seat']:,.0f}K", "Predicted Total Comp")
            for c in DISPLAY_RATIO_COLS:
                v=p["ratios"].get(c,0)
                if v>0.005:
                    st.markdown(f'<div style="font-size:.8rem;color:{GRAY};">{PAY_RATIO_LABELS.get(c,c)}: '
                                f'<strong style="color:{NAVY};">{v*100:.1f}%</strong></div>', unsafe_allow_html=True)
    st.divider()

    ch1,ch2 = st.columns(2)
    with ch1:
        if has_comp:
            r_ok=[p for p in preds if "comp_per_seat" in p]
            fig=go.Figure(go.Bar(x=[p["role"] for p in r_ok], y=[p["comp_per_seat"] for p in r_ok],
                marker_color=[ROLE_COLORS.get(p["role"],"#999") for p in r_ok],
                text=[f"€{p['comp_per_seat']:,.0f}K" for p in r_ok], textposition="outside"))
            fig.update_layout(title="Predicted Total Comp per Seat (€K)", yaxis_title="€K",
                              height=400, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
    with ch2:
        bar_colors={"ratio_salary":"#4C72B0","ratio_sti":"#DD8452","ratio_lti":"#55A868",
                    "ratio_equity":"#C44E52","ratio_other":"#8172B3","ratio_pension":"#CCB974"}
        fig2=go.Figure()
        for c in DISPLAY_RATIO_COLS:
            vals=[p["ratios"].get(c,0)*100 for p in preds]
            if max(vals)<0.1: continue
            fig2.add_trace(go.Bar(x=[p["role"] for p in preds], y=vals,
                                  name=PAY_RATIO_LABELS.get(c,c), marker_color=bar_colors.get(c)))
        fig2.update_layout(barmode="stack", title="Predicted Pay Structure by Role",
                           yaxis=dict(title="% of Total",range=[0,105]), height=400, plot_bgcolor="white",
                           legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
        st.plotly_chart(fig2, use_container_width=True)

    if has_comp:
        st.divider()
        _section("Actual vs Predicted — Individual Companies",
                 "Companies above the dashed line are paid more than predicted")
        yr_data=peer_df[(peer_df["year"]==pred_year)&(peer_df["sector"]==pred_sector)&(peer_df[INDEX_COL]==pred_index)]
        if not yr_data.empty:
            rows_avp=[]
            for _,row in yr_data.iterrows():
                if pd.isna(row.get(SIZE_COL)) or pd.isna(row.get("comp_per_seat")): continue
                pr=_predict(models,pred_sector,pred_index,pred_year,row[SIZE_COL],row["role"])
                if "comp_per_seat" in pr:
                    act,pred_v=row["comp_per_seat"],pr["comp_per_seat"]
                    dev=(act-pred_v)/pred_v*100 if pred_v>0 else 0
                    rows_avp.append({"company_shortname":row["company_shortname"],"role":row["role"],
                                     "actual":act,"predicted":pred_v,"deviation_pct":dev})
            if rows_avp:
                avp=pd.DataFrame(rows_avp)
                fig3=go.Figure()
                for role in avp["role"].unique():
                    sub=avp[avp["role"]==role]
                    fig3.add_trace(go.Scatter(x=sub["predicted"],y=sub["actual"],mode="markers+text",
                        marker=dict(size=10,color=ROLE_COLORS.get(role,"#999")),
                        text=sub["company_shortname"],textposition="top center",textfont=dict(size=8),name=role))
                vals=pd.concat([avp["predicted"],avp["actual"]]); lo,hi=vals.min()*0.8,vals.max()*1.1
                fig3.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",
                    line=dict(dash="dash",color="rgba(0,0,0,0.3)"),showlegend=False))
                fig3.update_layout(title="Actual vs Predicted",xaxis_title="Predicted (€K)",
                    yaxis_title="Actual (€K)",height=500,plot_bgcolor="white",
                    legend=dict(orientation="h",yanchor="bottom",y=1.02))
                st.plotly_chart(fig3, use_container_width=True)
                tbl=avp.sort_values("deviation_pct",key=abs,ascending=False).copy()
                tbl["Deviation"]=tbl["deviation_pct"].map(lambda x: f"{'🔺' if x>0 else '🔻'} {abs(x):.0f}%")
                st.dataframe(tbl[["company_shortname","role","actual","predicted","Deviation"]].rename(
                    columns={"company_shortname":"Company","role":"Role","actual":"Actual (€K)","predicted":"Predicted (€K)"}),
                    use_container_width=True, hide_index=True)

    with st.expander("📊 Model performance", expanded=True):
        perf=[]
        for k in ["comp_per_seat"]+_TARGETS:
            if k in models:
                perf.append({"Target":PAY_RATIO_LABELS.get(k,k.replace("_"," ").title()),
                             "R²":f"{models[k]['r2']:.3f}","MAE":f"{models[k]['mae']:.4f}"})
        if perf: st.dataframe(pd.DataFrame(perf), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# PAGE 3: OUTLIER DETECTION
# ══════════════════════════════════════════════
def show_peer_outlier_detection(nav_fn=None):
    _sidebar()

    peer_df = load_peer_data()
    if peer_df is None:
        st.warning("Peer data not found."); return
    has_comp = "comp_per_seat" in peer_df.columns and peer_df["comp_per_seat"].notna().sum()>50
    if not has_comp:
        st.error("Requires `comp_per_seat` column."); return
    with st.spinner("Loading models…"):
        models = _train_models(peer_df)
    if models is None:
        st.error("Not enough data."); return

    _section("Filters")
    oc1,oc2,oc3,oc4 = st.columns(4)
    with oc1: ot_year=st.selectbox("Year",sorted(peer_df["year"].dropna().unique().astype(int)),
                                    index=len(peer_df["year"].dropna().unique())-1,key="ot_year")
    with oc2: ot_index=st.selectbox("Index",["All"]+sorted(peer_df[INDEX_COL].unique()),key="ot_index")
    with oc3: ot_sector=st.selectbox("Sector",["All"]+sorted(peer_df["sector"].dropna().unique()),key="ot_sector")
    with oc4: ot_thresh=st.slider("Deviation threshold (%)",10,200,50,5,key="ot_thresh",
                                   help="Flag if deviation exceeds this %.")

    yr_df=peer_df[peer_df["year"]==ot_year].copy()
    if ot_index!="All": yr_df=yr_df[yr_df[INDEX_COL]==ot_index]
    if ot_sector!="All": yr_df=yr_df[yr_df["sector"]==ot_sector]
    if yr_df.empty: st.warning("No data for this filter combination."); return

    rows_out=[]
    for _,row in yr_df.iterrows():
        if pd.isna(row.get(SIZE_COL)) or pd.isna(row.get("comp_per_seat")): continue
        pr=_predict(models,row["sector"],row[INDEX_COL],ot_year,row[SIZE_COL],row["role"])
        if "comp_per_seat" not in pr: continue
        act,pred_v=row["comp_per_seat"],pr["comp_per_seat"]
        dev=(act-pred_v)/pred_v*100 if pred_v>0 else 0
        rows_out.append({"company_shortname":row["company_shortname"],"role":row["role"],
            "sector":row["sector"],INDEX_COL:row[INDEX_COL],"actual":act,"predicted":pred_v,
            "deviation_pct":dev,"is_outlier":abs(dev)>ot_thresh,
            **{c:row.get(c,0) for c in DISPLAY_RATIO_COLS}, SIZE_COL:row.get(SIZE_COL,0)})
    if not rows_out: st.warning("No valid data."); return

    out_df=pd.DataFrame(rows_out); outliers=out_df[out_df["is_outlier"]]
    n_total,n_out=len(out_df),len(outliers)
    n_over=len(outliers[outliers["deviation_pct"]>0])
    n_under=len(outliers[outliers["deviation_pct"]<0])

    _section(f"Results  ·  {ot_year}")
    m1,m2,m3,m4=st.columns(4)
    with m1: _kpi_card(n_total, "Positions Analyzed")
    with m2: _kpi_card(n_out, "Outliers Flagged", color=RED if n_out>0 else GREEN)
    with m3: _kpi_card(n_over, "🔺 Overpaid", color=RED if n_over>0 else GRAY)
    with m4: _kpi_card(n_under, "🔻 Underpaid", color="#2563eb" if n_under>0 else GRAY)
    st.divider()

    if n_out>0:
        _section("Flagged Outliers")
        df_t=outliers.sort_values("deviation_pct",ascending=True)
        colors=["#C44E52" if d>0 else "#4C72B0" for d in df_t["deviation_pct"]]
        labels=df_t["company_shortname"]+" ("+df_t["role"]+")"
        fig=go.Figure(go.Bar(y=labels,x=df_t["deviation_pct"],orientation="h",marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Deviation: %{x:+.0f}%<br>"
                          "Actual: €%{customdata[0]:,.0f}K<br>Predicted: €%{customdata[1]:,.0f}K<extra></extra>",
            customdata=df_t[["actual","predicted"]].values))
        fig.add_vline(x=ot_thresh,line_dash="dash",line_color="#C44E52",annotation_text=f"+{ot_thresh}%")
        fig.add_vline(x=-ot_thresh,line_dash="dash",line_color="#4C72B0",annotation_text=f"−{ot_thresh}%")
        fig.add_vline(x=0,line_color="rgba(0,0,0,0.3)",line_width=1)
        fig.update_layout(title="Compensation Deviation from Predicted",xaxis_title="Deviation %",
                          height=max(400,28*n_out),margin=dict(t=50,l=200),plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        tbl=outliers.sort_values("deviation_pct",key=abs,ascending=False)
        tbl_d=pd.DataFrame({"Company":tbl["company_shortname"],"Role":tbl["role"],
            "Sector":tbl["sector"],"Index":tbl[INDEX_COL],
            "Actual (€K)":tbl["actual"].map(lambda x: f"{x:,.0f}"),
            "Predicted (€K)":tbl["predicted"].map(lambda x: f"{x:,.0f}"),
            "Deviation":tbl["deviation_pct"].map(lambda x: f"{'🔺' if x>0 else '🔻'} {abs(x):.0f}%")})
        st.dataframe(tbl_d, use_container_width=True, hide_index=True)

        st.divider()
        _section("Drill Down — Compare Outlier vs Peers")
        names=(outliers["company_shortname"]+" ("+outliers["role"]+")").tolist()
        sel=st.selectbox("Select an outlier",names,key="ot_drill")
        sel_co=sel.rsplit(" (",1)[0]; sel_role=sel.rsplit("(",1)[1].rstrip(")")
        sel_row=outliers[(outliers["company_shortname"]==sel_co)&(outliers["role"]==sel_role)].iloc[0]
        peers=out_df[(out_df["role"]==sel_role)&(out_df["sector"]==sel_row["sector"])
                     &(out_df[INDEX_COL]==sel_row[INDEX_COL])&(out_df["company_shortname"]!=sel_co)]
        d1,d2=st.columns(2)
        with d1:
            st.markdown(f'<div style="font-size:1.1rem;font-weight:700;color:{NAVY};">{sel_co} — {sel_role}</div>', unsafe_allow_html=True)
            _kpi_card(f"€{sel_row['actual']:,.0f}K", "Actual")
            _kpi_card(f"€{sel_row['predicted']:,.0f}K", "Predicted")
            dv=sel_row["deviation_pct"]
            _kpi_card(f"{'🔺' if dv>0 else '🔻'} {abs(dv):.0f}%", "Deviation",
                      color=RED if dv>0 else "#2563eb")
        with d2:
            if not peers.empty:
                peer_avg={c:peers[c].mean() for c in DISPLAY_RATIO_COLS}
                co_vals=[sel_row.get(c,0)*100 for c in DISPLAY_RATIO_COLS]
                pe_vals=[peer_avg.get(c,0)*100 for c in DISPLAY_RATIO_COLS]
                labs=[PAY_RATIO_LABELS.get(c,c) for c in DISPLAY_RATIO_COLS]
                fig_c=go.Figure()
                fig_c.add_trace(go.Bar(x=labs,y=co_vals,name=sel_co,marker_color="#C44E52"))
                fig_c.add_trace(go.Bar(x=labs,y=pe_vals,name="Peer Avg",marker_color="#4C72B0",opacity=0.6))
                fig_c.update_layout(barmode="group",title=f"{sel_co} vs Peers",yaxis_title="%",
                                    height=350,plot_bgcolor="white",
                                    legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=0.5))
                st.plotly_chart(fig_c, use_container_width=True)
            else: st.info("No peers to compare.")
    else:
        st.success(f"✅ No outliers at ±{ot_thresh}% threshold.")