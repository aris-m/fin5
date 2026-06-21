"""
ExComp — AI-Powered Executive Compensation Benchmarking & Red Flag Detection
TUM Science Hackathon 2026 | Chair of Financial Accounting — Prof. Dr. Jürgen Ernstberger

Navigation:
  Landing → Overview (company snapshot + 6 module cards) → Module deep-dive
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from peer_pages import show_peer_grouping, show_gbr_predictor, show_peer_outlier_detection

st.set_page_config(page_title="ExComp", page_icon="🌱", layout="wide",
                   initial_sidebar_state="expanded")

BASE = Path(__file__).parent
NAVY, NAVYLT = "#0a1628", "#0f2744"
ORANGE, ORANGEBG = "#f97316", "#fff7ed"
GREEN, RED, AMBER, GRAY, GRAYLT = "#16a34a", "#dc2626", "#d97706", "#64748b", "#f8fafc"

# ── GLOBAL CSS ────────────────────────────────────────────────
st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
html,body,.stApp{{background:{GRAYLT}!important;font-family:'Inter',system-ui,-apple-system,sans-serif!important}}
[data-testid="stSidebar"]{{background:{NAVY}!important;border-right:1px solid rgba(255,255,255,.04)!important}}
[data-testid="stSidebar"] *{{color:#cbd5e1!important}}
[data-testid="stSidebar"] .stSelectbox>div>div{{background:#132035!important;border-color:#1e3352!important;color:white!important}}
[data-testid="stSidebar"] .stButton>button{{background:linear-gradient(135deg,#1e3a5f,#132035)!important;color:white!important;border:1px solid #2d4a72!important;border-radius:10px!important}}
section[data-testid="stSidebar"] hr{{border-color:rgba(255,255,255,.07)!important}}
.block-container{{padding-top:1.5rem!important;padding-bottom:2rem!important}}

/* Fix: widget labels in main content */
.stApp [data-testid="stAppViewContainer"] label p{{color:{NAVY}!important;font-weight:600!important;font-size:.85rem!important}}
.stApp [data-testid="stAppViewContainer"] .stSelectbox label p,
.stApp [data-testid="stAppViewContainer"] .stMultiSelect label p,
.stApp [data-testid="stAppViewContainer"] .stSlider label p,
.stApp [data-testid="stAppViewContainer"] .stNumberInput label p,
.stApp [data-testid="stAppViewContainer"] .stTextInput label p,
.stApp [data-testid="stAppViewContainer"] .stCheckbox label p{{color:{NAVY}!important;font-weight:600!important}}

/* ─── HERO ─── */
.hero{{
  background:linear-gradient(135deg,{NAVY} 0%,#0f2744 55%,#162d52 100%);
  border-radius:20px;padding:36px 48px;margin-bottom:22px;
  position:relative;overflow:hidden;
  box-shadow:0 8px 40px rgba(0,0,0,.22);
}}
.hero::before{{
  content:'';position:absolute;top:-30%;right:-5%;
  width:420px;height:420px;border-radius:50%;
  background:radial-gradient(circle,rgba(249,115,22,.18) 0%,transparent 68%);
  pointer-events:none;
}}
.hero::after{{
  content:'';position:absolute;bottom:-40%;left:10%;
  width:280px;height:280px;border-radius:50%;
  background:radial-gradient(circle,rgba(99,102,241,.10) 0%,transparent 70%);
  pointer-events:none;
}}

/* ─── KPI CARDS ─── */
.kpi-card{{
  background:white;border-radius:16px;padding:18px 14px;
  text-align:center;
  border:1px solid rgba(15,39,68,.06);
  box-shadow:0 2px 20px rgba(0,0,0,.055),0 1px 4px rgba(0,0,0,.04);
  height:116px;display:flex;flex-direction:column;justify-content:center;
  position:relative;overflow:hidden;
}}
.kpi-card::before{{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,{ORANGE},{AMBER});
}}
.kpi-number{{font-size:2.1rem;font-weight:800;line-height:1;color:{NAVY};letter-spacing:-.03em}}
.kpi-label{{font-size:0.66rem;color:{GRAY};text-transform:uppercase;letter-spacing:.07em;margin-top:5px;font-weight:600}}
.kpi-delta{{font-size:0.74rem;margin-top:4px;font-weight:500}}

/* ─── KPI INFO TOOLTIP ─── */
.kpi-info{{
  position:absolute;top:8px;right:8px;
  width:17px;height:17px;border-radius:50%;
  background:#e2e8f0;color:{GRAY};
  font-size:9px;font-weight:800;
  display:flex;align-items:center;justify-content:center;
  cursor:help;z-index:10;line-height:1;
  border:1px solid #cbd5e1;
  font-family:serif;font-style:italic;
}}
.kpi-info:hover{{background:{NAVY};color:white;border-color:{NAVY};}}
.kpi-info .kpi-tooltip{{
  visibility:hidden;opacity:0;
  position:absolute;top:22px;right:0;
  width:210px;
  background:#1e293b;color:#f1f5f9;
  font-size:11px;font-family:sans-serif;font-style:normal;font-weight:400;
  border-radius:8px;padding:9px 11px;
  z-index:9999;pointer-events:none;
  transition:opacity .18s ease;
  text-align:left;line-height:1.45;
  box-shadow:0 8px 24px rgba(0,0,0,.35);
}}
.kpi-info:hover .kpi-tooltip{{visibility:visible;opacity:1;}}

/* ─── SECTION HEADERS ─── */
.sec{{
  position:relative;
  padding-left:16px;
  margin:30px 0 14px 0;
}}
.sec::before{{
  content:'';
  position:absolute;left:0;top:2px;bottom:2px;
  width:4px;border-radius:2px;
  background:linear-gradient(180deg,{ORANGE},{AMBER});
}}
.sec-title{{font-size:1.05rem;font-weight:700;color:{NAVY};margin:0;letter-spacing:-.01em}}
.sec-sub{{font-size:0.78rem;color:{GRAY};margin:3px 0 0 0;font-weight:400}}

/* ─── INSIGHT BOX ─── */
.insight{{
  background:linear-gradient(135deg,{ORANGEBG},{GRAYLT});
  border:1px solid #fed7aa;border-left:4px solid {ORANGE};
  border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:14px;
  font-size:0.85rem;color:#431407;line-height:1.55;
}}

/* ─── LANDING CARDS ─── */
.lcard{{
  background:white;border-radius:18px;padding:26px;margin-bottom:16px;
  border:1.5px solid rgba(15,39,68,.07);
  box-shadow:0 2px 20px rgba(0,0,0,.05);
  transition:box-shadow .2s,transform .2s;
}}
.lcard:hover{{box-shadow:0 6px 32px rgba(0,0,0,.10);}}
.lcard-active{{
  border:2px solid {ORANGE}!important;
  box-shadow:0 6px 32px rgba(249,115,22,.22)!important;
  background:linear-gradient(135deg,#fffdf9 0%,white 100%)!important;
}}
.lcard-locked{{opacity:.50;filter:grayscale(.3)}}
.tag{{display:inline-block;background:{GRAYLT};border:1px solid #e2e8f0;border-radius:20px;padding:3px 10px;font-size:0.73rem;color:#475569;margin:2px 2px 0 0;font-weight:500}}
.tag-orange{{background:{ORANGEBG};border-color:#fed7aa;color:#9a3412}}

/* ─── LIVE BADGE ─── */
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.6}}}}
.badge-live{{
  background:linear-gradient(135deg,{ORANGEBG},{ORANGE}18);
  color:#9a3412;border:1px solid #fed7aa;
  border-radius:20px;padding:3px 12px;font-size:0.7rem;font-weight:700;
  animation:pulse 2s ease-in-out infinite;
  display:inline-flex;align-items:center;gap:4px;
}}
.badge-soon{{background:{GRAYLT};color:#94a3b8;border-radius:20px;padding:3px 10px;font-size:0.7rem;border:1px solid #e2e8f0}}

/* ─── MODULE CARDS ─── */
.mod-card{{
  background:white;border-radius:18px;padding:24px;
  border:1.5px solid rgba(15,39,68,.07);
  box-shadow:0 2px 16px rgba(0,0,0,.05);
  height:100%;cursor:pointer;
  transition:all .25s ease;
}}
.mod-card:hover{{
  border-color:{ORANGE};
  box-shadow:0 8px 36px rgba(249,115,22,.16);
  transform:translateY(-2px);
}}
.mod-icon-wrap{{
  width:44px;height:44px;border-radius:12px;
  display:flex;align-items:center;justify-content:center;
  font-size:1.35rem;margin-bottom:12px;
}}
.mod-title{{font-size:0.93rem;font-weight:700;color:{NAVY};margin:0 0 5px 0;letter-spacing:-.01em}}
.mod-desc{{font-size:0.77rem;color:{GRAY};margin:0 0 10px 0;line-height:1.45}}
.mod-stat{{font-size:1.35rem;font-weight:800;margin:8px 0 2px 0;letter-spacing:-.02em}}
.mod-statlbl{{font-size:0.66rem;color:{GRAY};text-transform:uppercase;letter-spacing:.05em;font-weight:600}}

/* ─── MISC ─── */
.kpi-tile{{border-radius:10px;padding:12px 15px;margin-bottom:9px}}
.metric-row{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f1f5f9;font-size:0.82rem}}
.metric-row:last-child{{border-bottom:none}}
.flag-red{{background:#fef2f2;border:1px solid #fca5a5;border-left:4px solid {RED};border-radius:0 10px 10px 0;padding:11px 15px;margin-bottom:9px;font-size:0.82rem;color:#7f1d1d;line-height:1.5}}
.flag-amber{{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid {AMBER};border-radius:0 10px 10px 0;padding:11px 15px;margin-bottom:9px;font-size:0.82rem;color:#713f12;line-height:1.5}}
.flag-green{{background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid {GREEN};border-radius:0 10px 10px 0;padding:11px 15px;margin-bottom:9px;font-size:0.82rem;color:#14532d;line-height:1.5}}
.wash-card{{background:linear-gradient(135deg,#fff1f2,#fef2f2);border:1.5px solid #fca5a5;border-radius:14px;padding:18px 20px;margin-bottom:10px}}
.wash-card-ok{{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1.5px solid #bbf7d0;border-radius:14px;padding:18px 20px;margin-bottom:10px}}
.ext-score-badge{{
  display:inline-flex;align-items:center;gap:6px;
  border-radius:8px;padding:6px 14px;font-weight:700;font-size:0.84rem;
}}
</style>""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # features_dax_all.csv: all 54 DAX universe companies × 2006-2024 (from pipeline_v2.py)
    # Falls back to the original features_dax15.csv if the new file is absent
    feat_path = BASE / "features_dax_all.csv"
    if not feat_path.exists():
        feat_path = BASE / "features_dax15.csv"
    features  = pd.read_csv(feat_path)

    coefs     = pd.read_csv(BASE / "model_coefs.csv")
    anomaly_u = pd.read_csv(BASE / "anomaly_scores_universe.csv")

    # model_universe.csv: 54 companies × 2007-2024, full predictions already computed
    mu = pd.read_csv(BASE / "model_universe.csv")
    # Add peer ranking per year from the universe
    mu["peer_rank_total_comp"] = mu.groupby("year")["total_comp_bt"].rank(
        ascending=False, method="min")
    mu["peer_pct_total_comp"] = mu.groupby("year")["total_comp_bt"].rank(
        pct=True, ascending=True) * 100

    # df = features (rich columns, all companies 2006-2024) merged with universe predictions
    pred_cols = ["company_shortname","year","pred_comp","pred_comp_low","pred_comp_high",
                 "actual_vs_expected_pct","benchmark_signal","peer_rank_total_comp","peer_pct_total_comp",
                 "total_comp_bt"]
    df = features.merge(mu[pred_cols], on=["company_shortname","year"], how="left",
                        suffixes=("","_mu"))
    # Fill total_comp_bt from universe for years where features file has NaN (2022-2024)
    if "total_comp_bt_mu" in df.columns:
        df["total_comp_bt"] = df["total_comp_bt"].fillna(df["total_comp_bt_mu"])
        df.drop(columns=["total_comp_bt_mu"], inplace=True)

    esg = pd.DataFrame()
    p = BASE / "esg_dashboard_data.csv"
    if p.exists():
        esg = pd.read_csv(p)

    ext_esg = pd.DataFrame()
    pe = BASE / "external_esg_ratings.csv"
    if pe.exists():
        ext_esg = pd.read_csv(pe)

    return df, coefs, anomaly_u, esg, ext_esg, mu

df, coefs, anomaly_u, esg, ext_esg, mu = load_data()
COMPANIES  = sorted(esg["company_shortname"].unique()) if len(esg) > 0 else sorted(mu["company_shortname"].unique())
MODEL_COS  = set(df["company_shortname"].unique())          # 54 companies with rich features
MU_COS     = set(mu["company_shortname"].unique())          # 54 companies with predictions
YEARS_M    = sorted(df["year"].unique())                    # 2006–2024 (features_dax_all)
YEARS_MU   = sorted(mu["year"].unique())                    # 2007–2024 (model universe)
YEARS_ESG  = sorted(esg["year"].unique()) if len(esg) > 0 else []
YEARS_ALL  = sorted(set(YEARS_MU + YEARS_ESG))             # 2007–2024 (union)
MODEL_YEAR_MAX = max(YEARS_MU)                              # 2024 — full model coverage
FEATURES_YEAR_MAX = max(YEARS_M)                            # 2024 — features cover full range
COMP_DATA_MAX = 2021                                        # actual comp amounts available through 2021

# ── SESSION STATE ─────────────────────────────────────────────
_defaults = {"screen":"landing","module":None,
             "company": "Volkswagen" if "Volkswagen" in COMPANIES else COMPANIES[0],
             "year": 2024}   # default to 2024 — full model + ESG data available
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def nav(screen, module=None):
    st.session_state.screen = screen
    if module: st.session_state.module = module
    st.rerun()

# ── HELPERS ───────────────────────────────────────────────────
def last(series):
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) > 0 else None

def kpi_html(number, label, delta, num_color=None, delta_color=GRAY, bar_color=None, info=None):
    num_color = num_color or NAVY
    bar = f"background:linear-gradient(90deg,{bar_color},{bar_color}99);" if bar_color else f"background:linear-gradient(90deg,{ORANGE},{AMBER});"
    overflow = "visible" if info else "hidden"
    info_html = f'<div class="kpi-info">i<span class="kpi-tooltip">{info}</span></div>' if info else ""
    return f"""<div class="kpi-card" style="overflow:{overflow};">
        {info_html}
        <div style="position:absolute;top:0;left:0;right:0;height:3px;{bar}border-radius:3px 3px 0 0;"></div>
        <div class="kpi-number" style="color:{num_color};">{number}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-delta" style="color:{delta_color};">{delta}</div>
    </div>"""

def sec_html(title, sub=""):
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    return f'<div class="sec"><div class="sec-title">{title}</div>{sub_html}</div>'

def traffic(pct):
    if pct is None or np.isnan(pct): return "⚪", GRAY, "No Data"
    if pct >  40: return "🔴", RED,   f"+{pct:.0f}% significantly overpaid"
    if pct >  15: return "🟡", AMBER, f"+{pct:.0f}% above market"
    if pct > -15: return "🟢", GREEN, f"{pct:+.0f}% market-aligned"
    if pct > -40: return "🟡", AMBER, f"{pct:.0f}% below market"
    return "🔵", "#2563eb", f"{pct:.0f}% significantly underpaid"

def risk_color(val, hi=60, mid=40):
    if val >= hi: return RED
    if val >= mid: return AMBER
    return GREEN

def sidebar_nav(back_screen, back_label="← Back"):
    with st.sidebar:
        st.markdown(f"""<div style="padding:16px 0 8px 0;">
            <div style="font-size:1.5rem;font-weight:800;color:white;">ExComp</div>
            <div style="font-size:0.72rem;color:#64748b;">AI-Powered Pay Intelligence</div>
        </div>""", unsafe_allow_html=True)
        if st.button(back_label, use_container_width=True):
            nav(back_screen)
        st.divider()
        st.markdown('<p style="color:#94a3b8;font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;">Company</p>', unsafe_allow_html=True)
        idx = COMPANIES.index(st.session_state.company) if st.session_state.company in COMPANIES else 0
        co = st.selectbox("", COMPANIES, index=idx, label_visibility="collapsed", key="sb_co")
        if co != st.session_state.company:
            st.session_state.company = co
            st.rerun()
        st.markdown('<p style="color:#94a3b8;font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em;margin-top:10px;margin-bottom:4px;">Year</p>', unsafe_allow_html=True)
        yr_idx = YEARS_ALL.index(st.session_state.year) if st.session_state.year in YEARS_ALL else len(YEARS_ALL)-1
        yr = st.selectbox("", YEARS_ALL, index=yr_idx, label_visibility="collapsed", key="sb_yr")
        if yr != st.session_state.year:
            st.session_state.year = yr
            st.rerun()
        has_rich = yr <= FEATURES_YEAR_MAX
        has_mu   = yr in YEARS_MU
        has_esg  = yr in YEARS_ESG
        if has_rich and has_mu and has_esg:
            st.markdown(f'<div style="background:#1e3a5f;border-radius:6px;padding:6px 10px;margin-top:4px;font-size:.7rem;color:#4ade80;">📊 Full Data Available<br><span style="color:#64748b;">Model + ESG + All Features</span></div>', unsafe_allow_html=True)
        elif has_mu and has_esg:
            st.markdown(f'<div style="background:#1e3a5f;border-radius:6px;padding:6px 10px;margin-top:4px;font-size:.7rem;color:{ORANGE};">🌱 Model + ESG Available<br><span style="color:#64748b;">Governance features until 2021 only</span></div>', unsafe_allow_html=True)
        elif has_mu:
            st.markdown(f'<div style="background:#1e3a5f;border-radius:6px;padding:6px 10px;margin-top:4px;font-size:.7rem;color:#94a3b8;">📈 Model Data Only<br><span style="color:#64748b;">ESG data not available</span></div>', unsafe_allow_html=True)
        st.divider()
        st.markdown('<p style="color:#475569;font-size:0.73rem;line-height:1.7;">43 DAX Companies<br>7,500+ Exec Observations<br>OLS R²=0.71 · 2006–2024</p>', unsafe_allow_html=True)

# ── UNIVERSAL SNAPSHOT ────────────────────────────────────────
def show_universal_snapshot(sel_co, sel_year):
    """Universal company snapshot shown at top of every stakeholder view."""
    in_model = sel_co in MODEL_COS
    co_df    = df[df["company_shortname"] == sel_co].sort_values("year")
    co_mu    = mu[mu["company_shortname"] == sel_co].sort_values("year")
    co_mu_yr = co_mu[co_mu["year"] == sel_year]
    dax_avg  = df.groupby("year")["total_comp_bt"].mean().reset_index()

    total_c  = float(co_mu_yr["total_comp_bt"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["total_comp_bt"].notna().any() else None
    aep      = float(co_mu_yr["actual_vs_expected_pct"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["actual_vs_expected_pct"].notna().any() else None
    peer_pct = float(co_mu_yr["peer_pct_total_comp"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["peer_pct_total_comp"].notna().any() else None
    avg_lti  = float(co_df["lti_pct"].mean()) if in_model and len(co_df) > 0 and "lti_pct" in co_df.columns else None
    avg_yoy  = float(co_df["comp_yoy_pct"].mean()) if in_model and len(co_df) > 0 else None
    _, t_col, t_lbl = traffic(aep)

    st.markdown(f"""<div style="background:linear-gradient(135deg,#f1f5f9,#e8edf4);
        border:1.5px solid #cbd5e1;border-radius:16px;padding:20px 24px 6px 24px;margin-bottom:4px;">
        <div style="font-size:.68rem;color:{GRAY};text-transform:uppercase;letter-spacing:.1em;
            font-weight:700;margin-bottom:14px;">
            📊 Company Snapshot — {sel_co} · {sel_year}
        </div>""", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_html(f"€{total_c/1000:.1f}M" if total_c else "—",
            "Total Compensation", f"Exec. Board {sel_year}", NAVY, GRAY, "#3b82f6",
            info="Sum of all executive board members' compensation (fixed salary + short-term + long-term incentives) before taxes. Provides a single headline figure for the company's total executive pay spend."), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_html(f"Top {100-peer_pct:.0f}%" if peer_pct else "—",
            "DAX Peer Rank", f"Percentile {peer_pct:.0f}" if peer_pct else "n/a", NAVY, GRAY, "#8b5cf6",
            info="Where this company sits in the DAX compensation ranking for the selected year. 'Top 10%' means only 10% of DAX companies pay their board more. Important for benchmarking against true peers."), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_html(f"{aep:+.0f}%" if aep is not None else "—",
            "vs. Model Expectation", t_lbl, t_col, t_col, t_col,
            info="Deviation of actual compensation from the OLS model's fair-value estimate, which controls for company size, sector, board size, and prior-year pay. Values above +15% suggest the board is paid above what fundamentals justify."), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_html(f"{avg_lti:.0f}%" if avg_lti else "—",
            "Avg. LTI Share", "Long-term incentive", NAVY, GRAY, "#06b6d4",
            info="Average share of total compensation paid as Long-Term Incentives (LTI) over the company's history. Higher LTI share aligns executives with long-term shareholder value creation rather than short-term results."), unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")
    with col_l:
        if in_model and len(co_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dax_avg["year"], y=dax_avg["total_comp_bt"],
                fill="tozeroy", fillcolor="rgba(148,163,184,.07)",
                line=dict(color="#94a3b8", dash="dot", width=1.5), name="DAX Average",
                hovertemplate="DAX avg: €%{y:,.0f}K<extra></extra>"))
            fig.add_trace(go.Scatter(x=co_df["year"], y=co_df["total_comp_bt"],
                line=dict(color="#3b82f6", width=2.5), mode="lines+markers",
                marker=dict(size=5), name=sel_co,
                hovertemplate=f"{sel_co}: €%{{y:,.0f}}K<extra></extra>"))
            for yr, lbl in [(2009, "GFC"), (2020, "COVID")]:
                if yr in co_df["year"].values:
                    fig.add_vline(x=yr, line_dash="dot", line_color="#e2e8f0", line_width=1,
                        annotation_text=lbl, annotation_font=dict(size=7, color=GRAY),
                        annotation_position="top")
            fig.update_layout(height=200, margin=dict(l=0, r=0, t=8, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Total Comp. (€K)", gridcolor=GRAYLT, tickfont=dict(size=9)),
                xaxis=dict(gridcolor=GRAYLT, tickfont=dict(size=9)),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=9)),
                hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
    with col_r:
        if in_model and len(co_df) > 0:
            comp_first = co_df["total_comp_bt"].dropna().iloc[0] if co_df["total_comp_bt"].notna().any() else None
            comp_last  = co_df["total_comp_bt"].dropna().iloc[-1] if co_df["total_comp_bt"].notna().any() else None
            total_growth = ((comp_last / comp_first) - 1) * 100 if comp_first and comp_last and comp_first > 0 else None
            sz_cnt = int(co_df["schlechte_zeiten"].sum()) if "schlechte_zeiten" in co_df.columns else 0
            avg_ebit = float(co_df["ebit_yoy_pct"].mean()) if "ebit_yoy_pct" in co_df.columns else None
            board_size = int(co_df["n_executives"].dropna().iloc[-1]) if "n_executives" in co_df.columns and co_df["n_executives"].notna().any() else None
            rows = [
                ("Avg. Pay Growth p.a.", f"{avg_yoy:+.1f}%" if avg_yoy and pd.notna(avg_yoy) else "—", ORANGE),
                ("Total Growth (period)", f"{total_growth:+.0f}%" if total_growth else "—", GRAY),
                ("Avg. EBIT Growth p.a.", f"{avg_ebit:+.1f}%" if avg_ebit and pd.notna(avg_ebit) else "—", GRAY),
                ("Board Size (latest)", str(board_size) if board_size else "—", GRAY),
                ("Bad Times Events", str(sz_cnt), RED if sz_cnt > 0 else GREEN),
            ]
            st.markdown('<div style="background:white;border-radius:10px;padding:12px 14px;border:1px solid #e2e8f0;margin-top:4px;">', unsafe_allow_html=True)
            for lbl, val, col_c in rows:
                st.markdown(f'<div class="metric-row"><span style="color:{GRAY};font-size:.8rem;">{lbl}</span>'
                            f'<span style="font-weight:700;color:{col_c};font-size:.8rem;">{val}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def stakeholder_divider(label):
    """Visual separator between universal snapshot and stakeholder-specific content."""
    st.markdown(f"""<div style="position:relative;margin:22px 0 20px 0;border-top:2px solid #e2e8f0;">
        <span style="position:absolute;top:-11px;left:50%;transform:translateX(-50%);
            background:{GRAYLT};padding:0 16px;color:{GRAY};font-size:.7rem;
            text-transform:uppercase;letter-spacing:.1em;font-weight:700;white-space:nowrap;">
            {label}
        </span>
    </div><div style="height:8px;"></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LAYER 1 — LANDING
# ══════════════════════════════════════════════════════════════
def show_landing():
    with st.sidebar:
        st.markdown(f"""<div style="padding:20px 0 12px 0;">
            <div style="font-size:1.7rem;font-weight:800;color:white;">ExComp</div>
            <div style="font-size:0.73rem;color:#64748b;margin-top:2px;">AI-Powered Pay Intelligence</div>
        </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown('<p style="color:#475569;font-size:0.75rem;line-height:1.7;">TUM Science Hackathon 2026<br>Chair of Financial Accounting<br>Prof. Dr. Jürgen Ernstberger<br><br>43 DAX Companies<br>7,500+ Exec Observations<br>2006–2024</p>', unsafe_allow_html=True)

    st.markdown(f"""<div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:20px;">
            <div style="flex:1;min-width:260px;">
                <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(249,115,22,.15);border:1px solid rgba(249,115,22,.3);border-radius:20px;padding:3px 12px;margin-bottom:14px;">
                    <span style="width:6px;height:6px;background:{ORANGE};border-radius:50%;display:inline-block;"></span>
                    <span style="font-size:.68rem;color:{ORANGE};text-transform:uppercase;letter-spacing:.12em;font-weight:700;">TUM Science Hackathon 2026</span>
                </div>
                <div style="font-size:2.8rem;font-weight:900;color:white;letter-spacing:-1.5px;line-height:1.05;">ExComp</div>
                <div style="font-size:.98rem;color:#94a3b8;margin-top:10px;font-weight:400;">AI-Powered Executive Compensation Benchmarking &amp; Red Flag Detection</div>
                <div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:6px;">
                    <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.73rem;color:#cbd5e1;">📈 15 Years of Data</span>
                    <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.73rem;color:#cbd5e1;">🤖 OLS Prediction Model</span>
                    <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.73rem;color:#cbd5e1;">🌱 CSRD Compliance</span>
                    <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.73rem;color:#cbd5e1;">🔍 Anomaly Detector</span>
                </div>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;">
                <div style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:16px 22px;text-align:center;min-width:90px;">
                    <div style="font-size:2.4rem;font-weight:900;color:{ORANGE};letter-spacing:-1px;line-height:1;">16<span style="font-size:1.2rem;color:#64748b;">/43</span></div>
                    <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:5px;font-weight:600;">DAX with ESG-Pay</div>
                </div>
                <div style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:16px 22px;text-align:center;min-width:90px;">
                    <div style="font-size:2.4rem;font-weight:900;color:white;letter-spacing:-1px;line-height:1;">0.71</div>
                    <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:5px;font-weight:600;">Model R²</div>
                </div>
                <div style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:16px 22px;text-align:center;min-width:90px;">
                    <div style="font-size:2.4rem;font-weight:900;color:white;letter-spacing:-1px;line-height:1;">74x</div>
                    <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:5px;font-weight:600;">Max Pay Ratio</div>
                </div>
                <div style="background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.25);border-radius:16px;padding:16px 22px;text-align:center;min-width:90px;">
                    <div style="font-size:2.4rem;font-weight:900;color:{ORANGE};letter-spacing:-1px;line-height:1;">63%</div>
                    <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:5px;font-weight:600;">Pay 0% on ESG</div>
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;margin:6px 0 22px 0;">
        <div style="font-size:1.05rem;font-weight:700;color:#0a1628;">Choose Your Stakeholder Perspective</div>
        <div style="font-size:.82rem;color:#64748b;margin-top:4px;">Six perspectives on executive compensation — all backed by real data</div>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""<div class="lcard">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">🏦</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">Capital Allocators</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">Follow the money — want risk signals</div></div>
                <span class="badge-live">✦ Live</span>
            </div>
            <div><span class="tag tag-orange">📈 Pay/EBIT Gap</span><span class="tag tag-orange">📉 LTI Share</span><span class="tag tag-orange">⚠️ CEO Premium</span></div>
            <div style="margin-top:10px;font-size:.78rem;color:#94a3b8;font-style:italic;">"Is this company governed well enough to trust with my capital?"</div>
        </div>""", unsafe_allow_html=True)
        if st.button("→ Capital Allocators", key="nav_capital", use_container_width=True):
            nav("capital")
        st.markdown("""<div class="lcard" style="margin-top:8px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">🏛</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">Board &amp; HR</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">Set and validate compensation</div></div>
                <span class="badge-live">✦ Live</span>
            </div>
            <div><span class="tag tag-orange">👥 Peer Rank</span><span class="tag tag-orange">📊 STI/LTI Mix</span><span class="tag tag-orange">🏛 Say-on-Pay</span></div>
            <div style="margin-top:10px;font-size:.78rem;color:#94a3b8;font-style:italic;">"Where do we stand vs. the market — and is our structure defensible?"</div>
        </div>""", unsafe_allow_html=True)
        if st.button("→ Board & HR", key="nav_board", use_container_width=True):
            nav("board")

    with col2:
        st.markdown(f"""<div class="lcard lcard-active">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">🌱</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">ESG &amp; CSRD Governance</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">ESG pay integration &amp; governance risks</div></div>
                <span class="badge-live">✦ Full Analysis</span>
            </div>
            <div>
                <span class="tag tag-orange">🌱 ESG Rating</span>
                <span class="tag tag-orange">🔬 Pay-Washing</span>
                <span class="tag tag-orange">♀ Gender Equity</span>
                <span class="tag tag-orange">📋 CSRD Score</span>
            </div>
            <div style="margin-top:10px;font-size:.78rem;color:#9a3412;font-style:italic;font-weight:500;">"Is this board compensated fairly — and sustainably?"</div>
        </div>""", unsafe_allow_html=True)
        if st.button("→ ESG & CSRD Governance", key="nav_esg", type="primary", use_container_width=True):
            nav("esg")
        st.markdown("""<div class="lcard" style="margin-top:8px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">👷</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">Employees &amp; Labor</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">Question fairness and distribution</div></div>
                <span class="badge-live">✦ Live</span>
            </div>
            <div><span class="tag tag-orange">📉 Bad Times</span><span class="tag tag-orange">👥 CEO/Worker Ratio</span><span class="tag tag-orange">⚡ Headcount</span></div>
            <div style="margin-top:10px;font-size:.78rem;color:#94a3b8;font-style:italic;">"Did executive bonuses rise while jobs were being cut?"</div>
        </div>""", unsafe_allow_html=True)
        if st.button("→ Employees & Labor", key="nav_employees", use_container_width=True):
            nav("employees")

    with col3:
        st.markdown("""<div class="lcard">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">⚖️</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">Accountability Actors</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">Hold companies accountable</div></div>
                <span class="badge-live">✦ Live</span>
            </div>
            <div><span class="tag tag-orange">🚨 Outlier Rank</span><span class="tag tag-orange">⚠️ Violations</span><span class="tag tag-orange">🏛 Proxy View</span></div>
            <div style="margin-top:10px;font-size:.78rem;color:#94a3b8;font-style:italic;">"Who earns too much — and can we prove it?"</div>
        </div>""", unsafe_allow_html=True)
        if st.button("→ Accountability Actors", key="nav_accountability", use_container_width=True):
            nav("accountability")
        st.markdown("""<div class="lcard" style="margin-top:8px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">🧮</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">Compensation Consultants</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">Design and validate compensation systems</div></div>
                <span class="badge-live">✦ Live</span>
            </div>
            <div><span class="tag tag-orange">🤖 Model Estimate</span><span class="tag tag-orange">🔬 Sector Fit</span><span class="tag tag-orange">📐 Structure</span></div>
            <div style="margin-top:10px;font-size:.78rem;color:#94a3b8;font-style:italic;">"Does the recommendation I gave hold up against the model?"</div>
        </div>""", unsafe_allow_html=True)
        if st.button("→ Compensation Consultants", key="nav_consultants", use_container_width=True):
            nav("consultants")

    st.markdown('<div style="text-align:center;color:#94a3b8;font-size:.74rem;margin-top:8px;">ORBIS/Bureau van Dijk · DGAP Compensation Reports · 43 DAX Companies · 2006–2024</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)
    bm1, bm2, bm3, bm4 = st.columns(4)
    with bm1:
        if st.button("🧮 Compensation Predictor", key="nav_predictor", use_container_width=True, type="primary"):
            nav("predictor")
    with bm2:
        if st.button("👥 Peer Grouping", key="nav_peer_grouping", use_container_width=True, type="primary"):
            nav("peer_grouping")
    with bm3:
        if st.button("⚠️ Outlier Detection", key="nav_outlier_detect", use_container_width=True, type="primary"):
            nav("outlier_detect")
    with bm4:
        if st.button("📖 Methodology", key="nav_methodology", use_container_width=True):
            nav("methodology")


# ══════════════════════════════════════════════════════════════
# LAYER 2 — OVERVIEW (Company Snapshot + 6 Module Cards)
# ══════════════════════════════════════════════════════════════
def show_overview():
    sidebar_nav("landing", "← Back to Home")
    sel_co   = st.session_state.company
    sel_year = st.session_state.year
    in_model   = sel_co in MODEL_COS     # has rich feature data (54 companies, 2006-2024)
    in_mu      = sel_co in MU_COS        # has model predictions (54 companies, 2007-2024)

    co_esg    = esg[esg["company_shortname"] == sel_co].sort_values("year") if len(esg) > 0 else pd.DataFrame()
    co_df     = df[df["company_shortname"] == sel_co].sort_values("year")   # rich features 2006-2024
    co_mu     = mu[mu["company_shortname"] == sel_co].sort_values("year")   # model universe 2007-2024

    # Universe row for selected year (predictions available for all MU_COS 2007-2024)
    co_mu_yr  = co_mu[co_mu["year"] == sel_year]
    # Rich features row (all 54 companies, full year range)
    has_rich  = sel_co in MODEL_COS
    co_yr     = df[(df["company_shortname"] == sel_co) & (df["year"] == sel_year)] if has_rich else pd.DataFrame()
    # ESG: prefer selected year, fall back to nearest available
    co_esg_yr = co_esg[co_esg["year"] == sel_year] if len(co_esg) > 0 else pd.DataFrame()
    if len(co_esg_yr) == 0 and len(co_esg) > 0:
        co_esg_yr = co_esg.iloc[[-1]]

    # ── Hero KPIs from model_universe (works for all 54 companies 2007-2024) ──
    aep      = float(co_mu_yr["actual_vs_expected_pct"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["actual_vs_expected_pct"].notna().any() else None
    peer_pct = float(co_mu_yr["peer_pct_total_comp"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["peer_pct_total_comp"].notna().any() else None
    total_c  = float(co_mu_yr["total_comp_bt"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["total_comp_bt"].notna().any() else None
    # Governance score + anomaly from rich features (54 companies, 2006-2024)
    gov_sc   = float(co_df["anomaly_score_pct"].mean()) if in_model and len(co_df) > 0 else None
    is_anom  = bool(co_yr["is_anomaly"].iloc[0]) if has_rich and len(co_yr) > 0 and "is_anomaly" in co_yr.columns and co_yr["is_anomaly"].notna().any() else None
    # ESG metrics: from the selected year's ESG row
    sti_esg  = float(co_esg_yr["sti_esg_share"].iloc[0]) if len(co_esg_yr) > 0 and co_esg_yr["sti_esg_share"].notna().any() else None
    lti_esg  = float(co_esg_yr["lti_esg_share"].iloc[0]) if len(co_esg_yr) > 0 and co_esg_yr["lti_esg_share"].notna().any() else None
    esg_tot  = (sti_esg or 0) + (lti_esg or 0)

    t_icon, t_col, t_label = traffic(aep)

    year_note = f"Model + ESG Available · {sel_year}" if sel_year > FEATURES_YEAR_MAX else f"Full Data · {sel_year}"
    st.markdown(f"""<div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">Company Overview</div>
                <div style="font-size:2rem;font-weight:900;color:white;letter-spacing:-.5px;margin:4px 0;">{sel_co}</div>
                <div style="font-size:.84rem;color:#94a3b8;">DAX · {year_note} · Select a Module</div>
            </div>
            <div style="background:rgba(255,255,255,.06);border-radius:12px;padding:12px 20px;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:{t_col};">{t_icon} {f'{aep:+.0f}%' if aep is not None else '—'}</div>
                <div style="font-size:.72rem;color:#94a3b8;margin-top:3px;">vs. Model Expectation {sel_year}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    # Banner when actual compensation amounts are not yet available (2022+)
    if sel_year > COMP_DATA_MAX:
        st.markdown(f"""<div style="background:{ORANGEBG};border:1px solid #fed7aa;border-left:4px solid {ORANGE};border-radius:0 10px 10px 0;padding:10px 16px;margin-bottom:16px;font-size:.84rem;color:#431407;">
            📅 <strong>Year {sel_year}:</strong> Prediction model, peer ranking and ESG KPI data fully available.
            Compensation structure (STI/LTI mix, Bad Times) from this reporting year not yet in the dataset — AI forecast active.
        </div>""", unsafe_allow_html=True)

    # ── Snapshot KPIs ──
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.markdown(kpi_html(f"€{total_c/1000:.1f}M" if total_c else "—", "Total Compensation", f"Executive Board {sel_year}", NAVY,
        info="Sum of all executive board members' total compensation (fixed + STI + LTI) before taxes for the selected year."), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html(f"{gov_sc:.0f}/100" if gov_sc else "—", "Governance Risk", "⚠ High" if gov_sc and gov_sc>60 else "✅ Normal", risk_color(gov_sc or 0), risk_color(gov_sc or 0),
        info="Composite governance risk score (0–100) combining pay-performance misalignment, structural anomalies, and Bad Times events. Scores above 60 signal material governance concerns."), unsafe_allow_html=True)
    with k3: st.markdown(kpi_html(f"Top {100-peer_pct:.0f}%" if peer_pct else "—", "DAX Peer Rank", f"Percentile {peer_pct:.0f}" if peer_pct else "n/a", NAVY,
        info="Where this company ranks among all DAX companies in absolute compensation for the selected year. Benchmarks pay level against the full index."), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html("🚨 Yes" if is_anom else ("✅ No" if is_anom is not None else "—"), "Anomaly", "Unusual Structure" if is_anom else "Within Normal Range", RED if is_anom else GREEN, RED if is_anom else GREEN,
        info="Statistical anomaly flag based on the compensation structure and level. Flagged when the company's pay pattern deviates significantly from what the model expects given its characteristics."), unsafe_allow_html=True)
    with k5: st.markdown(kpi_html(f"{esg_tot:.0f}%" if esg_tot else "0%", "ESG in Compensation", "CSRD-relevant" if esg_tot > 0 else "No ESG Link", ORANGE if esg_tot > 0 else "#94a3b8", GREEN if esg_tot > 0 else RED,
        info="Share of variable compensation (STI + LTI) linked to ESG/sustainability targets. Increasingly mandated by CSRD and expected by institutional investors. Zero ESG link is a pay-washing red flag."), unsafe_allow_html=True)
    with k6:
        sz_cnt = int(co_df["schlechte_zeiten"].sum()) if in_model and "schlechte_zeiten" in co_df.columns else 0
        st.markdown(kpi_html(str(sz_cnt), "Bad Times Events", "Pay ↑ when EBIT ↓", RED if sz_cnt > 0 else GREEN, RED if sz_cnt > 0 else GREEN,
            info="Count of years where executive pay increased while both EBIT AND headcount fell simultaneously. These 'Bad Times' events are the clearest pay-for-performance failures in the dataset."), unsafe_allow_html=True)

    st.markdown('<div class="sec"><div class="sec-title">Modules — Select an Analysis Level</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Each module covers one of the five hackathon requirement areas</div></div>', unsafe_allow_html=True)

    # ── Module Cards 3×2 ──
    MODULES = [
        ("historical", "📈", "Historical Trends",
         "How has compensation evolved over 15 years? Crisis resilience, STI/LTI mix, structural change.",
         f"+{co_df['comp_yoy_pct'].mean():.1f}% avg/yr" if in_model and len(co_df)>0 else "2006–2024",
         "Avg. Growth p.a.", "#dbeafe", "#1d4ed8"),
        ("prediction", "🤖", "Prediction Model",
         "OLS model estimates expected compensation based on company size, sector, performance & peer group.",
         f"{aep:+.0f}% vs. Expected" if aep is not None else "R²=0.71",
         f"Model Deviation {sel_year}", "#ede9fe", "#6d28d9"),
        ("governance", "⚠️", "Governance Risk Score",
         "Composite score from compensation volatility, CEO premium & 'Bad Times' — warning signal for investors.",
         f"{gov_sc:.0f}/100" if gov_sc else "n/a",
         "Avg. Governance Risk", "#fef2f2", "#dc2626"),
        ("peer", "👥", "Peer Benchmarking",
         "Where does this company stand vs. DAX peers? Sector & size comparison with model expectation.",
         f"Top {100-peer_pct:.0f}%" if peer_pct else "DAX",
         "Peer Rank in DAX", "#dcfce7", "#15803d"),
        ("anomaly", "🔍", "Anomaly Detector",
         "Detects unusual compensation structures: atypical STI/LTI mix, extreme bonus-to-salary ratios.",
         "🚨 Anomaly" if is_anom else ("✅ Normal" if is_anom is not None else "—"),
         f"Status {sel_year}", "#fffbeb", "#d97706"),
        ("esg", "🌱", "ESG Rating",
         "CSRD compliance, sustainability pay integration & external ESG validation check.",
         f"{esg_tot:.0f}% ESG" if esg_tot else "No ESG Link",
         "ESG Share STI+LTI", "#f0fdf4", "#16a34a"),
    ]

    cols_r1 = st.columns(3, gap="medium")
    cols_r2 = st.columns(3, gap="medium")

    for i, (key, icon, title, desc, stat, statlbl, icon_bg, icon_fg) in enumerate(MODULES):
        col = cols_r1[i] if i < 3 else cols_r2[i-3]
        with col:
            stat_color = RED if ("Anomaly" in stat or "n/a" in str(stat)) else ORANGE if "%" in str(stat) else NAVY
            st.markdown(f"""<div class="mod-card">
                <div class="mod-icon-wrap" style="background:{icon_bg};">{icon}</div>
                <div class="mod-title">{title}</div>
                <div class="mod-desc">{desc}</div>
                <div style="border-top:1px solid #f1f5f9;padding-top:10px;margin-top:auto;">
                    <div class="mod-stat" style="color:{stat_color};">{stat}</div>
                    <div class="mod-statlbl">{statlbl}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            if st.button(f"→ {title}", key=f"mod_{key}", use_container_width=True):
                nav("module", key)


# ══════════════════════════════════════════════════════════════
# LAYER 3 — MODULES
# ══════════════════════════════════════════════════════════════

def module_header(icon, title, subtitle, sel_co, sel_year):
    st.markdown(f"""<div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">{icon} {title}</div>
                <div style="font-size:1.8rem;font-weight:900;color:white;letter-spacing:-.5px;margin:4px 0;">{sel_co}</div>
                <div style="font-size:.82rem;color:#94a3b8;">{subtitle}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODULE 1 — HISTORISCHE TRENDS
# ─────────────────────────────────────────────
def show_historical():
    sidebar_nav("overview", "← Back to Overview")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    co_df  = df[df["company_shortname"] == sel_co].sort_values("year")
    co_esg = esg[esg["company_shortname"] == sel_co].sort_values("year") if len(esg) > 0 else pd.DataFrame()
    dax_avg = df.groupby("year")["total_comp_bt"].mean().reset_index()

    module_header("📈", "Historical Trends", f"Compensation development 2006–2024 · DAX comparison · Structural change", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>What does this module show?</strong> 15 years of compensation data reveal: How crisis-resistant is pay?
        Does it grow faster than operating performance? Has the STI/LTI mix shifted over time?
        These questions are central to CSRD reporting and governance assessment.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="sec"><div class="sec-title">Compensation Trend vs. DAX Average</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Total board compensation (€K) — orange line = selected company</div></div>', unsafe_allow_html=True)
        if in_model and len(co_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dax_avg["year"], y=dax_avg["total_comp_bt"],
                fill="tozeroy", fillcolor="rgba(148,163,184,.08)",
                line=dict(color="#94a3b8", dash="dot", width=1.5),
                name="DAX Average",
                hovertemplate="DAX avg: €%{y:,.0f}K<extra></extra>"))
            fig.add_trace(go.Scatter(x=co_df["year"], y=co_df["total_comp_bt"],
                line=dict(color=ORANGE, width=3),
                mode="lines+markers", marker=dict(size=5),
                name=sel_co,
                hovertemplate=f"{sel_co}: €%{{y:,.0f}}K<extra></extra>"))
            # Crisis annotations
            crises = [(2009,"GFC"),(2012,"Euro Crisis"),(2020,"COVID")]
            for yr, lbl in crises:
                if yr in co_df["year"].values:
                    fig.add_vline(x=yr, line_dash="dot", line_color="#e2e8f0", line_width=1,
                                  annotation_text=lbl, annotation_font=dict(size=8, color=GRAY),
                                  annotation_position="top")
            fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Total Comp. (€K)", gridcolor=GRAYLT),
                xaxis=dict(gridcolor=GRAYLT),
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
                hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Use esg data for non-model companies
            if len(co_esg) > 0:
                esg_avg = esg.groupby("year")["avg_comp"].mean().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=esg_avg["year"], y=esg_avg["avg_comp"],
                    line=dict(color="#94a3b8", dash="dot", width=1.5),
                    name="DAX Avg. Compensation"))
                fig.add_trace(go.Scatter(x=co_esg["year"], y=co_esg["avg_comp"],
                    line=dict(color=ORANGE, width=3), mode="lines+markers",
                    name=sel_co))
                fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Avg. Compensation (€K)", gridcolor=GRAYLT),
                    xaxis=dict(gridcolor=GRAYLT),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01))
                st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec"><div class="sec-title">Key Metrics</div></div>', unsafe_allow_html=True)
        if in_model and len(co_df) > 0:
            comp_06 = co_df[co_df["year"]==2006]["total_comp_bt"].values
            comp_last = co_df["total_comp_bt"].dropna().iloc[-1] if co_df["total_comp_bt"].notna().any() else None
            total_growth = ((comp_last / comp_06[0]) - 1)*100 if len(comp_06)>0 and comp_last else None

            avg_yoy  = co_df["comp_yoy_pct"].mean()
            avg_ebit = co_df["ebit_yoy_pct"].mean() if "ebit_yoy_pct" in co_df.columns else None
            vol_3yr  = co_df["comp_volatility_3yr"].mean() if "comp_volatility_3yr" in co_df.columns else None
            crisis_covid = co_df[co_df["year"]==2020]["comp_yoy_pct"].values
            covid_chg = float(crisis_covid[0]) if len(crisis_covid)>0 else None

            rows = [
                ("Total Growth", f"+{total_growth:.0f}%" if total_growth else "—",
                 RED if total_growth and total_growth>150 else GREEN),
                ("Avg. Growth p.a.", f"{avg_yoy:+.1f}%" if pd.notna(avg_yoy) else "—", ORANGE),
                ("Avg. EBIT Growth p.a.", f"{avg_ebit:+.1f}%" if avg_ebit and pd.notna(avg_ebit) else "—", GRAY),
                ("Pay > EBIT Growth", "⚠ Yes" if avg_ebit and pd.notna(avg_ebit) and avg_yoy > avg_ebit else "✅ No",
                 RED if avg_ebit and pd.notna(avg_ebit) and avg_yoy > avg_ebit else GREEN),
                ("Compensation in COVID Year", f"{covid_chg:+.1f}%" if covid_chg and pd.notna(covid_chg) else "—",
                 RED if covid_chg and covid_chg > 0 else GREEN),
                ("3Y Volatility", f"{vol_3yr:.1f}%" if vol_3yr and pd.notna(vol_3yr) else "—",
                 RED if vol_3yr and vol_3yr > 25 else GRAY),
            ]
            st.markdown('<div style="background:white;border-radius:12px;padding:14px 16px;border:1px solid #e2e8f0;">', unsafe_allow_html=True)
            for lbl, val, col in rows:
                st.markdown(f'<div class="metric-row"><span style="color:{GRAY};">{lbl}</span><span style="font-weight:700;color:{col};">{val}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if avg_ebit and pd.notna(avg_ebit) and avg_yoy > avg_ebit + 5:
                st.markdown(f"""<div class="flag-red" style="margin-top:12px;">
                    <strong>⚠ Pay-Performance Gap:</strong> Compensation growth ({avg_yoy:.1f}% p.a.)
                    significantly exceeds EBIT growth ({avg_ebit:.1f}% p.a.) — a classic
                    governance warning signal.
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Detailed feature data only available for the core group.")

    # Comp Structure Evolution
    if in_model and len(co_df) > 0 and all(c in co_df.columns for c in ["fixed_pct","sti_pct","lti_pct"]):
        st.markdown('<div class="sec"><div class="sec-title">Compensation Structure Evolution (Fixed / STI / LTI)</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">How has the mix of fixed and variable compensation changed over time?</div></div>', unsafe_allow_html=True)
        st.markdown("""<div class="insight">
            <strong>Why relevant?</strong> A rising LTI share strengthens long-term pay-for-performance incentives.
            Strong STI dominance can encourage short-term thinking. CSRD requires disclosure of the incentive structure.
        </div>""", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Fixed", x=co_df["year"], y=co_df["fixed_pct"], marker_color="#3b82f6"))
        fig2.add_trace(go.Bar(name="STI (Short-term)", x=co_df["year"], y=co_df["sti_pct"], marker_color=ORANGE))
        fig2.add_trace(go.Bar(name="LTI (Long-term)", x=co_df["year"], y=co_df["lti_pct"], marker_color=NAVY))
        fig2.update_layout(barmode="stack", height=260, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Share (%)", range=[0,100], gridcolor=GRAYLT),
            xaxis=dict(gridcolor=GRAYLT),
            legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 2 — PREDICTION MODEL
# ─────────────────────────────────────────────
def show_prediction():
    sidebar_nav("overview", "← Back to Overview")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    in_mu    = sel_co in MU_COS
    # Use model_universe for predictions (2007-2024, 42 companies)
    co_mu_all = mu[mu["company_shortname"] == sel_co].sort_values("year")
    co_df     = df[df["company_shortname"] == sel_co].sort_values("year")  # rich features 2006-2024

    module_header("🤖", "Prediction Model", f"OLS Regression Model · R²=0.71 · Expected vs. Actual Pay · 2007–{MODEL_YEAR_MAX}", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>How does the model work?</strong> Our OLS model estimates expected executive compensation
        based on: prior-year compensation (pay stickiness), board size, sector and year trend.
        If a company is >15% above expectation, this indicates a potential governance issue.
        R²=0.71 means: 71% of compensation variation is explained by the model.
        The model covers all 42 DAX companies from 2007 to 2024.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="sec"><div class="sec-title">Actual vs. Model Expectation</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Orange band = 80% expectation interval · Navy line = Actual compensation</div></div>', unsafe_allow_html=True)
        if in_mu and len(co_mu_all) > 0 and co_mu_all["pred_comp"].notna().any():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(co_mu_all["year"])+list(co_mu_all["year"])[::-1],
                y=list(co_mu_all["pred_comp_high"])+list(co_mu_all["pred_comp_low"])[::-1],
                fill="toself", fillcolor="rgba(249,115,22,.10)",
                line=dict(color="rgba(0,0,0,0)"), name="80% Expectation Band"))
            fig.add_trace(go.Scatter(x=co_mu_all["year"], y=co_mu_all["pred_comp"],
                line=dict(color=ORANGE, dash="dash", width=2),
                name="Model Expectation",
                hovertemplate="Expected: €%{y:,.0f}K<extra></extra>"))
            fig.add_trace(go.Scatter(x=co_mu_all["year"], y=co_mu_all["total_comp_bt"],
                line=dict(color=NAVY, width=3), mode="lines+markers",
                name="Actual",
                hovertemplate="Actual: €%{y:,.0f}K<extra></extra>"))
            over = co_mu_all[co_mu_all["actual_vs_expected_pct"] > 40]
            if len(over) > 0:
                fig.add_trace(go.Scatter(x=over["year"], y=over["total_comp_bt"],
                    mode="markers", marker=dict(symbol="x", size=14, color=RED, line=dict(width=2.5)),
                    name="⚠ Significantly Overpaid",
                    hovertemplate="⚠ Year %{x}: significantly above expectation<extra></extra>"))
            fig.update_layout(height=330, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Total Compensation (€K)", gridcolor=GRAYLT),
                xaxis=dict(gridcolor=GRAYLT),
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
                hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"{sel_co} not in the model universe — no prediction available.")

    with col_r:
        st.markdown('<div class="sec"><div class="sec-title">Deviation from Model</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Positive = overpaid, Negative = underpaid</div></div>', unsafe_allow_html=True)
        if in_mu and co_mu_all["actual_vs_expected_pct"].notna().any():
            aep_series = co_mu_all.dropna(subset=["actual_vs_expected_pct"])
            colors_aep = [RED if v>15 else GREEN if v<-15 else AMBER for v in aep_series["actual_vs_expected_pct"]]
            fig2 = go.Figure(go.Bar(
                x=aep_series["year"], y=aep_series["actual_vs_expected_pct"],
                marker_color=colors_aep,
                hovertemplate="Jahr %{x}: %{y:+.1f}%<extra></extra>"))
            fig2.add_hline(y=0, line_color=GRAY, line_width=1)
            fig2.add_hline(y=15, line_dash="dot", line_color=AMBER, line_width=1,
                           annotation_text="+15% Threshold", annotation_font=dict(size=8))
            fig2.add_hline(y=-15, line_dash="dot", line_color="#2563eb", line_width=1)
            fig2.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="vs. Expectation (%)", gridcolor=GRAYLT),
                xaxis=dict(gridcolor=GRAYLT))
            st.plotly_chart(fig2, use_container_width=True)

            co_mu_yr = co_mu_all[co_mu_all["year"] == sel_year]
            aep_now = float(co_mu_yr["actual_vs_expected_pct"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["actual_vs_expected_pct"].notna().any() else None
            if aep_now is not None:
                icon, col, lbl = traffic(aep_now)
                st.markdown(f"""<div style="background:{NAVY};border-radius:10px;padding:14px;text-align:center;">
                    <div style="font-size:1.8rem;font-weight:800;color:{col};">{aep_now:+.1f}%</div>
                    <div style="font-size:.72rem;color:#94a3b8;text-transform:uppercase;margin-top:4px;">vs. Model {sel_year}</div>
                    <div style="font-size:.82rem;color:{col};margin-top:6px;font-weight:600;">{icon} {lbl}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Not in the model universe.")

    # Feature Importance
    st.markdown('<div class="sec"><div class="sec-title">What drives compensation? — Model Coefficients</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">How strongly does each factor influence the compensation level? (effect in %)</div></div>', unsafe_allow_html=True)
    st.markdown("""<div class="insight">
        <strong>Interpretation:</strong> "log_comp_lag1 +131%" means: prior-year compensation is the
        strongest single driver — pay is sticky. Sector dummies show how much industry membership alone
        moves compensation up or down, independent of performance.
    </div>""", unsafe_allow_html=True)
    top_c = coefs[coefs["feature"] != "intercept"].sort_values("exp_effect_pct", ascending=True)
    lbl_map = {"log_comp_lag1":"Prior-Year Compensation (Stickiness)","year_trend":"Year Trend",
               "log_board_size":"Board Size (log)"}
    top_c["label"] = top_c["feature"].apply(lambda x: lbl_map.get(x, x.replace("sector_","Sector: ").replace("_"," ")))
    fig3 = go.Figure(go.Bar(
        x=top_c["exp_effect_pct"], y=top_c["label"], orientation="h",
        marker_color=[GREEN if v>0 else RED for v in top_c["exp_effect_pct"]],
        text=[f"{v:+.0f}%" for v in top_c["exp_effect_pct"]], textposition="outside",
        hovertemplate="%{y}: %{x:+.0f}%<extra></extra>"))
    fig3.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Effect on Compensation Level (%)", gridcolor=GRAYLT),
        yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 3 — GOVERNANCE RISK SCORE
# ─────────────────────────────────────────────
def show_governance():
    sidebar_nav("overview", "← Back to Overview")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    co_df = df[df["company_shortname"] == sel_co].sort_values("year")

    module_header("⚠️", "Governance Risk Score", "Composite Score · Bad Times Analysis · CEO Premiums · Red Flags", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>What is the Governance Risk Score?</strong> A composite of: compensation volatility,
        CEO premium over the rest of the board, peer deviation and "Bad Times" events
        (compensation rose despite EBIT <em>and</em> headcount falling). High scores are a warning signal
        for institutional investors and proxy advisors at Say-on-Pay votes.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="sec"><div class="sec-title">Governance Risk Heatmap — DAX Core Group</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Red = high risk · Green = low risk · Orange outline = selected company</div></div>', unsafe_allow_html=True)
        if "anomaly_score_pct" in df.columns:
            pivot = df.pivot_table(index="company_shortname", columns="year", values="anomaly_score_pct")
            last_yr = pivot.columns.max()
            pivot = pivot.sort_values(by=last_yr, ascending=False)
            fig = go.Figure(go.Heatmap(
                z=pivot.values, x=[str(int(y)) for y in pivot.columns], y=pivot.index.tolist(),
                colorscale=[[0,"#f0fdf4"],[0.35,"#fef9c3"],[0.65,"#fed7aa"],[1,"#fca5a5"]],
                colorbar=dict(title="Risk", tickfont=dict(size=9), thickness=12),
                hovertemplate="<b>%{y}</b> · %{x}: <b>%{z:.1f}</b><extra></extra>",
                zmin=0, zmax=100))
            if sel_co in pivot.index and in_model:
                idx = list(pivot.index).index(sel_co)
                fig.add_shape(type="rect", x0=-0.5, x1=len(pivot.columns)-.5,
                    y0=idx-.5, y1=idx+.5, line=dict(color=ORANGE, width=2.5), fillcolor="rgba(0,0,0,0)")
            fig.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(f'<div class="sec"><div class="sec-title">Red Flags — {sel_co}</div></div>', unsafe_allow_html=True)
        if in_model and len(co_df) > 0:
            avg_sc   = co_df["anomaly_score_pct"].mean()
            sz_cnt   = int(co_df["schlechte_zeiten"].sum()) if "schlechte_zeiten" in co_df.columns else 0
            sz_rows  = co_df[co_df.get("schlechte_zeiten", pd.Series(0, index=co_df.index)) == 1]
            ceo_prem = co_df["ceo_board_premium_ratio"].mean() if "ceo_board_premium_ratio" in co_df.columns else None
            aep_over = (co_df["actual_vs_expected_pct"] > 40).sum() if co_df["actual_vs_expected_pct"].notna().any() else 0
            vol      = co_df["comp_volatility_3yr"].mean() if "comp_volatility_3yr" in co_df.columns else None

            # Score badge
            sc_fg = "#4ade80" if avg_sc < 40 else "#fbbf24" if avg_sc < 60 else "#f87171"
            sc_bg = "#14532d" if avg_sc < 40 else "#713f12" if avg_sc < 60 else "#7f1d1d"
            sc_lbl = "Low Risk" if avg_sc < 40 else "Moderate Risk" if avg_sc < 60 else "High Risk"
            st.markdown(f"""<div style="background:{NAVY};border-radius:12px;padding:16px;text-align:center;margin-bottom:14px;">
                <div style="font-size:2.6rem;font-weight:800;color:{sc_fg};line-height:1;">{avg_sc:.0f}<span style="font-size:1.2rem;color:#475569;">/100</span></div>
                <div style="font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:4px;">Ø Governance Risk Score</div>
                <div style="background:{sc_bg};border-radius:6px;padding:3px 12px;display:inline-block;margin-top:8px;font-size:.78rem;font-weight:700;color:{sc_fg};">{sc_lbl}</div>
            </div>""", unsafe_allow_html=True)

            # Red flags
            flags = [
                (sz_cnt > 0,
                 f"⚠ {sz_cnt}× Bad Times Event(s): Pay ↑ when EBIT ↓ ({list(sz_rows['year'].astype(int)) if len(sz_rows)>0 else ''})",
                 "red"),
                (ceo_prem and ceo_prem > 2.5,
                 f"⚠ High CEO Premium: {ceo_prem:.1f}x avg. board (DAX median ~2x)",
                 "amber"),
                (aep_over > 0,
                 f"⚠ {aep_over} year(s) with >+40% above model expectation",
                 "red"),
                (vol and vol > 25,
                 f"⚠ High compensation volatility: avg. {vol:.0f}% std. over 3 years",
                 "amber"),
            ]
            has_flag = False
            for condition, text, severity in flags:
                if condition:
                    cls = f"flag-{severity}"
                    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)
                    has_flag = True
            if not has_flag:
                st.markdown(f'<div class="flag-green">✅ No significant governance red flags detected.</div>', unsafe_allow_html=True)
        else:
            st.info(f"Governance Risk Score only available for the core group. {sel_co} is not included.")

    # Bad Times Table
    st.markdown('<div class="sec"><div class="sec-title">Bad Times Events — DAX Core Group</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Compensation rose despite falling EBIT and headcount — a clear pay-for-performance violation</div></div>', unsafe_allow_html=True)
    if "schlechte_zeiten" in df.columns:
        sz_all = df[df["schlechte_zeiten"]==1][["company_shortname","year","comp_yoy_pct","ebit_yoy_pct","empl_yoy_pct","schlechte_zeiten_score"]].sort_values("schlechte_zeiten_score", ascending=False)
        if len(sz_all) > 0:
            st.dataframe(sz_all.rename(columns={
                "company_shortname":"Company","year":"Year",
                "comp_yoy_pct":"Comp %","ebit_yoy_pct":"EBIT %",
                "empl_yoy_pct":"Headcount %","schlechte_zeiten_score":"Severity"
            }).round(1), use_container_width=True, hide_index=True)
        else:
            st.success("No Bad Times events found.")

# ─────────────────────────────────────────────
# MODULE 4 — PEER BENCHMARKING
# ─────────────────────────────────────────────
def show_peer():
    sidebar_nav("overview", "← Back to Overview")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    # Use model_universe for full 42-company peer comparison (2007-2024)
    year_data = mu[mu["year"] == sel_year].sort_values("total_comp_bt", ascending=False)
    co_df = df[df["company_shortname"] == sel_co]  # rich features for sector comparison

    module_header("👥", "Peer Benchmarking", f"DAX Universe Comparison · {sel_year} · 42 Companies · Sector Positioning", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>Peer Benchmarking:</strong> Compares actual compensation against the model expectation band
        for all DAX companies in the same year. Color = benchmark signal:
        🔴 significantly overpaid · 🟡 above/below market · 🟢 market-aligned.
        Additionally: how does the company rank in its sector peer group?
    </div>""", unsafe_allow_html=True)

    if len(year_data) > 0:
        k1,k2,k3,k4 = st.columns(4)
        co_mu_yr = year_data[year_data["company_shortname"]==sel_co]
        rank   = int(co_mu_yr["peer_rank_total_comp"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["peer_rank_total_comp"].notna().any() else None
        pctile = float(co_mu_yr["peer_pct_total_comp"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["peer_pct_total_comp"].notna().any() else None
        aep    = float(co_mu_yr["actual_vs_expected_pct"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["actual_vs_expected_pct"].notna().any() else None
        total  = float(co_mu_yr["total_comp_bt"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["total_comp_bt"].notna().any() else None
        _, t_col, t_lbl = traffic(aep)
        with k1: st.markdown(kpi_html(f"#{rank}" if rank else "—", "DAX Peer Rank", f"of {len(year_data)} companies",
            info="Absolute rank by total board compensation among all DAX companies in the selected year. Lower number = higher pay."), unsafe_allow_html=True)
        with k2: st.markdown(kpi_html(f"Top {100-pctile:.0f}%" if pctile else "—", "Percentile", f"Compensation Level {sel_year}",
            info="Percentile position in the DAX compensation distribution. Percentile 80 means this company pays more than 80% of DAX peers."), unsafe_allow_html=True)
        with k3: st.markdown(kpi_html(f"{aep:+.0f}%" if aep is not None else "—", "vs. Model", t_lbl, t_col, t_col,
            info="How much actual compensation deviates from the OLS model's sector- and size-adjusted fair value. Positive = overpays vs. fundamentals; negative = underpays."), unsafe_allow_html=True)
        with k4: st.markdown(kpi_html(f"€{total/1000:.1f}M" if total else "—", "Total Compensation", "Total Executive Board",
            info="Sum of all executive board members' compensation (fixed + STI + LTI) before taxes for the selected year."), unsafe_allow_html=True)

    st.markdown('<div class="sec"><div class="sec-title">Compensation Comparison — All DAX Companies</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Bars = actual compensation · Diamonds = model expectation band · Color = benchmark signal</div></div>', unsafe_allow_html=True)

    if len(year_data) > 0:
        def bar_color(row):
            p = row.get("actual_vs_expected_pct", np.nan)
            if row["company_shortname"] == sel_co: return NAVY
            if pd.isna(p): return GRAY
            if p > 40: return RED
            if p > 15: return AMBER
            if p > -15: return "#4ade80"
            return "#2563eb"
        colors = [bar_color(r) for _, r in year_data.iterrows()]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=year_data["company_shortname"], y=year_data["pred_comp"],
            error_y=dict(type="data", symmetric=False,
                array=(year_data["pred_comp_high"]-year_data["pred_comp"]).fillna(0),
                arrayminus=(year_data["pred_comp"]-year_data["pred_comp_low"]).fillna(0)),
            mode="markers", marker=dict(symbol="diamond", size=9, color=ORANGE),
            name="Model Expectation (80% Band)"))
        fig.add_trace(go.Bar(
            x=year_data["company_shortname"], y=year_data["total_comp_bt"],
            marker_color=colors, opacity=0.85, name="Actual",
            hovertemplate="%{x}: €%{y:,.0f}K<extra></extra>"))
        fig.update_layout(height=360, barmode="overlay",
            margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Total Compensation (€K)", gridcolor=GRAYLT),
            xaxis=dict(tickangle=-30, gridcolor=GRAYLT),
            legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig, use_container_width=True)

    # Sector comparison
    if in_model and "sector" in df.columns:
        co_sector = df[df["company_shortname"]==sel_co]["sector"].iloc[0] if len(df[df["company_shortname"]==sel_co])>0 else None
        if co_sector:
            st.markdown(f'<div class="sec"><div class="sec-title">Sector Comparison: {co_sector}</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">How does {sel_co} compare vs. companies in the same sector?</div></div>', unsafe_allow_html=True)
            sect_data = df[(df["sector"]==co_sector) & (df["year"]==sel_year)].sort_values("total_comp_bt", ascending=False)
            if len(sect_data) > 1:
                sect_colors = [ORANGE if c==sel_co else "#cbd5e1" for c in sect_data["company_shortname"]]
                fig2 = go.Figure(go.Bar(
                    x=sect_data["company_shortname"], y=sect_data["total_comp_bt"],
                    marker_color=sect_colors,
                    hovertemplate="%{x}: €%{y:,.0f}K<extra></extra>"))
                fig2.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Total Compensation (€K)", gridcolor=GRAYLT),
                    xaxis=dict(tickangle=-20))
                st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 5 — ANOMALY DETECTOR
# ─────────────────────────────────────────────
def show_anomaly():
    sidebar_nav("overview", "← Back to Overview")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    co_df = df[df["company_shortname"] == sel_co].sort_values("year")
    co_yr = df[(df["company_shortname"]==sel_co) & (df["year"]==sel_year)]

    module_header("🔍", "Anomaly Detector", "Unusual Compensation Structures · Outlier Detection · Universe Comparison", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>What is an anomaly?</strong> Our detector flags companies with an atypical
        STI/LTI mix, extreme bonus-to-salary ratios, or unusually large compensation jumps —
        even when the absolute level appears market-aligned. Anomalies can indicate short-term bias,
        hidden one-time payments, or opaque incentive designs.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="sec"><div class="sec-title">Anomaly Score vs. Compensation Level — Universe</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">All companies in dataset · Orange = selected · Red = flagged as anomaly</div></div>', unsafe_allow_html=True)
        au_yr = anomaly_u[anomaly_u["year"]==sel_year]
        if len(au_yr) > 0:
            is_anom_mask  = au_yr["is_anomaly"] == 1
            not_anom_mask = ~is_anom_mask
            sel_mask      = au_yr["company_shortname"] == sel_co

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=au_yr[not_anom_mask & ~sel_mask]["total_comp_bt"],
                y=au_yr[not_anom_mask & ~sel_mask]["anomaly_score_pct"],
                mode="markers", marker=dict(size=7, color="#cbd5e1", opacity=0.6),
                name="Normal", hovertemplate="<b>%{text}</b><br>Verg.: €%{x:,.0f}K<br>Score: %{y:.1f}<extra></extra>",
                text=au_yr[not_anom_mask & ~sel_mask]["company_shortname"]))
            fig.add_trace(go.Scatter(
                x=au_yr[is_anom_mask & ~sel_mask]["total_comp_bt"],
                y=au_yr[is_anom_mask & ~sel_mask]["anomaly_score_pct"],
                mode="markers", marker=dict(size=9, color=RED, opacity=0.75, symbol="triangle-up"),
                name="🚨 Anomaly", hovertemplate="<b>%{text}</b><br>Comp.: €%{x:,.0f}K<br>Score: %{y:.1f}<extra></extra>",
                text=au_yr[is_anom_mask & ~sel_mask]["company_shortname"]))
            sel_row = au_yr[sel_mask]
            if len(sel_row) > 0:
                r = sel_row.iloc[0]
                fig.add_trace(go.Scatter(
                    x=[r["total_comp_bt"]], y=[r["anomaly_score_pct"]],
                    mode="markers+text",
                    marker=dict(size=16, color=NAVY, line=dict(width=3, color=ORANGE)),
                    text=[f"▶ {sel_co}"], textposition="top center",
                    textfont=dict(size=10, color=NAVY),
                    name=f"▶ {sel_co}"))
            fig.add_hline(y=50, line_dash="dot", line_color=AMBER, line_width=1,
                          annotation_text="Anomaly Threshold", annotation_font=dict(size=8, color=AMBER))
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Total Compensation (€K)", gridcolor=GRAYLT),
                yaxis=dict(title="Anomaly Score (0–100)", gridcolor=GRAYLT),
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
                hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(f'<div class="sec"><div class="sec-title">Anomaly Profile — {sel_co}</div></div>', unsafe_allow_html=True)
        if in_model and len(co_yr) > 0:
            anom_sc  = float(co_yr["anomaly_score_pct"].iloc[0]) if co_yr["anomaly_score_pct"].notna().any() else None
            is_anom  = bool(co_yr["is_anomaly"].iloc[0]) if "is_anomaly" in co_yr.columns and co_yr["is_anomaly"].notna().any() else None
            fix_p    = float(co_yr["fixed_pct"].iloc[0]) if co_yr["fixed_pct"].notna().any() else None
            sti_p    = float(co_yr["sti_pct"].iloc[0]) if co_yr["sti_pct"].notna().any() else None
            lti_p    = float(co_yr["lti_pct"].iloc[0]) if co_yr["lti_pct"].notna().any() else None
            bsr      = float(co_yr["bonus_salary_ratio"].iloc[0]) if "bonus_salary_ratio" in co_yr.columns and co_yr["bonus_salary_ratio"].notna().any() else None
            comp_yoy = float(co_yr["comp_yoy_pct"].iloc[0]) if co_yr["comp_yoy_pct"].notna().any() else None

            # Status badge
            badge_c = "#f87171" if is_anom else "#4ade80"
            badge_bg = "#7f1d1d" if is_anom else "#14532d"
            badge_txt = "🚨 Anomaly Detected" if is_anom else "✅ No Anomaly"
            st.markdown(f"""<div style="background:{NAVY};border-radius:12px;padding:16px;text-align:center;margin-bottom:14px;">
                <div style="font-size:2.4rem;font-weight:800;color:{badge_c};line-height:1;">{anom_sc:.0f}<span style="font-size:1.1rem;color:#475569;">/100</span></div>
                <div style="font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-top:4px;">Anomaly Score {sel_year}</div>
                <div style="background:{badge_bg};border-radius:6px;padding:3px 12px;display:inline-block;margin-top:8px;font-size:.78rem;font-weight:700;color:{badge_c};">{badge_txt}</div>
            </div>""", unsafe_allow_html=True)

            # Structure breakdown
            dax_fix = df[df["year"]==sel_year]["fixed_pct"].mean()
            dax_sti = df[df["year"]==sel_year]["sti_pct"].mean()
            dax_lti = df[df["year"]==sel_year]["lti_pct"].mean()
            rows = [
                ("Fixed Share", f"{fix_p:.0f}%" if fix_p else "—", f"DAX avg. {dax_fix:.0f}%",
                 RED if fix_p and abs(fix_p-dax_fix)>15 else GRAY),
                ("STI Share", f"{sti_p:.0f}%" if sti_p else "—", f"DAX avg. {dax_sti:.0f}%",
                 RED if sti_p and abs(sti_p-dax_sti)>15 else GRAY),
                ("LTI Share", f"{lti_p:.0f}%" if lti_p else "—", f"DAX avg. {dax_lti:.0f}%",
                 RED if lti_p and abs(lti_p-dax_lti)>15 else GRAY),
                ("Bonus/Salary", f"{bsr:.1f}x" if bsr else "—", "DAX avg. ~1.5x",
                 RED if bsr and bsr > 3 else GRAY),
                ("Compensation Growth YoY", f"{comp_yoy:+.1f}%" if comp_yoy and pd.notna(comp_yoy) else "—", "",
                 RED if comp_yoy and comp_yoy > 30 else GRAY),
            ]
            st.markdown('<div style="background:white;border-radius:12px;padding:14px 16px;border:1px solid #e2e8f0;">', unsafe_allow_html=True)
            for lbl, val, ref, col in rows:
                st.markdown(f'<div class="metric-row"><span style="color:{GRAY};">{lbl}<br><span style="font-size:.7rem;color:#94a3b8;">{ref}</span></span><span style="font-weight:700;color:{col};">{val}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(f"Anomaly score for {sel_co} not available in core group.")

    # Timeline
    if in_model and len(co_df) > 0 and "anomaly_score_pct" in co_df.columns:
        st.markdown('<div class="sec"><div class="sec-title">Anomaly Score Over Time</div></div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=co_df["year"], y=co_df["anomaly_score_pct"],
            fill="tozeroy", fillcolor="rgba(249,115,22,.10)",
            line=dict(color=ORANGE, width=2.5), mode="lines+markers",
            marker=dict(size=6, color=[RED if v else ORANGE for v in co_df.get("is_anomaly", pd.Series(False, index=co_df.index))]),
            hovertemplate="Jahr %{x}: Score %{y:.1f}<extra></extra>"))
        fig2.add_hline(y=50, line_dash="dot", line_color=AMBER, line_width=1,
                       annotation_text="Anomaly Threshold", annotation_font=dict(size=8))
        fig2.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(range=[0,105], gridcolor=GRAYLT), xaxis=dict(gridcolor=GRAYLT))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 6 — ESG RATING
# ─────────────────────────────────────────────
def show_esg():
    sidebar_nav("landing", "← Back to Home")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    co_esg = esg[esg["company_shortname"] == sel_co].sort_values("year") if len(esg) > 0 else pd.DataFrame()

    # For ESG metrics: prefer selected year; fall back to latest available
    co_esg_yr = co_esg[co_esg["year"] == sel_year] if len(co_esg) > 0 else pd.DataFrame()
    if len(co_esg_yr) == 0 and len(co_esg) > 0:
        co_esg_yr = co_esg.iloc[[-1]]
    actual_esg_year = int(co_esg_yr["year"].iloc[0]) if len(co_esg_yr) > 0 else sel_year

    module_header("🌱", "ESG & CSRD Governance", f"ESG Integration in Compensation · CSRD Compliance · Gender Equity · Pay Ratio · Data Year: {actual_esg_year}", sel_co, sel_year)

    show_universal_snapshot(sel_co, sel_year)
    stakeholder_divider("ESG & CSRD Governance View — Pay Integration & External Validation")

    def esg_val(col):
        v = co_esg_yr[col].iloc[0] if len(co_esg_yr) > 0 and col in co_esg_yr.columns and co_esg_yr[col].notna().any() else None
        return float(v) if v is not None else None

    sti_esg = esg_val("sti_esg_share")
    lti_esg = esg_val("lti_esg_share")
    fem_pct = esg_val("female_pct")
    w_ratio = esg_val("ceo_worker_ratio")
    esg_tot = (sti_esg or 0) + (lti_esg or 0)

    # Warn if showing fallback year
    if actual_esg_year != sel_year:
        st.markdown(f'<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:8px 14px;margin-bottom:10px;font-size:.8rem;color:#713f12;">ℹ️ No ESG data for {sel_year} — showing last available year: <strong>{actual_esg_year}</strong></div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="insight">
        <strong>ESG & CSRD:</strong> The EU Corporate Sustainability Reporting Directive (CSRD) has been
        mandatory since 2025. It requires explicit ESG compensation targets, verifiable KPI texts and
        disclosure of the incentive structure. Only <strong>16 of 43 DAX companies (37%)</strong> meet
        this standard. {sel_co} currently has <strong>{esg_tot:.0f}% ESG share</strong> in STI+LTI.
    </div>""", unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(kpi_html(f"{sti_esg:.0f}%" if sti_esg else "0%", "ESG Share STI", "✅ CSRD-relevant" if sti_esg and sti_esg>0 else "❌ No ESG", ORANGE if sti_esg and sti_esg>0 else "#94a3b8", GREEN if sti_esg and sti_esg>0 else RED,
        info="Share of the Short-Term Incentive (annual bonus) tied to ESG/sustainability KPIs. Under CSRD, companies must disclose and justify sustainability links in executive pay. Zero = no short-term accountability for ESG performance."), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html(f"{lti_esg:.0f}%" if lti_esg else "0%", "ESG Share LTI", "Long-term sustainability targets", ORANGE if lti_esg and lti_esg>0 else "#94a3b8",
        info="Share of Long-Term Incentives (multi-year programs) linked to ESG targets such as emissions reduction or social KPIs. LTI ESG links are considered more credible as they span multiple years and can't be gamed short-term."), unsafe_allow_html=True)
    with k3: st.markdown(kpi_html(f"{fem_pct:.0f}%" if fem_pct else "—", "Women on Board", "✅ ARUG II ≥30%" if fem_pct and fem_pct>=30 else "⚠ Below 30% Target", NAVY, GREEN if fem_pct and fem_pct>=30 else AMBER,
        info="Percentage of female executive board members. German law (ARUG II) mandates ≥30% for supervisory boards of large listed companies. Research shows board diversity correlates with better risk management."), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html(f"{w_ratio:.0f}x" if w_ratio else "—", "Exec/Worker Pay", "⚠ Very High" if w_ratio and w_ratio>60 else "Income Inequality", RED if w_ratio and w_ratio>60 else NAVY, RED if w_ratio and w_ratio>60 else GRAY,
        info="Ratio of CEO total compensation to the median employee wage. A high ratio (>60x) signals internal pay inequality and is increasingly scrutinized by ESG raters and employees. Required disclosure under CSRD."), unsafe_allow_html=True)

    # DAX Bubble Chart
    st.markdown('<div class="sec"><div class="sec-title">DAX Overview: Who Actually Integrates ESG?</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">X/Y = ESG share STI/LTI · Dot size = avg. compensation · Orange = with ESG · Grey = without ESG</div></div>', unsafe_allow_html=True)

    if len(esg) > 0:
        recent = esg[esg["year"]>=2022]
        bub = recent.groupby("company_shortname").agg(
            sti=("sti_esg_share","mean"), lti=("lti_esg_share","mean"),
            avg_comp=("avg_comp","mean")).reset_index().fillna(0)
        bub["has_esg"] = (bub["sti"]+bub["lti"]) > 0
        bub["is_sel"]  = bub["company_shortname"] == sel_co

        fig = go.Figure()
        for has, color, name in [(False,"#cbd5e1","No ESG Link"),(True,ORANGE,"With ESG Link")]:
            d = bub[~bub["is_sel"] & (bub["has_esg"]==has)]
            fig.add_trace(go.Scatter(x=d["sti"], y=d["lti"], mode="markers+text",
                marker=dict(size=d["avg_comp"].clip(50,2000).apply(lambda v: max(8,min(26,v/80))),
                    color=color, opacity=0.7, line=dict(width=1, color="white")),
                text=d["company_shortname"], textposition="top center",
                textfont=dict(size=8, color="#94a3b8"),
                hovertemplate="<b>%{text}</b><br>STI: %{x:.0f}% · LTI: %{y:.0f}%<extra></extra>",
                showlegend=True, name=name))
        sel_row = bub[bub["is_sel"]]
        if len(sel_row)>0:
            r = sel_row.iloc[0]
            fig.add_trace(go.Scatter(x=[r["sti"]], y=[r["lti"]], mode="markers+text",
                marker=dict(size=max(18,min(34,r["avg_comp"]/70)), color=NAVY, line=dict(width=3,color=ORANGE)),
                text=[f"▶ {sel_co}"], textposition="top center",
                textfont=dict(size=11, color=NAVY),
                hovertemplate=f"<b>{sel_co}</b><extra></extra>",
                showlegend=True, name=f"▶ {sel_co}"))
        for x,y,txt,col in [(45,45,"ESG Leader","#15803d"),(3,45,"LTI-focused","#1d4ed8"),(45,3,"STI-focused",AMBER),(3,3,"No ESG Link",RED)]:
            fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(color=col,size=9), opacity=0.5)
        fig.add_hline(y=20,line_dash="dot",line_color="#e2e8f0",line_width=1)
        fig.add_vline(x=20,line_dash="dot",line_color="#e2e8f0",line_width=1)
        fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="ESG Share STI (%)", range=[-5,68], gridcolor=GRAYLT),
            yaxis=dict(title="ESG Share LTI (%)", range=[-5,68], gridcolor=GRAYLT),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig, use_container_width=True)

    col_kpi, col_csrd = st.columns([3, 2], gap="large")

    with col_kpi:
        st.markdown(f"**KPI Texts from the Compensation Report — {sel_co} ({actual_esg_year})**")
        # Use year-specific row first; else search all years
        co_kpi_rows = co_esg_yr[co_esg_yr[["stipi1","stipi2","ltipi1","ltipi2"]].notna().any(axis=1)] if len(co_esg_yr)>0 else pd.DataFrame()
        if len(co_kpi_rows) == 0:
            co_kpi_rows = co_esg[co_esg[["stipi1","stipi2","ltipi1","ltipi2"]].notna().any(axis=1)] if len(co_esg)>0 else pd.DataFrame()
        CAT = {"Environmental":("#dcfce7","#15803d","🌍"),"Social":("#dbeafe","#1d4ed8","🤝"),
               "Governance":("#f3e8ff","#6d28d9","🏛"),"Financial":("#fef9c3","#92400e","💰")}
        if len(co_kpi_rows) > 0:
            latest = co_kpi_rows.iloc[-1]
            for lbl, txt, cat, w in [("STI-KPI 1",latest.get("stipi1"),latest.get("stipi1_cat"),latest.get("stipi1weight_num")),
                                      ("STI-KPI 2",latest.get("stipi2"),latest.get("stipi2_cat"),latest.get("stipi2weight_num")),
                                      ("LTI-KPI 1",latest.get("ltipi1"),latest.get("ltipi1_cat"),latest.get("ltipi1weight_num")),
                                      ("LTI-KPI 2",latest.get("ltipi2"),latest.get("ltipi2_cat"),latest.get("ltipi2weight_num"))]:
                if pd.notna(txt) and str(txt).strip():
                    bg,fg,icon = CAT.get(str(cat),("#f1f5f9","#475569","💰"))
                    w_s = f"<span style='font-weight:700;color:{fg};float:right;'>{w:.0f}%</span>" if pd.notna(w) else ""
                    st.markdown(f"""<div class="kpi-tile" style="background:{bg};">
                        <div style="font-size:.7rem;font-weight:700;color:{fg};text-transform:uppercase;letter-spacing:.05em;">{icon} {cat} · {lbl} {w_s}</div>
                        <div style="font-size:.84rem;color:#1e293b;margin-top:4px;line-height:1.4;">{str(txt)[:130]}</div>
                    </div>""", unsafe_allow_html=True)
            if sel_co in {"BMW","Volkswagen","Mercedes-Benz"}:
                st.markdown(f"""<div class="flag-red">
                    <strong>⚠️ Disclosure Opacity:</strong> {sel_co} uses abstract KPI terms —
                    external ESG content cannot be verified. This is exactly what CSRD addresses.
                </div>""", unsafe_allow_html=True)
        else:
            st.info("KPI texts available for 2022–2024. No text data collected for earlier years.")

    with col_csrd:
        st.markdown(f"**CSRD Compliance Check ({actual_esg_year})**")
        st.caption("EU Corporate Sustainability Reporting Directive — mandatory from 2025")
        OPACITY = {"BMW","Volkswagen","Mercedes-Benz"}
        has_kpi_text = len(co_kpi_rows) > 0
        checks = [
            ("ESG target anchored in STI",          sti_esg is not None and sti_esg > 0),
            ("ESG target anchored in LTI",          lti_esg is not None and lti_esg > 0),
            ("KPI text publicly verifiable",        has_kpi_text and sel_co not in OPACITY),
            ("No abstract blanket KPIs",            sel_co not in OPACITY),
            ("Female quota ≥30% (ARUG II)",         fem_pct is not None and fem_pct >= 30),
        ]
        score = sum(1 for _,ok in checks if ok)
        s_fg  = "#4ade80" if score>=4 else "#fbbf24" if score>=3 else "#f87171"
        s_bg  = "#14532d" if score>=4 else "#713f12" if score>=3 else "#7f1d1d"
        s_lbl = "Compliant" if score>=4 else "Partial" if score>=3 else "Not Compliant"
        st.markdown(f"""<div style="background:{NAVY};border-radius:12px;padding:16px;text-align:center;margin-bottom:14px;">
            <div style="font-size:2.6rem;font-weight:800;color:{s_fg};line-height:1;">{score}<span style="font-size:1.2rem;color:#475569;">/5</span></div>
            <div style="font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:4px;">CSRD-Score</div>
            <div style="background:{s_bg};border-radius:6px;padding:3px 12px;display:inline-block;margin-top:8px;font-size:.78rem;font-weight:700;color:{s_fg};">{s_lbl}</div>
        </div>""", unsafe_allow_html=True)
        for label, ok in checks:
            icon = "✅" if ok else "❌"
            col  = "#1e293b" if ok else RED
            st.markdown(f'<div style="font-size:.82rem;color:{col};padding:6px 0;border-bottom:1px solid #f1f5f9;">{icon} {label}</div>', unsafe_allow_html=True)

    # Pay Equity
    st.markdown('<div class="sec"><div class="sec-title">Pay Equity — Gender &amp; CEO/Worker Ratio</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Female share vs. ARUG II target · Income inequality across DAX</div></div>', unsafe_allow_html=True)
    col_f, col_r2 = st.columns(2)
    with col_f:
        latest_fem = (esg.sort_values("year").groupby("company_shortname").last()
                      .reset_index()[["company_shortname","female_pct"]]
                      .sort_values("female_pct", ascending=True).dropna())
        colors_f = [ORANGE if c==sel_co else ("#4ade80" if v>=30 else "#fbbf24" if v>=20 else "#f87171") for c,v in zip(latest_fem["company_shortname"],latest_fem["female_pct"])]
        fig_f = go.Figure(go.Bar(x=latest_fem["female_pct"], y=latest_fem["company_shortname"],
            orientation="h", marker_color=colors_f,
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig_f.add_vline(x=30, line_dash="dash", line_color=GREEN, line_width=1.5,
            annotation_text="30% (ARUG II)", annotation_font=dict(size=8, color=GREEN))
        fig_f.update_layout(height=480, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Female Share (%)", range=[0,65], gridcolor=GRAYLT),
            yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_f, use_container_width=True)
    with col_r2:
        latest_rat = (esg.sort_values("year").groupby("company_shortname").last()
                      .reset_index()[["company_shortname","ceo_worker_ratio"]]
                      .sort_values("ceo_worker_ratio", ascending=True).dropna())
        colors_r = [ORANGE if c==sel_co else (RED if v>65 else AMBER if v>45 else "#94a3b8") for c,v in zip(latest_rat["company_shortname"],latest_rat["ceo_worker_ratio"])]
        fig_r = go.Figure(go.Bar(x=latest_rat["ceo_worker_ratio"], y=latest_rat["company_shortname"],
            orientation="h", marker_color=colors_r,
            text=[f"{v:.0f}x" for v in latest_rat["ceo_worker_ratio"]],
            textposition="outside", textfont=dict(size=8),
            hovertemplate="%{y}: %{x:.0f}x<extra></extra>"))
        med = latest_rat["ceo_worker_ratio"].median()
        fig_r.add_vline(x=med, line_dash="dot", line_color=GRAY, line_width=1.5,
            annotation_text=f"Median {med:.0f}x", annotation_font=dict(size=8, color=GRAY))
        fig_r.update_layout(height=480, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Exec/Worker Pay (x)", gridcolor=GRAYLT),
            yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_r, use_container_width=True)

    # ── ESG Pay-Washing Detector ──────────────────────────────────
    st.markdown('<div class="sec"><div class="sec-title">🔬 External ESG Validation — Pay-Washing Detector</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Compares our derived ESG-pay link with Sustainalytics risk rating & MSCI ESG score</div></div>', unsafe_allow_html=True)

    if len(ext_esg) > 0:
        co_ext = ext_esg[ext_esg["company_shortname"] == sel_co]
        if len(co_ext) > 0:
            ext_row = co_ext.iloc[0]
            sust_score  = float(ext_row["sustainalytics_score"])
            sust_cat    = str(ext_row["sustainalytics_risk_category"])
            msci_rating = str(ext_row["msci_rating"])
            msci_score  = int(ext_row["msci_score"])
            cdp_score   = str(ext_row.get("cdp_score", "—"))
            msci_score_norm = (msci_score / 7) * 100  # normalize to 0-100
            sust_score_inv  = max(0, 100 - (sust_score / 50) * 100)  # invert (lower risk = higher score)

            # ESG Pay Score: normalized 0-100
            pay_esg_score = min(100, esg_tot * 2.0)  # 50% ESG in pay = 100 pts
            if sti_esg is None:  # BMW opacity
                pay_esg_score = None

            sust_col = {"Negligible": GREEN, "Low": GREEN, "Medium": AMBER,
                        "High": RED, "Severe": "#7f1d1d"}.get(sust_cat, GRAY)
            msci_col = {"AAA": GREEN,"AA": GREEN,"A": "#4ade80","BBB": AMBER,
                        "BB": AMBER,"B": RED,"CCC": RED}.get(msci_rating, GRAY)

            # Determine Pay-Washing Risk
            if pay_esg_score is None:
                wash_risk = "opacity"
            elif pay_esg_score >= 20 and sust_score_inv >= 60:
                wash_risk = "aligned"
            elif pay_esg_score < 10 and sust_score_inv >= 65:
                wash_risk = "washing"  # high external score, low pay ESG = performative
            elif pay_esg_score >= 20 and sust_score_inv < 50:
                wash_risk = "disconnect"  # paying for ESG but still poor performer
            else:
                wash_risk = "neutral"

            cA, cB, cC = st.columns([1, 1, 1], gap="large")
            with cA:
                st.markdown(f"""<div style="background:{NAVY};border-radius:16px;padding:20px;text-align:center;">
                    <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:8px;">Sustainalytics Risk</div>
                    <div style="font-size:2.6rem;font-weight:900;color:{sust_col};line-height:1;">{sust_score:.1f}</div>
                    <div style="font-size:.7rem;color:#94a3b8;margin-top:3px;">/ 50 · {sust_cat} Risk</div>
                    <div style="background:{sust_col}22;border:1px solid {sust_col}44;border-radius:8px;padding:4px 12px;display:inline-block;margin-top:8px;font-size:.72rem;font-weight:700;color:{sust_col};">{"✓ Good" if sust_cat in ("Low","Negligible") else "⚠ Medium" if sust_cat=="Medium" else "✗ High"}</div>
                </div>""", unsafe_allow_html=True)
            with cB:
                st.markdown(f"""<div style="background:{NAVY};border-radius:16px;padding:20px;text-align:center;">
                    <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:8px;">MSCI ESG Rating</div>
                    <div style="font-size:2.6rem;font-weight:900;color:{msci_col};line-height:1;">{msci_rating}</div>
                    <div style="font-size:.7rem;color:#94a3b8;margin-top:3px;">MSCI ESG Leaders Index</div>
                    <div style="background:{msci_col}22;border:1px solid {msci_col}44;border-radius:8px;padding:4px 12px;display:inline-block;margin-top:8px;font-size:.72rem;font-weight:700;color:{msci_col};">{"AAA–A: Leader" if msci_score>=5 else "BBB: Average" if msci_score==4 else "BB–CCC: Laggard"}</div>
                </div>""", unsafe_allow_html=True)
            with cC:
                if pay_esg_score is not None:
                    pcolor = GREEN if pay_esg_score >= 25 else AMBER if pay_esg_score >= 10 else RED
                    st.markdown(f"""<div style="background:{NAVY};border-radius:16px;padding:20px;text-align:center;">
                        <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:8px;">ExComp ESG-Pay Score</div>
                        <div style="font-size:2.6rem;font-weight:900;color:{pcolor};line-height:1;">{pay_esg_score:.0f}</div>
                        <div style="font-size:.7rem;color:#94a3b8;margin-top:3px;">/ 100 · derived from KPI analysis</div>
                        <div style="background:{pcolor}22;border:1px solid {pcolor}44;border-radius:8px;padding:4px 12px;display:inline-block;margin-top:8px;font-size:.72rem;font-weight:700;color:{pcolor};">{"ESG Payer" if pay_esg_score>=25 else "Low" if pay_esg_score>=5 else "No ESG Link"}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="background:{NAVY};border-radius:16px;padding:20px;text-align:center;">
                        <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:8px;">ExComp ESG-Pay Score</div>
                        <div style="font-size:2.6rem;font-weight:900;color:{AMBER};line-height:1;">N/V</div>
                        <div style="font-size:.7rem;color:#94a3b8;margin-top:3px;">Not Verifiable</div>
                        <div style="background:{AMBER}22;border:1px solid {AMBER}44;border-radius:8px;padding:4px 12px;display:inline-block;margin-top:8px;font-size:.72rem;font-weight:700;color:{AMBER};">⚠ Opacity</div>
                    </div>""", unsafe_allow_html=True)

            # Pay-Washing verdict
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if wash_risk == "aligned":
                st.markdown(f"""<div class="flag-green">
                    <strong>✅ ESG-Pay Alignment:</strong> {sel_co} shows consistency between external ESG ratings
                    (Sustainalytics: {sust_score:.1f} · MSCI: {msci_rating}) and ESG pay integration ({esg_tot:.0f}% in pay).
                    The compensation system supports the ESG leadership position.
                </div>""", unsafe_allow_html=True)
            elif wash_risk == "washing":
                st.markdown(f"""<div class="flag-red">
                    <strong>🚨 ESG Pay-Washing Risk:</strong> {sel_co} enjoys a strong external ESG rating
                    (Sustainalytics: {sust_score:.1f} Low/Medium · MSCI: {msci_rating}), yet links
                    <strong>0% of executive pay</strong> to ESG targets. External reputation without internal incentives —
                    exactly what CSRD is designed to expose.
                </div>""", unsafe_allow_html=True)
            elif wash_risk == "disconnect":
                st.markdown(f"""<div class="flag-amber">
                    <strong>⚠ ESG Pay-Performance Gap:</strong> {sel_co} ties {esg_tot:.0f}% of pay
                    to ESG targets (Sustainalytics: {sust_score:.1f} · MSCI: {msci_rating}),
                    yet still shows above-average ESG risk. Pay-for-ESG alone is not enough — target quality matters.
                </div>""", unsafe_allow_html=True)
            elif wash_risk == "opacity":
                st.markdown(f"""<div class="flag-amber">
                    <strong>⚠ Disclosure Opacity:</strong> {sel_co} uses abstract compensation terms —
                    external verification of the ESG share is not possible. CSRD §29a
                    requires explicit, verifiable ESG KPI texts. External rating: MSCI {msci_rating} ·
                    Sustainalytics {sust_score:.1f}.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="flag-amber">
                    <strong>ℹ ESG-Pay in Progress:</strong> Low ESG share in compensation with moderate
                    external ratings (Sustainalytics: {sust_score:.1f} · MSCI: {msci_rating}).
                    CSRD compliance pressure will require adjustments.
                </div>""", unsafe_allow_html=True)

        # DAX Pay-Washing Heatmap
        st.markdown('<div class="sec"><div class="sec-title">DAX Overview: ESG-Pay vs. External ESG Score</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Quadrants: Who pays for ESG — and has it improved external scores?</div></div>', unsafe_allow_html=True)
        recent = esg[esg["year"]>=2022]
        bub2 = recent.groupby("company_shortname").agg(
            sti=("sti_esg_share","mean"), lti=("lti_esg_share","mean"),
            avg_comp=("avg_comp","mean")).reset_index().fillna(0)
        bub2["esg_pay_total"] = bub2["sti"] + bub2["lti"]
        bub2 = bub2.merge(ext_esg[["company_shortname","sustainalytics_score","msci_score","sustainalytics_risk_category"]], on="company_shortname", how="left")
        bub2["sust_inv"] = (100 - (bub2["sustainalytics_score"] / 50) * 100).clip(0, 100)
        bub2["is_sel"] = bub2["company_shortname"] == sel_co

        def quadrant_color(row):
            pay = row["esg_pay_total"]
            perf = row["sust_inv"]
            if pay >= 15 and perf >= 60: return "#16a34a"    # ESG-Leader
            if pay < 10  and perf >= 65: return "#dc2626"    # Pay-Washer
            if pay >= 15 and perf < 55:  return "#d97706"    # Pay without results
            return "#94a3b8"

        bub2["color"] = bub2.apply(quadrant_color, axis=1)

        fig_w = go.Figure()
        for color, label in [("#16a34a","✅ ESG Leader"), ("#dc2626","🚨 Pay-Washing Risk"),
                              ("#d97706","⚠ Pay Without Results"), ("#94a3b8","Neutral")]:
            d = bub2[~bub2["is_sel"] & (bub2["color"]==color)]
            if len(d)>0:
                fig_w.add_trace(go.Scatter(
                    x=d["esg_pay_total"], y=d["sust_inv"],
                    mode="markers+text",
                    marker=dict(size=d["avg_comp"].clip(100,2000).apply(lambda v: max(9,min(28,v/70))),
                        color=color, opacity=0.75, line=dict(width=1.5, color="white")),
                    text=d["company_shortname"], textposition="top center",
                    textfont=dict(size=7.5, color="#64748b"),
                    hovertemplate="<b>%{text}</b><br>ESG-Pay: %{x:.0f}%<br>ESG-Score: %{y:.0f}<extra></extra>",
                    name=label, showlegend=True))
        # Selected company
        sel_w = bub2[bub2["is_sel"]]
        if len(sel_w)>0:
            r = sel_w.iloc[0]
            fig_w.add_trace(go.Scatter(x=[r["esg_pay_total"]], y=[r["sust_inv"]],
                mode="markers+text",
                marker=dict(size=max(18,min(34,r["avg_comp"]/70)), color=NAVY,
                    line=dict(width=3, color=ORANGE)),
                text=[f"▶ {sel_co}"], textposition="top center",
                textfont=dict(size=11, color=NAVY),
                hovertemplate=f"<b>{sel_co}</b><extra></extra>",
                name=f"▶ {sel_co}", showlegend=True))
        # Quadrant lines
        fig_w.add_hline(y=65, line_dash="dot", line_color="#e2e8f0", line_width=1)
        fig_w.add_vline(x=12, line_dash="dot", line_color="#e2e8f0", line_width=1)
        # Quadrant labels
        for x, y, txt, col in [(35,85,"ESG Leader","#15803d"), (2,85,"🚨 Pay-Washing","#dc2626"),
                                (35,35,"Pay Without Results","#d97706"), (2,35,"Neutral","#94a3b8")]:
            fig_w.add_annotation(x=x, y=y, text=txt, showarrow=False,
                font=dict(color=col, size=9), opacity=0.6)
        fig_w.update_layout(
            height=440, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="ESG Share in Compensation STI+LTI (%)", range=[-5, 68], gridcolor=GRAYLT),
            yaxis=dict(title="ESG Performance Score (Sustainalytics, inverted)", range=[20, 100], gridcolor=GRAYLT),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=9)))
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown('<div style="text-align:center;color:#94a3b8;font-size:.74rem;padding:16px 0 4px 0;border-top:1px solid #e2e8f0;margin-top:12px;">ExComp · TUM Science Hackathon 2026 · ORBIS/Bureau van Dijk · DGAP · Sustainalytics · MSCI ESG · 43 DAX Companies · 2006–2024</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# STAKEHOLDER PAGES
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# STAKEHOLDER 1 — CAPITAL ALLOCATORS
# ─────────────────────────────────────────────
def show_capital_allocators():
    sidebar_nav("landing", "← Back to Home")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    co_df    = df[df["company_shortname"] == sel_co].sort_values("year")
    co_mu_yr = mu[(mu["company_shortname"] == sel_co) & (mu["year"] == sel_year)]

    st.markdown(f"""<div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">🏦 Capital Allocators View</div>
                <div style="font-size:1.8rem;font-weight:900;color:white;letter-spacing:-.5px;margin:4px 0;">{sel_co}</div>
                <div style="font-size:.82rem;color:#94a3b8;">Pay-for-Performance · Governance Risk · Long-term Alignment · 2006–{FEATURES_YEAR_MAX}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    show_universal_snapshot(sel_co, sel_year)
    stakeholder_divider("Capital Allocator View — Pay-for-Performance & Governance Risk")

    st.markdown("""<div class="insight">
        <strong>What do capital allocators care about?</strong> Institutional investors and ESG funds need to
        assess whether executive pay aligns with company performance. Key signals: pay growing faster than
        EBIT (misalignment), excessive CEO premiums, compensation rising during bad years, and short-term
        incentive bias that rewards quarterly thinking over long-term value creation.
    </div>""", unsafe_allow_html=True)

    if in_model and len(co_df) > 0:
        avg_yoy  = float(co_df["comp_yoy_pct"].mean()) if "comp_yoy_pct" in co_df.columns else None
        avg_ebit = float(co_df["ebit_yoy_pct"].mean()) if "ebit_yoy_pct" in co_df.columns and co_df["ebit_yoy_pct"].notna().any() else None
        lti_avg  = float(co_df["lti_pct"].mean()) if "lti_pct" in co_df.columns and co_df["lti_pct"].notna().any() else None
        ceo_prem = float(co_df["ceo_board_premium_ratio"].mean()) if "ceo_board_premium_ratio" in co_df.columns and co_df["ceo_board_premium_ratio"].notna().any() else None
        gov_sc   = float(co_df["anomaly_score_pct"].mean()) if "anomaly_score_pct" in co_df.columns and co_df["anomaly_score_pct"].notna().any() else None
        sz_cnt   = int(co_df["schlechte_zeiten"].sum()) if "schlechte_zeiten" in co_df.columns else 0
        pay_ebit_gap = (avg_yoy - avg_ebit) if avg_yoy and avg_ebit and pd.notna(avg_ebit) else None
        aep_over = int((co_df["actual_vs_expected_pct"] > 40).sum()) if "actual_vs_expected_pct" in co_df.columns and co_df["actual_vs_expected_pct"].notna().any() else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(kpi_html(
                f"{pay_ebit_gap:+.1f}pp" if pay_ebit_gap is not None else "—",
                "Pay vs. EBIT Growth Gap",
                "⚠ Pay outpaces earnings" if pay_ebit_gap and pay_ebit_gap > 5 else "✅ Aligned",
                RED if pay_ebit_gap and pay_ebit_gap > 5 else GREEN,
                RED if pay_ebit_gap and pay_ebit_gap > 5 else GREEN,
                info="Average annual exec pay growth minus average annual EBIT growth (in percentage points). A positive gap means pay has grown faster than earnings over the period — a core pay-for-performance concern for investors."
            ), unsafe_allow_html=True)
        with k2:
            st.markdown(kpi_html(
                f"{lti_avg:.0f}%" if lti_avg else "—",
                "Avg. LTI Share",
                "✅ Long-term focused" if lti_avg and lti_avg > 40 else "⚠ Short-term bias" if lti_avg and lti_avg < 25 else "Balanced",
                GREEN if lti_avg and lti_avg > 40 else RED if lti_avg and lti_avg < 25 else AMBER,
                GREEN if lti_avg and lti_avg > 40 else RED if lti_avg and lti_avg < 25 else AMBER,
                info="Average share of total executive pay delivered as Long-Term Incentives. Higher LTI = executives rewarded for multi-year performance, reducing short-termism. Best practice in DAX is ≥40% LTI."
            ), unsafe_allow_html=True)
        with k3:
            st.markdown(kpi_html(
                f"{ceo_prem:.1f}x" if ceo_prem else "—",
                "CEO Board Premium",
                "⚠ Very high (>2.5x)" if ceo_prem and ceo_prem > 2.5 else "✅ Within range",
                RED if ceo_prem and ceo_prem > 2.5 else GREEN,
                RED if ceo_prem and ceo_prem > 2.5 else GREEN,
                info="Ratio of CEO compensation to the average compensation of other executive board members. Values >2.5x may signal excessive CEO pay relative to peers or governance issues in the remuneration committee."
            ), unsafe_allow_html=True)
        with k4:
            st.markdown(kpi_html(
                f"{gov_sc:.0f}/100" if gov_sc else "—",
                "Avg. Governance Risk",
                "⚠ High Risk" if gov_sc and gov_sc > 60 else "✅ Normal",
                risk_color(gov_sc or 0), risk_color(gov_sc or 0),
                info="Composite governance risk score (0–100) averaged over recent years, combining pay-performance gaps, structural anomalies, and Bad Times events. Used to flag systemic governance issues."
            ), unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2], gap="large")
        with col_l:
            st.markdown(sec_html("Pay Growth vs. EBIT Growth",
                "Compensation growth outpacing earnings is a pay-for-performance misalignment signal"), unsafe_allow_html=True)
            if "ebit_yoy_pct" in co_df.columns:
                co_clean = co_df.dropna(subset=["comp_yoy_pct", "ebit_yoy_pct"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=co_clean["year"], y=co_clean["ebit_yoy_pct"].clip(-100, 200),
                    line=dict(color="#3b82f6", width=2), mode="lines+markers", marker=dict(size=5),
                    name="EBIT Growth %", hovertemplate="EBIT: %{y:+.1f}%<extra></extra>"))
                fig.add_trace(go.Scatter(x=co_clean["year"], y=co_clean["comp_yoy_pct"].clip(-100, 200),
                    line=dict(color=ORANGE, width=2.5), mode="lines+markers", marker=dict(size=5),
                    name="Exec Pay Growth %", hovertemplate="Pay: %{y:+.1f}%<extra></extra>"))
                fig.add_hline(y=0, line_color=GRAY, line_width=1)
                fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Growth (%)", gridcolor=GRAYLT),
                    xaxis=dict(gridcolor=GRAYLT),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                    hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown(sec_html("Governance Risk Timeline"), unsafe_allow_html=True)
            if "anomaly_score_pct" in co_df.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=co_df["year"], y=co_df["anomaly_score_pct"],
                    fill="tozeroy", fillcolor="rgba(220,38,38,.08)",
                    line=dict(color=RED, width=2), mode="lines+markers", marker=dict(size=5),
                    name="Risk Score"))
                fig2.add_hline(y=60, line_dash="dot", line_color=RED, line_width=1,
                    annotation_text="High Risk", annotation_font=dict(size=8, color=RED))
                fig2.add_hline(y=40, line_dash="dot", line_color=AMBER, line_width=1)
                fig2.update_layout(height=170, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(range=[0, 105], gridcolor=GRAYLT, tickfont=dict(size=9)),
                    xaxis=dict(gridcolor=GRAYLT, tickfont=dict(size=9)))
                st.plotly_chart(fig2, use_container_width=True)

            flags = [
                (sz_cnt > 0,      f"⚠ {sz_cnt}× pay ↑ when EBIT ↓ (Bad Times)", "red"),
                (pay_ebit_gap and pay_ebit_gap > 5, f"⚠ Pay outpaced EBIT by {pay_ebit_gap:.1f}pp on avg.", "red"),
                (aep_over > 0,    f"⚠ {aep_over} year(s) >+40% above model expectation", "amber"),
                (ceo_prem and ceo_prem > 2.5, f"⚠ CEO premium {ceo_prem:.1f}x (DAX median ~2x)", "amber"),
                (lti_avg and lti_avg < 25, f"⚠ Low LTI share ({lti_avg:.0f}%) — short-termism risk", "amber"),
            ]
            has_flag = False
            for cond, txt, sev in flags:
                if cond:
                    st.markdown(f'<div class="flag-{sev}">{txt}</div>', unsafe_allow_html=True)
                    has_flag = True
            if not has_flag:
                st.markdown('<div class="flag-green">✅ No major pay-for-performance red flags detected.</div>', unsafe_allow_html=True)

        # ── Row 2: quantitative deep-dive ──────────────────────────────
        st.markdown(sec_html("Quantitative Pay-for-Performance Analysis",
            "Statistical measures of compensation alignment — pay-β, cumulative excess, LTI trend, crisis resilience"), unsafe_allow_html=True)

        co_mu_all = mu[mu["company_shortname"] == sel_co].sort_values("year")

        # Pay-performance beta
        co_clean = co_df.dropna(subset=["comp_yoy_pct", "ebit_yoy_pct"])
        pay_beta, pay_corr = None, None
        if len(co_clean) > 4:
            pay_beta = float(np.polyfit(
                co_clean["ebit_yoy_pct"].clip(-100, 200),
                co_clean["comp_yoy_pct"].clip(-100, 200), 1)[0])
            pay_corr = float(co_clean["comp_yoy_pct"].corr(co_clean["ebit_yoy_pct"]))

        # Cumulative excess pay in €M
        cum_excess_m = None
        co_mu_known = co_mu_all.dropna(subset=["total_comp_bt", "pred_comp"])
        if len(co_mu_known) > 0:
            cum_excess_m = float((co_mu_known["total_comp_bt"] - co_mu_known["pred_comp"]).sum() / 1000)

        # Consecutive years above +15% threshold
        max_streak = 0
        if "actual_vs_expected_pct" in co_mu_all.columns and co_mu_all["actual_vs_expected_pct"].notna().any():
            curr = 0
            for v in (co_mu_all["actual_vs_expected_pct"] > 15).fillna(False):
                if v: curr += 1; max_streak = max(max_streak, curr)
                else: curr = 0

        # LTI trend slope (pp per year)
        lti_slope = None
        lti_clean = co_df.dropna(subset=["lti_pct"])
        if len(lti_clean) > 4:
            lti_slope = float(np.polyfit(range(len(lti_clean)), lti_clean["lti_pct"].values, 1)[0])

        # Crisis resilience: comp change in GFC (2009) and COVID (2020)
        gfc_chg   = float(co_df[co_df["year"] == 2009]["comp_yoy_pct"].values[0]) if 2009 in co_df["year"].values and co_df[co_df["year"] == 2009]["comp_yoy_pct"].notna().any() else None
        covid_chg = float(co_df[co_df["year"] == 2020]["comp_yoy_pct"].values[0]) if 2020 in co_df["year"].values and co_df[co_df["year"] == 2020]["comp_yoy_pct"].notna().any() else None

        qa1, qa2, qa3, qa4, qa5 = st.columns(5)
        with qa1:
            beta_col = GREEN if pay_beta and 0 < pay_beta < 0.5 else RED if pay_beta and pay_beta > 0.8 else AMBER if pay_beta else GRAY
            beta_lbl = "✅ Low coupling" if pay_beta and pay_beta < 0.3 else "⚠ Tight coupling" if pay_beta and pay_beta > 0.7 else "Moderate" if pay_beta else "n/a"
            st.markdown(kpi_html(f"{pay_beta:.2f}" if pay_beta is not None else "—",
                "Pay-Performance β",
                f"r={pay_corr:.2f}" if pay_corr is not None else beta_lbl,
                beta_col, beta_col,
                info="Linear regression slope of exec pay growth (Y) on EBIT growth (X). β=1 means pay moves 1:1 with earnings. β close to 0 or negative means pay is essentially decoupled from operating performance — a key investor concern."), unsafe_allow_html=True)
        with qa2:
            exc_col = RED if cum_excess_m and cum_excess_m > 50 else AMBER if cum_excess_m and cum_excess_m > 10 else GREEN if cum_excess_m else GRAY
            st.markdown(kpi_html(
                f"€{cum_excess_m:+.0f}M" if cum_excess_m is not None else "—",
                "Cumulative Excess Pay",
                "vs. model expectation (all yrs)",
                exc_col, exc_col,
                info="Total euros of excess pay above the OLS model's fair-value prediction, summed over all years with available data. Represents the aggregate shareholder value transferred to executives beyond what is justified by company fundamentals."), unsafe_allow_html=True)
        with qa3:
            str_col = RED if max_streak >= 3 else AMBER if max_streak >= 2 else GREEN
            str_lbl = "⚠ Persistent offender" if max_streak >= 3 else "⚠ 2 consecutive yrs" if max_streak >= 2 else "✅ No persistence"
            st.markdown(kpi_html(f"{max_streak} yr{'s' if max_streak != 1 else ''}",
                "Longest Overpay Streak",
                str_lbl, str_col, str_col,
                info="Maximum number of consecutive years where actual pay exceeded the model expectation by more than +15%. Long streaks indicate structural overpayment rather than one-off events, and are a strong red flag for governance activists."), unsafe_allow_html=True)
        with qa4:
            sl_col = GREEN if lti_slope and lti_slope > 0.3 else RED if lti_slope and lti_slope < -0.3 else GRAY
            sl_lbl = f"{'↑' if lti_slope and lti_slope > 0 else '↓'} {abs(lti_slope):.2f}pp/yr" if lti_slope else "n/a"
            st.markdown(kpi_html(sl_lbl if lti_slope else "—",
                "LTI Trend Slope",
                "✅ Rising long-term bias" if lti_slope and lti_slope > 0.3 else "⚠ Declining LTI" if lti_slope and lti_slope < -0.3 else "Stable structure",
                sl_col, sl_col,
                info="Year-over-year trend in the LTI share of total compensation (pp/year), estimated via linear regression. A positive slope means the company is progressively shifting pay toward long-term incentives — a positive governance signal."), unsafe_allow_html=True)
        with qa5:
            st.markdown(kpi_html(
                f"{gfc_chg:+.1f}% / {covid_chg:+.1f}%" if gfc_chg is not None and covid_chg is not None else (f"{gfc_chg:+.1f}% / —" if gfc_chg is not None else "—"),
                "Crisis Pay Change",
                "GFC 2009 / COVID 2020",
                RED if (gfc_chg and gfc_chg > 5) or (covid_chg and covid_chg > 5) else GREEN,
                RED if (gfc_chg and gfc_chg > 5) or (covid_chg and covid_chg > 5) else GREEN,
                info="Year-on-year change in exec pay during the two major crises: Global Financial Crisis (2009) and COVID-19 (2020). Positive values mean pay rose during downturns — a strong pay-performance disconnect signal."), unsafe_allow_html=True)

        # Beta scatter + LTI trend line
        if pay_beta is not None or lti_slope is not None:
            col_b, col_lti = st.columns(2, gap="large")
            with col_b:
                st.markdown(sec_html("Pay-Performance Scatter (β regression)",
                    "Each dot = one year · slope = pay-β · flat or negative slope = pay decoupled from earnings"), unsafe_allow_html=True)
                if pay_beta is not None:
                    x_vals = co_clean["ebit_yoy_pct"].clip(-100, 200)
                    y_vals = co_clean["comp_yoy_pct"].clip(-100, 200)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                    y_line = pay_beta * x_line + float(np.polyfit(x_vals, y_vals, 1)[1])
                    fig_b = go.Figure()
                    fig_b.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="markers+text",
                        marker=dict(size=8, color=ORANGE, opacity=0.8),
                        text=co_clean["year"].astype(str), textposition="top center",
                        textfont=dict(size=7), name="Year"))
                    fig_b.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                        line=dict(color=NAVY, width=2, dash="dash"),
                        name=f"β={pay_beta:.2f}  r={pay_corr:.2f}"))
                    fig_b.add_hline(y=0, line_color=GRAY, line_width=1)
                    fig_b.add_vline(x=0, line_color=GRAY, line_width=1)
                    fig_b.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(title="EBIT Growth (%)", gridcolor=GRAYLT),
                        yaxis=dict(title="Exec Pay Growth (%)", gridcolor=GRAYLT),
                        legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=9)))
                    st.plotly_chart(fig_b, use_container_width=True)
            with col_lti:
                st.markdown(sec_html("LTI Share Trend",
                    "Is the board moving toward long-term incentives over time?"), unsafe_allow_html=True)
                if lti_slope is not None:
                    lti_yrs = lti_clean["year"].values
                    lti_line = lti_slope * np.arange(len(lti_yrs)) + float(np.polyfit(range(len(lti_yrs)), lti_clean["lti_pct"].values, 1)[1])
                    fig_lti = go.Figure()
                    fig_lti.add_trace(go.Bar(x=lti_yrs, y=lti_clean["lti_pct"],
                        marker_color=[GREEN if v > 35 else AMBER if v > 20 else RED for v in lti_clean["lti_pct"]],
                        name="LTI %"))
                    fig_lti.add_trace(go.Scatter(x=lti_yrs, y=lti_line, mode="lines",
                        line=dict(color=NAVY, width=2, dash="dash"),
                        name=f"Trend {lti_slope:+.2f}pp/yr"))
                    fig_lti.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                        plot_bgcolor="white", paper_bgcolor="white",
                        yaxis=dict(title="LTI Share (%)", gridcolor=GRAYLT),
                        xaxis=dict(gridcolor=GRAYLT),
                        legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=9)))
                    st.plotly_chart(fig_lti, use_container_width=True)
    else:
        st.info(f"Detailed feature data not available for {sel_co}.")


# ─────────────────────────────────────────────
# STAKEHOLDER 2 — BOARD & HR
# ─────────────────────────────────────────────
def show_board_hr():
    sidebar_nav("landing", "← Back to Home")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    in_mu    = sel_co in MU_COS
    co_df    = df[df["company_shortname"] == sel_co].sort_values("year")
    co_yr    = df[(df["company_shortname"] == sel_co) & (df["year"] == sel_year)]
    co_mu_yr = mu[(mu["company_shortname"] == sel_co) & (mu["year"] == sel_year)]
    year_mu  = mu[mu["year"] == sel_year].sort_values("total_comp_bt", ascending=False)

    st.markdown(f"""<div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">🏛 Board & HR View</div>
                <div style="font-size:1.8rem;font-weight:900;color:white;letter-spacing:-.5px;margin:4px 0;">{sel_co}</div>
                <div style="font-size:.82rem;color:#94a3b8;">Peer Benchmarking · Compensation Structure · Market Positioning · {sel_year}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    show_universal_snapshot(sel_co, sel_year)
    stakeholder_divider("Board & HR View — Compensation Structure & Market Benchmarking")

    st.markdown("""<div class="insight">
        <strong>What does the Board & HR Committee care about?</strong> Setting defensible compensation
        requires peer benchmarking, understanding where the company stands vs. the model expectation,
        and validating the STI/LTI structure against sector peers. Overpaying risks shareholder backlash
        at Say-on-Pay; underpaying risks talent loss. The OLS model provides an objective anchor.
    </div>""", unsafe_allow_html=True)

    rank   = int(co_mu_yr["peer_rank_total_comp"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["peer_rank_total_comp"].notna().any() else None
    pctile = float(co_mu_yr["peer_pct_total_comp"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["peer_pct_total_comp"].notna().any() else None
    aep    = float(co_mu_yr["actual_vs_expected_pct"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["actual_vs_expected_pct"].notna().any() else None
    n_exec = int(co_yr["n_executives"].iloc[0]) if len(co_yr) > 0 and "n_executives" in co_yr.columns and co_yr["n_executives"].notna().any() else None
    ceo_pr = float(co_yr["ceo_board_premium_ratio"].iloc[0]) if len(co_yr) > 0 and "ceo_board_premium_ratio" in co_yr.columns and co_yr["ceo_board_premium_ratio"].notna().any() else None
    _, t_col, t_lbl = traffic(aep)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(kpi_html(f"#{rank}" if rank else "—", "DAX Peer Rank",
        f"of {len(year_mu)} companies", NAVY,
        info="Absolute compensation rank among all DAX companies for the selected year. Lower = higher total board pay. Useful for the compensation committee to position the company against explicit peer benchmarks."), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html(f"{aep:+.0f}%" if aep is not None else "—", "vs. Model Expectation",
        t_lbl, t_col, t_col,
        info="Percentage deviation of actual pay from the OLS model's size-, sector-, and board-size-adjusted expectation. Used by compensation committees to justify pay decisions — values above +15% require a compelling narrative."), unsafe_allow_html=True)
    with k3: st.markdown(kpi_html(str(n_exec) if n_exec else "—", "Board Members",
        f"Exec. board size {sel_year}", NAVY,
        info="Number of executive board members in the selected year. Board size directly affects total compensation spend and is a significant driver in the OLS model (larger boards = higher total pay, but usually lower pay-per-head)."), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html(f"{ceo_pr:.1f}x" if ceo_pr else "—", "CEO/Board Premium",
        "⚠ High (>2.5x)" if ceo_pr and ceo_pr > 2.5 else "Within range",
        RED if ceo_pr and ceo_pr > 2.5 else GREEN, RED if ceo_pr and ceo_pr > 2.5 else GREEN,
        info="CEO total compensation divided by the average compensation of other executive board members. A premium >2.5x is at the high end for DAX companies and can trigger questions about pay equity within the board."), unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")
    with col_l:
        st.markdown(sec_html(f"Compensation Structure vs. DAX Average — {sel_year}",
            "Fixed / STI / LTI share compared to DAX peers — deviation > 15pp warrants discussion"), unsafe_allow_html=True)
        if in_model and len(co_yr) > 0:
            fix_co  = float(co_yr["fixed_pct"].iloc[0]) if co_yr["fixed_pct"].notna().any() else None
            sti_co  = float(co_yr["sti_pct"].iloc[0]) if co_yr["sti_pct"].notna().any() else None
            lti_co  = float(co_yr["lti_pct"].iloc[0]) if co_yr["lti_pct"].notna().any() else None
            dax_yr  = df[df["year"] == sel_year]
            fix_dax = float(dax_yr["fixed_pct"].mean()) if "fixed_pct" in dax_yr.columns else None
            sti_dax = float(dax_yr["sti_pct"].mean()) if "sti_pct" in dax_yr.columns else None
            lti_dax = float(dax_yr["lti_pct"].mean()) if "lti_pct" in dax_yr.columns else None

            if all(v is not None for v in [fix_co, sti_co, lti_co, fix_dax, sti_dax, lti_dax]):
                cats = ["Fixed", "STI (Short-term)", "LTI (Long-term)"]
                fig = go.Figure()
                fig.add_trace(go.Bar(name=sel_co, x=cats, y=[fix_co, sti_co, lti_co],
                    marker_color=ORANGE, text=[f"{v:.0f}%" for v in [fix_co, sti_co, lti_co]],
                    textposition="outside"))
                fig.add_trace(go.Bar(name="DAX Average", x=cats, y=[fix_dax, sti_dax, lti_dax],
                    marker_color="#94a3b8", text=[f"{v:.0f}%" for v in [fix_dax, sti_dax, lti_dax]],
                    textposition="outside"))
                fig.update_layout(height=280, barmode="group", margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Share (%)", gridcolor=GRAYLT),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01))
                st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(sec_html("Peer Rank & Say-on-Pay Assessment"), unsafe_allow_html=True)
        if in_model and len(co_yr) > 0:
            dax_yr  = df[df["year"] == sel_year]
            lti_co  = float(co_yr["lti_pct"].iloc[0]) if co_yr["lti_pct"].notna().any() else None
            sti_co  = float(co_yr["sti_pct"].iloc[0]) if co_yr["sti_pct"].notna().any() else None
            fix_co  = float(co_yr["fixed_pct"].iloc[0]) if co_yr["fixed_pct"].notna().any() else None
            rows = [
                ("DAX Rank", f"#{rank} of {len(year_mu)}" if rank else "—", NAVY),
                ("Percentile", f"Top {100-pctile:.0f}%" if pctile else "—", ORANGE),
                ("Fixed vs. DAX avg", f"{fix_co:.0f}% vs. {dax_yr['fixed_pct'].mean():.0f}%" if fix_co else "—",
                 RED if fix_co and abs(fix_co - dax_yr["fixed_pct"].mean()) > 15 else GRAY),
                ("STI vs. DAX avg", f"{sti_co:.0f}% vs. {dax_yr['sti_pct'].mean():.0f}%" if sti_co else "—", GRAY),
                ("LTI vs. DAX avg", f"{lti_co:.0f}% vs. {dax_yr['lti_pct'].mean():.0f}%" if lti_co else "—",
                 RED if lti_co and abs(lti_co - dax_yr["lti_pct"].mean()) > 15 else GRAY),
                ("Say-on-Pay Risk", "🔴 High — >+15% above model" if aep and aep > 15 else "🟢 Low", RED if aep and aep > 15 else GREEN),
            ]
            st.markdown('<div style="background:white;border-radius:12px;padding:14px 16px;border:1px solid #e2e8f0;">', unsafe_allow_html=True)
            for lbl, val, col_c in rows:
                st.markdown(f'<div class="metric-row"><span style="color:{GRAY};">{lbl}</span>'
                            f'<span style="font-weight:700;color:{col_c};">{val}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(f"Detailed structure data not available for {sel_co}.")

    if in_model and len(co_df) > 0 and all(c in co_df.columns for c in ["fixed_pct", "sti_pct", "lti_pct"]):
        st.markdown(sec_html("Compensation Structure Over Time",
            "Shifting LTI share signals evolving governance standards — rising LTI aligns with long-term value creation"), unsafe_allow_html=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="Fixed", x=co_df["year"], y=co_df["fixed_pct"], marker_color="#3b82f6"))
        fig3.add_trace(go.Bar(name="STI", x=co_df["year"], y=co_df["sti_pct"], marker_color=ORANGE))
        fig3.add_trace(go.Bar(name="LTI", x=co_df["year"], y=co_df["lti_pct"], marker_color=NAVY))
        fig3.update_layout(barmode="stack", height=240, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Share (%)", range=[0, 100], gridcolor=GRAYLT),
            xaxis=dict(gridcolor=GRAYLT),
            legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Advanced Board & HR metrics ────────────────────────────────
    st.markdown(sec_html("Advanced Benchmarking & Decision Support",
        "P25/P50/P75 positioning · AGM defensibility score · model drift · succession cost estimate"), unsafe_allow_html=True)

    co_mu_all = mu[mu["company_shortname"] == sel_co].sort_values("year")
    actual_comp = float(co_mu_yr["total_comp_bt"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["total_comp_bt"].notna().any() else None

    # DAX percentile distribution for the year
    year_comps = mu[mu["year"] == sel_year]["total_comp_bt"].dropna()
    p25 = float(year_comps.quantile(0.25)) if len(year_comps) > 0 else None
    p50 = float(year_comps.quantile(0.50)) if len(year_comps) > 0 else None
    p75 = float(year_comps.quantile(0.75)) if len(year_comps) > 0 else None

    # Sector-adjusted comp per exec
    dax_yr_full = df[df["year"] == sel_year]
    sector_co = co_df["sector"].iloc[0] if len(co_df) > 0 and "sector" in co_df.columns else None
    per_exec_co = float(co_yr["comp_per_exec"].iloc[0]) if len(co_yr) > 0 and "comp_per_exec" in co_yr.columns and co_yr["comp_per_exec"].notna().any() else (actual_comp / n_exec if actual_comp and n_exec else None)
    per_exec_sector = float(dax_yr_full[dax_yr_full["sector"] == sector_co]["comp_per_exec"].median()) if sector_co and "comp_per_exec" in dax_yr_full.columns else None

    # AGM defensibility score (0–100, higher = more defensible)
    def_score = None
    if aep is not None and pctile is not None:
        dev_pts = max(0, min(40, 40 - abs(aep) * 0.5))   # 40 pts if on model, 0 if >80% off
        rank_pts = min(30, (1 - (rank / len(year_mu) if rank and len(year_mu) > 0 else 0.5)) * 30)  # near median = good
        # Structure alignment: avg distance from DAX avg for each component
        if in_model and len(co_yr) > 0:
            dists = []
            for col_ in ["fixed_pct", "sti_pct", "lti_pct"]:
                if col_ in co_yr.columns and col_ in dax_yr_full.columns and co_yr[col_].notna().any():
                    dists.append(abs(float(co_yr[col_].iloc[0]) - dax_yr_full[col_].mean()))
            struct_pts = max(0, 30 - (np.mean(dists) * 0.5 if dists else 15))
        else:
            struct_pts = 15
        def_score = dev_pts + rank_pts + struct_pts

    # Model drift slope (pp/yr in actual_vs_expected_pct)
    drift_slope = None
    aep_series = co_mu_all.dropna(subset=["actual_vs_expected_pct"])
    if len(aep_series) > 4:
        drift_slope = float(np.polyfit(range(len(aep_series)), aep_series["actual_vs_expected_pct"].values, 1)[0])

    # Succession cost estimate using log_board_size coefficient
    succ_plus1, succ_minus1 = None, None
    pred_val = float(co_mu_yr["pred_comp"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["pred_comp"].notna().any() else None
    if pred_val and n_exec and len(coefs) > 0:
        coef_row = coefs[coefs["feature"] == "log_board_size"]
        if len(coef_row) > 0:
            eff_pct = float(coef_row["exp_effect_pct"].iloc[0])
            raw_beta = np.log(1 + eff_pct / 100)
            succ_plus1  = pred_val * np.exp(raw_beta * (np.log(n_exec + 1) - np.log(n_exec)))
            succ_minus1 = pred_val * np.exp(raw_beta * (np.log(max(1, n_exec - 1)) - np.log(n_exec)))

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        pos_lbl = "At P75 (top quartile)" if actual_comp and p75 and actual_comp > p75 else \
                  "At P50–P75" if actual_comp and p50 and actual_comp > p50 else \
                  "At P25–P50" if actual_comp and p25 and actual_comp > p25 else "Below P25"
        pos_col = RED if actual_comp and p75 and actual_comp > p75 else ORANGE if actual_comp and p50 and actual_comp > p50 else GREEN
        st.markdown(kpi_html(
            f"P{int(pctile)}" if pctile else "—", "DAX Percentile Position", pos_lbl, pos_col, pos_col,
            info="Exact percentile of this company's total compensation in the DAX distribution for the selected year, with P25/P50/P75 quartile boundaries. Compensation committees typically target P50–P75 to attract talent without triggering governance pushback."), unsafe_allow_html=True)
    with b2:
        st.markdown(kpi_html(
            f"€{per_exec_co:.0f}K" if per_exec_co else "—", "Comp per Exec Member",
            f"Sector med. €{per_exec_sector:.0f}K" if per_exec_sector else f"Board of {n_exec}" if n_exec else "—",
            RED if per_exec_co and per_exec_sector and per_exec_co > per_exec_sector * 1.25 else GREEN if per_exec_co else GRAY,
            RED if per_exec_co and per_exec_sector and per_exec_co > per_exec_sector * 1.25 else GREEN if per_exec_co else GRAY,
            info="Average compensation per executive board member, compared to the sector median. Values >25% above sector median suggest the company is setting a generous per-head baseline, which compounds with board size to drive total cost."), unsafe_allow_html=True)
    with b3:
        def_col = GREEN if def_score and def_score > 65 else AMBER if def_score and def_score > 40 else RED if def_score else GRAY
        def_lbl = "✅ Highly defensible" if def_score and def_score > 65 else "⚠ Moderate risk" if def_score and def_score > 40 else "⚠ Hard to defend" if def_score else "—"
        st.markdown(kpi_html(
            f"{def_score:.0f}/100" if def_score else "—", "AGM Defensibility Score",
            def_lbl, def_col, def_col,
            info="Composite score (0–100) estimating how easily the compensation committee can defend this pay package at the AGM. Combines: closeness to model expectation (40pts), DAX rank position (30pts), and structural consistency vs. peers (30pts)."), unsafe_allow_html=True)
    with b4:
        drift_col = RED if drift_slope and drift_slope > 2 else GREEN if drift_slope and drift_slope < -2 else GRAY
        drift_lbl = "⚠ Drifting above model" if drift_slope and drift_slope > 2 else "✅ Moving toward model" if drift_slope and drift_slope < -2 else "Stable"
        st.markdown(kpi_html(
            f"{drift_slope:+.1f}pp/yr" if drift_slope is not None else "—", "Model Deviation Drift",
            drift_lbl, drift_col, drift_col,
            info="Year-over-year trend in the deviation from model expectation (pp/year). A rising drift means pay is progressively exceeding the model's fair value over time — a warning sign that compensation practices are decoupling from fundamentals."), unsafe_allow_html=True)

    # P25/P50/P75 chart + succession cost
    col_pct, col_succ = st.columns(2, gap="large")
    with col_pct:
        st.markdown(sec_html(f"DAX Compensation Distribution — {sel_year}",
            "Where does this company sit in the full DAX distribution?"), unsafe_allow_html=True)
        if len(year_comps) > 0:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=year_comps, nbinsx=15,
                marker_color="#e2e8f0", name="DAX Distribution"))
            if actual_comp:
                fig_dist.add_vline(x=actual_comp, line_color=ORANGE, line_width=2.5,
                    annotation_text=sel_co, annotation_font=dict(color=ORANGE, size=9))
            for val, lbl, col_v in [(p25, "P25", "#94a3b8"), (p50, "P50", GRAY), (p75, "P75", NAVY)]:
                if val:
                    fig_dist.add_vline(x=val, line_color=col_v, line_width=1, line_dash="dot",
                        annotation_text=lbl, annotation_font=dict(color=col_v, size=8))
            fig_dist.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Total Comp. (€K)", gridcolor=GRAYLT),
                yaxis=dict(title="# Companies", gridcolor=GRAYLT),
                showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
            if p25 and p50 and p75:
                st.markdown(f'<div style="font-size:.78rem;color:{GRAY};text-align:center;">P25: €{p25/1000:.1f}M &nbsp;·&nbsp; P50: €{p50/1000:.1f}M &nbsp;·&nbsp; P75: €{p75/1000:.1f}M</div>', unsafe_allow_html=True)

    with col_succ:
        st.markdown(sec_html("Board Size Scenario (Succession Cost Estimate)",
            "Model-implied total compensation if board grows or shrinks by one member"), unsafe_allow_html=True)
        if succ_plus1 and succ_minus1 and pred_val:
            fig_succ = go.Figure(go.Bar(
                x=[f"n–1 ({n_exec - 1} members)", f"Current (n={n_exec})", f"n+1 ({n_exec + 1} members)"],
                y=[succ_minus1, pred_val, succ_plus1],
                marker_color=["#3b82f6", ORANGE, RED],
                text=[f"€{v/1000:.1f}M" for v in [succ_minus1, pred_val, succ_plus1]],
                textposition="outside"))
            fig_succ.update_layout(height=240, margin=dict(l=0, r=0, t=10, b=40),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Expected Comp. (€K)", gridcolor=GRAYLT),
                showlegend=False)
            st.plotly_chart(fig_succ, use_container_width=True)
            delta_up   = succ_plus1 - pred_val
            delta_down = pred_val - succ_minus1
            st.markdown(f'<div style="font-size:.78rem;color:{GRAY};text-align:center;">+1 member adds est. <strong style="color:{RED};">€{delta_up/1000:.1f}M</strong> · −1 member saves est. <strong style="color:{GREEN};">€{delta_down/1000:.1f}M</strong></div>', unsafe_allow_html=True)
        else:
            st.info("Succession scenario requires model prediction data.")


# ─────────────────────────────────────────────
# STAKEHOLDER 3 — EMPLOYEES & LABOR
# ─────────────────────────────────────────────
def show_employees_labor():
    sidebar_nav("landing", "← Back to Home")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    co_df  = df[df["company_shortname"] == sel_co].sort_values("year")
    co_esg = esg[esg["company_shortname"] == sel_co].sort_values("year") if len(esg) > 0 else pd.DataFrame()

    st.markdown(f"""<div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">👷 Employees & Labor View</div>
                <div style="font-size:1.8rem;font-weight:900;color:white;letter-spacing:-.5px;margin:4px 0;">{sel_co}</div>
                <div style="font-size:.82rem;color:#94a3b8;">Pay Fairness · Headcount vs. Exec Pay · Bad Times · CEO/Worker Ratio</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    show_universal_snapshot(sel_co, sel_year)
    stakeholder_divider("Employees & Labor View — Fairness & Pay Distribution")

    st.markdown("""<div class="insight">
        <strong>What do employees and labor representatives care about?</strong> Did executive bonuses
        rise while jobs were cut? Does the CEO earn 50× or 150× the average worker? These questions drive
        Works Council scrutiny and collective bargaining leverage. Bad Times events — executive pay rising
        despite falling EBIT <em>and</em> headcount — are the clearest evidence of unfair distribution.
    </div>""", unsafe_allow_html=True)

    if in_model and len(co_df) > 0:
        sz_cnt   = int(co_df["schlechte_zeiten"].sum()) if "schlechte_zeiten" in co_df.columns else 0
        avg_yoy  = float(co_df["comp_yoy_pct"].mean()) if "comp_yoy_pct" in co_df.columns and co_df["comp_yoy_pct"].notna().any() else None
        avg_empl = float(co_df["empl_yoy_pct"].mean()) if "empl_yoy_pct" in co_df.columns and co_df["empl_yoy_pct"].notna().any() else None
        fairness_gap = (avg_yoy - avg_empl) if avg_yoy and avg_empl and pd.notna(avg_empl) else None
        co_esg_latest = co_esg.iloc[[-1]] if len(co_esg) > 0 else pd.DataFrame()
        w_ratio = float(co_esg_latest["ceo_worker_ratio"].iloc[0]) if len(co_esg_latest) > 0 and "ceo_worker_ratio" in co_esg_latest.columns and co_esg_latest["ceo_worker_ratio"].notna().any() else None

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(kpi_html(str(sz_cnt), "Bad Times Events",
            "⚠ Pay ↑ while EBIT+Jobs ↓" if sz_cnt > 0 else "✅ None detected",
            RED if sz_cnt > 0 else GREEN, RED if sz_cnt > 0 else GREEN,
            info="Number of years where executive pay increased while both EBIT AND employee headcount fell simultaneously. From a workforce perspective, these events represent the starkest form of pay-for-performance failure: management rewarded while workers bear the cost."), unsafe_allow_html=True)
        with k2: st.markdown(kpi_html(
            f"{fairness_gap:+.1f}pp" if fairness_gap is not None else "—",
            "Pay vs. Headcount Gap",
            "⚠ Exec pay grew much faster" if fairness_gap and fairness_gap > 5 else "✅ Within range",
            RED if fairness_gap and fairness_gap > 5 else GREEN,
            RED if fairness_gap and fairness_gap > 5 else GREEN,
            info="Average annual exec pay growth minus average annual employee headcount growth (in percentage points). A large positive gap suggests executive pay has systematically outpaced workforce expansion, raising fairness concerns and internal equity issues."), unsafe_allow_html=True)
        with k3: st.markdown(kpi_html(
            f"{avg_empl:+.1f}%/yr" if avg_empl and pd.notna(avg_empl) else "—",
            "Avg. Headcount Growth", "Historical p.a. trend", NAVY,
            info="Average annual percentage change in total employee headcount over the company's history in the dataset. Contextualize executive pay growth against actual workforce growth to assess whether labor investment matches executive rewards."), unsafe_allow_html=True)
        with k4: st.markdown(kpi_html(
            f"{w_ratio:.0f}x" if w_ratio else "—",
            "CEO/Worker Pay Ratio",
            "⚠ Very High" if w_ratio and w_ratio > 60 else "Income inequality",
            RED if w_ratio and w_ratio > 60 else AMBER if w_ratio and w_ratio > 40 else NAVY,
            RED if w_ratio and w_ratio > 60 else AMBER if w_ratio and w_ratio > 40 else GRAY,
            info="CEO total compensation divided by the median employee wage. Required disclosure under CSRD. The DAX average is around 50–80x; values above 100x are outliers. High ratios correlate with lower employee satisfaction and are a reputational risk."), unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2], gap="large")
        with col_l:
            st.markdown(sec_html("Exec Pay Growth vs. Headcount Growth",
                "Orange line = exec comp growth · Blue bars = headcount growth · Red shading = Bad Times years"), unsafe_allow_html=True)
            if "empl_yoy_pct" in co_df.columns:
                co_clean = co_df.dropna(subset=["comp_yoy_pct", "empl_yoy_pct"])
                fig = go.Figure()
                fig.add_trace(go.Bar(x=co_clean["year"], y=co_clean["empl_yoy_pct"].clip(-50, 50),
                    name="Headcount Growth %", marker_color="#3b82f6", opacity=0.7))
                fig.add_trace(go.Scatter(x=co_clean["year"], y=co_clean["comp_yoy_pct"].clip(-100, 150),
                    line=dict(color=ORANGE, width=2.5), mode="lines+markers", marker=dict(size=6),
                    name="Exec Pay Growth %"))
                fig.add_hline(y=0, line_color=GRAY, line_width=1)
                if "schlechte_zeiten" in co_df.columns:
                    for yr in co_df[co_df["schlechte_zeiten"] == 1]["year"].tolist():
                        fig.add_vrect(x0=yr - 0.4, x1=yr + 0.4, fillcolor="rgba(220,38,38,.12)",
                            line_width=0, annotation_text="BT", annotation_font=dict(size=7, color=RED))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Growth (%)", gridcolor=GRAYLT),
                    xaxis=dict(gridcolor=GRAYLT),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                    hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown(sec_html("Bad Times Events"), unsafe_allow_html=True)
            if "schlechte_zeiten" in co_df.columns:
                sz_rows = co_df[co_df["schlechte_zeiten"] == 1]
                if len(sz_rows) > 0:
                    for _, row in sz_rows.iterrows():
                        ebit_s = f"EBIT: {row['ebit_yoy_pct']:+.1f}% · " if "ebit_yoy_pct" in row.index and pd.notna(row.get("ebit_yoy_pct")) else ""
                        empl_s = f"Headcount: {row['empl_yoy_pct']:+.1f}%" if "empl_yoy_pct" in row.index and pd.notna(row.get("empl_yoy_pct")) else ""
                        st.markdown(f"""<div class="flag-red">
                            <strong>Year {int(row['year'])}:</strong> Exec Pay {row['comp_yoy_pct']:+.1f}%
                            &nbsp;·&nbsp; {ebit_s}{empl_s}
                        </div>""", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="flag-green">✅ No Bad Times events detected.</div>', unsafe_allow_html=True)

        if len(esg) > 0 and "ceo_worker_ratio" in esg.columns:
            st.markdown(sec_html("CEO/Worker Pay Ratio — DAX Comparison",
                "Latest available year per company · Orange = selected company"), unsafe_allow_html=True)
            latest_rat = (esg.sort_values("year").groupby("company_shortname").last()
                          .reset_index()[["company_shortname", "ceo_worker_ratio"]]
                          .sort_values("ceo_worker_ratio", ascending=True).dropna())
            if len(latest_rat) > 0:
                colors_r = [ORANGE if c == sel_co else (RED if v > 65 else AMBER if v > 45 else "#94a3b8")
                            for c, v in zip(latest_rat["company_shortname"], latest_rat["ceo_worker_ratio"])]
                fig_r = go.Figure(go.Bar(
                    x=latest_rat["ceo_worker_ratio"], y=latest_rat["company_shortname"],
                    orientation="h", marker_color=colors_r,
                    text=[f"{v:.0f}x" for v in latest_rat["ceo_worker_ratio"]],
                    textposition="outside", textfont=dict(size=8),
                    hovertemplate="%{y}: %{x:.0f}x<extra></extra>"))
                med = latest_rat["ceo_worker_ratio"].median()
                fig_r.add_vline(x=med, line_dash="dot", line_color=GRAY, line_width=1.5,
                    annotation_text=f"Median {med:.0f}x", annotation_font=dict(size=8, color=GRAY))
                fig_r.update_layout(height=440, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Exec/Worker Pay (x)", gridcolor=GRAYLT),
                    yaxis=dict(tickfont=dict(size=9)))
                st.plotly_chart(fig_r, use_container_width=True)

        # ── Advanced Labor metrics ──────────────────────────────────────
        st.markdown(sec_html("Cumulative Divergence, Pay-Headcount Elasticity & Restructuring Flag",
            "Long-run fairness signals across the full data period"), unsafe_allow_html=True)

        col_div, col_ela = st.columns([3, 2], gap="large")
        with col_div:
            st.markdown(sec_html("Cumulative Indexed Growth — Exec Pay vs. Headcount",
                "Base 100 = first common year · gap between lines = accumulated fairness divergence"), unsafe_allow_html=True)
            if "empl_yoy_pct" in co_df.columns and "comp_yoy_pct" in co_df.columns:
                idx_df = co_df.dropna(subset=["comp_yoy_pct", "empl_yoy_pct"]).sort_values("year").copy()
                idx_df["comp_idx"] = np.cumprod(1 + idx_df["comp_yoy_pct"].clip(-60, 120) / 100) * 100
                idx_df["empl_idx"] = np.cumprod(1 + idx_df["empl_yoy_pct"].clip(-30, 30) / 100) * 100
                final_gap = float(idx_df["comp_idx"].iloc[-1] - idx_df["empl_idx"].iloc[-1]) if len(idx_df) > 0 else None
                fig_div = go.Figure()
                fig_div.add_trace(go.Scatter(x=idx_df["year"], y=idx_df["empl_idx"],
                    line=dict(color="#3b82f6", width=2), mode="lines",
                    name="Headcount Index", hovertemplate="Headcount: %{y:.0f}<extra></extra>"))
                fig_div.add_trace(go.Scatter(x=idx_df["year"], y=idx_df["comp_idx"],
                    line=dict(color=ORANGE, width=2.5), mode="lines",
                    fill="tonexty", fillcolor="rgba(249,115,22,.07)",
                    name="Exec Pay Index", hovertemplate="Exec Pay: %{y:.0f}<extra></extra>"))
                fig_div.add_hline(y=100, line_color=GRAY, line_width=1, line_dash="dot",
                    annotation_text="Base 100", annotation_font=dict(size=8))
                fig_div.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Index (Base 100)", gridcolor=GRAYLT),
                    xaxis=dict(gridcolor=GRAYLT),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                    hovermode="x unified")
                st.plotly_chart(fig_div, use_container_width=True)
                if final_gap is not None:
                    gap_col = RED if final_gap > 50 else AMBER if final_gap > 20 else GREEN
                    st.markdown(f'<div class="flag-{"red" if final_gap > 50 else "amber" if final_gap > 20 else "green"}">'
                                f'Cumulative divergence: exec pay index is <strong>{final_gap:+.0f} points</strong> above the headcount index over this period.</div>', unsafe_allow_html=True)

        with col_ela:
            st.markdown(sec_html("Pay-Headcount Elasticity"), unsafe_allow_html=True)
            ela_df = co_df.dropna(subset=["comp_yoy_pct", "empl_yoy_pct"])
            if len(ela_df) > 4:
                beta_ela = float(np.polyfit(ela_df["empl_yoy_pct"].clip(-30, 30),
                                            ela_df["comp_yoy_pct"].clip(-60, 120), 1)[0])
                corr_ela = float(ela_df["comp_yoy_pct"].corr(ela_df["empl_yoy_pct"]))
                ela_col = RED if beta_ela < -0.3 else AMBER if beta_ela < 0.1 else GREEN
                ela_lbl = "⚠ Pay rises as jobs fall" if beta_ela < -0.3 else "⚠ Weak coupling" if beta_ela < 0.1 else "✅ Co-moves with employment"
                st.markdown(f"""<div style="background:{NAVY};border-radius:12px;padding:16px;text-align:center;margin-bottom:10px;">
                    <div style="font-size:2.2rem;font-weight:800;color:{ela_col};line-height:1;">{beta_ela:.2f}</div>
                    <div style="font-size:.68rem;color:#94a3b8;text-transform:uppercase;margin-top:4px;">Pay-Headcount Elasticity β</div>
                    <div style="font-size:.78rem;color:{ela_col};margin-top:6px;font-weight:600;">{ela_lbl}</div>
                    <div style="font-size:.72rem;color:#64748b;margin-top:4px;">r = {corr_ela:.2f}</div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f'<div style="font-size:.76rem;color:{GRAY};line-height:1.55;">Interpretation: when headcount grows by 1%, exec pay moves by <strong>{beta_ela:.2f}%</strong>. A negative β means executive pay tends to rise precisely when the workforce shrinks.</div>', unsafe_allow_html=True)

            # Restructuring-bonus flag
            st.markdown(sec_html("Restructuring-Bonus Flag", "Years with headcount cuts ≥ 2%"), unsafe_allow_html=True)
            restr = co_df[co_df["empl_yoy_pct"] < -2].copy() if "empl_yoy_pct" in co_df.columns else pd.DataFrame()
            if len(restr) > 0:
                bonus_col_name = "one_year_bonus_bt" if "one_year_bonus_bt" in restr.columns else None
                restr["bonus_up"] = False
                if bonus_col_name:
                    restr["bonus_prev"] = restr[bonus_col_name].shift(1)
                    restr["bonus_up"] = restr[bonus_col_name] > restr["bonus_prev"]
                for _, row in restr.iterrows():
                    empl_s = f"Headcount: {row['empl_yoy_pct']:+.1f}%"
                    comp_s = f"Exec Pay: {row['comp_yoy_pct']:+.1f}%"
                    bonus_flag = " · 🚨 Bonus ↑" if row.get("bonus_up", False) else ""
                    sev = "flag-red" if row.get("bonus_up", False) or (row["comp_yoy_pct"] > 0) else "flag-amber"
                    st.markdown(f'<div class="{sev}"><strong>{int(row["year"])}:</strong> {empl_s} · {comp_s}{bonus_flag}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="flag-green">✅ No major restructuring years (headcount cut ≥2%) detected.</div>', unsafe_allow_html=True)

    else:
        st.info(f"Detailed feature data not available for {sel_co}.")


# ─────────────────────────────────────────────
# STAKEHOLDER 4 — ACCOUNTABILITY ACTORS
# ─────────────────────────────────────────────
def show_accountability_actors():
    sidebar_nav("landing", "← Back to Home")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    co_df    = df[df["company_shortname"] == sel_co].sort_values("year")
    co_yr    = df[(df["company_shortname"] == sel_co) & (df["year"] == sel_year)]
    co_mu_yr = mu[(mu["company_shortname"] == sel_co) & (mu["year"] == sel_year)]
    co_esg   = esg[esg["company_shortname"] == sel_co] if len(esg) > 0 else pd.DataFrame()

    st.markdown(f"""<div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">⚖️ Accountability Actors View</div>
                <div style="font-size:1.8rem;font-weight:900;color:white;letter-spacing:-.5px;margin:4px 0;">{sel_co}</div>
                <div style="font-size:.82rem;color:#94a3b8;">BaFin · Proxy Advisors · NGOs · Outlier Detection · DAX-wide Ranking</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    show_universal_snapshot(sel_co, sel_year)
    stakeholder_divider("Accountability View — Outliers, Violations & DAX-wide Red Flags")

    st.markdown("""<div class="insight">
        <strong>What do accountability actors care about?</strong> Proxy advisors (ISS, Glass Lewis) and BaFin
        identify the most egregious pay outliers across the full market — not just one company in isolation,
        but who ranks worst overall. NGOs and journalists focus on Bad Times violations and ESG pay-washing:
        companies claiming sustainability credentials without linking executive pay to measurable targets.
    </div>""", unsafe_allow_html=True)

    aep     = float(co_mu_yr["actual_vs_expected_pct"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["actual_vs_expected_pct"].notna().any() else None
    sz_cnt  = int(co_df["schlechte_zeiten"].sum()) if in_model and "schlechte_zeiten" in co_df.columns else 0
    is_anom = bool(co_yr["is_anomaly"].iloc[0]) if in_model and len(co_yr) > 0 and "is_anomaly" in co_yr.columns and co_yr["is_anomaly"].notna().any() else None
    co_esg_latest = co_esg.iloc[[-1]] if len(co_esg) > 0 else pd.DataFrame()
    sti_esg = float(co_esg_latest["sti_esg_share"].iloc[0]) if len(co_esg_latest) > 0 and "sti_esg_share" in co_esg_latest.columns and co_esg_latest["sti_esg_share"].notna().any() else None
    lti_esg = float(co_esg_latest["lti_esg_share"].iloc[0]) if len(co_esg_latest) > 0 and "lti_esg_share" in co_esg_latest.columns and co_esg_latest["lti_esg_share"].notna().any() else None
    esg_tot = (sti_esg or 0) + (lti_esg or 0)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(kpi_html(
        f"{aep:+.0f}%" if aep is not None else "—", "vs. Model Expectation",
        "⚠ Significant Outlier" if aep and aep > 40 else "Above market" if aep and aep > 15 else "Normal",
        RED if aep and aep > 40 else AMBER if aep and aep > 15 else GREEN,
        RED if aep and aep > 40 else AMBER if aep and aep > 15 else GREEN,
        info="How much actual compensation deviates from the OLS model's fair-value estimate. Values above +40% typically trigger a 'vote against' recommendation from proxy advisors like ISS or Glass Lewis. Values above +15% warrant scrutiny."), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html(str(sz_cnt), "Bad Times Violations",
        "⚠ Pay-for-performance breach" if sz_cnt > 0 else "✅ None detected",
        RED if sz_cnt > 0 else GREEN, RED if sz_cnt > 0 else GREEN,
        info="Years where exec pay increased while both EBIT AND headcount fell. Each occurrence is a concrete, documentable pay-for-performance failure that accountability actors (regulators, activists, proxy advisors) can point to directly."), unsafe_allow_html=True)
    with k3: st.markdown(kpi_html(
        "🚨 Yes" if is_anom else ("✅ No" if is_anom is not None else "—"),
        f"Anomaly Flag {sel_year}",
        "Unusual comp. structure" if is_anom else "Within normal range",
        RED if is_anom else GREEN, RED if is_anom else GREEN,
        info="Statistical flag indicating an unusual compensation structure or level for the selected year, beyond what company characteristics explain. Anomalies may indicate one-time payments, structural changes, or governance failures worth investigating."), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html(
        f"{esg_tot:.0f}%" if esg_tot else "0%", "ESG in Pay",
        "⚠ ESG Pay-Washing Risk" if esg_tot == 0 else "CSRD-relevant",
        RED if esg_tot == 0 else ORANGE, RED if esg_tot == 0 else GREEN,
        info="Combined ESG-linked share of variable pay (STI + LTI). Zero ESG link is increasingly untenable under CSRD and signals potential 'pay-washing' — where sustainability claims in annual reports are not backed by actual incentive structures."), unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")
    with col_l:
        st.markdown(sec_html(f"DAX Overpayment Ranking — {sel_year}",
            "Companies sorted by excess pay vs. model — navy = selected company · red = significant outlier"), unsafe_allow_html=True)
        mu_yr = mu[mu["year"] == sel_year].dropna(subset=["actual_vs_expected_pct"]).sort_values("actual_vs_expected_pct", ascending=False)
        if len(mu_yr) > 0:
            bar_colors = [NAVY if c == sel_co else (RED if v > 40 else AMBER if v > 15 else "#4ade80" if v > -15 else "#2563eb")
                          for c, v in zip(mu_yr["company_shortname"], mu_yr["actual_vs_expected_pct"])]
            fig = go.Figure(go.Bar(
                x=mu_yr["actual_vs_expected_pct"], y=mu_yr["company_shortname"],
                orientation="h", marker_color=bar_colors,
                text=[f"{v:+.0f}%" for v in mu_yr["actual_vs_expected_pct"]],
                textposition="outside", textfont=dict(size=7),
                hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
            fig.add_vline(x=40, line_dash="dot", line_color=RED, line_width=1,
                annotation_text="Outlier Threshold +40%", annotation_font=dict(size=7, color=RED))
            fig.add_vline(x=0, line_color=GRAY, line_width=1)
            fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=30),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Excess Pay vs. Model (%)", gridcolor=GRAYLT),
                yaxis=dict(tickfont=dict(size=8)))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(sec_html("DAX-wide Bad Times Events (Top 10)"), unsafe_allow_html=True)
        if "schlechte_zeiten" in df.columns:
            sz_all = df[df["schlechte_zeiten"] == 1][
                ["company_shortname", "year", "comp_yoy_pct", "schlechte_zeiten_score"]
            ].sort_values("schlechte_zeiten_score", ascending=False).head(10)
            if len(sz_all) > 0:
                for _, row in sz_all.iterrows():
                    sc  = float(row["schlechte_zeiten_score"]) if pd.notna(row.get("schlechte_zeiten_score")) else 0
                    sev = "flag-red" if sc > 0.5 else "flag-amber"
                    is_sel = row["company_shortname"] == sel_co
                    bold_s = "font-weight:900;" if is_sel else ""
                    st.markdown(f'<div class="{sev}" style="{bold_s}"><strong>{row["company_shortname"]} '
                                f'({int(row["year"])}):</strong> Pay {row["comp_yoy_pct"]:+.1f}% · Severity {sc:.2f}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="flag-green">✅ No Bad Times events in dataset.</div>', unsafe_allow_html=True)

        st.markdown(sec_html("ESG Pay Integration — DAX Summary"), unsafe_allow_html=True)
        if len(esg) > 0:
            recent = esg[esg["year"] >= 2022]
            esg_bub = recent.groupby("company_shortname").agg(
                sti=("sti_esg_share", "mean"), lti=("lti_esg_share", "mean")).reset_index().fillna(0)
            esg_bub["total_esg"] = esg_bub["sti"] + esg_bub["lti"]
            n_esg  = int((esg_bub["total_esg"] > 0).sum())
            n_total = len(esg_bub)
            st.markdown(f"""<div style="background:{NAVY};border-radius:12px;padding:16px;text-align:center;margin-top:4px;">
                <div style="font-size:2.2rem;font-weight:900;color:{ORANGE};line-height:1;">{n_esg}<span style="font-size:1.1rem;color:#64748b;">/{n_total}</span></div>
                <div style="font-size:.7rem;color:#94a3b8;text-transform:uppercase;margin-top:4px;">DAX Companies with ESG Pay Link</div>
                <div style="font-size:.82rem;color:#f87171;margin-top:8px;font-weight:600;">⚠ {n_total - n_esg} of {n_total} still 0% ESG in pay</div>
            </div>""", unsafe_allow_html=True)

    # ── Advanced Accountability metrics ────────────────────────────
    st.markdown(sec_html("Proxy Advisor Score · Persistent Offenders · Cumulative Violations · Anomaly Rate",
        "Composite metrics used by ISS/Glass Lewis — DAX-wide violation league table"), unsafe_allow_html=True)

    co_mu_all = mu[mu["company_shortname"] == sel_co].sort_values("year")

    # Persistent offender streak for selected company
    persist_streak = 0
    if "actual_vs_expected_pct" in co_mu_all.columns and co_mu_all["actual_vs_expected_pct"].notna().any():
        curr_s = 0
        for v in (co_mu_all["actual_vs_expected_pct"] > 15).fillna(False):
            if v: curr_s += 1; persist_streak = max(persist_streak, curr_s)
            else: curr_s = 0

    # Anomaly persistence rate
    anom_yrs  = int(co_df["is_anomaly"].sum()) if in_model and "is_anomaly" in co_df.columns else 0
    anom_rate = float(co_df["is_anomaly"].mean() * 100) if in_model and "is_anomaly" in co_df.columns and len(co_df) > 0 else None

    # Proxy advisor red-flag score (0-100)
    ceo_prem_acc = float(co_df["ceo_board_premium_ratio"].mean()) if in_model and "ceo_board_premium_ratio" in co_df.columns and co_df["ceo_board_premium_ratio"].notna().any() else None
    proxy_score = 0
    if aep is not None:        proxy_score += min(35, max(0, aep * 0.5))       # excess pay: up to 35 pts
    proxy_score += min(25, sz_cnt * 8)                                           # bad times: 8 pts each, max 25
    if is_anom:                proxy_score += 20                                 # anomaly flag: 20 pts
    if esg_tot == 0:           proxy_score += 15                                 # zero ESG: 15 pts
    if ceo_prem_acc and ceo_prem_acc > 2.5: proxy_score += min(5, (ceo_prem_acc - 2.5) * 3)
    proxy_score = min(100, proxy_score)
    proxy_lbl = "🔴 Recommend Against" if proxy_score > 65 else "🟡 Abstain" if proxy_score > 35 else "🟢 Support"
    proxy_col = RED if proxy_score > 65 else AMBER if proxy_score > 35 else GREEN

    # DAX-wide persistent offender calculation
    def _streak(series):
        mx, cur = 0, 0
        for v in series:
            if v: cur += 1; mx = max(mx, cur)
            else: cur = 0
        return mx

    mu_streaks = mu.dropna(subset=["actual_vs_expected_pct"]).groupby("company_shortname").apply(
        lambda g: _streak((g.sort_values("year")["actual_vs_expected_pct"] > 15).values)).reset_index()
    mu_streaks.columns = ["company_shortname", "max_streak"]
    persistent_cos = mu_streaks[mu_streaks["max_streak"] >= 3].sort_values("max_streak", ascending=False)

    pa1, pa2, pa3, pa4 = st.columns(4)
    with pa1: st.markdown(kpi_html(f"{proxy_score:.0f}/100", "Proxy Advisor Score",
        proxy_lbl, proxy_col, proxy_col,
        info="Proxy advisor concern score (0–100) modeled on ISS/Glass Lewis methodology. Combines: excess pay vs. model (up to 35pts), Bad Times count (up to 25pts), anomaly flag (20pts), zero ESG link (15pts), and extreme CEO premium (5pts). Scores above 50 typically trigger 'vote against' recommendations."), unsafe_allow_html=True)
    with pa2: st.markdown(kpi_html(f"{persist_streak} yr{'s' if persist_streak != 1 else ''}",
        "Longest Overpay Streak",
        "⚠ Persistent offender" if persist_streak >= 3 else "⚠ Borderline" if persist_streak >= 2 else "✅ No pattern",
        RED if persist_streak >= 3 else AMBER if persist_streak >= 2 else GREEN,
        RED if persist_streak >= 3 else AMBER if persist_streak >= 2 else GREEN,
        info="Longest consecutive run of years where pay exceeded the model expectation by more than +15%. Streaks ≥3 years are used by shareholder activists as evidence of a systemic governance failure rather than a one-off anomaly."), unsafe_allow_html=True)
    with pa3: st.markdown(kpi_html(f"{anom_yrs} / {len(co_df)}",
        "Anomaly Years",
        f"{anom_rate:.0f}% of years flagged" if anom_rate is not None else "—",
        RED if anom_rate and anom_rate > 40 else AMBER if anom_rate and anom_rate > 20 else GREEN if anom_rate is not None else GRAY,
        RED if anom_rate and anom_rate > 40 else AMBER if anom_rate and anom_rate > 20 else GREEN if anom_rate is not None else GRAY,
        info="Number of years flagged as statistical anomalies out of all years with data. A high anomaly rate (>40%) suggests systemic structural irregularities beyond isolated events, which may warrant a formal investigation or shareholder resolution."), unsafe_allow_html=True)
    with pa4: st.markdown(kpi_html(str(len(persistent_cos)), "DAX Persistent Offenders",
        "Companies ≥3 consecutive years >+15%",
        RED if len(persistent_cos) > 5 else AMBER,
        info="Number of DAX companies with a longest overpay streak of 3+ consecutive years above model expectation. Provides market context — if many companies are persistent offenders, it may indicate a systemic DAX-wide issue with governance norms."), unsafe_allow_html=True)

    col_proxy, col_cum = st.columns([2, 3], gap="large")
    with col_proxy:
        st.markdown(sec_html("Proxy Advisor Score Breakdown",
            "Mimics ISS/Glass Lewis methodology — higher = more likely to receive 'vote against' recommendation"), unsafe_allow_html=True)
        components = {
            "Excess Pay vs. Model": min(35, max(0, aep * 0.5) if aep else 0),
            "Bad Times Events": min(25, sz_cnt * 8),
            "Anomaly Flag": 20 if is_anom else 0,
            "Zero ESG in Pay": 15 if esg_tot == 0 else 0,
            "CEO Premium": min(5, (ceo_prem_acc - 2.5) * 3) if ceo_prem_acc and ceo_prem_acc > 2.5 else 0,
        }
        fig_proxy = go.Figure(go.Bar(
            x=list(components.values()), y=list(components.keys()),
            orientation="h",
            marker_color=[RED if v > 10 else AMBER if v > 0 else "#e2e8f0" for v in components.values()],
            text=[f"{v:.0f} pts" for v in components.values()], textposition="outside",
            hovertemplate="%{y}: %{x:.0f} pts<extra></extra>"))
        fig_proxy.add_vline(x=proxy_score, line_color=proxy_col, line_width=2, line_dash="dot",
            annotation_text=f"Total: {proxy_score:.0f}", annotation_font=dict(color=proxy_col, size=9))
        fig_proxy.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Points", range=[0, 40], gridcolor=GRAYLT),
            yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_proxy, use_container_width=True)

    with col_cum:
        st.markdown(sec_html("DAX Cumulative Bad Times Severity — All-Time League Table",
            "Sum of schlechte_zeiten_score across all years · who are the chronic violators?"), unsafe_allow_html=True)
        if "schlechte_zeiten_score" in df.columns:
            cum_bt = df.groupby("company_shortname")["schlechte_zeiten_score"].sum().reset_index()
            cum_bt = cum_bt[cum_bt["schlechte_zeiten_score"] > 0].sort_values("schlechte_zeiten_score", ascending=True)
            if len(cum_bt) > 0:
                bt_colors = [ORANGE if c == sel_co else (RED if v > 2 else AMBER if v > 1 else "#fde68a")
                             for c, v in zip(cum_bt["company_shortname"], cum_bt["schlechte_zeiten_score"])]
                fig_cum = go.Figure(go.Bar(
                    x=cum_bt["schlechte_zeiten_score"], y=cum_bt["company_shortname"],
                    orientation="h", marker_color=bt_colors,
                    text=[f"{v:.2f}" for v in cum_bt["schlechte_zeiten_score"]],
                    textposition="outside", textfont=dict(size=7),
                    hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
                fig_cum.update_layout(height=max(300, len(cum_bt) * 22), margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Cumulative Severity Score", gridcolor=GRAYLT),
                    yaxis=dict(tickfont=dict(size=8)))
                st.plotly_chart(fig_cum, use_container_width=True)

    # Persistent offenders table
    if len(persistent_cos) > 0:
        st.markdown(sec_html("DAX Persistent Offenders — ≥3 Consecutive Years Above +15% Model Expectation"), unsafe_allow_html=True)
        po_display = persistent_cos.copy()
        po_display["Is Selected"] = po_display["company_shortname"].apply(lambda x: "▶ " + x if x == sel_co else x)
        st.dataframe(po_display[["Is Selected", "max_streak"]].rename(
            columns={"Is Selected": "Company", "max_streak": "Longest Streak (years)"}
        ).set_index("Company"), use_container_width=True)


# ─────────────────────────────────────────────
# STAKEHOLDER 5 — COMPENSATION CONSULTANTS
# ─────────────────────────────────────────────
def show_consultants():
    sidebar_nav("landing", "← Back to Home")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    in_mu    = sel_co in MU_COS
    co_df     = df[df["company_shortname"] == sel_co].sort_values("year")
    co_yr     = df[(df["company_shortname"] == sel_co) & (df["year"] == sel_year)]
    co_mu_all = mu[mu["company_shortname"] == sel_co].sort_values("year")
    co_mu_yr  = co_mu_all[co_mu_all["year"] == sel_year]

    st.markdown(f"""<div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">🧮 Compensation Consultants View</div>
                <div style="font-size:1.8rem;font-weight:900;color:white;letter-spacing:-.5px;margin:4px 0;">{sel_co}</div>
                <div style="font-size:.82rem;color:#94a3b8;">Model-Implied Fair Pay · Sector Benchmarks · Structure Optimization · {sel_year}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    show_universal_snapshot(sel_co, sel_year)
    stakeholder_divider("Compensation Consultant View — Model-Implied Fair Pay & Sector Fit")

    st.markdown("""<div class="insight">
        <strong>What do compensation consultants care about?</strong> They need to validate whether their
        recommendation holds up against an objective benchmark. The OLS model (R²=0.71) provides an
        evidence-based "fair compensation" range from company size, sector, and pay stickiness.
        Consultants can use this to back-test past recommendations and calibrate proposals to avoid
        shareholder backlash at Say-on-Pay votes.
    </div>""", unsafe_allow_html=True)

    aep     = float(co_mu_yr["actual_vs_expected_pct"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["actual_vs_expected_pct"].notna().any() else None
    pred    = float(co_mu_yr["pred_comp"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["pred_comp"].notna().any() else None
    pred_lo = float(co_mu_yr["pred_comp_low"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["pred_comp_low"].notna().any() else None
    pred_hi = float(co_mu_yr["pred_comp_high"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["pred_comp_high"].notna().any() else None
    actual  = float(co_mu_yr["total_comp_bt"].iloc[0]) if len(co_mu_yr) > 0 and co_mu_yr["total_comp_bt"].notna().any() else None
    sector  = co_df["sector"].iloc[0] if in_model and len(co_df) > 0 and "sector" in co_df.columns else "—"
    _, t_col, t_lbl = traffic(aep)

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(kpi_html(f"€{pred/1000:.1f}M" if pred else "—", "Model-Fair Comp.",
        f"OLS Estimate {sel_year}", NAVY,
        info="The OLS model's point estimate of 'fair' compensation given this company's size, sector, board size, year trend, and prior-year pay. This is the anchor for the compensation consultant's recommendation — the defensible midpoint."), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html(f"€{actual/1000:.1f}M" if actual else "—", "Actual Compensation",
        f"Board Total {sel_year}", NAVY,
        info="Reported total executive board compensation before taxes for the selected year, as disclosed in the annual report. This is the starting point for any benchmarking or fairness analysis."), unsafe_allow_html=True)
    with k3: st.markdown(kpi_html(f"{aep:+.0f}%" if aep is not None else "—", "Deviation from Model",
        t_lbl, t_col, t_col,
        info="How much actual pay deviates from the model's fair-value estimate. Consultants use this to quantify how much adjustment (up or down) would be needed to reach the model midpoint. Values beyond ±15% require explicit justification in the remuneration report."), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html(str(sector), "Sector",
        "Sector-adjusted baseline", NAVY,
        info="The company's sector classification used in the OLS model. Sector dummies capture structural differences in pay norms (e.g., Financials and Technology tend to pay more). The model adjusts fair-value estimates accordingly."), unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")
    with col_l:
        st.markdown(sec_html("Actual vs. Model Expectation Band",
            "Orange band = 80% expectation interval · Navy line = actual compensation"), unsafe_allow_html=True)
        if in_mu and len(co_mu_all) > 0 and co_mu_all["pred_comp"].notna().any():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(co_mu_all["year"]) + list(co_mu_all["year"])[::-1],
                y=list(co_mu_all["pred_comp_high"]) + list(co_mu_all["pred_comp_low"])[::-1],
                fill="toself", fillcolor="rgba(249,115,22,.10)",
                line=dict(color="rgba(0,0,0,0)"), name="80% Expectation Band"))
            fig.add_trace(go.Scatter(x=co_mu_all["year"], y=co_mu_all["pred_comp"],
                line=dict(color=ORANGE, dash="dash", width=2), name="Model Expectation",
                hovertemplate="Expected: €%{y:,.0f}K<extra></extra>"))
            fig.add_trace(go.Scatter(x=co_mu_all["year"], y=co_mu_all["total_comp_bt"],
                line=dict(color=NAVY, width=3), mode="lines+markers", name="Actual",
                hovertemplate="Actual: €%{y:,.0f}K<extra></extra>"))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Total Compensation (€K)", gridcolor=GRAYLT),
                xaxis=dict(gridcolor=GRAYLT),
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
                hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            if pred_lo and pred_hi and pred:
                st.markdown(f"""<div style="background:{NAVY};border-radius:10px;padding:12px 16px;">
                    <div style="font-size:.62rem;color:#64748b;text-transform:uppercase;margin-bottom:6px;">Model Range {sel_year}</div>
                    <div style="display:flex;justify-content:space-between;color:white;font-size:.84rem;">
                        <span>Low: <strong>€{pred_lo/1000:.1f}M</strong></span>
                        <span style="color:{ORANGE};">Estimate: <strong>€{pred/1000:.1f}M</strong></span>
                        <span>High: <strong>€{pred_hi/1000:.1f}M</strong></span>
                    </div>
                </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown(sec_html("Sector Peer Structure"), unsafe_allow_html=True)
        if in_model and sector and sector != "—" and "sector" in df.columns:
            sect_yr = df[(df["sector"] == sector) & (df["year"] == sel_year)].dropna(subset=["fixed_pct", "sti_pct", "lti_pct"])
            if len(sect_yr) > 0:
                co_fix = float(co_yr["fixed_pct"].iloc[0]) if len(co_yr) > 0 and co_yr["fixed_pct"].notna().any() else None
                co_sti = float(co_yr["sti_pct"].iloc[0]) if len(co_yr) > 0 and co_yr["sti_pct"].notna().any() else None
                co_lti = float(co_yr["lti_pct"].iloc[0]) if len(co_yr) > 0 and co_yr["lti_pct"].notna().any() else None
                s_fix  = sect_yr["fixed_pct"].mean()
                s_sti  = sect_yr["sti_pct"].mean()
                s_lti  = sect_yr["lti_pct"].mean()
                rows = [
                    ("Fixed Pay", f"{co_fix:.0f}%", f"Sector avg {s_fix:.0f}%",
                     RED if co_fix and abs(co_fix - s_fix) > 12 else GRAY),
                    ("STI Share", f"{co_sti:.0f}%", f"Sector avg {s_sti:.0f}%",
                     RED if co_sti and abs(co_sti - s_sti) > 12 else GRAY),
                    ("LTI Share", f"{co_lti:.0f}%", f"Sector avg {s_lti:.0f}%",
                     RED if co_lti and abs(co_lti - s_lti) > 12 else GRAY),
                    ("Sector Peers (N)", str(len(sect_yr)), f"{sel_year}", GRAY),
                ]
                st.markdown('<div style="background:white;border-radius:12px;padding:14px 16px;border:1px solid #e2e8f0;">', unsafe_allow_html=True)
                for lbl, val, ref, col_c in rows:
                    st.markdown(f'<div class="metric-row"><span style="color:{GRAY};">{lbl}'
                                f'<br><span style="font-size:.7rem;color:#94a3b8;">{ref}</span></span>'
                                f'<span style="font-weight:700;color:{col_c};">{val}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(f"Sector data not available for {sel_co}.")

    st.markdown(sec_html("Model Drivers — What Moves Compensation?",
        "OLS coefficients: how strongly each factor shifts expected pay · back-test your proposal against these weights"), unsafe_allow_html=True)
    top_c = coefs[coefs["feature"] != "intercept"].sort_values("exp_effect_pct", ascending=True).copy()
    lbl_map = {"log_comp_lag1": "Prior-Year Pay (Stickiness)", "year_trend": "Year Trend", "log_board_size": "Board Size (log)"}
    top_c["label"] = top_c["feature"].apply(lambda x: lbl_map.get(x, x.replace("sector_", "Sector: ").replace("_", " ")))
    fig3 = go.Figure(go.Bar(
        x=top_c["exp_effect_pct"], y=top_c["label"], orientation="h",
        marker_color=[GREEN if v > 0 else RED for v in top_c["exp_effect_pct"]],
        text=[f"{v:+.0f}%" for v in top_c["exp_effect_pct"]], textposition="outside",
        hovertemplate="%{y}: %{x:+.0f}%<extra></extra>"))
    fig3.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Effect on Compensation Level (%)", gridcolor=GRAYLT),
        yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Advanced Consultant metrics ─────────────────────────────────
    st.markdown(sec_html("Advanced Tools — RMSE · Natural Peer Group · Structure Gap · Pay Forecast · Stickiness",
        "Quantitative consultant toolkit: back-test accuracy, data-driven comps, scenario analysis, forward projection"), unsafe_allow_html=True)

    # Per-company model RMSE
    co_mu_known = co_mu_all.dropna(subset=["total_comp_bt", "pred_comp"])
    rmse_k, mae_k, n_obs = None, None, 0
    if len(co_mu_known) > 0:
        resid = co_mu_known["total_comp_bt"] - co_mu_known["pred_comp"]
        rmse_k = float(np.sqrt((resid ** 2).mean()))
        mae_k  = float(resid.abs().mean())
        n_obs  = len(co_mu_known)
        resid_pct = (resid / co_mu_known["pred_comp"] * 100).dropna()

    # Natural comparator group (5 closest pred_comp in same year)
    mu_yr_all = mu[mu["year"] == sel_year].dropna(subset=["pred_comp"])
    comparators = pd.DataFrame()
    if pred_val and len(mu_yr_all) > 1:
        mu_yr_all = mu_yr_all.copy()
        mu_yr_all["dist"] = (mu_yr_all["pred_comp"] - pred_val).abs()
        comparators = mu_yr_all[mu_yr_all["company_shortname"] != sel_co].nsmallest(5, "dist")[
            ["company_shortname", "pred_comp", "total_comp_bt", "actual_vs_expected_pct"]]

    # Per-company stickiness (lag1 correlation)
    co_mu_lag = co_mu_all.dropna(subset=["total_comp_bt"]).copy()
    co_mu_lag["lag1"] = co_mu_lag["total_comp_bt"].shift(1)
    co_mu_lag = co_mu_lag.dropna(subset=["lag1"])
    stick_corr = float(co_mu_lag["total_comp_bt"].corr(co_mu_lag["lag1"])) if len(co_mu_lag) > 3 else None
    stick_beta = float(np.polyfit(co_mu_lag["lag1"], co_mu_lag["total_comp_bt"], 1)[0]) if len(co_mu_lag) > 3 else None

    # Structure optimization gap (vs sector MEDIAN on lti_pct)
    lti_gap, sti_gap = None, None
    if in_model and sector and sector != "—" and "sector" in df.columns:
        sect_yr_opt = df[(df["sector"] == sector) & (df["year"] == sel_year)].dropna(subset=["lti_pct", "sti_pct"])
        if len(sect_yr_opt) > 0 and len(co_yr) > 0:
            sect_lti_med = float(sect_yr_opt["lti_pct"].median())
            sect_sti_med = float(sect_yr_opt["sti_pct"].median())
            curr_lti = float(co_yr["lti_pct"].iloc[0]) if co_yr["lti_pct"].notna().any() else None
            curr_sti = float(co_yr["sti_pct"].iloc[0]) if co_yr["sti_pct"].notna().any() else None
            if curr_lti is not None: lti_gap = sect_lti_med - curr_lti
            if curr_sti is not None: sti_gap = sect_sti_med - curr_sti

    # Pay trend forecast using year_trend coefficient
    forecast_rows = []
    yr_coef_row = coefs[coefs["feature"] == "year_trend"]
    last_mu_pred = co_mu_all.dropna(subset=["pred_comp"])
    if len(last_mu_pred) > 0 and len(yr_coef_row) > 0:
        last_yr   = int(last_mu_pred.iloc[-1]["year"])
        last_pred = float(last_mu_pred.iloc[-1]["pred_comp"])
        yr_eff    = float(yr_coef_row["exp_effect_pct"].iloc[0]) / 100
        raw_yr_coef = np.log(1 + yr_eff)
        cur = last_pred
        for yr_f in range(last_yr + 1, last_yr + 4):
            cur = cur * np.exp(raw_yr_coef)
            forecast_rows.append({"year": yr_f, "pred_comp": cur, "type": "Forecast"})

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_html(
        f"€{rmse_k:.0f}K" if rmse_k else "—", f"Model RMSE ({n_obs} obs)",
        f"MAE €{mae_k:.0f}K" if mae_k else "Prediction accuracy",
        GREEN if rmse_k and rmse_k < 500 else AMBER if rmse_k and rmse_k < 1500 else RED if rmse_k else GRAY,
        GREEN if rmse_k and rmse_k < 500 else AMBER if rmse_k and rmse_k < 1500 else RED if rmse_k else GRAY,
        info="Root Mean Squared Error of the OLS model for this specific company over all available years. Lower RMSE means the global model fits this company's pay history well. High RMSE (>€1.5M) suggests company-specific factors not captured by the model."
    ), unsafe_allow_html=True)
    with c2: st.markdown(kpi_html(
        str(len(comparators)) if len(comparators) > 0 else "—", "Natural Comparators",
        "Data-driven peer group", NAVY,
        info="Number of DAX companies identified as natural comparators — those with the closest model-implied fair compensation value in the same year. These are the most defensible peers for benchmarking, as they are identified by the data rather than manual selection."), unsafe_allow_html=True)
    with c3:
        lti_gap_col = RED if lti_gap and lti_gap > 10 else AMBER if lti_gap and lti_gap > 5 else GREEN if lti_gap is not None else GRAY
        st.markdown(kpi_html(
            f"{lti_gap:+.0f}pp LTI" if lti_gap is not None else "—",
            "Structure Gap to Sector Med.",
            f"STI gap: {sti_gap:+.0f}pp" if sti_gap is not None else "vs. sector median",
            lti_gap_col, lti_gap_col,
            info="Difference between this company's LTI share and the sector median LTI share for the selected year (in percentage points). A positive gap means the company uses more LTI than peers; negative means it relies more on short-term pay. Useful for structure optimization recommendations."), unsafe_allow_html=True)
    with c4: st.markdown(kpi_html(
        f"{stick_corr:.2f}" if stick_corr is not None else "—", "Pay Stickiness (r)",
        f"β={stick_beta:.2f}" if stick_beta is not None else "lag-1 correlation",
        GREEN if stick_corr and stick_corr > 0.85 else AMBER if stick_corr and stick_corr > 0.65 else RED if stick_corr else GRAY,
        GREEN if stick_corr and stick_corr > 0.85 else AMBER if stick_corr and stick_corr > 0.65 else RED if stick_corr else GRAY,
        info="Correlation between this year's pay and last year's pay (lag-1 autocorrelation). High stickiness (r>0.85) means pay is slow to adjust to performance — once set, it rarely drops. The β coefficient shows by how many euros pay changes for every €1 increase in last year's pay."
    ), unsafe_allow_html=True)

    col_comp, col_fore = st.columns([2, 3], gap="large")
    with col_comp:
        st.markdown(sec_html("Natural Comparator Group",
            "5 companies with the closest model-implied fair compensation in the same year"), unsafe_allow_html=True)
        if len(comparators) > 0:
            comp_colors = ["#94a3b8"] * len(comparators)
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                x=comparators["pred_comp"] / 1000, y=comparators["company_shortname"],
                orientation="h", marker_color="#94a3b8",
                text=[f"€{v/1000:.1f}M" for v in comparators["pred_comp"]],
                textposition="outside", textfont=dict(size=8), name="Comparators"))
            if pred_val:
                fig_comp.add_vline(x=pred_val / 1000, line_color=ORANGE, line_width=2,
                    annotation_text=sel_co, annotation_font=dict(color=ORANGE, size=9))
            fig_comp.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Model-Fair Comp. (€M)", gridcolor=GRAYLT),
                yaxis=dict(tickfont=dict(size=9)), showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(comparators.rename(columns={
                "company_shortname": "Comparator", "pred_comp": "Model-Fair (€K)",
                "total_comp_bt": "Actual (€K)", "actual_vs_expected_pct": "vs. Model %"
            }).round(0).set_index("Comparator"), use_container_width=True)

    with col_fore:
        st.markdown(sec_html("Pay Trend Forecast & Model Accuracy",
            "Dashed = model-based forecast using year trend coefficient · dots = residuals (actual - predicted)"), unsafe_allow_html=True)
        if len(co_mu_all) > 0:
            fig_fore = go.Figure()
            hist_data = co_mu_all.dropna(subset=["pred_comp"])
            fig_fore.add_trace(go.Scatter(x=hist_data["year"], y=hist_data["pred_comp"],
                line=dict(color=ORANGE, dash="dash", width=2), name="Model Expectation"))
            fig_fore.add_trace(go.Scatter(x=co_mu_all["year"], y=co_mu_all["total_comp_bt"],
                line=dict(color=NAVY, width=2.5), mode="lines+markers", marker=dict(size=5),
                name="Actual"))
            if forecast_rows:
                fore_df = pd.DataFrame(forecast_rows)
                # Connect last actual to first forecast
                last_pt = last_mu_pred.iloc[-1]
                fig_fore.add_trace(go.Scatter(
                    x=[last_pt["year"]] + fore_df["year"].tolist(),
                    y=[last_pt["pred_comp"]] + fore_df["pred_comp"].tolist(),
                    line=dict(color=RED, width=2, dash="dot"), mode="lines+markers",
                    marker=dict(size=7, symbol="diamond"), name="Forecast (yr-trend only)"))
            if len(co_mu_known) > 0:
                fig_fore.add_trace(go.Bar(
                    x=co_mu_known["year"],
                    y=(co_mu_known["total_comp_bt"] - co_mu_known["pred_comp"]).clip(-3000, 3000),
                    marker_color=["rgba(220,38,38,.25)" if v > 0 else "rgba(22,163,74,.25)"
                                  for v in (co_mu_known["total_comp_bt"] - co_mu_known["pred_comp"])],
                    name="Residual (actual−model)", yaxis="y2"))
            fig_fore.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Total Comp. (€K)", gridcolor=GRAYLT),
                yaxis2=dict(title="Residual (€K)", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(size=8)),
                xaxis=dict(gridcolor=GRAYLT),
                legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=9)),
                hovermode="x unified")
            st.plotly_chart(fig_fore, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# COMPENSATION PREDICTOR
# ══════════════════════════════════════════════════════════════
_COEF = dict(zip(coefs["feature"], coefs["coefficient"]))
_SECTORS_MU = sorted(mu["sector"].dropna().unique().tolist())  # reference = "Other"
_RESID_STD  = 0.268  # median resid_std from model universe (80% PI: ±1.28σ in log space)

def _predict(sector: str, n_exec: int, year: int, lag1_comp_k: float) -> tuple:
    """Return (point, low, high) in €K using the OLS model."""
    year_centered  = year - 2012
    sector_dummy   = f"sector_{sector}"
    sector_coef    = _COEF.get(sector_dummy, 0.0)   # 0 for reference sector "Other"
    log_pred = (_COEF["intercept"]
                + _COEF["log_comp_lag1"]  * np.log(lag1_comp_k)
                + _COEF["year_trend"]     * year_centered
                + _COEF["log_board_size"] * np.log(n_exec)
                + sector_coef)
    point = np.exp(log_pred)
    low   = point * np.exp(-1.28 * _RESID_STD)
    high  = point * np.exp( 1.28 * _RESID_STD)
    return point, low, high


def show_predictor():
    with st.sidebar:
        st.markdown(f'<div style="padding:20px 0 12px 0;font-size:1.4rem;font-weight:800;color:white;">ExComp</div>', unsafe_allow_html=True)
        st.divider()
        if st.button("← Back to Home", key="pred_home", use_container_width=True): nav("landing")

    # ── Hero ──────────────────────────────────────────────────
    st.markdown(f"""<div class="hero" style="padding:26px 32px 22px;">
        <div style="font-size:1.9rem;font-weight:900;color:white;letter-spacing:-1px;">Compensation Predictor</div>
        <div style="font-size:.88rem;color:#94a3b8;margin-top:6px;">
            Enter company characteristics → receive a model-based fair-compensation estimate with an 80% prediction interval, structure recommendation, and peer context
        </div>
        <div style="margin-top:12px;display:flex;flex-wrap:wrap;gap:6px;">
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.71rem;color:#cbd5e1;">OLS R²=0.71</span>
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.71rem;color:#cbd5e1;">80% Prediction Interval</span>
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.71rem;color:#cbd5e1;">Sector Structure Benchmark</span>
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.71rem;color:#cbd5e1;">Peer Overlay</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Layout: inputs left, results right ────────────────────
    col_in, col_out = st.columns([2, 3], gap="large")

    with col_in:
        st.markdown(sec_html("Company Characteristics", "Fill in the fields below — all inputs feed directly into the OLS model"), unsafe_allow_html=True)

        with st.form("predictor_form"):
            p_sector = st.selectbox("Sector", _SECTORS_MU,
                index=_SECTORS_MU.index("Auto") if "Auto" in _SECTORS_MU else 0,
                help="Sector classification matching the OLS model's sector dummies. 'Other' is the model's reference sector (no dummy).")
            p_year = st.selectbox("Prediction Year", [2027, 2026, 2025, 2024, 2023, 2022, 2021, 2020],
                index=3, help="For 2025–2027, prior-year compensation is the required input (no historical data available).")
            p_board = st.slider("Executive Board Size (# members)", min_value=2, max_value=20, value=7,
                help="Number of executive board members. The model uses log(board_size) — larger boards cost more in total but less per head.")

            st.markdown("---")
            st.markdown("**Prior-Year Total Compensation**")
            use_avg = st.checkbox("Use DAX sector average as prior-year anchor", value=True,
                help="If checked, the sector median from the prior year is used for the stickiness (lag-1) term. Uncheck to enter a specific value.")
            p_lag1 = None
            if not use_avg:
                p_lag1 = st.number_input("Prior-year total board compensation (€K)", min_value=500, max_value=100_000,
                    value=10_000, step=500,
                    help="Total executive board compensation in the year before the prediction year (€ thousands). The single strongest predictor in the model.")

            st.markdown("---")
            st.markdown("**Optional: Manual Peer Benchmarks**")
            st.caption("Enter up to 3 comparable companies for the comparison chart.")
            peers = []
            for i in range(3):
                pc, pv = st.columns([3, 2])
                with pc: pname = st.text_input(f"Peer {i+1} name", key=f"pname{i}", placeholder=f"Company {i+1}", label_visibility="collapsed")
                with pv: pcomp = st.number_input(f"Comp (€K)", min_value=0, max_value=200_000, value=0, step=500, key=f"pcomp{i}", label_visibility="collapsed")
                if pname and pcomp > 0:
                    peers.append((pname, float(pcomp)))

            submitted = st.form_submit_button("▶ Run Prediction", type="primary", use_container_width=True)

    # ── Compute ───────────────────────────────────────────────
    with col_out:
        if not submitted and "pred_result" not in st.session_state:
            st.markdown(f"""<div style="background:#f8fafc;border:2px dashed #e2e8f0;border-radius:16px;
                padding:48px 32px;text-align:center;margin-top:48px;">
                <div style="font-size:2.5rem;margin-bottom:12px;">🧮</div>
                <div style="font-size:1rem;font-weight:700;color:{NAVY};margin-bottom:6px;">Ready to Predict</div>
                <div style="font-size:.83rem;color:{GRAY};">Fill in the company characteristics on the left and click <strong>Run Prediction</strong>.</div>
            </div>""", unsafe_allow_html=True)
        else:
            # Cache result in session state so it persists across re-runs without re-submit
            if submitted:
                # Resolve lag1
                if use_avg or p_lag1 is None:
                    prior_year = p_year - 1
                    mu_prior = mu[(mu["year"] == prior_year) & (mu["sector"] == p_sector)]
                    if len(mu_prior) > 0:
                        lag1_val = float(mu_prior["total_comp_bt"].dropna().median())
                    else:
                        mu_prior_all = mu[mu["year"] == prior_year]
                        lag1_val = float(mu_prior_all["total_comp_bt"].dropna().median()) if len(mu_prior_all) > 0 else 10_000.0
                else:
                    lag1_val = float(p_lag1)

                point, low, high = _predict(p_sector, p_board, p_year, lag1_val)

                # Sector context from mu
                mu_sector_yr = mu[(mu["year"] == p_year) & (mu["sector"] == p_sector)].dropna(subset=["pred_comp"])
                mu_all_yr    = mu[mu["year"] == p_year].dropna(subset=["pred_comp"])
                sector_med   = float(mu_sector_yr["pred_comp"].median()) if len(mu_sector_yr) > 0 else None
                dax_med      = float(mu_all_yr["pred_comp"].median())    if len(mu_all_yr) > 0 else None
                sector_p25   = float(mu_sector_yr["pred_comp"].quantile(.25)) if len(mu_sector_yr) > 1 else None
                sector_p75   = float(mu_sector_yr["pred_comp"].quantile(.75)) if len(mu_sector_yr) > 1 else None

                # Comp structure from sector peers
                sector_cos  = mu[mu["sector"] == p_sector]["company_shortname"].unique()
                struct_df   = df[df["company_shortname"].isin(sector_cos) & (df["year"] == min(p_year, FEATURES_YEAR_MAX))].dropna(subset=["fixed_pct","sti_pct","lti_pct"])
                if len(struct_df) == 0:
                    struct_df = df[df["company_shortname"].isin(sector_cos)].dropna(subset=["fixed_pct","sti_pct","lti_pct"])
                fix_med = float(struct_df["fixed_pct"].median()) if len(struct_df) > 0 else 33.0
                sti_med = float(struct_df["sti_pct"].median())   if len(struct_df) > 0 else 33.0
                lti_med = float(struct_df["lti_pct"].median())   if len(struct_df) > 0 else 34.0

                st.session_state["pred_result"] = dict(
                    sector=p_sector, year=p_year, board=p_board, lag1=lag1_val,
                    point=point, low=low, high=high,
                    sector_med=sector_med, dax_med=dax_med,
                    sector_p25=sector_p25, sector_p75=sector_p75,
                    fix_med=fix_med, sti_med=sti_med, lti_med=lti_med,
                    mu_sector_yr=mu_sector_yr, mu_all_yr=mu_all_yr,
                    peers=peers,
                    use_avg=use_avg,
                )

            r = st.session_state.get("pred_result", {})
            if not r:
                st.info("Submit the form to see results.")
                st.stop()

            point      = r["point"]
            low        = r["low"]
            high       = r["high"]
            sector_med = r["sector_med"]
            dax_med    = r["dax_med"]
            fix_med, sti_med, lti_med = r["fix_med"], r["sti_med"], r["lti_med"]
            mu_sector_yr = r["mu_sector_yr"]
            mu_all_yr    = r["mu_all_yr"]
            peers_res    = r["peers"]
            res_sector   = r["sector"]
            res_year     = r["year"]
            res_board    = r["board"]

            vs_sector = (point - sector_med) / sector_med * 100 if sector_med else None
            vs_dax    = (point - dax_med)    / dax_med    * 100 if dax_med    else None
            _, t_col_s, t_lbl_s = traffic(vs_sector)
            per_exec = point / res_board

            st.markdown(sec_html(f"Prediction Result — {res_sector} · {res_board} board members · {res_year}",
                "Point estimate with 80% prediction interval based on the OLS model"), unsafe_allow_html=True)

            # Main result hero
            st.markdown(f"""<div style="background:linear-gradient(135deg,{NAVY} 0%,#1e3a5f 100%);border-radius:20px;padding:28px 30px;margin-bottom:16px;">
                <div style="display:flex;align-items:flex-end;gap:12px;margin-bottom:4px;">
                    <div style="font-size:3.2rem;font-weight:900;color:white;letter-spacing:-2px;line-height:1;">€{point/1000:.2f}M</div>
                    <div style="font-size:.85rem;color:#94a3b8;margin-bottom:8px;">total board comp</div>
                </div>
                <div style="font-size:.82rem;color:#64748b;margin-bottom:16px;">
                    80% Prediction Interval: <span style="color:#94a3b8;font-weight:600;">€{low/1000:.2f}M – €{high/1000:.2f}M</span>
                </div>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">
                    <div style="background:rgba(255,255,255,.07);border-radius:10px;padding:12px;text-align:center;">
                        <div style="font-size:1.25rem;font-weight:800;color:white;">€{per_exec:.0f}K</div>
                        <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;margin-top:3px;">per exec</div>
                    </div>
                    <div style="background:rgba(255,255,255,.07);border-radius:10px;padding:12px;text-align:center;">
                        <div style="font-size:1.25rem;font-weight:800;color:{t_col_s};">{vs_sector:+.0f}%</div>
                        <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;margin-top:3px;">vs. sector median</div>
                    </div>
                    <div style="background:rgba(255,255,255,.07);border-radius:10px;padding:12px;text-align:center;">
                        <div style="font-size:1.25rem;font-weight:800;color:white;">{vs_dax:+.0f}%</div>
                        <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;margin-top:3px;">vs. DAX median</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # KPI row: range width, structure rec, sector rank
            r1, r2, r3 = st.columns(3)
            range_width = (high - low) / point * 100
            with r1: st.markdown(kpi_html(f"€{low/1000:.1f}M – €{high/1000:.1f}M", "80% Prediction Range",
                f"±{range_width/2:.0f}% around point estimate", NAVY,
                info="The 80% prediction interval derived from the model's empirical residual standard deviation (σ=0.27 in log space). There is an 80% probability that the 'true' fair compensation for a company with these characteristics falls within this range."),
                unsafe_allow_html=True)
            with r2: st.markdown(kpi_html(f"{fix_med:.0f}% / {sti_med:.0f}% / {lti_med:.0f}%",
                "Recommended Structure", f"Fix / STI / LTI — {res_sector} median",
                ORANGE, info="Sector-median compensation structure (Fixed / Short-Term Incentive / Long-Term Incentive) based on DAX companies in this sector. Use as a starting point for the pay mix recommendation."),
                unsafe_allow_html=True)
            rank_in_sector = int((mu_sector_yr["pred_comp"] < point).sum()) + 1 if len(mu_sector_yr) > 0 else None
            with r3: st.markdown(kpi_html(
                f"#{rank_in_sector} of {len(mu_sector_yr)}" if rank_in_sector else "—",
                "Sector Peer Rank", f"By model-fair comp — {res_year}", NAVY,
                info="Where this company's predicted fair compensation ranks among DAX peers in the same sector for the selected year. Based on model-expected values, not actual reported pay."),
                unsafe_allow_html=True)

            # ── Structure chart ───────────────────────────────
            st.markdown(sec_html("Recommended Compensation Structure",
                f"Sector median split ({res_sector}) — adjust based on company-specific governance objectives"), unsafe_allow_html=True)

            struct_fig = go.Figure()
            # Suggested amounts
            fix_amt = point * fix_med / 100
            sti_amt = point * sti_med / 100
            lti_amt = point * lti_med / 100
            struct_fig.add_trace(go.Bar(name="Fixed Salary", x=["Fixed", "STI", "LTI"],
                y=[fix_amt/1000, sti_amt/1000, lti_amt/1000],
                marker_color=[NAVY, ORANGE, GREEN],
                text=[f"€{v/1000:.1f}M<br>({p:.0f}%)" for v, p in
                      [(fix_amt, fix_med), (sti_amt, sti_med), (lti_amt, lti_med)]],
                textposition="outside", textfont=dict(size=10), showlegend=False))
            struct_fig.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Amount (€M)", gridcolor=GRAYLT),
                xaxis=dict(title="Compensation Component"))
            st.plotly_chart(struct_fig, use_container_width=True)

            # Structure donut
            donut_fig = go.Figure(go.Pie(
                labels=["Fixed", "STI", "LTI"],
                values=[fix_med, sti_med, lti_med],
                hole=0.55,
                marker_colors=[NAVY, ORANGE, GREEN],
                textinfo="label+percent", textfont=dict(size=11)))
            donut_fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False, paper_bgcolor="white")

            # ── Peer comparison chart ─────────────────────────
            st.markdown(sec_html(f"Peer Context — {res_sector} Sector ({res_year})",
                "Distribution of model-fair compensation among DAX peers in the same sector · orange = your prediction"), unsafe_allow_html=True)

            if len(mu_sector_yr) > 0:
                peer_sorted = mu_sector_yr.sort_values("pred_comp", ascending=True).copy()
                bar_cols = [ORANGE if abs(v - point) < 1 else "#94a3b8" for v in peer_sorted["pred_comp"]]
                # Insert predicted point as separate trace
                fig_peer = go.Figure()
                fig_peer.add_trace(go.Bar(
                    x=peer_sorted["pred_comp"] / 1000, y=peer_sorted["company_shortname"],
                    orientation="h", marker_color="#cbd5e1",
                    text=[f"€{v/1000:.1f}M" for v in peer_sorted["pred_comp"]],
                    textposition="outside", textfont=dict(size=8),
                    name=f"DAX {res_sector} peers",
                    hovertemplate="%{y}: €%{x:.1f}M<extra></extra>"))
                # Prediction vlines
                fig_peer.add_vline(x=point/1000, line_color=ORANGE, line_width=3,
                    annotation_text=f"Prediction €{point/1000:.1f}M",
                    annotation_font=dict(color=ORANGE, size=9))
                fig_peer.add_vline(x=low/1000, line_color=ORANGE, line_width=1, line_dash="dot")
                fig_peer.add_vline(x=high/1000, line_color=ORANGE, line_width=1, line_dash="dot",
                    annotation_text="80% PI", annotation_font=dict(color=ORANGE, size=8))
                # Manual peers
                for pname, pcomp in peers_res:
                    fig_peer.add_vline(x=pcomp/1000, line_color=NAVY, line_width=2, line_dash="dash",
                        annotation_text=pname, annotation_font=dict(color=NAVY, size=8))
                fig_peer.update_layout(
                    height=max(220, len(peer_sorted) * 28 + 40),
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title="Model-Fair Comp. (€M)", gridcolor=GRAYLT),
                    yaxis=dict(tickfont=dict(size=9)), showlegend=False)
                st.plotly_chart(fig_peer, use_container_width=True)

            # ── Full DAX scatter if ≥2 sectors ────────────────
            if len(mu_all_yr) > 5:
                st.markdown(sec_html("Full DAX Context", "All sectors · orange = prediction · shade = 80% interval"), unsafe_allow_html=True)
                dax_sorted = mu_all_yr.sort_values("pred_comp", ascending=False).head(40)
                dax_cols = [ORANGE if r == res_sector else "#e2e8f0" for r in dax_sorted["sector"]]
                fig_dax = go.Figure()
                fig_dax.add_trace(go.Bar(
                    x=dax_sorted["company_shortname"], y=dax_sorted["pred_comp"]/1000,
                    marker_color=dax_cols, name="DAX peers",
                    hovertemplate="%{x}: €%{y:.1f}M<extra></extra>"))
                fig_dax.add_hline(y=point/1000, line_color=ORANGE, line_width=2,
                    annotation_text=f"Prediction €{point/1000:.1f}M",
                    annotation_font=dict(color=ORANGE, size=9))
                fig_dax.add_hrect(y0=low/1000, y1=high/1000, fillcolor="rgba(249,115,22,.08)",
                    line_width=0, annotation_text="80% PI", annotation_font=dict(color=ORANGE, size=8))
                for pname, pcomp in peers_res:
                    fig_dax.add_hline(y=pcomp/1000, line_color=NAVY, line_dash="dash", line_width=1.5,
                        annotation_text=pname, annotation_font=dict(color=NAVY, size=8))
                fig_dax.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Model-Fair Comp. (€M)", gridcolor=GRAYLT),
                    xaxis=dict(tickangle=-40, tickfont=dict(size=8)))
                st.plotly_chart(fig_dax, use_container_width=True)

            # ── Input sensitivity ─────────────────────────────
            st.markdown(sec_html("Board Size Sensitivity",
                "How does the prediction change as board size varies? Everything else held constant."), unsafe_allow_html=True)
            board_range = list(range(2, 21))
            sens_pts = [_predict(res_sector, b, res_year, r["lag1"])[0] for b in board_range]
            sens_lo  = [_predict(res_sector, b, res_year, r["lag1"])[1] for b in board_range]
            sens_hi  = [_predict(res_sector, b, res_year, r["lag1"])[2] for b in board_range]
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(x=board_range, y=[v/1000 for v in sens_hi],
                mode="lines", line=dict(width=0), showlegend=False,
                hovertemplate=""))
            fig_sens.add_trace(go.Scatter(x=board_range, y=[v/1000 for v in sens_lo],
                mode="lines", fill="tonexty", fillcolor="rgba(249,115,22,.12)",
                line=dict(width=0), name="80% PI", showlegend=False))
            fig_sens.add_trace(go.Scatter(x=board_range, y=[v/1000 for v in sens_pts],
                mode="lines+markers", line=dict(color=ORANGE, width=2.5),
                marker=dict(size=5), name="Point estimate"))
            fig_sens.add_vline(x=res_board, line_color=NAVY, line_dash="dash", line_width=2,
                annotation_text=f"Your input: {res_board}",
                annotation_font=dict(color=NAVY, size=9))
            fig_sens.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Board Size (# members)", gridcolor=GRAYLT, dtick=2),
                yaxis=dict(title="Total Comp. (€M)", gridcolor=GRAYLT),
                legend=dict(orientation="h"))
            st.plotly_chart(fig_sens, use_container_width=True)

            # ── Model inputs table ────────────────────────────
            st.markdown(sec_html("Model Inputs Summary", "What was fed into the OLS model to produce this prediction"), unsafe_allow_html=True)
            inp_rows = [
                ("Sector", res_sector, f"Dummy coef: {_COEF.get(f'sector_{res_sector}', 0):.4f}"),
                ("Board Size", str(res_board), f"log({res_board}) = {np.log(res_board):.3f}"),
                ("Year", str(res_year), f"Year-centered = {res_year - 2012}"),
                ("Prior-Year Comp (lag-1)", f"€{r['lag1']:.0f}K", f"{'Sector avg' if r['use_avg'] else 'User input'} · log = {np.log(r['lag1']):.3f}"),
            ]
            rows_h = "".join([f"""<tr>
                <td style="padding:8px 12px;font-weight:600;color:{NAVY};font-size:.82rem;border-bottom:1px solid #f1f5f9;">{a}</td>
                <td style="padding:8px 12px;color:#374151;font-size:.82rem;border-bottom:1px solid #f1f5f9;">{b}</td>
                <td style="padding:8px 12px;color:{GRAY};font-size:.78rem;border-bottom:1px solid #f1f5f9;font-family:monospace;">{c}</td>
            </tr>""" for a, b, c in inp_rows])
            st.markdown(f"""<div style="background:white;border-radius:12px;border:1px solid #e2e8f0;overflow:hidden;">
                <table style="width:100%;border-collapse:collapse;">
                    <thead><tr style="background:{NAVY};">
                        <th style="color:white;font-size:.7rem;padding:9px 12px;text-align:left;">Input</th>
                        <th style="color:white;font-size:.7rem;padding:9px 12px;text-align:left;">Value</th>
                        <th style="color:white;font-size:.7rem;padding:9px 12px;text-align:left;">Model term</th>
                    </tr></thead>
                    <tbody>{rows_h}</tbody>
                </table>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# METHODOLOGY PAGE
# ══════════════════════════════════════════════════════════════
def show_methodology():
    with st.sidebar:
        st.markdown(f'<div style="padding:20px 0 12px 0;font-size:1.4rem;font-weight:800;color:white;">ExComp</div>', unsafe_allow_html=True)
        st.divider()
        if st.button("← Back to Home", use_container_width=True): nav("landing")

    # ── Hero ──────────────────────────────────────────────────
    st.markdown(f"""<div class="hero" style="padding:28px 32px 24px;">
        <div style="font-size:2rem;font-weight:900;color:white;letter-spacing:-1px;">Methodology &amp; KPI Reference</div>
        <div style="font-size:.9rem;color:#94a3b8;margin-top:8px;">Every metric, how it is computed, and why it matters — all in one place</div>
        <div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:6px;">
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.72rem;color:#cbd5e1;">📐 OLS Regression (R²=0.71)</span>
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.72rem;color:#cbd5e1;">📈 Trend Regression (polyfit)</span>
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.72rem;color:#cbd5e1;">🔴 Anomaly Detection</span>
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.72rem;color:#cbd5e1;">📦 Rule-Based Flags</span>
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.72rem;color:#cbd5e1;">📊 Percentile / Distribution</span>
            <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.72rem;color:#cbd5e1;">🔗 Correlation Analysis</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Section 1: Core OLS Model ─────────────────────────────
    st.markdown(sec_html("The Core Prediction Model",
        "All 'vs. model' and 'fair-value' metrics derive from this single OLS regression estimated on the full DAX panel"), unsafe_allow_html=True)

    st.markdown(f"""<div style="background:white;border-radius:16px;padding:22px 26px;border:1px solid #e2e8f0;box-shadow:0 2px 12px rgba(0,0,0,.05);">
        <div style="font-size:1rem;font-weight:700;color:{NAVY};margin-bottom:8px;">Model Formula</div>
        <div style="background:#f8fafc;border-radius:8px;padding:14px 18px;font-family:monospace;font-size:.88rem;color:{NAVY};border:1px solid #e2e8f0;line-height:1.9;">
            log(total_comp) = β₀ + β₁·log(total_comp<sub>t-1</sub>) + β₂·log(board_size) + β₃·year_trend + Σ βₛ·sector_dummy<sub>s</sub> + ε
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:16px;">
            <div style="background:#f1f5f9;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:{NAVY};">0.71</div>
                <div style="font-size:.68rem;color:{GRAY};text-transform:uppercase;margin-top:3px;">R² (in-sample)</div>
            </div>
            <div style="background:#f1f5f9;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:{NAVY};">7,500+</div>
                <div style="font-size:.68rem;color:{GRAY};text-transform:uppercase;margin-top:3px;">Exec Observations</div>
            </div>
            <div style="background:#f1f5f9;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:{NAVY};">43</div>
                <div style="font-size:.68rem;color:{GRAY};text-transform:uppercase;margin-top:3px;">DAX Companies</div>
            </div>
            <div style="background:#f1f5f9;border-radius:10px;padding:12px;text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:{NAVY};">2006–2024</div>
                <div style="font-size:.68rem;color:{GRAY};text-transform:uppercase;margin-top:3px;">Data Period</div>
            </div>
        </div>
        <div style="margin-top:14px;font-size:.82rem;color:{GRAY};line-height:1.65;">
            <strong>Why log-linear?</strong> Compensation data is right-skewed. Taking logs symmetrizes residuals and means coefficients are interpretable as percentage effects.
            The <em>exp_effect_pct</em> column in the coefficient table = (exp(β)−1)×100, i.e. the % change in comp for a 1-unit increase in the log predictor.
            Sector dummies capture structural pay norms (e.g., Financials and Technology tend to pay more). Prior-year pay (β₁) captures <em>stickiness</em> — the single strongest predictor.
        </div>
    </div>""", unsafe_allow_html=True)

    # OLS coefficients table from real data
    st.markdown(sec_html("Live OLS Coefficients", "Estimated from the full DAX panel — updated each time the data is refreshed"), unsafe_allow_html=True)
    if len(coefs) > 0:
        lbl_map = {
            "log_comp_lag1": "Prior-Year Pay (log)", "year_trend": "Year Trend",
            "log_board_size": "Board Size (log)", "intercept": "Intercept"
        }
        coefs_disp = coefs.copy()
        coefs_disp["Feature"] = coefs_disp["feature"].apply(
            lambda x: lbl_map.get(x, x.replace("sector_", "Sector: ").replace("_", " ").title()))
        coefs_disp["Effect on Pay"] = coefs_disp["exp_effect_pct"].apply(
            lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
        coefs_disp["Direction"] = coefs_disp["exp_effect_pct"].apply(
            lambda v: "↑ Increases pay" if v > 0 else ("↓ Decreases pay" if v < 0 else "—") if pd.notna(v) else "—")
        coefs_disp["Interpretation"] = coefs_disp["feature"].apply(lambda x: {
            "log_comp_lag1": "1% higher prior-year pay → this much % higher current pay (stickiness)",
            "year_trend":    "Each additional calendar year → structural pay growth trend",
            "log_board_size":"1% larger board → this much % change in total board pay",
            "intercept":     "Baseline level when all other predictors are zero (log scale)",
        }.get(x, "Sector-level pay premium/discount vs. reference sector"))
        st.dataframe(
            coefs_disp[["Feature", "Effect on Pay", "Direction", "Interpretation"]].set_index("Feature"),
            use_container_width=True, height=320)

    # ── Section 2: Methods reference ─────────────────────────
    st.markdown(sec_html("Statistical Methods Used Across the App",
        "All methods and what they contribute to the analysis"), unsafe_allow_html=True)

    METHODS = [
        ("📐 OLS Regression (Panel)", NAVY,
         "Ordinary Least Squares on the log-linear model above, estimated on the full DAX panel (2006–2024).",
         "Produces the fair-value estimate, model deviation, and all 'vs. model' percentages. R²=0.71 means the model explains 71% of pay variation.",
         ["vs. Model Expectation", "Cumulative Excess Pay", "Model RMSE", "Pay Forecast 2025–2027", "Natural Comparator Group"]),
        ("📈 Linear Trend Regression (polyfit)", "#7c3aed",
         "numpy.polyfit(x, y, 1) fit on company-specific time series to estimate year-over-year trend slopes.",
         "Used whenever we need a direction and speed metric rather than a point value. Slope = change per year in percentage points or euros.",
         ["LTI Trend Slope", "Model Deviation Drift", "Pay-Headcount Elasticity β", "Pay Stickiness β"]),
        ("📊 Percentile / Distribution Analysis", "#0891b2",
         "Rank and quantile operations over the cross-sectional DAX distribution for each year.",
         "Tells where a company stands relative to its peers in absolute terms. P25/P50/P75 quartile boundaries are computed fresh for each selected year.",
         ["DAX Peer Rank", "DAX Percentile Position", "Comp per Exec vs. Sector Median"]),
        ("🔗 Correlation & Scatter Analysis", "#059669",
         "Pearson correlation and OLS slope estimated on company-specific (pay_yoy, ebit_yoy) or (comp, lag1_comp) pairs.",
         "Quantifies how tightly pay co-moves with earnings or prior-year pay. Key signal for pay-for-performance alignment.",
         ["Pay-Performance β (pay-EBIT coupling)", "Pay Stickiness (r)", "Crisis Pay Change"]),
        ("📦 Rule-Based Flagging", ORANGE,
         "Boolean conditions applied year-by-year to the raw data: e.g. (comp_yoy > 0) AND (ebit_yoy < 0) AND (empl_yoy < 0).",
         "Produces the clearest, most defensible governance signals because they are directly auditable from the annual report. No model assumptions required.",
         ["Bad Times Events", "Restructuring-Bonus Flag", "Anomaly Flag (threshold-based)"]),
        ("🔴 Statistical Anomaly Detection", RED,
         "Z-score / percentile-based outlier detection on the compensation structure vector (fixed/STI/LTI shares + level). Pre-computed as anomaly_score_pct.",
         "Flags company-years where the compensation structure is statistically unusual even after controlling for size and sector — may indicate one-time payments, settlements, or governance failures.",
         ["Anomaly Flag", "Anomaly Years / Rate", "Governance Risk Score"]),
        ("🧮 Composite Score Construction", AMBER,
         "Weighted sum of sub-scores from multiple signals, calibrated to produce 0–100 indices.",
         "Converts multi-dimensional evidence into a single actionable number. Used for the Governance Risk Score, Proxy Advisor Score, and AGM Defensibility Score.",
         ["Proxy Advisor Score (ISS/GL modeled)", "AGM Defensibility Score", "Governance Risk Score"]),
        ("📉 Cumulative Index (base-100)", "#64748b",
         "np.cumprod(1 + yoy/100) * 100 starting from the first common year at index=100.",
         "Converts year-on-year growth rates into an accumulated index that makes long-run divergence between two series directly visible as a gap on the chart.",
         ["Cumulative Exec Pay vs. Headcount Index", "Pay Trend Visualization"]),
    ]

    for icon_label, color, what, why, used_in in METHODS:
        used_tags = " ".join([f'<span style="background:#f1f5f9;border-radius:5px;padding:2px 8px;font-size:.72rem;color:{NAVY};margin:2px 2px 2px 0;display:inline-block;">{m}</span>' for m in used_in])
        st.markdown(f"""<div style="background:white;border-radius:14px;padding:18px 22px;border:1px solid #e2e8f0;margin-bottom:10px;border-left:4px solid {color};">
            <div style="font-size:.95rem;font-weight:700;color:{NAVY};margin-bottom:6px;">{icon_label}</div>
            <div style="font-size:.82rem;color:#374151;margin-bottom:4px;"><strong>What:</strong> {what}</div>
            <div style="font-size:.82rem;color:{GRAY};margin-bottom:8px;"><strong>Why it matters:</strong> {why}</div>
            <div style="font-size:.72rem;color:{GRAY};margin-bottom:4px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">Used in</div>
            <div>{used_tags}</div>
        </div>""", unsafe_allow_html=True)

    # ── Section 3: Full KPI table ─────────────────────────────
    st.markdown(sec_html("Full KPI Reference Table",
        "Every metric in the app — name, calculation method, interpretation, and which stakeholder pages show it"), unsafe_allow_html=True)

    KPI_TABLE = [
        # (KPI Name, Method, Green signal, Red signal, Pages)
        ("Total Compensation (€M)", "Sum of all exec board members' pay (fixed+STI+LTI) before tax", "—", "Very high vs. peers", "All pages"),
        ("DAX Peer Rank / Percentile", "Cross-sectional rank & quantile of total_comp_bt per year", "Below P50", "Above P75", "All pages"),
        ("vs. Model Expectation (%)", "(actual − pred_comp) / pred_comp × 100; pred from OLS model", "−15% to +15%", "> +40%", "All pages"),
        ("Governance Risk Score (0–100)", "Weighted composite: pay gap + anomaly flag + Bad Times count", "< 40", "> 60", "Overview, Capital"),
        ("Anomaly Flag", "Pre-computed anomaly_score_pct threshold (structural outlier detection)", "Not flagged", "Flagged", "Overview, Accountability"),
        ("Bad Times Events", "Rule: comp_yoy > 0 AND ebit_yoy < 0 AND empl_yoy < 0, summed", "0 events", "≥ 2 events", "All pages"),
        ("ESG Share STI / LTI (%)", "sti_esg_share / lti_esg_share from DGAP ESG disclosures", "> 20%", "0% (no link)", "ESG, Overview"),
        ("Women on Board (%)", "n_female_executives / n_executives × 100", "≥ 30% (ARUG II)", "< 20%", "ESG"),
        ("CEO/Worker Pay Ratio", "CEO total comp / median employee wage (CSRD field)", "< 40×", "> 80×", "ESG, Employees"),
        ("Pay vs. EBIT Growth Gap (pp)", "Avg comp_yoy_pct minus avg ebit_yoy_pct over all years", "< 2pp", "> 10pp", "Capital Allocators"),
        ("Avg. LTI Share (%)", "Mean of lti_pct over all available years for the company", "> 40%", "< 25%", "Capital, Board & HR"),
        ("CEO/Board Premium (×)", "CEO total comp / avg of other exec board members' total comp", "< 2×", "> 3×", "Capital, Board & HR"),
        ("Pay-Performance β", "polyfit slope: comp_yoy ~ ebit_yoy (annual % changes)", "0.3–0.8", "< 0 or > 1.2", "Capital Allocators"),
        ("Cumulative Excess Pay (€M)", "Sum of (actual − pred_comp) over all known years", "< €10M", "> €50M", "Capital Allocators"),
        ("Longest Overpay Streak (yrs)", "Max consecutive years with actual_vs_expected_pct > +15%", "0–1 yr", "≥ 3 yrs", "Capital, Accountability"),
        ("LTI Trend Slope (pp/yr)", "polyfit slope on lti_pct time series", "> 0 (rising)", "< −0.5 (falling)", "Capital Allocators"),
        ("Crisis Pay Change (GFC/COVID)", "comp_yoy_pct in 2009 and 2020 from the raw data", "< 0 (cut)", "> +5% (rose)", "Capital Allocators"),
        ("DAX Percentile Position (P25/50/75)", "Quantile position in cross-sectional distribution per year", "P25–P50", "Above P75", "Board & HR"),
        ("Comp per Exec vs. Sector Median", "total_comp / n_exec vs. sector median of same metric", "Within ±20%", "> +25% above", "Board & HR"),
        ("AGM Defensibility Score (0–100)", "Composite: 40pts model fit + 30pts rank + 30pts structure", "> 65", "< 40", "Board & HR"),
        ("Model Deviation Drift (pp/yr)", "polyfit slope on actual_vs_expected_pct time series", "< 0 (improving)", "> +2 (worsening)", "Board & HR"),
        ("Succession Cost Estimate (€K)", "OLS log_board_size coef applied to board size ±1", "—", "—", "Board & HR"),
        ("Pay vs. Headcount Gap (pp/yr)", "Avg comp_yoy_pct minus avg empl_yoy_pct", "< 2pp", "> 10pp", "Employees"),
        ("Avg. Headcount Growth (%/yr)", "Mean of empl_yoy_pct over all years", "Positive", "Negative + rising pay", "Employees"),
        ("Cumulative Indexed Growth (base 100)", "np.cumprod(1 + yoy/100)×100 for pay and headcount separately", "Converging", "Large gap (>50 pts)", "Employees"),
        ("Pay-Headcount Elasticity β", "polyfit slope: comp_yoy ~ empl_yoy", "> 0.2", "< 0 (inverse)", "Employees"),
        ("Restructuring-Bonus Flag", "Years with empl_yoy < −2% AND one_year_bonus increasing", "None flagged", "Any year flagged", "Employees"),
        ("Proxy Advisor Score (0–100)", "Weighted sum: excess pay + Bad Times + anomaly + ESG + CEO prem", "< 30", "> 60", "Accountability"),
        ("Anomaly Years / Rate (%)", "Count of anomaly-flagged years / total years × 100", "< 20%", "> 40%", "Accountability"),
        ("DAX Persistent Offenders (count)", "Companies with longest overpay streak ≥ 3 yrs (DAX-wide)", "Few (<5)", "Many (>10)", "Accountability"),
        ("Model-Fair Comp. (€K)", "OLS point estimate pred_comp for selected year", "—", "—", "Consultants"),
        ("Model RMSE / MAE (€K)", "sqrt(mean(residuals²)) over all known years for this company", "< €500K", "> €1.5M", "Consultants"),
        ("Natural Comparator Group", "5 companies with min |pred_comp − sel_pred| in same year", "—", "—", "Consultants"),
        ("Structure Gap to Sector Med. (pp)", "sector_median_lti − company_lti for selected year", "Within ±5pp", "> 15pp", "Consultants"),
        ("Pay Stickiness (r / β)", "Pearson r and slope: comp_t ~ comp_{t-1} (lag-1 autocorrelation)", "r > 0.7", "—", "Consultants"),
        ("Pay Forecast 2025–2027", "last_pred × exp(raw_year_coef)^n using OLS year_trend coefficient", "—", "—", "Consultants"),
    ]

    rows_html = ""
    for kpi, method, green, red, pages in KPI_TABLE:
        rows_html += f"""<tr>
            <td style="font-weight:600;color:{NAVY};font-size:.8rem;padding:8px 10px;border-bottom:1px solid #f1f5f9;white-space:nowrap;">{kpi}</td>
            <td style="font-size:.77rem;color:#374151;padding:8px 10px;border-bottom:1px solid #f1f5f9;">{method}</td>
            <td style="font-size:.77rem;color:{GREEN};font-weight:600;padding:8px 10px;border-bottom:1px solid #f1f5f9;">{green}</td>
            <td style="font-size:.77rem;color:{RED};font-weight:600;padding:8px 10px;border-bottom:1px solid #f1f5f9;">{red}</td>
            <td style="font-size:.72rem;color:{GRAY};padding:8px 10px;border-bottom:1px solid #f1f5f9;white-space:nowrap;">{pages}</td>
        </tr>"""

    st.markdown(f"""<div style="background:white;border-radius:16px;border:1px solid #e2e8f0;overflow:hidden;box-shadow:0 2px 12px rgba(0,0,0,.04);">
        <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="background:{NAVY};">
                    <th style="color:white;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;padding:10px 10px;text-align:left;">KPI / Metric</th>
                    <th style="color:white;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;padding:10px 10px;text-align:left;">How It Is Calculated</th>
                    <th style="color:#4ade80;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;padding:10px 10px;text-align:left;">Green Signal</th>
                    <th style="color:#f87171;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;padding:10px 10px;text-align:left;">Red Signal</th>
                    <th style="color:#94a3b8;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em;padding:10px 10px;text-align:left;">Pages</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;color:#94a3b8;font-size:.74rem;margin-top:16px;">ORBIS/Bureau van Dijk · DGAP Compensation Reports · 43 DAX Companies · 2006–2024</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
MODULE_FUNCS = {
    "historical": show_historical,
    "prediction": show_prediction,
    "governance": show_governance,
    "peer":       show_peer,
    "anomaly":    show_anomaly,
    "esg":        show_esg,
}

STAKEHOLDER_FUNCS = {
    "capital":       show_capital_allocators,
    "board":         show_board_hr,
    "employees":     show_employees_labor,
    "accountability": show_accountability_actors,
    "consultants":   show_consultants,
    "esg":           show_esg,
}

screen = st.session_state.screen
if screen == "landing":
    show_landing()
elif screen == "overview":
    show_overview()
elif screen == "methodology":
    show_methodology()
elif screen == "predictor":
    show_gbr_predictor()
elif screen in STAKEHOLDER_FUNCS:
    STAKEHOLDER_FUNCS[screen]()
elif screen == "module" and st.session_state.module in MODULE_FUNCS:
    MODULE_FUNCS[st.session_state.module]()
elif screen == "peer_grouping":
    show_peer_grouping()
elif screen == "outlier_detect":
    show_peer_outlier_detection()
else:
    show_landing()
