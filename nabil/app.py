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

def kpi_html(number, label, delta, num_color=None, delta_color=GRAY, bar_color=None):
    num_color = num_color or NAVY
    bar = f"background:linear-gradient(90deg,{bar_color},{bar_color}99);" if bar_color else f"background:linear-gradient(90deg,{ORANGE},{AMBER});"
    return f"""<div class="kpi-card" style="overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;{bar}border-radius:3px 3px 0 0;"></div>
        <div class="kpi-number" style="color:{num_color};">{number}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-delta" style="color:{delta_color};">{delta}</div>
    </div>"""

def sec_html(title, sub=""):
    sub_html = f'<div class="sec-sub">{sub}</div>' if sub else ""
    return f'<div class="sec"><div class="sec-title">{title}</div>{sub_html}</div>'

def traffic(pct):
    if pct is None or np.isnan(pct): return "⚪", GRAY, "Keine Daten"
    if pct >  40: return "🔴", RED,   f"+{pct:.0f}% deutlich überbezahlt"
    if pct >  15: return "🟡", AMBER, f"+{pct:.0f}% über Markt"
    if pct > -15: return "🟢", GREEN, f"{pct:+.0f}% marktkonform"
    if pct > -40: return "🟡", AMBER, f"{pct:.0f}% unter Markt"
    return "🔵", "#2563eb", f"{pct:.0f}% deutlich unterbezahlt"

def risk_color(val, hi=60, mid=40):
    if val >= hi: return RED
    if val >= mid: return AMBER
    return GREEN

def sidebar_nav(back_screen, back_label="← Zurück"):
    with st.sidebar:
        st.markdown(f"""<div style="padding:16px 0 8px 0;">
            <div style="font-size:1.5rem;font-weight:800;color:white;">ExComp</div>
            <div style="font-size:0.72rem;color:#64748b;">AI-Powered Pay Intelligence</div>
        </div>""", unsafe_allow_html=True)
        if st.button(back_label, use_container_width=True):
            nav(back_screen)
        st.divider()
        st.markdown('<p style="color:#94a3b8;font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;">Unternehmen</p>', unsafe_allow_html=True)
        idx = COMPANIES.index(st.session_state.company) if st.session_state.company in COMPANIES else 0
        co = st.selectbox("", COMPANIES, index=idx, label_visibility="collapsed", key="sb_co")
        if co != st.session_state.company:
            st.session_state.company = co
            st.rerun()
        st.markdown('<p style="color:#94a3b8;font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em;margin-top:10px;margin-bottom:4px;">Jahr</p>', unsafe_allow_html=True)
        yr_idx = YEARS_ALL.index(st.session_state.year) if st.session_state.year in YEARS_ALL else len(YEARS_ALL)-1
        yr = st.selectbox("", YEARS_ALL, index=yr_idx, label_visibility="collapsed", key="sb_yr")
        if yr != st.session_state.year:
            st.session_state.year = yr
            st.rerun()
        # Show data-availability hint
        has_rich = yr <= FEATURES_YEAR_MAX
        has_mu   = yr in YEARS_MU
        has_esg  = yr in YEARS_ESG
        if has_rich and has_mu and has_esg:
            st.markdown(f'<div style="background:#1e3a5f;border-radius:6px;padding:6px 10px;margin-top:4px;font-size:.7rem;color:#4ade80;">📊 Volle Daten verfügbar<br><span style="color:#64748b;">Modell + ESG + Alle Features</span></div>', unsafe_allow_html=True)
        elif has_mu and has_esg:
            st.markdown(f'<div style="background:#1e3a5f;border-radius:6px;padding:6px 10px;margin-top:4px;font-size:.7rem;color:{ORANGE};">🌱 Modell + ESG verfügbar<br><span style="color:#64748b;">Governance-Features nur bis 2021</span></div>', unsafe_allow_html=True)
        elif has_mu:
            st.markdown(f'<div style="background:#1e3a5f;border-radius:6px;padding:6px 10px;margin-top:4px;font-size:.7rem;color:#94a3b8;">📈 Nur Modelldaten<br><span style="color:#64748b;">ESG-Daten nicht verfügbar</span></div>', unsafe_allow_html=True)
        st.divider()
        st.markdown('<p style="color:#475569;font-size:0.73rem;line-height:1.7;">43 DAX-Unternehmen<br>7.500+ Exec-Beobachtungen<br>OLS R²=0.71 · 2006–2024</p>', unsafe_allow_html=True)

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
        st.markdown('<p style="color:#475569;font-size:0.75rem;line-height:1.7;">TUM Science Hackathon 2026<br>Chair of Financial Accounting<br>Prof. Dr. Jürgen Ernstberger<br><br>43 DAX-Unternehmen<br>7.500+ Exec-Beobachtungen<br>2006–2024</p>', unsafe_allow_html=True)

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
                    <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.73rem;color:#cbd5e1;">📈 15 Jahre Daten</span>
                    <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.73rem;color:#cbd5e1;">🤖 OLS Prognosemodell</span>
                    <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.73rem;color:#cbd5e1;">🌱 CSRD-Compliance</span>
                    <span style="background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);border-radius:20px;padding:3px 10px;font-size:.73rem;color:#cbd5e1;">🔍 Anomalie-Detektor</span>
                </div>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;">
                <div style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:16px 22px;text-align:center;min-width:90px;">
                    <div style="font-size:2.4rem;font-weight:900;color:{ORANGE};letter-spacing:-1px;line-height:1;">16<span style="font-size:1.2rem;color:#64748b;">/43</span></div>
                    <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:5px;font-weight:600;">DAX mit ESG-Pay</div>
                </div>
                <div style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:16px 22px;text-align:center;min-width:90px;">
                    <div style="font-size:2.4rem;font-weight:900;color:white;letter-spacing:-1px;line-height:1;">0.71</div>
                    <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:5px;font-weight:600;">Modell R²</div>
                </div>
                <div style="background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);border-radius:16px;padding:16px 22px;text-align:center;min-width:90px;">
                    <div style="font-size:2.4rem;font-weight:900;color:white;letter-spacing:-1px;line-height:1;">74x</div>
                    <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:5px;font-weight:600;">Max Pay Ratio</div>
                </div>
                <div style="background:rgba(249,115,22,.12);border:1px solid rgba(249,115,22,.25);border-radius:16px;padding:16px 22px;text-align:center;min-width:90px;">
                    <div style="font-size:2.4rem;font-weight:900;color:{ORANGE};letter-spacing:-1px;line-height:1;">63%</div>
                    <div style="font-size:.65rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:5px;font-weight:600;">Zahlen 0% auf ESG</div>
                </div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style="text-align:center;margin:6px 0 22px 0;">
        <div style="font-size:1.05rem;font-weight:700;color:#0a1628;">Wähle deinen Stakeholder-Blickwinkel</div>
        <div style="font-size:.82rem;color:#64748b;margin-top:4px;">Vier Perspektiven auf Executive Compensation — eine vollständig verfügbare Analyse</div>
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("""<div class="lcard lcard-locked">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">🏦</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">Capital Allocators</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">Folgen dem Geld — wollen Risikosignale</div></div>
                <span class="badge-soon">In Entwicklung</span>
            </div>
            <div><span class="tag">📈 Inst. Investoren</span><span class="tag">🌱 ESG Fonds</span><span class="tag">🏛 Banken</span></div>
            <div style="margin-top:10px;font-size:.78rem;color:#94a3b8;font-style:italic;">"Ist diese Firma gut genug regiert, um mein Geld zu riskieren?"</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="lcard lcard-locked">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">🏢</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">Corporate Insiders</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">Setzen oder validieren Vergütung</div></div>
                <span class="badge-soon">In Entwicklung</span>
            </div>
            <div><span class="tag">🏛 Aufsichtsrat</span><span class="tag">👷 Betriebsrat</span><span class="tag">📊 HR</span></div>
            <div style="margin-top:10px;font-size:.78rem;color:#94a3b8;font-style:italic;">"Wo stehen wir im Marktvergleich?"</div>
        </div>""", unsafe_allow_html=True)
    with col_r:
        st.markdown(f"""<div class="lcard lcard-active">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">🌱</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">ESG &amp; CSRD Governance</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">Misst ESG-Vergütungsintegration &amp; Governance-Risiken</div></div>
                <span class="badge-live">✦ Vollständig verfügbar</span>
            </div>
            <div>
                <span class="tag tag-orange">📈 Historische Trends</span>
                <span class="tag tag-orange">🤖 Prognosemodell</span>
                <span class="tag tag-orange">⚠️ Governance Risk</span>
                <span class="tag tag-orange">👥 Peer Benchmarking</span>
                <span class="tag tag-orange">🔍 Anomalie-Detektor</span>
                <span class="tag tag-orange">🌱 ESG-Rating</span>
            </div>
            <div style="margin-top:10px;font-size:.78rem;color:#9a3412;font-style:italic;font-weight:500;">"Verdient dieser Vorstand angemessen — und warum?"</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Analyse starten →", type="primary", use_container_width=True):
            nav("overview")
        st.markdown("""<div class="lcard lcard-locked">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">
                <div><div style="font-size:1.3rem;margin-bottom:2px;">⚖️</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f2744;">Accountability Actors</div>
                <div style="font-size:.78rem;color:#64748b;margin-top:2px;">Halten Unternehmen rechenschaftspflichtig</div></div>
                <span class="badge-soon">In Entwicklung</span>
            </div>
            <div><span class="tag">🏛 BaFin</span><span class="tag">🎯 Proxy Advisors</span><span class="tag">📰 NGOs</span></div>
            <div style="margin-top:10px;font-size:.78rem;color:#94a3b8;font-style:italic;">"Wer verdient zu viel — und können wir das beweisen?"</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;color:#94a3b8;font-size:.74rem;margin-top:8px;">ORBIS/Bureau van Dijk · DGAP Vergütungsberichte · 43 DAX-Unternehmen · 2006–2024</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LAYER 2 — OVERVIEW (Company Snapshot + 6 Module Cards)
# ══════════════════════════════════════════════════════════════
def show_overview():
    sidebar_nav("landing", "← Zur Startseite")
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

    year_note = f"Modell + ESG verfügbar · {sel_year}" if sel_year > FEATURES_YEAR_MAX else f"Volle Daten · {sel_year}"
    st.markdown(f"""<div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:.7rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;">Unternehmens-Übersicht</div>
                <div style="font-size:2rem;font-weight:900;color:white;letter-spacing:-.5px;margin:4px 0;">{sel_co}</div>
                <div style="font-size:.84rem;color:#94a3b8;">DAX · {year_note} · Wähle ein Modul</div>
            </div>
            <div style="background:rgba(255,255,255,.06);border-radius:12px;padding:12px 20px;text-align:center;">
                <div style="font-size:1.4rem;font-weight:800;color:{t_col};">{t_icon} {f'{aep:+.0f}%' if aep is not None else '—'}</div>
                <div style="font-size:.72rem;color:#94a3b8;margin-top:3px;">vs. Modell-Erwartung {sel_year}</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
    # Banner when actual compensation amounts are not yet available (2022+)
    if sel_year > COMP_DATA_MAX:
        st.markdown(f"""<div style="background:{ORANGEBG};border:1px solid #fed7aa;border-left:4px solid {ORANGE};border-radius:0 10px 10px 0;padding:10px 16px;margin-bottom:16px;font-size:.84rem;color:#431407;">
            📅 <strong>Jahr {sel_year}:</strong> Prognosemodell, Peer-Ranking und ESG-KPI-Daten vollständig verfügbar.
            Vergütungsstruktur (STI/LTI-Mix, Schlechte-Zeiten) aus dem Berichtsjahr noch nicht im Datensatz — KI-Prognose aktiv.
        </div>""", unsafe_allow_html=True)

    # ── Snapshot KPIs ──
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    with k1: st.markdown(kpi_html(f"€{total_c/1000:.1f}M" if total_c else "—", "Gesamtvergütung", f"Vorstandsboard {sel_year}", NAVY), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html(f"{gov_sc:.0f}/100" if gov_sc else "—", "Governance Risk", "⚠ Hoch" if gov_sc and gov_sc>60 else "✅ Normal", risk_color(gov_sc or 0), risk_color(gov_sc or 0)), unsafe_allow_html=True)
    with k3: st.markdown(kpi_html(f"Top {100-peer_pct:.0f}%" if peer_pct else "—", "DAX Peer Rang", f"Perzentil {peer_pct:.0f}" if peer_pct else "n/a", NAVY), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html("🚨 Ja" if is_anom else ("✅ Nein" if is_anom is not None else "—"), "Anomalie", "Ungewöhnliche Struktur" if is_anom else "Im Normbereich", RED if is_anom else GREEN, RED if is_anom else GREEN), unsafe_allow_html=True)
    with k5: st.markdown(kpi_html(f"{esg_tot:.0f}%" if esg_tot else "0%", "ESG in Vergütung", "CSRD-relevant" if esg_tot > 0 else "Kein ESG-Bezug", ORANGE if esg_tot > 0 else "#94a3b8", GREEN if esg_tot > 0 else RED), unsafe_allow_html=True)
    with k6:
        sz_cnt = int(co_df["schlechte_zeiten"].sum()) if in_model and "schlechte_zeiten" in co_df.columns else 0
        st.markdown(kpi_html(str(sz_cnt), "Schlechte-Zeiten-Events", "Pay ↑ bei EBIT ↓", RED if sz_cnt > 0 else GREEN, RED if sz_cnt > 0 else GREEN), unsafe_allow_html=True)

    st.markdown('<div class="sec"><div class="sec-title">Module — wähle eine Analyseebene</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Jedes Modul deckt einen der fünf Hackathon-Anforderungsbereiche ab</div></div>', unsafe_allow_html=True)

    # ── Module Cards 3×2 ──
    MODULES = [
        ("historical", "📈", "Historische Trends",
         "Wie hat sich die Vergütung über 15 Jahre entwickelt? Krisenresistenz, STI/LTI-Mix, Strukturwandel.",
         f"+{co_df['comp_yoy_pct'].mean():.1f}% Ø/Jahr" if in_model and len(co_df)>0 else "2006–2024",
         "Ø Wachstum p.a.", "#dbeafe", "#1d4ed8"),
        ("prediction", "🤖", "Prognosemodell",
         "OLS-Modell schätzt erwartete Vergütung basierend auf Firmengröße, Sektor, Performance & Peer-Gruppe.",
         f"{aep:+.0f}% vs. Erwartung" if aep is not None else "R²=0.71",
         f"Modell-Abweichung {sel_year}", "#ede9fe", "#6d28d9"),
        ("governance", "⚠️", "Governance Risk Score",
         "Composite-Score aus Vergütungsvolatilität, CEO-Prämie & 'Schlechte Zeiten' — Warnsignal für Investoren.",
         f"{gov_sc:.0f}/100" if gov_sc else "n/a",
         "Ø Governance Risk", "#fef2f2", "#dc2626"),
        ("peer", "👥", "Peer Benchmarking",
         "Wo steht dieses Unternehmen vs. DAX-Peers? Sektor- & Größenvergleich mit Modell-Erwartung.",
         f"Top {100-peer_pct:.0f}%" if peer_pct else "DAX-15",
         "Peer-Rang im DAX", "#dcfce7", "#15803d"),
        ("anomaly", "🔍", "Anomalie-Detektor",
         "Erkennt ungewöhnliche Vergütungsstrukturen: atypischer STI/LTI-Mix, extreme Bonus-Gehalts-Verhältnisse.",
         "🚨 Anomalie" if is_anom else ("✅ Normal" if is_anom is not None else "—"),
         f"Status {sel_year}", "#fffbeb", "#d97706"),
        ("esg", "🌱", "ESG-Rating",
         "CSRD-Compliance, Nachhaltigkeits-Pay-Integration & externer ESG-Validierungscheck.",
         f"{esg_tot:.0f}% ESG" if esg_tot else "Kein ESG-Bezug",
         "ESG-Anteil STI+LTI", "#f0fdf4", "#16a34a"),
    ]

    cols_r1 = st.columns(3, gap="medium")
    cols_r2 = st.columns(3, gap="medium")

    for i, (key, icon, title, desc, stat, statlbl, icon_bg, icon_fg) in enumerate(MODULES):
        col = cols_r1[i] if i < 3 else cols_r2[i-3]
        with col:
            stat_color = RED if ("Anomalie" in stat or "n/a" in str(stat)) else ORANGE if "%" in str(stat) else NAVY
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
    sidebar_nav("overview", "← Zur Übersicht")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    co_df  = df[df["company_shortname"] == sel_co].sort_values("year")
    co_esg = esg[esg["company_shortname"] == sel_co].sort_values("year") if len(esg) > 0 else pd.DataFrame()
    dax_avg = df.groupby("year")["total_comp_bt"].mean().reset_index()

    module_header("📈", "Historische Trends", f"Vergütungsentwicklung 2006–2024 · DAX-Vergleich · Strukturwandel", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>Was zeigt dieses Modul?</strong> 15 Jahre Vergütungsdaten offenbaren: Wie krisenresistent ist die Vergütung?
        Wächst sie schneller als die operative Leistung? Hat sich der STI/LTI-Mix über Zeit verändert?
        Diese Fragen sind zentral für die CSRD-Berichterstattung und die Governance-Bewertung.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="sec"><div class="sec-title">Vergütungstrend vs. DAX-Durchschnitt</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Gesamtvergütung Vorstand (€ Tsd.) — blaue Linie = ausgewähltes Unternehmen</div></div>', unsafe_allow_html=True)
        if in_model and len(co_df) > 0:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dax_avg["year"], y=dax_avg["total_comp_bt"],
                fill="tozeroy", fillcolor="rgba(148,163,184,.08)",
                line=dict(color="#94a3b8", dash="dot", width=1.5),
                name="DAX-Durchschnitt",
                hovertemplate="DAX ø: €%{y:,.0f}K<extra></extra>"))
            fig.add_trace(go.Scatter(x=co_df["year"], y=co_df["total_comp_bt"],
                line=dict(color=ORANGE, width=3),
                mode="lines+markers", marker=dict(size=5),
                name=sel_co,
                hovertemplate=f"{sel_co}: €%{{y:,.0f}}K<extra></extra>"))
            # Crisis annotations
            crises = [(2009,"GFC"),(2012,"Euro-Krise"),(2020,"COVID")]
            for yr, lbl in crises:
                if yr in co_df["year"].values:
                    fig.add_vline(x=yr, line_dash="dot", line_color="#e2e8f0", line_width=1,
                                  annotation_text=lbl, annotation_font=dict(size=8, color=GRAY),
                                  annotation_position="top")
            fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Gesamtverg. (€ Tsd.)", gridcolor=GRAYLT),
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
                    name="DAX ø Ø Vergütung"))
                fig.add_trace(go.Scatter(x=co_esg["year"], y=co_esg["avg_comp"],
                    line=dict(color=ORANGE, width=3), mode="lines+markers",
                    name=sel_co))
                fig.update_layout(height=320, margin=dict(l=0,r=0,t=10,b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Ø Vergütung (€ Tsd.)", gridcolor=GRAYLT),
                    xaxis=dict(gridcolor=GRAYLT),
                    legend=dict(orientation="h", yanchor="bottom", y=1.01))
                st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="sec"><div class="sec-title">Key Metriken</div></div>', unsafe_allow_html=True)
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
                ("Gesamtwachstum", f"+{total_growth:.0f}%" if total_growth else "—",
                 RED if total_growth and total_growth>150 else GREEN),
                ("Ø Wachstum p.a.", f"{avg_yoy:+.1f}%" if pd.notna(avg_yoy) else "—", ORANGE),
                ("Ø EBIT-Wachstum p.a.", f"{avg_ebit:+.1f}%" if avg_ebit and pd.notna(avg_ebit) else "—", GRAY),
                ("Pay > EBIT Wachstum", "⚠ Ja" if avg_ebit and pd.notna(avg_ebit) and avg_yoy > avg_ebit else "✅ Nein",
                 RED if avg_ebit and pd.notna(avg_ebit) and avg_yoy > avg_ebit else GREEN),
                ("Vergütung in COVID-Jahr", f"{covid_chg:+.1f}%" if covid_chg and pd.notna(covid_chg) else "—",
                 RED if covid_chg and covid_chg > 0 else GREEN),
                ("3J-Volatilität", f"{vol_3yr:.1f}%" if vol_3yr and pd.notna(vol_3yr) else "—",
                 RED if vol_3yr and vol_3yr > 25 else GRAY),
            ]
            st.markdown('<div style="background:white;border-radius:12px;padding:14px 16px;border:1px solid #e2e8f0;">', unsafe_allow_html=True)
            for lbl, val, col in rows:
                st.markdown(f'<div class="metric-row"><span style="color:{GRAY};">{lbl}</span><span style="font-weight:700;color:{col};">{val}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if avg_ebit and pd.notna(avg_ebit) and avg_yoy > avg_ebit + 5:
                st.markdown(f"""<div class="flag-red" style="margin-top:12px;">
                    <strong>⚠ Pay-Performance-Lücke:</strong> Vergütungswachstum ({avg_yoy:.1f}% p.a.)
                    übertrifft EBIT-Wachstum ({avg_ebit:.1f}% p.a.) deutlich — ein klassisches
                    Governance-Warnsignal.
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Detaillierte Feature-Daten nur für Kerngruppe verfügbar.")

    # Comp Structure Evolution
    if in_model and len(co_df) > 0 and all(c in co_df.columns for c in ["fixed_pct","sti_pct","lti_pct"]):
        st.markdown('<div class="sec"><div class="sec-title">Vergütungsstruktur-Evolution (Fix / STI / LTI)</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Wie hat sich der Mix aus fixer und variabler Vergütung verändert?</div></div>', unsafe_allow_html=True)
        st.markdown("""<div class="insight">
            <strong>Warum relevant?</strong> Ein steigender LTI-Anteil stärkt langfristige Pay-for-Performance-Anreize.
            Starke STI-Dominanz kann kurzfristiges Denken fördern. CSRD verlangt Offenlegung der Incentive-Struktur.
        </div>""", unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Fix", x=co_df["year"], y=co_df["fixed_pct"], marker_color="#3b82f6"))
        fig2.add_trace(go.Bar(name="STI (Kurzfrist)", x=co_df["year"], y=co_df["sti_pct"], marker_color=ORANGE))
        fig2.add_trace(go.Bar(name="LTI (Langfrist)", x=co_df["year"], y=co_df["lti_pct"], marker_color=NAVY))
        fig2.update_layout(barmode="stack", height=260, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Anteil (%)", range=[0,100], gridcolor=GRAYLT),
            xaxis=dict(gridcolor=GRAYLT),
            legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 2 — PROGNOSEMODELL
# ─────────────────────────────────────────────
def show_prediction():
    sidebar_nav("overview", "← Zur Übersicht")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    in_mu    = sel_co in MU_COS
    # Use model_universe for predictions (2007-2024, 42 companies)
    co_mu_all = mu[mu["company_shortname"] == sel_co].sort_values("year")
    co_df     = df[df["company_shortname"] == sel_co].sort_values("year")  # rich features 2006-2024

    module_header("🤖", "Prognosemodell", f"OLS-Regressionsmodell · R²=0.71 · Expected vs. Actual Pay · 2007–{MODEL_YEAR_MAX}", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>Wie funktioniert das Modell?</strong> Unser OLS-Modell schätzt die erwartete Vorstandsvergütung
        basierend auf: Vorjahresvergütung (Pay Stickiness), Vorstandsgröße, Sektor und Jahrestrend.
        Liegt ein Unternehmen >15% über der Erwartung, deutet das auf einen möglichen Governance-Issue hin.
        R²=0.71 bedeutet: 71% der Vergütungsvariation werden durch das Modell erklärt.
        Das Modell gilt für alle 42 DAX-Unternehmen von 2007 bis 2024.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="sec"><div class="sec-title">Tatsächlich vs. Modell-Erwartung</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Oranger Bereich = 80% Erwartungsintervall · Blaue Linie = Tatsächliche Vergütung</div></div>', unsafe_allow_html=True)
        if in_mu and len(co_mu_all) > 0 and co_mu_all["pred_comp"].notna().any():
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(co_mu_all["year"])+list(co_mu_all["year"])[::-1],
                y=list(co_mu_all["pred_comp_high"])+list(co_mu_all["pred_comp_low"])[::-1],
                fill="toself", fillcolor="rgba(249,115,22,.10)",
                line=dict(color="rgba(0,0,0,0)"), name="80% Erwartungsband"))
            fig.add_trace(go.Scatter(x=co_mu_all["year"], y=co_mu_all["pred_comp"],
                line=dict(color=ORANGE, dash="dash", width=2),
                name="Modell-Erwartung",
                hovertemplate="Erwartet: €%{y:,.0f}K<extra></extra>"))
            fig.add_trace(go.Scatter(x=co_mu_all["year"], y=co_mu_all["total_comp_bt"],
                line=dict(color=NAVY, width=3), mode="lines+markers",
                name="Tatsächlich",
                hovertemplate="Tatsächlich: €%{y:,.0f}K<extra></extra>"))
            # Flag overpaid years
            over = co_mu_all[co_mu_all["actual_vs_expected_pct"] > 40]
            if len(over) > 0:
                fig.add_trace(go.Scatter(x=over["year"], y=over["total_comp_bt"],
                    mode="markers", marker=dict(symbol="x", size=14, color=RED, line=dict(width=2.5)),
                    name="⚠ Deutlich überbezahlt",
                    hovertemplate="⚠ Jahr %{x}: deutlich über Erwartung<extra></extra>"))
            fig.update_layout(height=330, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="Gesamtvergütung (€ Tsd.)", gridcolor=GRAYLT),
                xaxis=dict(gridcolor=GRAYLT),
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
                hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"{sel_co} nicht im Modell-Universum — keine Prediction verfügbar.")

    with col_r:
        st.markdown('<div class="sec"><div class="sec-title">Abweichung vom Modell</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Positiv = überbezahlt, Negativ = unterbezahlt</div></div>', unsafe_allow_html=True)
        if in_mu and co_mu_all["actual_vs_expected_pct"].notna().any():
            aep_series = co_mu_all.dropna(subset=["actual_vs_expected_pct"])
            colors_aep = [RED if v>15 else GREEN if v<-15 else AMBER for v in aep_series["actual_vs_expected_pct"]]
            fig2 = go.Figure(go.Bar(
                x=aep_series["year"], y=aep_series["actual_vs_expected_pct"],
                marker_color=colors_aep,
                hovertemplate="Jahr %{x}: %{y:+.1f}%<extra></extra>"))
            fig2.add_hline(y=0, line_color=GRAY, line_width=1)
            fig2.add_hline(y=15, line_dash="dot", line_color=AMBER, line_width=1,
                           annotation_text="+15% Schwelle", annotation_font=dict(size=8))
            fig2.add_hline(y=-15, line_dash="dot", line_color="#2563eb", line_width=1)
            fig2.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(title="vs. Erwartung (%)", gridcolor=GRAYLT),
                xaxis=dict(gridcolor=GRAYLT))
            st.plotly_chart(fig2, use_container_width=True)

            co_mu_yr = co_mu_all[co_mu_all["year"] == sel_year]
            aep_now = float(co_mu_yr["actual_vs_expected_pct"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["actual_vs_expected_pct"].notna().any() else None
            if aep_now is not None:
                icon, col, lbl = traffic(aep_now)
                st.markdown(f"""<div style="background:{NAVY};border-radius:10px;padding:14px;text-align:center;">
                    <div style="font-size:1.8rem;font-weight:800;color:{col};">{aep_now:+.1f}%</div>
                    <div style="font-size:.72rem;color:#94a3b8;text-transform:uppercase;margin-top:4px;">vs. Modell {sel_year}</div>
                    <div style="font-size:.82rem;color:{col};margin-top:6px;font-weight:600;">{icon} {lbl}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Nicht im Modell-Universum.")

    # Feature Importance
    st.markdown('<div class="sec"><div class="sec-title">Was treibt die Vergütung? — Modell-Koeffizienten</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Wie stark beeinflusst jeder Faktor das Vergütungsniveau? (Effekt in %)</div></div>', unsafe_allow_html=True)
    st.markdown("""<div class="insight">
        <strong>Interpretation:</strong> "log_comp_lag1 +131%" bedeutet: die Vorjahresvergütung ist der
        stärkste Einzeltreiber — Vergütung klebt (Pay Stickiness). Branchendummies zeigen, wie viel
        Industrie-Zugehörigkeit allein die Vergütung nach oben oder unten bewegt, unabhängig von Performance.
    </div>""", unsafe_allow_html=True)
    top_c = coefs[coefs["feature"] != "intercept"].sort_values("exp_effect_pct", ascending=True)
    lbl_map = {"log_comp_lag1":"Vorjahresvergütung (Stickiness)","year_trend":"Jahrestrend",
               "log_board_size":"Vorstandsgröße (log)"}
    top_c["label"] = top_c["feature"].apply(lambda x: lbl_map.get(x, x.replace("sector_","Sektor: ").replace("_"," ")))
    fig3 = go.Figure(go.Bar(
        x=top_c["exp_effect_pct"], y=top_c["label"], orientation="h",
        marker_color=[GREEN if v>0 else RED for v in top_c["exp_effect_pct"]],
        text=[f"{v:+.0f}%" for v in top_c["exp_effect_pct"]], textposition="outside",
        hovertemplate="%{y}: %{x:+.0f}%<extra></extra>"))
    fig3.update_layout(height=380, margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Effekt auf Vergütungsniveau (%)", gridcolor=GRAYLT),
        yaxis=dict(tickfont=dict(size=10)))
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 3 — GOVERNANCE RISK SCORE
# ─────────────────────────────────────────────
def show_governance():
    sidebar_nav("overview", "← Zur Übersicht")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    co_df = df[df["company_shortname"] == sel_co].sort_values("year")

    module_header("⚠️", "Governance Risk Score", "Composite-Score · Schlechte-Zeiten-Analyse · CEO-Prämien · Red Flags", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>Was ist der Governance Risk Score?</strong> Ein Composite aus: Vergütungsvolatilität,
        CEO-Aufschlag gegenüber dem Restvorstand, Peer-Abweichung und "Schlechte-Zeiten"-Events
        (Vergütung stieg, obwohl EBIT <em>und</em> Headcount sanken). Hohe Werte sind ein Warnsignal
        für institutionelle Investoren und Proxy Advisors bei Say-on-Pay-Abstimmungen.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="sec"><div class="sec-title">Governance Risk Heatmap — DAX Kerngruppe</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Rot = hohes Risiko · Grün = niedriges Risiko · Orangener Rahmen = ausgewähltes Unternehmen</div></div>', unsafe_allow_html=True)
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
            sc_lbl = "Niedriges Risiko" if avg_sc < 40 else "Moderates Risiko" if avg_sc < 60 else "Hohes Risiko"
            st.markdown(f"""<div style="background:{NAVY};border-radius:12px;padding:16px;text-align:center;margin-bottom:14px;">
                <div style="font-size:2.6rem;font-weight:800;color:{sc_fg};line-height:1;">{avg_sc:.0f}<span style="font-size:1.2rem;color:#475569;">/100</span></div>
                <div style="font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin-top:4px;">Ø Governance Risk Score</div>
                <div style="background:{sc_bg};border-radius:6px;padding:3px 12px;display:inline-block;margin-top:8px;font-size:.78rem;font-weight:700;color:{sc_fg};">{sc_lbl}</div>
            </div>""", unsafe_allow_html=True)

            # Red flags
            flags = [
                (sz_cnt > 0,
                 f"⚠ {sz_cnt}× Schlechte-Zeiten-Event(s): Vergütung ↑ bei EBIT ↓ ({list(sz_rows['year'].astype(int)) if len(sz_rows)>0 else ''})",
                 "red"),
                (ceo_prem and ceo_prem > 2.5,
                 f"⚠ Hohe CEO-Prämie: {ceo_prem:.1f}x Ø Vorstand (DAX-Median ~2x)",
                 "amber"),
                (aep_over > 0,
                 f"⚠ {aep_over} Jahr(e) mit >+40% über Modell-Erwartung",
                 "red"),
                (vol and vol > 25,
                 f"⚠ Hohe Vergütungsvolatilität: Ø {vol:.0f}% Std. über 3 Jahre",
                 "amber"),
            ]
            has_flag = False
            for condition, text, severity in flags:
                if condition:
                    cls = f"flag-{severity}"
                    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)
                    has_flag = True
            if not has_flag:
                st.markdown(f'<div class="flag-green">✅ Keine wesentlichen Governance Red Flags erkannt.</div>', unsafe_allow_html=True)
        else:
            st.info(f"Governance Risk Score nur für Kerngruppe. {sel_co} ist nicht enthalten.")

    # Schlechte Zeiten Tabelle
    st.markdown('<div class="sec"><div class="sec-title">Schlechte-Zeiten-Ereignisse — DAX Kerngruppe</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Vergütung stieg, obwohl EBIT und Headcount sanken — klarer Pay-for-Performance-Verstoß</div></div>', unsafe_allow_html=True)
    if "schlechte_zeiten" in df.columns:
        sz_all = df[df["schlechte_zeiten"]==1][["company_shortname","year","comp_yoy_pct","ebit_yoy_pct","empl_yoy_pct","schlechte_zeiten_score"]].sort_values("schlechte_zeiten_score", ascending=False)
        if len(sz_all) > 0:
            st.dataframe(sz_all.rename(columns={
                "company_shortname":"Unternehmen","year":"Jahr",
                "comp_yoy_pct":"Vergütung %","ebit_yoy_pct":"EBIT %",
                "empl_yoy_pct":"Headcount %","schlechte_zeiten_score":"Schwere"
            }).round(1), use_container_width=True, hide_index=True)
        else:
            st.success("Keine Schlechte-Zeiten-Ereignisse gefunden.")

# ─────────────────────────────────────────────
# MODULE 4 — PEER BENCHMARKING
# ─────────────────────────────────────────────
def show_peer():
    sidebar_nav("overview", "← Zur Übersicht")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    # Use model_universe for full 42-company peer comparison (2007-2024)
    year_data = mu[mu["year"] == sel_year].sort_values("total_comp_bt", ascending=False)
    co_df = df[df["company_shortname"] == sel_co]  # rich features for sector comparison

    module_header("👥", "Peer Benchmarking", f"DAX-Universum Vergleich · {sel_year} · 42 Unternehmen · Sektorpositioning", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>Peer Benchmarking:</strong> Vergleicht die tatsächliche Vergütung mit dem Modell-Erwartungsband
        aller DAX-Unternehmen im gleichen Jahr. Farbe = Benchmark-Signal:
        🔴 deutlich überbezahlt · 🟡 über/unter Markt · 🟢 marktkonform.
        Zusätzlich: Wie steht das Unternehmen im Sektor-Peer-Vergleich?
    </div>""", unsafe_allow_html=True)

    if len(year_data) > 0:
        k1,k2,k3,k4 = st.columns(4)
        co_mu_yr = year_data[year_data["company_shortname"]==sel_co]
        rank   = int(co_mu_yr["peer_rank_total_comp"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["peer_rank_total_comp"].notna().any() else None
        pctile = float(co_mu_yr["peer_pct_total_comp"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["peer_pct_total_comp"].notna().any() else None
        aep    = float(co_mu_yr["actual_vs_expected_pct"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["actual_vs_expected_pct"].notna().any() else None
        total  = float(co_mu_yr["total_comp_bt"].iloc[0]) if len(co_mu_yr)>0 and co_mu_yr["total_comp_bt"].notna().any() else None
        _, t_col, t_lbl = traffic(aep)
        with k1: st.markdown(kpi_html(f"#{rank}" if rank else "—", "Peer-Rang DAX", f"von {len(year_data)} Unternehmen"), unsafe_allow_html=True)
        with k2: st.markdown(kpi_html(f"Top {100-pctile:.0f}%" if pctile else "—", "Perzentil", f"Vergütungshöhe {sel_year}"), unsafe_allow_html=True)
        with k3: st.markdown(kpi_html(f"{aep:+.0f}%" if aep is not None else "—", "vs. Modell", t_lbl, t_col, t_col), unsafe_allow_html=True)
        with k4: st.markdown(kpi_html(f"€{total/1000:.1f}M" if total else "—", "Gesamtvergütung", "Vorstandsboard gesamt"), unsafe_allow_html=True)

    st.markdown('<div class="sec"><div class="sec-title">Vergütungsvergleich — alle DAX-Unternehmen</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Balken = tatsächliche Vergütung · Rauten = Modell-Erwartungsband · Farbe = Benchmark-Signal</div></div>', unsafe_allow_html=True)

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
            name="Modell-Erwartung (80%-Band)"))
        fig.add_trace(go.Bar(
            x=year_data["company_shortname"], y=year_data["total_comp_bt"],
            marker_color=colors, opacity=0.85, name="Tatsächlich",
            hovertemplate="%{x}: €%{y:,.0f}K<extra></extra>"))
        fig.update_layout(height=360, barmode="overlay",
            margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Gesamtvergütung (€ Tsd.)", gridcolor=GRAYLT),
            xaxis=dict(tickangle=-30, gridcolor=GRAYLT),
            legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig, use_container_width=True)

    # Sector comparison
    if in_model and "sector" in df.columns:
        co_sector = df[df["company_shortname"]==sel_co]["sector"].iloc[0] if len(df[df["company_shortname"]==sel_co])>0 else None
        if co_sector:
            st.markdown(f'<div class="sec"><div class="sec-title">Sektorvergleich: {co_sector}</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Wie steht {sel_co} vs. Unternehmen im gleichen Sektor?</div></div>', unsafe_allow_html=True)
            sect_data = df[(df["sector"]==co_sector) & (df["year"]==sel_year)].sort_values("total_comp_bt", ascending=False)
            if len(sect_data) > 1:
                sect_colors = [ORANGE if c==sel_co else "#cbd5e1" for c in sect_data["company_shortname"]]
                fig2 = go.Figure(go.Bar(
                    x=sect_data["company_shortname"], y=sect_data["total_comp_bt"],
                    marker_color=sect_colors,
                    hovertemplate="%{x}: €%{y:,.0f}K<extra></extra>"))
                fig2.update_layout(height=240, margin=dict(l=0,r=0,t=10,b=0),
                    plot_bgcolor="white", paper_bgcolor="white",
                    yaxis=dict(title="Gesamtvergütung (€ Tsd.)", gridcolor=GRAYLT),
                    xaxis=dict(tickangle=-20))
                st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 5 — ANOMALIE-DETEKTOR
# ─────────────────────────────────────────────
def show_anomaly():
    sidebar_nav("overview", "← Zur Übersicht")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    in_model = sel_co in MODEL_COS
    co_df = df[df["company_shortname"] == sel_co].sort_values("year")
    co_yr = df[(df["company_shortname"]==sel_co) & (df["year"]==sel_year)]

    module_header("🔍", "Anomalie-Detektor", "Ungewöhnliche Vergütungsstrukturen · Outlier-Erkennung · Universum-Vergleich", sel_co, sel_year)

    st.markdown("""<div class="insight">
        <strong>Was ist eine Anomalie?</strong> Unser Detektor flaggt Unternehmen mit atypischem
        STI/LTI-Mix, extremem Bonus-Gehalts-Verhältnis oder ungewöhnlich starken Vergütungssprüngen —
        auch wenn die absolute Höhe marktkonform wäre. Anomalien können auf Short-Term-Bias,
        versteckte One-Time-Payments oder intransparente Incentive-Designs hinweisen.
    </div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="sec"><div class="sec-title">Anomalie-Score vs. Vergütungshöhe — Universum</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Alle Unternehmen im Datensatz · Orange markiert = ausgewählt · Rot = flagged als Anomalie</div></div>', unsafe_allow_html=True)
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
                name="🚨 Anomalie", hovertemplate="<b>%{text}</b><br>Verg.: €%{x:,.0f}K<br>Score: %{y:.1f}<extra></extra>",
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
                          annotation_text="Anomalie-Schwelle", annotation_font=dict(size=8, color=AMBER))
            fig.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Gesamtvergütung (€ Tsd.)", gridcolor=GRAYLT),
                yaxis=dict(title="Anomalie-Score (0–100)", gridcolor=GRAYLT),
                legend=dict(orientation="h", yanchor="bottom", y=1.01),
                hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(f'<div class="sec"><div class="sec-title">Anomalie-Profil — {sel_co}</div></div>', unsafe_allow_html=True)
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
            badge_txt = "🚨 Anomalie erkannt" if is_anom else "✅ Keine Anomalie"
            st.markdown(f"""<div style="background:{NAVY};border-radius:12px;padding:16px;text-align:center;margin-bottom:14px;">
                <div style="font-size:2.4rem;font-weight:800;color:{badge_c};line-height:1;">{anom_sc:.0f}<span style="font-size:1.1rem;color:#475569;">/100</span></div>
                <div style="font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-top:4px;">Anomalie-Score {sel_year}</div>
                <div style="background:{badge_bg};border-radius:6px;padding:3px 12px;display:inline-block;margin-top:8px;font-size:.78rem;font-weight:700;color:{badge_c};">{badge_txt}</div>
            </div>""", unsafe_allow_html=True)

            # Structure breakdown
            dax_fix = df[df["year"]==sel_year]["fixed_pct"].mean()
            dax_sti = df[df["year"]==sel_year]["sti_pct"].mean()
            dax_lti = df[df["year"]==sel_year]["lti_pct"].mean()
            rows = [
                ("Fix-Anteil", f"{fix_p:.0f}%" if fix_p else "—", f"DAX ø {dax_fix:.0f}%",
                 RED if fix_p and abs(fix_p-dax_fix)>15 else GRAY),
                ("STI-Anteil", f"{sti_p:.0f}%" if sti_p else "—", f"DAX ø {dax_sti:.0f}%",
                 RED if sti_p and abs(sti_p-dax_sti)>15 else GRAY),
                ("LTI-Anteil", f"{lti_p:.0f}%" if lti_p else "—", f"DAX ø {dax_lti:.0f}%",
                 RED if lti_p and abs(lti_p-dax_lti)>15 else GRAY),
                ("Bonus/Gehalt", f"{bsr:.1f}x" if bsr else "—", "DAX ø ~1.5x",
                 RED if bsr and bsr > 3 else GRAY),
                ("Vergütungswachstum YoY", f"{comp_yoy:+.1f}%" if comp_yoy and pd.notna(comp_yoy) else "—", "",
                 RED if comp_yoy and comp_yoy > 30 else GRAY),
            ]
            st.markdown('<div style="background:white;border-radius:12px;padding:14px 16px;border:1px solid #e2e8f0;">', unsafe_allow_html=True)
            for lbl, val, ref, col in rows:
                st.markdown(f'<div class="metric-row"><span style="color:{GRAY};">{lbl}<br><span style="font-size:.7rem;color:#94a3b8;">{ref}</span></span><span style="font-weight:700;color:{col};">{val}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info(f"Anomalie-Score für {sel_co} in Kerngruppe nicht verfügbar.")

    # Timeline
    if in_model and len(co_df) > 0 and "anomaly_score_pct" in co_df.columns:
        st.markdown('<div class="sec"><div class="sec-title">Anomalie-Score im Zeitverlauf</div></div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=co_df["year"], y=co_df["anomaly_score_pct"],
            fill="tozeroy", fillcolor="rgba(249,115,22,.10)",
            line=dict(color=ORANGE, width=2.5), mode="lines+markers",
            marker=dict(size=6, color=[RED if v else ORANGE for v in co_df.get("is_anomaly", pd.Series(False, index=co_df.index))]),
            hovertemplate="Jahr %{x}: Score %{y:.1f}<extra></extra>"))
        fig2.add_hline(y=50, line_dash="dot", line_color=AMBER, line_width=1,
                       annotation_text="Anomalie-Schwelle", annotation_font=dict(size=8))
        fig2.update_layout(height=200, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(range=[0,105], gridcolor=GRAYLT), xaxis=dict(gridcolor=GRAYLT))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# MODULE 6 — ESG RATING
# ─────────────────────────────────────────────
def show_esg():
    sidebar_nav("overview", "← Zur Übersicht")
    sel_co, sel_year = st.session_state.company, st.session_state.year
    co_esg = esg[esg["company_shortname"] == sel_co].sort_values("year") if len(esg) > 0 else pd.DataFrame()

    # For ESG metrics: prefer selected year; fall back to latest available
    co_esg_yr = co_esg[co_esg["year"] == sel_year] if len(co_esg) > 0 else pd.DataFrame()
    if len(co_esg_yr) == 0 and len(co_esg) > 0:
        co_esg_yr = co_esg.iloc[[-1]]
    actual_esg_year = int(co_esg_yr["year"].iloc[0]) if len(co_esg_yr) > 0 else sel_year

    module_header("🌱", "ESG-Rating", f"ESG-Integration in Vergütung · CSRD-Compliance · Gender Equity · Pay Ratio · Datenjahr: {actual_esg_year}", sel_co, sel_year)

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
        st.markdown(f'<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:8px 14px;margin-bottom:10px;font-size:.8rem;color:#713f12;">ℹ️ Keine ESG-Daten für {sel_year} — zeige letztverfügbares Jahr: <strong>{actual_esg_year}</strong></div>', unsafe_allow_html=True)

    st.markdown(f"""<div class="insight">
        <strong>ESG & CSRD:</strong> Die EU Corporate Sustainability Reporting Directive (CSRD) ist seit 2025
        verpflichtend. Sie verlangt explizite ESG-Vergütungsziele, verifizierbare KPI-Texte und
        Offenlegung der Anreizstruktur. Nur <strong>16 von 43 DAX-Unternehmen (37%)</strong> erfüllen
        diesen Standard. {sel_co} hat aktuell <strong>{esg_tot:.0f}% ESG-Anteil</strong> in STI+LTI.
    </div>""", unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    with k1: st.markdown(kpi_html(f"{sti_esg:.0f}%" if sti_esg else "0%", "ESG-Anteil STI", "✅ CSRD-relevant" if sti_esg and sti_esg>0 else "❌ Kein ESG", ORANGE if sti_esg and sti_esg>0 else "#94a3b8", GREEN if sti_esg and sti_esg>0 else RED), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html(f"{lti_esg:.0f}%" if lti_esg else "0%", "ESG-Anteil LTI", "Langfristige Nachhaltigkeitsziele", ORANGE if lti_esg and lti_esg>0 else "#94a3b8"), unsafe_allow_html=True)
    with k3: st.markdown(kpi_html(f"{fem_pct:.0f}%" if fem_pct else "—", "Frauen im Vorstand", "✅ ARUG II ≥30%" if fem_pct and fem_pct>=30 else "⚠ Unter 30%-Ziel", NAVY, GREEN if fem_pct and fem_pct>=30 else AMBER), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html(f"{w_ratio:.0f}x" if w_ratio else "—", "Exec/Worker Pay", "⚠ Sehr hoch" if w_ratio and w_ratio>60 else "Einkommensungleichheit", RED if w_ratio and w_ratio>60 else NAVY, RED if w_ratio and w_ratio>60 else GRAY), unsafe_allow_html=True)

    # DAX Bubble Chart
    st.markdown('<div class="sec"><div class="sec-title">DAX-Überblick: Wer integriert ESG wirklich?</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">X/Y = ESG-Anteil STI/LTI · Punktgröße = Ø Vergütung · Orange = mit ESG · Grau = ohne ESG</div></div>', unsafe_allow_html=True)

    if len(esg) > 0:
        recent = esg[esg["year"]>=2022]
        bub = recent.groupby("company_shortname").agg(
            sti=("sti_esg_share","mean"), lti=("lti_esg_share","mean"),
            avg_comp=("avg_comp","mean")).reset_index().fillna(0)
        bub["has_esg"] = (bub["sti"]+bub["lti"]) > 0
        bub["is_sel"]  = bub["company_shortname"] == sel_co

        fig = go.Figure()
        for has, color, name in [(False,"#cbd5e1","Kein ESG-Bezug"),(True,ORANGE,"Mit ESG-Bezug")]:
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
        for x,y,txt,col in [(45,45,"ESG-Leader","#15803d"),(3,45,"LTI-fokussiert","#1d4ed8"),(45,3,"STI-fokussiert",AMBER),(3,3,"Kein ESG-Bezug",RED)]:
            fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(color=col,size=9), opacity=0.5)
        fig.add_hline(y=20,line_dash="dot",line_color="#e2e8f0",line_width=1)
        fig.add_vline(x=20,line_dash="dot",line_color="#e2e8f0",line_width=1)
        fig.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="ESG-Anteil STI (%)", range=[-5,68], gridcolor=GRAYLT),
            yaxis=dict(title="ESG-Anteil LTI (%)", range=[-5,68], gridcolor=GRAYLT),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.01))
        st.plotly_chart(fig, use_container_width=True)

    col_kpi, col_csrd = st.columns([3, 2], gap="large")

    with col_kpi:
        st.markdown(f"**KPI-Texte aus dem Vergütungsbericht — {sel_co} ({actual_esg_year})**")
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
                    <strong>⚠️ Disclosure-Opacity:</strong> {sel_co} verwendet abstrakte KPI-Begriffe —
                    ein externer ESG-Inhalt ist nicht verifizierbar. Genau das adressiert die CSRD.
                </div>""", unsafe_allow_html=True)
        else:
            st.info("KPI-Texte verfügbar für 2022–2024. Für frühere Jahre wurden keine Textdaten erhoben.")

    with col_csrd:
        st.markdown(f"**CSRD-Compliance-Check ({actual_esg_year})**")
        st.caption("EU Corporate Sustainability Reporting Directive — verpflichtend ab 2025")
        OPACITY = {"BMW","Volkswagen","Mercedes-Benz"}
        has_kpi_text = len(co_kpi_rows) > 0
        checks = [
            ("ESG-Ziel in STI verankert",           sti_esg is not None and sti_esg > 0),
            ("ESG-Ziel in LTI verankert",           lti_esg is not None and lti_esg > 0),
            ("KPI-Text öffentlich verifizierbar",   has_kpi_text and sel_co not in OPACITY),
            ("Keine abstrakten Blanket-KPIs",       sel_co not in OPACITY),
            ("Frauenquote ≥30% (ARUG II)",          fem_pct is not None and fem_pct >= 30),
        ]
        score = sum(1 for _,ok in checks if ok)
        s_fg  = "#4ade80" if score>=4 else "#fbbf24" if score>=3 else "#f87171"
        s_bg  = "#14532d" if score>=4 else "#713f12" if score>=3 else "#7f1d1d"
        s_lbl = "Compliant" if score>=4 else "Teilweise" if score>=3 else "Nicht compliant"
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
    st.markdown('<div class="sec"><div class="sec-title">Pay Equity — Gender &amp; CEO/Worker Ratio</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Frauenanteil vs. ARUG II-Ziel · Einkommensungleichheit im DAX-Vergleich</div></div>', unsafe_allow_html=True)
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
            xaxis=dict(title="Frauenanteil (%)", range=[0,65], gridcolor=GRAYLT),
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
    st.markdown('<div class="sec"><div class="sec-title">🔬 Externer ESG-Validierungscheck — Pay-Washing Detektor</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Vergleicht unsere abgeleitete ESG-Pay-Verknüpfung mit Sustainalytics-Risikorating & MSCI ESG-Score</div></div>', unsafe_allow_html=True)

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
                    <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:8px;">Sustainalytics ESG Risk</div>
                    <div style="font-size:2.6rem;font-weight:900;color:{sust_col};line-height:1;">{sust_score:.1f}</div>
                    <div style="font-size:.7rem;color:#94a3b8;margin-top:3px;">/ 50 · {sust_cat} Risk</div>
                    <div style="background:{sust_col}22;border:1px solid {sust_col}44;border-radius:8px;padding:4px 12px;display:inline-block;margin-top:8px;font-size:.72rem;font-weight:700;color:{sust_col};">{"✓ Gut" if sust_cat in ("Low","Negligible") else "⚠ Mittel" if sust_cat=="Medium" else "✗ Hoch"}</div>
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
                        <div style="font-size:.7rem;color:#94a3b8;margin-top:3px;">/ 100 · abgeleitet aus KPI-Analyse</div>
                        <div style="background:{pcolor}22;border:1px solid {pcolor}44;border-radius:8px;padding:4px 12px;display:inline-block;margin-top:8px;font-size:.72rem;font-weight:700;color:{pcolor};">{"ESG-Zahler" if pay_esg_score>=25 else "Niedrig" if pay_esg_score>=5 else "Kein ESG-Bezug"}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="background:{NAVY};border-radius:16px;padding:20px;text-align:center;">
                        <div style="font-size:.65rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:8px;">ExComp ESG-Pay Score</div>
                        <div style="font-size:2.6rem;font-weight:900;color:{AMBER};line-height:1;">N/V</div>
                        <div style="font-size:.7rem;color:#94a3b8;margin-top:3px;">Nicht verifizierbar</div>
                        <div style="background:{AMBER}22;border:1px solid {AMBER}44;border-radius:8px;padding:4px 12px;display:inline-block;margin-top:8px;font-size:.72rem;font-weight:700;color:{AMBER};">⚠ Opacity</div>
                    </div>""", unsafe_allow_html=True)

            # Pay-Washing verdict
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            if wash_risk == "aligned":
                st.markdown(f"""<div class="flag-green">
                    <strong>✅ ESG-Pay Alignment:</strong> {sel_co} zeigt Konsistenz zwischen externen ESG-Ratings
                    (Sustainalytics: {sust_score:.1f} · MSCI: {msci_rating}) und ESG-Vergütungsintegration ({esg_tot:.0f}% in Pay).
                    Das Vergütungssystem stützt die ESG-Leadership-Position.
                </div>""", unsafe_allow_html=True)
            elif wash_risk == "washing":
                st.markdown(f"""<div class="flag-red">
                    <strong>🚨 ESG Pay-Washing Risiko:</strong> {sel_co} genießt ein starkes externes ESG-Rating
                    (Sustainalytics: {sust_score:.1f} Low/Medium · MSCI: {msci_rating}), verknüpft jedoch
                    <strong>0% der Vorstandsvergütung</strong> mit ESG-Zielen. Externe Reputation ohne interne Anreize —
                    exakt das, was die CSRD sichtbar machen will.
                </div>""", unsafe_allow_html=True)
            elif wash_risk == "disconnect":
                st.markdown(f"""<div class="flag-amber">
                    <strong>⚠ ESG Pay-Performance Lücke:</strong> {sel_co} zahlt {esg_tot:.0f}% der Vergütung
                    auf ESG-Ziele ({AMBER}Sustainalytics: {sust_score:.1f} · MSCI: {msci_rating}),
                    zeigt jedoch noch überdurchschnittliches ESG-Risiko. Pay-for-ESG allein reicht nicht — die Zielqualität entscheidet.
                </div>""", unsafe_allow_html=True)
            elif wash_risk == "opacity":
                st.markdown(f"""<div class="flag-amber">
                    <strong>⚠ Disclosure-Opacity:</strong> {sel_co} verwendet abstrakte Vergütungsbegriffe —
                    eine externe Verifizierung des ESG-Anteils ist nicht möglich. CSRD §29a
                    verlangt explizite, überprüfbare ESG-KPI-Texte. Externes Rating: MSCI {msci_rating} ·
                    Sustainalytics {sust_score:.1f}.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="flag-amber">
                    <strong>ℹ ESG-Pay im Aufbau:</strong> Niedriger ESG-Anteil in der Vergütung bei moderaten
                    externen Ratings (Sustainalytics: {sust_score:.1f} · MSCI: {msci_rating}).
                    CSRD-Compliance-Druck wird Anpassungen erfordern.
                </div>""", unsafe_allow_html=True)

        # DAX Pay-Washing Heatmap
        st.markdown('<div class="sec"><div class="sec-title">DAX-Überblick: ESG-Pay vs. Externer ESG-Score</div><div class="sec-sub" style="font-size:.78rem;color:#64748b;margin-top:2px;">Quadranten: Wer zahlt für ESG — und hat das externe Scores verbessert?</div></div>', unsafe_allow_html=True)
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
        for color, label in [("#16a34a","✅ ESG-Leader"), ("#dc2626","🚨 Pay-Washing-Risiko"),
                              ("#d97706","⚠ Pay ohne Ergebnis"), ("#94a3b8","Neutral")]:
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
        for x, y, txt, col in [(35,85,"ESG-Leader","#15803d"), (2,85,"🚨 Pay-Washing","#dc2626"),
                                (35,35,"Pay ohne Ergebnis","#d97706"), (2,35,"Neutral","#94a3b8")]:
            fig_w.add_annotation(x=x, y=y, text=txt, showarrow=False,
                font=dict(color=col, size=9), opacity=0.6)
        fig_w.update_layout(
            height=440, margin=dict(l=0,r=0,t=10,b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="ESG-Anteil in Vergütung STI+LTI (%)", range=[-5, 68], gridcolor=GRAYLT),
            yaxis=dict(title="ESG-Performance-Score (Sustainalytics, invertiert)", range=[20, 100], gridcolor=GRAYLT),
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, font=dict(size=9)))
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown('<div style="text-align:center;color:#94a3b8;font-size:.74rem;padding:16px 0 4px 0;border-top:1px solid #e2e8f0;margin-top:12px;">ExComp · TUM Science Hackathon 2026 · ORBIS/Bureau van Dijk · DGAP · Sustainalytics · MSCI ESG · 43 DAX-Unternehmen · 2006–2024</div>', unsafe_allow_html=True)


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

screen = st.session_state.screen
if screen == "landing":
    show_landing()
elif screen == "overview":
    show_overview()
elif screen == "module" and st.session_state.module in MODULE_FUNCS:
    MODULE_FUNCS[st.session_state.module]()
else:
    show_landing()
