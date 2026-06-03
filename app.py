import os, sys, datetime
import pandas as pd
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.db_setup import get_data_from_db, setup_database
from models.ml_models import run_all_ml_models
from models.dl_model import train_dl_model, get_model_summary, predict_churn, preprocess_data
from visualizations.nn_visualizer import draw_neural_network

# ══════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TelecomAI — Predicción de Churn",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════
#  CSS PROFESIONAL
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  /* ── Reset base ── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp {
    background: #060B18;
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(14,90,170,0.18) 0%, transparent 60%),
      radial-gradient(ellipse 40% 30% at 90% 80%, rgba(6,182,212,0.07) 0%, transparent 50%);
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #060E1C 100%) !important;
    border-right: 1px solid rgba(14,90,170,0.35) !important;
  }
  section[data-testid="stSidebar"] > div { padding-top: 0 !important; }

  /* ── Logo sidebar ── */
  .logo-wrap {
    padding: 28px 20px 20px;
    border-bottom: 1px solid rgba(14,90,170,0.3);
    margin-bottom: 8px;
  }
  .logo-title {
    font-size: 22px; font-weight: 700; color: #E8F4FF;
    letter-spacing: -0.5px; line-height: 1;
  }
  .logo-sub {
    font-size: 10px; color: #4A90D9; text-transform: uppercase;
    letter-spacing: 2px; margin-top: 4px;
  }
  .logo-dot { color: #0EA5E9; }

  /* ── Nav radio ── */
  div[data-testid="stRadio"] label {
    display: flex !important; align-items: center;
    padding: 9px 14px !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #8BAAC8 !important; transition: all 0.18s !important;
    cursor: pointer !important;
  }
  div[data-testid="stRadio"] label:hover {
    background: rgba(14,90,170,0.18) !important;
    color: #C8E0F4 !important;
  }
  div[data-testid="stRadio"] [aria-checked="true"] + label,
  div[data-testid="stRadio"] label[data-checked="true"] {
    background: linear-gradient(90deg,rgba(14,90,170,0.35),rgba(6,182,212,0.12)) !important;
    color: #7DD3FC !important; border-left: 2px solid #0EA5E9 !important;
  }
  div[data-testid="stRadio"] > div { gap: 2px !important; }

  /* ── Encabezado de página ── */
  .page-header {
    padding: 32px 0 8px;
    border-bottom: 1px solid rgba(14,90,170,0.3);
    margin-bottom: 28px;
  }
  .page-title {
    font-size: 28px; font-weight: 700; color: #E8F4FF;
    letter-spacing: -0.8px; margin: 0;
  }
  .page-title span { color: #0EA5E9; }
  .page-desc { font-size: 13px; color: #5A8AB0; margin-top: 6px; }

  /* ── KPI Cards ── */
  .kpi-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 32px; }
  .kpi-card {
    background: linear-gradient(135deg,rgba(10,22,40,0.9),rgba(6,18,34,0.95));
    border: 1px solid rgba(14,90,170,0.3);
    border-radius: 14px; padding: 22px 24px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
  }
  .kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg,#0EA5E9,#06B6D4);
  }
  .kpi-card:hover { border-color: rgba(14,165,233,0.5); transform: translateY(-2px); }
  .kpi-label { font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #4A7A9B; font-weight: 600; }
  .kpi-value { font-size: 36px; font-weight: 700; color: #E8F4FF; line-height: 1.1; margin: 6px 0 2px; letter-spacing: -1px; }
  .kpi-value.danger { color: #F87171; }
  .kpi-value.warning { color: #FBBF24; }
  .kpi-sub { font-size: 11px; color: #3A6080; }
  .kpi-icon { position:absolute; right:20px; top:20px; font-size:28px; opacity:0.15; }

  /* ── Section title ── */
  .sec-title {
    font-size: 15px; font-weight: 600; color: #7DD3FC;
    text-transform: uppercase; letter-spacing: 1px;
    margin: 28px 0 14px; display: flex; align-items: center; gap: 8px;
  }
  .sec-title::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg,rgba(14,90,170,0.4),transparent);
  }

  /* ── Info badge ── */
  .badge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:11px; font-weight:600; letter-spacing:0.5px;
  }
  .badge-blue { background:rgba(14,165,233,0.15); color:#7DD3FC; border:1px solid rgba(14,165,233,0.25); }
  .badge-green { background:rgba(34,197,94,0.12); color:#86EFAC; border:1px solid rgba(34,197,94,0.2); }
  .badge-red { background:rgba(239,68,68,0.12); color:#FCA5A5; border:1px solid rgba(239,68,68,0.2); }
  .badge-amber { background:rgba(251,191,36,0.12); color:#FDE68A; border:1px solid rgba(251,191,36,0.2); }

  /* ── SQL block ── */
  .sql-block {
    background: #0D1B2A; border: 1px solid rgba(14,90,170,0.35);
    border-left: 3px solid #0EA5E9;
    border-radius: 10px; padding: 18px 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
    color: #7DD3FC; line-height: 1.7; margin-bottom: 20px;
  }
  .sql-kw { color: #C084FC; font-weight: 600; }
  .sql-tbl { color: #86EFAC; }
  .sql-str { color: #FDE68A; }

  /* ── Result card ── */
  .result-card {
    border-radius: 16px; padding: 28px 32px; text-align: center;
    margin: 20px 0;
  }
  .result-riesgo {
    background: linear-gradient(135deg,rgba(239,68,68,0.1),rgba(220,38,38,0.05));
    border: 1px solid rgba(239,68,68,0.3);
  }
  .result-estable {
    background: linear-gradient(135deg,rgba(34,197,94,0.1),rgba(22,163,74,0.05));
    border: 1px solid rgba(34,197,94,0.3);
  }
  .result-icon { font-size: 48px; margin-bottom: 8px; }
  .result-label-risk { font-size: 26px; font-weight: 700; color: #F87171; letter-spacing: -0.5px; }
  .result-label-ok   { font-size: 26px; font-weight: 700; color: #86EFAC; letter-spacing: -0.5px; }
  .result-prob { font-size: 15px; color: #8BAAC8; margin-top: 8px; }
  .result-prob strong { color: #E8F4FF; font-size: 20px; }

  /* ── Gauge bar ── */
  .gauge-wrap { margin: 16px 0 8px; }
  .gauge-track {
    height: 10px; border-radius: 99px;
    background: rgba(255,255,255,0.06); overflow: hidden; position: relative;
  }
  .gauge-fill-risk { height:100%; border-radius:99px; background:linear-gradient(90deg,#F59E0B,#EF4444); transition:width 0.8s; }
  .gauge-fill-ok   { height:100%; border-radius:99px; background:linear-gradient(90deg,#10B981,#06B6D4); transition:width 0.8s; }
  .gauge-labels { display:flex; justify-content:space-between; margin-top:4px; font-size:10px; color:#3A6080; }

  /* ── Historial ── */
  .hist-row {
    display:flex; align-items:center; gap:12px;
    padding:10px 14px; border-radius:10px; margin-bottom:6px;
    background:rgba(10,22,40,0.8); border:1px solid rgba(14,90,170,0.2);
    font-size:13px;
  }
  .hist-name { font-weight:600; color:#C8E0F4; min-width:140px; }
  .hist-prob { font-family:'JetBrains Mono',monospace; font-weight:600; min-width:54px; }

  /* ── Tabs ── */
  div[data-testid="stTabs"] button {
    font-size:13px !important; font-weight:500 !important;
    color:#5A8AB0 !important; padding:8px 16px !important;
    border-radius:8px 8px 0 0 !important;
  }
  div[data-testid="stTabs"] button[aria-selected="true"] {
    color:#7DD3FC !important;
    border-bottom: 2px solid #0EA5E9 !important;
    background: rgba(14,90,170,0.15) !important;
  }

  /* ── Botones ── */
  .stButton > button {
    background: linear-gradient(135deg,#0369A1,#0EA5E9) !important;
    color: #fff !important; font-weight: 600 !important;
    border: none !important; border-radius: 10px !important;
    padding: 10px 28px !important; font-size: 14px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.25) !important;
    transition: opacity 0.15s !important;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* ── Inputs ── */
  .stTextInput input, .stSelectbox select, .stSlider {
    background: rgba(10,22,40,0.9) !important;
    border-color: rgba(14,90,170,0.35) !important;
    color: #C8E0F4 !important; border-radius: 10px !important;
  }
  label[data-testid="stWidgetLabel"] p { color: #8BAAC8 !important; font-size: 12px !important; font-weight: 500 !important; }

  /* ── Metric ── */
  div[data-testid="stMetric"] {
    background: rgba(10,22,40,0.7);
    border: 1px solid rgba(14,90,170,0.25);
    border-radius: 12px; padding: 14px 18px !important;
  }
  div[data-testid="stMetric"] label { color: #4A7A9B !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #E8F4FF !important; font-size: 24px !important; font-weight: 700 !important; }

  /* ── Info/success/warning ── */
  .stAlert { border-radius: 10px !important; border-left-width: 3px !important; }

  /* ── DataFrames ── */
  .stDataFrame { border: 1px solid rgba(14,90,170,0.25) !important; border-radius: 10px !important; }
  .stDataFrame iframe { border-radius: 10px !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #060B18; }
  ::-webkit-scrollbar-thumb { background: rgba(14,90,170,0.4); border-radius: 99px; }

  /* ── Hide Streamlit branding ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 2.5rem 3rem !important; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  HELPERS DE GRÁFICAS
# ══════════════════════════════════════════════════════════════════
BG   = "#060B18"
BG2  = "#0A1628"
GRID = "#0F2035"
AZ   = "#0EA5E9"
AZ2  = "#06B6D4"
RD   = "#F87171"
GR   = "#86EFAC"
AM   = "#FBBF24"
TX   = "#C8E0F4"
TX2  = "#5A8AB0"

def fig_style(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TX2, labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.label.set_color(TX2)
    ax.yaxis.label.set_color(TX2)
    ax.title.set_color(TX)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(color=GRID, linewidth=0.5, linestyle="--", alpha=0.6)
    return fig, ax


# ══════════════════════════════════════════════════════════════════
#  CARGA Y ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    if not os.path.exists(os.path.join("database", "telecom.db")):
        setup_database()
    return get_data_from_db()

@st.cache_resource
def train_models_cached(df):
    ml_results = run_all_ml_models(df)
    dl_results = train_dl_model(df)
    return ml_results, dl_results

df = load_data()


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div class="logo-wrap">
  <div class="logo-title">📡 Telecom<span class="logo-dot">AI</span></div>
  <div class="logo-sub">Churn Prediction Platform</div>
</div>
""", unsafe_allow_html=True)

nav_options = ["🏠  Inicio", "🗄️  Datos SQL", "🤖  Modelos ML", "🧠  Red Neuronal DL", "🔮  Predicción", "📊  Resultados"]
selected = st.sidebar.radio("", nav_options, index=0)

st.sidebar.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
churn_rate = (df["Churn"].value_counts().get("Yes", 0) / len(df) * 100) if len(df) > 0 else 0
st.sidebar.markdown(f"""
<div style="padding:14px 16px;background:rgba(10,22,40,0.8);border:1px solid rgba(14,90,170,0.25);border-radius:12px;margin:0 8px;">
  <div style="font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:#4A7A9B;font-weight:600;">Dataset activo</div>
  <div style="font-size:18px;font-weight:700;color:#E8F4FF;margin:4px 0;">7,043 clientes</div>
  <div style="font-size:11px;color:#F87171;">Tasa de churn: {churn_rate:.1f}%</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="padding:10px 16px;margin:10px 8px 0;font-size:10px;color:#2A4A60;text-align:center;">
  IBM Telco Dataset · SQLite · PyTorch · Keras<br>
  SENATI Sede Rio Negro · 2025
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PÁGINA: INICIO
# ══════════════════════════════════════════════════════════════════
if selected == "🏠  Inicio":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">📡 Telecom<span>AI</span> — Predicción de Churn</div>
      <div class="page-desc">Sistema inteligente de predicción de abandono de clientes · Machine Learning + Deep Learning</div>
    </div>
    """, unsafe_allow_html=True)

    at_risk = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum() if len(df) > 0 else 0
    avg_charge = df["MonthlyCharges"].mean() if len(df) > 0 else 0
    total_churn = df[df["Churn"] == "Yes"].shape[0]

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-icon">👥</div>
        <div class="kpi-label">Total clientes</div>
        <div class="kpi-value">7,043</div>
        <div class="kpi-sub">Dataset IBM Telco</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">⚠️</div>
        <div class="kpi-label">Tasa de churn</div>
        <div class="kpi-value warning">{churn_rate:.1f}%</div>
        <div class="kpi-sub">{total_churn:,} clientes abandonaron</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-icon">💸</div>
        <div class="kpi-label">Ingresos en riesgo</div>
        <div class="kpi-value danger">${at_risk:,.0f}</div>
        <div class="kpi-sub">Mensual · cargo promedio ${avg_charge:.0f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="sec-title">Distribución por tipo de contrato</div>', unsafe_allow_html=True)
        contract_churn = df.groupby("Contract")["Churn"].apply(lambda x: (x=="Yes").mean()*100).reset_index()
        contract_churn.columns = ["Contrato", "Churn %"]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = [AZ, AZ2, "#7C3AED"]
        bars = ax.barh(contract_churn["Contrato"], contract_churn["Churn %"], color=colors, height=0.5)
        for bar, val in zip(bars, contract_churn["Churn %"]):
            ax.text(val+0.5, bar.get_y()+bar.get_height()/2, f"{val:.1f}%", va="center", color=TX, fontsize=11, fontweight="600")
        ax.set_xlabel("Tasa de Churn (%)")
        ax.set_title("Churn por tipo de contrato", fontsize=13)
        fig, ax = fig_style(fig, ax)
        ax.set_xlim(0, max(contract_churn["Churn %"])*1.2)
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="sec-title">Churn por internet</div>', unsafe_allow_html=True)
        inet_churn = df.groupby("InternetService")["Churn"].apply(lambda x: (x=="Yes").mean()*100)
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
        colors2 = [RD, AZ, GR]
        wedges, texts, autotexts = ax2.pie(
            inet_churn.values, labels=inet_churn.index, autopct="%1.1f%%",
            colors=colors2, textprops={"color": TX, "fontsize": 10},
            wedgeprops={"linewidth": 2, "edgecolor": BG}, startangle=90
        )
        for at in autotexts: at.set_fontweight("700")
        ax2.set_facecolor(BG)
        fig2.patch.set_facecolor(BG)
        st.pyplot(fig2)

    st.markdown('<div class="sec-title">Arquitectura de la red neuronal</div>', unsafe_allow_html=True)
    nn_fig = draw_neural_network()
    st.pyplot(nn_fig)


# ══════════════════════════════════════════════════════════════════
#  PÁGINA: DATOS SQL
# ══════════════════════════════════════════════════════════════════
elif selected == "🗄️  Datos SQL":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">🗄️ Datos desde <span>SQLite</span></div>
      <div class="page-desc">Consulta SQL en tiempo real · Base de datos relacional telecom.db</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sql-block">
      <span class="sql-kw">SELECT</span>  customerID, gender, SeniorCitizen, tenure,<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PhoneService, InternetService, Contract,<br>
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MonthlyCharges, TotalCharges, Churn<br>
      <span class="sql-kw">FROM</span>    <span class="sql-tbl">customers</span><br>
      <span class="sql-kw">WHERE</span>   TotalCharges != <span class="sql-str">' '</span>;
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total filas", f"{len(df):,}")
    with col2: st.metric("Variables", f"{df.shape[1]}")
    with col3: st.metric("Clientes Churn", f"{(df['Churn']=='Yes').sum():,}")
    with col4: st.metric("Cargo promedio", f"${df['MonthlyCharges'].mean():.2f}")

    st.markdown('<div class="sec-title">Vista previa — primeras 100 filas</div>', unsafe_allow_html=True)
    st.dataframe(df.head(100), use_container_width=True, height=280)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec-title">Estadísticas descriptivas</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)

    with col_b:
        st.markdown('<div class="sec-title">Distribución de Churn</div>', unsafe_allow_html=True)
        churn_counts = df["Churn"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(["No Churn", "Churn"], churn_counts.values, color=[AZ, RD], width=0.5)
        for bar, val in zip(bars, churn_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30, f"{val:,}", ha="center", color=TX, fontsize=12, fontweight="700")
        ax.set_title("Distribución de abandono de clientes", fontsize=12)
        fig, ax = fig_style(fig, ax)
        st.pyplot(fig)

    st.markdown('<div class="sec-title">Análisis por variables clave</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for churn_val, color, lbl in [("No", AZ, "No Churn"), ("Yes", RD, "Churn")]:
            subset = df[df["Churn"]==churn_val]["tenure"]
            ax.hist(subset, bins=20, alpha=0.7, color=color, label=lbl, density=True)
        ax.set_xlabel("Antigüedad (meses)")
        ax.set_ylabel("Densidad")
        ax.set_title("Distribución de antigüedad", fontsize=12)
        ax.legend(facecolor=BG2, labelcolor=TX, fontsize=10)
        fig, ax = fig_style(fig, ax)
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for churn_val, color, lbl in [("No", AZ, "No Churn"), ("Yes", RD, "Churn")]:
            subset = df[df["Churn"]==churn_val]["MonthlyCharges"]
            ax.hist(subset, bins=20, alpha=0.7, color=color, label=lbl, density=True)
        ax.set_xlabel("Cargo mensual ($)")
        ax.set_ylabel("Densidad")
        ax.set_title("Distribución de cargos mensuales", fontsize=12)
        ax.legend(facecolor=BG2, labelcolor=TX, fontsize=10)
        fig, ax = fig_style(fig, ax)
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════════════
#  PÁGINA: MODELOS ML
# ══════════════════════════════════════════════════════════════════
elif selected == "🤖  Modelos ML":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">🤖 Modelos de <span>Machine Learning</span></div>
      <div class="page-desc">KNN · KMeans · Random Forest · ID3 · Regresión Lineal · Apriori</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Entrenando modelos…"):
        ml_results, _ = train_models_cached(df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["KNN", "KMeans", "Random Forest", "ID3 (Árbol)", "Regresión Lineal", "Apriori"])

    # ── KNN ──────────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="sec-title">K-Nearest Neighbors</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1,3])
        with col1:
            st.metric("Precisión", f"{ml_results['knn']['accuracy']:.4f}")
            st.markdown('<span class="badge badge-blue">Supervisado · Clasificación</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div style='font-size:12px;color:#5A8AB0;line-height:1.7'>
            Busca los <strong style='color:#7DD3FC'>5 vecinos</strong> más similares y predice por mayoría de votos.</div>""", unsafe_allow_html=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ConfusionMatrixDisplay(ml_results["knn"]["confusion"]).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Matriz de Confusión — KNN", fontsize=12)
            fig, ax = fig_style(fig, ax)
            st.pyplot(fig)

    # ── KMeans ───────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="sec-title">K-Means Clustering</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1,3])
        with col1:
            st.metric("Silhouette Score", f"{ml_results['kmeans']['silhouette_score']:.4f}")
            st.markdown('<span class="badge badge-amber">No supervisado · Clustering</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div style='font-size:12px;color:#5A8AB0;line-height:1.7'>
            Agrupa automáticamente a los clientes en <strong style='color:#7DD3FC'>2 segmentos</strong> según su perfil.</div>""", unsafe_allow_html=True)
        with col2:
            scatter_df = df.copy()
            scatter_df["TotalCharges"] = pd.to_numeric(scatter_df["TotalCharges"], errors="coerce")
            scatter_df.dropna(subset=["TotalCharges"], inplace=True)
            labels = ml_results["kmeans"]["labels"]
            scatter_df = scatter_df.iloc[:len(labels)].copy()
            scatter_df["Cluster"] = labels

            fig, ax = plt.subplots(figsize=(7, 4.5))
            for cl, color, lbl in [(0, AZ, "Cluster 0 — Perfil Estable"), (1, RD, "Cluster 1 — Perfil Riesgo")]:
                subset = scatter_df[scatter_df["Cluster"]==cl]
                ax.scatter(subset["tenure"], subset["MonthlyCharges"], c=color, label=lbl, alpha=0.4, s=14, edgecolors="none")
            ax.set_xlabel("Antigüedad (meses)")
            ax.set_ylabel("Cargo Mensual ($)")
            ax.set_title("Dispersión de clientes por cluster", fontsize=12)
            ax.legend(facecolor=BG2, labelcolor=TX, fontsize=10, framealpha=0.8)
            fig, ax = fig_style(fig, ax)
            st.pyplot(fig)

        conteo = scatter_df["Cluster"].value_counts().reset_index()
        conteo.columns = ["Cluster", "Clientes"]
        conteo["Perfil"] = conteo["Cluster"].map({0:"Perfil Estable", 1:"Perfil en Riesgo"})
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(conteo[["Perfil","Clientes"]], use_container_width=True, hide_index=True)
        with c2:
            fig, ax = plt.subplots(figsize=(5, 2.8))
            colors_b = [AZ if p=="Perfil Estable" else RD for p in conteo["Perfil"]]
            bars = ax.bar(conteo["Perfil"], conteo["Clientes"], color=colors_b, width=0.4)
            for bar, val in zip(bars, conteo["Clientes"]):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, str(val), ha="center", color=TX, fontsize=11, fontweight="700")
            ax.set_title("Clientes por cluster", fontsize=11)
            fig, ax = fig_style(fig, ax)
            st.pyplot(fig)

    # ── Random Forest ─────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="sec-title">Random Forest</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1,3])
        with col1:
            st.metric("Precisión", f"{ml_results['rf']['accuracy']:.4f}")
            st.markdown('<span class="badge badge-blue">Supervisado · Ensamble</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div style='font-size:12px;color:#5A8AB0;line-height:1.7'>
            Combina cientos de árboles de decisión por <strong style='color:#7DD3FC'>votación mayoritaria</strong>.</div>""", unsafe_allow_html=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ConfusionMatrixDisplay(ml_results["rf"]["confusion"]).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Matriz de Confusión — Random Forest", fontsize=12)
            fig, ax = fig_style(fig, ax)
            st.pyplot(fig)

        st.markdown('<div class="sec-title">Importancia de características — Top 10</div>', unsafe_allow_html=True)
        fi = ml_results["rf"]["feature_importance"]
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        fi_df = pd.DataFrame(fi_sorted, columns=["Característica", "Importancia"])
        fig, ax = plt.subplots(figsize=(10, 4))
        colors_fi = [AZ if i > 2 else AM if i == 1 else RD for i in range(len(fi_df))][::-1]
        bars = ax.barh(fi_df["Característica"][::-1], fi_df["Importancia"][::-1], color=colors_fi, height=0.55)
        for bar, val in zip(bars, fi_df["Importancia"][::-1]):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", color=TX, fontsize=10)
        ax.set_xlabel("Importancia relativa")
        ax.set_title("Variables más determinantes para predecir el churn", fontsize=12)
        fig, ax = fig_style(fig, ax)
        st.pyplot(fig)

    # ── ID3 ───────────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="sec-title">ID3 — Árbol de Decisión con Entropía</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1,3])
        with col1:
            st.metric("Precisión", f"{ml_results['id3']['accuracy']:.4f}")
            st.markdown('<span class="badge badge-green">Supervisado · Árbol</span>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""<div style='font-size:12px;color:#5A8AB0;line-height:1.7'>
            Usa <strong style='color:#7DD3FC'>entropía de información</strong> para elegir la variable más informativa en cada nodo.</div>""", unsafe_allow_html=True)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ConfusionMatrixDisplay(ml_results["id3"]["confusion"]).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Matriz de Confusión — ID3", fontsize=12)
            fig, ax = fig_style(fig, ax)
            st.pyplot(fig)

    # ── Regresión Lineal ──────────────────────────────────────────
    with tab5:
        st.markdown('<div class="sec-title">Regresión Lineal — Predicción de cargos</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("ECM (MSE)", f"{ml_results['linear_reg']['mse']:.4f}")
        with c2: st.metric("R² Score", f"{ml_results['linear_reg']['r2']:.4f}")
        with c3: st.markdown('<span class="badge badge-amber">Supervisado · Regresión</span><br><br>', unsafe_allow_html=True)
        st.info("Este modelo predice el **cargo mensual** esperado de un cliente según sus servicios contratados. Un R² alto indica que el modelo explica bien la variación en los cargos.")

    # ── Apriori ───────────────────────────────────────────────────
    with tab6:
        st.markdown('<div class="sec-title">Apriori — Reglas de Asociación</div>', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-amber">No supervisado · Asociación</span><br><br>', unsafe_allow_html=True)
        rules_df = ml_results["apriori"]
        if len(rules_df) > 0:
            st.markdown("""<div style='font-size:12px;color:#5A8AB0;margin-bottom:12px'>
            Descubre qué servicios se contratan juntos. <strong style='color:#7DD3FC'>Lift > 1</strong> indica asociación real, no aleatoria.</div>""", unsafe_allow_html=True)
            st.dataframe(rules_df, use_container_width=True)
        else:
            st.warning("No se encontraron reglas de asociación significativas.")


# ══════════════════════════════════════════════════════════════════
#  PÁGINA: RED NEURONAL DL
# ══════════════════════════════════════════════════════════════════
elif selected == "🧠  Red Neuronal DL":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">🧠 Red Neuronal — <span>Deep Learning</span></div>
      <div class="page-desc">PyTorch · TensorFlow/Keras · Arquitectura 14→64→32→16→1 · Adam · BCE · MSE</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Cargando resultados del entrenamiento…"):
        _, dl_results = train_models_cached(df)

    model = dl_results["model"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Accuracy PyTorch", f"{dl_results['accuracy']:.4f}")
    with c2: st.metric("Accuracy Keras", f"{dl_results['keras_accuracy']:.4f}")
    with c3: st.metric("MSE final PyTorch", f"{dl_results['mse_per_epoch'][-1]:.4f}")
    with c4: st.metric("MSE final Keras", f"{dl_results['keras_mse_final']:.4f}")

    st.markdown('<div class="sec-title">Arquitectura del modelo</div>', unsafe_allow_html=True)
    layers_data = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            act = "ReLU" if module.out_features > 1 else "Sigmoide"
            layers_data.append({"Capa": name, "Neuronas": module.out_features, "Activación": act,
                                 "Parámetros": sum(p.numel() for p in module.parameters())})
    st.dataframe(pd.DataFrame(layers_data), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-title">Evolución del entrenamiento</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(range(1,51), dl_results["train_losses"], color=AZ, linewidth=2, label="PyTorch BCE")
        ax.plot(range(1,51), dl_results["keras_train_losses"], color=AZ2, linewidth=2, linestyle="--", label="Keras BCE")
        ax.set_xlabel("Época"); ax.set_ylabel("Pérdida BCE")
        ax.set_title("Pérdida por época — PyTorch vs Keras", fontsize=12)
        ax.legend(facecolor=BG2, labelcolor=TX, fontsize=10)
        fig, ax = fig_style(fig, ax)
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(range(1,51), dl_results["mse_per_epoch"], color=AM, linewidth=2, label="PyTorch MSE")
        ax.plot(range(1,51), dl_results["keras_mse_per_epoch"], color=RD, linewidth=2, linestyle="--", label="Keras MSE")
        ax.set_xlabel("Época"); ax.set_ylabel("MSE")
        ax.set_title("MSE por época — PyTorch vs Keras", fontsize=12)
        ax.legend(facecolor=BG2, labelcolor=TX, fontsize=10)
        fig, ax = fig_style(fig, ax)
        st.pyplot(fig)

    st.markdown('<div class="sec-title">Diagrama de la red neuronal</div>', unsafe_allow_html=True)
    nn_fig = draw_neural_network()
    st.pyplot(nn_fig)


# ══════════════════════════════════════════════════════════════════
#  PÁGINA: PREDICCIÓN
# ══════════════════════════════════════════════════════════════════
elif selected == "🔮  Predicción":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">🔮 Predicción <span>Interactiva</span></div>
      <div class="page-desc">Ingresa el perfil de un cliente y obtén la probabilidad de churn en tiempo real</div>
    </div>
    """, unsafe_allow_html=True)

    if "historial" not in st.session_state:
        st.session_state["historial"] = []

    _, dl_results = train_models_cached(df)
    model = dl_results["model"]
    scaler = dl_results["scaler"]
    feature_cols = dl_results["feature_cols"]

    value_map = {
        "Sí": "Yes", "No": "No",
        "Masculino": "Male", "Femenino": "Female",
        "DSL": "DSL", "Fibra óptica": "Fiber optic", "Sin internet": "No",
        "Mes a mes": "Month-to-month", "Un año": "One year", "Dos años": "Two year",
        "Cheque electrónico": "Electronic check", "Cheque por correo": "Mailed check",
        "Transferencia bancaria": "Bank transfer (automatic)",
        "Tarjeta de crédito": "Credit card (automatic)",
        "Sí (tiene)": "Yes", "No (tiene)": "No"
    }

    with st.form("prediction_form"):
        nombre_cliente = st.text_input("👤  Nombre del cliente", placeholder="Ej: Juan Pérez García")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Antigüedad (meses)", 0, 72, 12)
            monthly_charges = st.slider("Cargo Mensual ($)", 0, 150, 70)
            senior_citizen = st.selectbox("Adulto Mayor", [0, 1])
        with col2:
            gender = st.selectbox("Género", ["Masculino", "Femenino"])
            phone_service = st.selectbox("Servicio Telefónico", ["Sí", "No"])
            internet_service = st.selectbox("Servicio de Internet", ["DSL", "Fibra óptica", "Sin internet"])
        with col3:
            contract = st.selectbox("Contrato", ["Mes a mes", "Un año", "Dos años"])
            payment_method = st.selectbox("Método de Pago", ["Cheque electrónico", "Cheque por correo", "Transferencia bancaria", "Tarjeta de crédito"])
            partner = st.selectbox("Pareja", ["Sí (tiene)", "No (tiene)"])
            dependents = st.selectbox("Dependientes", ["Sí (tiene)", "No (tiene)"])
        submitted = st.form_submit_button("🚀  Analizar cliente")

    if submitted:
        raw_input = {
            "tenure": tenure, "MonthlyCharges": monthly_charges,
            "SeniorCitizen": senior_citizen, "gender": value_map[gender],
            "PhoneService": value_map[phone_service], "InternetService": value_map[internet_service],
            "Contract": value_map[contract], "PaymentMethod": value_map[payment_method],
            "Partner": value_map[partner], "Dependents": value_map[dependents],
            "TotalCharges": tenure * monthly_charges, "Churn": "No"
        }
        input_df = pd.DataFrame([raw_input])
        input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce")
        input_df["Churn_binary"] = 0
        df_train_sample = df.copy().head(10)
        df_train_sample["Churn_binary"] = df_train_sample["Churn"].map({"Yes":1,"No":0})
        input_df_full = pd.concat([df_train_sample, input_df], ignore_index=True)
        cat_cols_full = input_df_full.select_dtypes(include=["object"]).columns.tolist()
        cat_cols_full = [c for c in cat_cols_full if c not in ["customerID","Churn","Churn_binary"]]
        encoded_full = pd.get_dummies(input_df_full, columns=cat_cols_full, drop_first=False)
        input_aligned = encoded_full.tail(1).reindex(columns=feature_cols, fill_value=0)
        X_input_aligned = input_aligned.values.astype(np.float32)
        X_input_scaled = scaler.transform(X_input_aligned)
        prob = predict_churn(model, scaler, feature_cols, X_input_scaled[0])

        pct = prob * 100
        es_riesgo = prob >= 0.5
        clase_card = "result-riesgo" if es_riesgo else "result-estable"
        icon = "🔴" if es_riesgo else "🟢"
        label_class = "result-label-risk" if es_riesgo else "result-label-ok"
        label_text = "CLIENTE EN RIESGO" if es_riesgo else "CLIENTE ESTABLE"
        fill_class = "gauge-fill-risk" if es_riesgo else "gauge-fill-ok"
        nombre_show = nombre_cliente.strip() if nombre_cliente.strip() else "Cliente"

        st.markdown(f"""
        <div class="result-card {clase_card}">
          <div class="result-icon">{icon}</div>
          <div class="{label_class}">{label_text}</div>
          <div style="font-size:14px;color:#8BAAC8;margin-top:6px">{nombre_show}</div>
          <div class="result-prob">Probabilidad de churn: <strong>{prob:.1%}</strong></div>
          <div class="gauge-wrap" style="max-width:400px;margin:16px auto 0">
            <div class="gauge-track"><div class="{fill_class}" style="width:{pct:.1f}%"></div></div>
            <div class="gauge-labels"><span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


        # ── Sugerencias personalizadas inteligentes ─────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # Datos del cliente para personalizar
        tiene_fibra     = internet_service == "Fibra óptica"
        tiene_dsl       = internet_service == "DSL"
        sin_internet    = internet_service == "Sin internet"
        contrato_mes    = contract == "Mes a mes"
        contrato_anual  = contract == "Un año"
        contrato_2anios = contract == "Dos años"
        cheque_elec     = payment_method == "Cheque electrónico"
        cargo_alto      = monthly_charges >= 80
        cargo_medio     = 40 <= monthly_charges < 80
        es_nuevo        = tenure <= 6
        es_veterano     = tenure >= 36
        tiene_familia   = partner == "Sí (tiene)" or dependents == "Sí (tiene)"

        # Construir sugerencias dinámicas
        def card(icon, titulo, desc, color):
            return f"""<div style="background:rgba({color},0.07);border-radius:10px;padding:14px 16px;border:1px solid rgba({color},0.18)">
              <div style="font-size:12px;font-weight:700;color:rgb({color});margin-bottom:6px">{icon} {titulo}</div>
              <div style="font-size:12px;color:#8BAAC8;line-height:1.6">{desc}</div>
            </div>"""

        if prob >= 0.75:
            # CRÍTICO
            ofertas = []
            ofertas.append(card("🚨","Descuento de emergencia",
                f"Ofrecer <strong style='color:#fff'>30% de descuento</strong> inmediato en el cargo mensual (${monthly_charges} → ${monthly_charges*0.7:.0f}).<br>Válido si firma contrato <strong style='color:#fff'>bianual hoy mismo</strong>.","239,68,68"))
            if contrato_mes:
                ofertas.append(card("📋","Cambio de contrato urgente",
                    f"Migrar de <strong style='color:#fff'>mes a mes</strong> a contrato anual con <strong style='color:#fff'>2 meses gratis</strong>.<br>Reducción inmediata del riesgo de abandono.","239,68,68"))
            if tiene_fibra and cargo_alto:
                ofertas.append(card("💡","Revisión de plan",
                    f"Cargo actual <strong style='color:#fff'>${monthly_charges}/mes</strong> es elevado para fibra.<br>Ofrecer plan <strong style='color:#fff'>fibra optimizada</strong> a ${monthly_charges*0.75:.0f}/mes con misma velocidad.","239,68,68"))
            if cheque_elec:
                ofertas.append(card("💳","Cambio de pago",
                    "El <strong style='color:#fff'>cheque electrónico</strong> está asociado a mayor tasa de churn.<br>Ofrecer <strong style='color:#fff'>descuento adicional 5%</strong> al migrar a débito automático.","239,68,68"))
            ofertas.append(card("📞","Llamada prioritaria 24h",
                f"Cliente <strong style='color:#fff'>{nombre_show}</strong> requiere contacto inmediato.<br>Probabilidad de abandono: <strong style='color:#F87171'>{prob:.1%}</strong> — intervención urgente.","239,68,68"))

            grid = "".join(ofertas)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(239,68,68,0.09),rgba(220,38,38,0.04));
                border:1px solid rgba(239,68,68,0.3);border-radius:14px;padding:22px 26px;margin-top:8px">
              <div style="font-size:13px;font-weight:700;color:#F87171;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px">
                🚨 RIESGO CRÍTICO — Intervención inmediata ({prob:.1%})
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{grid}</div>
            </div>
            """, unsafe_allow_html=True)

        elif prob >= 0.50:
            # ALTO
            ofertas = []
            ofertas.append(card("💰","Oferta de retención",
                f"Descuento del <strong style='color:#fff'>20%</strong> por 3 meses si renueva contrato.<br>Cargo: ${monthly_charges} → <strong style='color:#fff'>${monthly_charges*0.8:.0f}/mes</strong> durante 90 días.","239,68,68"))
            if contrato_mes:
                ofertas.append(card("📋","Upgrade de contrato",
                    "Migrar a <strong style='color:#fff'>contrato anual</strong> con precio congelado por 12 meses.<br>Ahorro estimado: <strong style='color:#fff'>${monthly_charges*0.15*12:.0f}/año</strong>.","239,68,68"))
            if sin_internet:
                ofertas.append(card("📶","Agregar internet",
                    "Cliente sin servicio de internet — alta oportunidad.<br>Ofrecer <strong style='color:#fff'>DSL básico gratis por 2 meses</strong> para fidelizar.","239,68,68"))
            if not tiene_familia:
                ofertas.append(card("👨‍👩‍👧","Plan familiar",
                    "Sin pareja ni dependientes registrados.<br>Proponer <strong style='color:#fff'>plan familiar</strong> con descuento al agregar una línea.","239,68,68"))
            ofertas.append(card("🎁","Servicio adicional gratis",
                "Agregar <strong style='color:#fff'>soporte técnico premium</strong> sin costo por 3 meses.<br>Aumenta percepción de valor y reduce intención de abandono.","239,68,68"))

            grid = "".join(ofertas)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(239,68,68,0.08),rgba(220,38,38,0.04));
                border:1px solid rgba(239,68,68,0.25);border-radius:14px;padding:22px 26px;margin-top:8px">
              <div style="font-size:13px;font-weight:700;color:#F87171;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px">
                🔴 RIESGO ALTO — Acciones de retención ({prob:.1%})
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{grid}</div>
            </div>
            """, unsafe_allow_html=True)

        elif prob >= 0.25:
            # MODERADO
            ofertas = []
            ofertas.append(card("🎯","Oferta preventiva",
                f"Descuento del <strong style='color:#fff'>10%</strong> al renovar por un año.<br>Cargo: ${monthly_charges} → <strong style='color:#fff'>${monthly_charges*0.9:.0f}/mes</strong> durante 12 meses.","251,191,36"))
            if contrato_mes or contrato_anual:
                ofertas.append(card("🔒","Fidelización por contrato",
                    f"Migrar de <strong style='color:#fff'>{contract.lower()}</strong> a contrato <strong style='color:#fff'>bianual</strong>.<br>Precio garantizado + 1 servicio adicional gratis.","251,191,36"))
            if tiene_dsl:
                ofertas.append(card("🚀","Upgrade a fibra óptica",
                    "Cliente con DSL — proponer migración a <strong style='color:#fff'>fibra óptica</strong>.<br>Mayor velocidad al mismo precio o con pequeño incremento.","251,191,36"))
            if cargo_medio:
                ofertas.append(card("📦","Paquete combo",
                    f"Cargo actual: <strong style='color:#fff'>${monthly_charges}/mes</strong>.<br>Ofrecer <strong style='color:#fff'>combo teléfono + internet + TV</strong> por precio similar.","251,191,36"))

            grid = "".join(ofertas)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(251,191,36,0.08),rgba(245,158,11,0.04));
                border:1px solid rgba(251,191,36,0.25);border-radius:14px;padding:22px 26px;margin-top:8px">
              <div style="font-size:13px;font-weight:700;color:#FBBF24;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px">
                ⚠️ RIESGO MODERADO — Prevención activa ({prob:.1%})
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{grid}</div>
            </div>
            """, unsafe_allow_html=True)

        elif prob >= 0.10:
            # BAJO — upsell suave
            ofertas = []
            if sin_internet:
                ofertas.append(card("📶","Agregar internet",
                    "Gran oportunidad: cliente sin internet.<br>Ofrecer <strong style='color:#fff'>DSL o fibra óptica</strong> con 1 mes gratis de prueba.","34,197,94"))
            elif tiene_dsl:
                ofertas.append(card("⚡","Upgrade a fibra",
                    "Proponer migración de DSL a <strong style='color:#fff'>fibra óptica</strong>.<br>Mayor velocidad, misma fidelidad, mayor ingreso.","34,197,94"))
            ofertas.append(card("📱","Segunda línea",
                f"Cliente estable con {tenure} meses de permanencia.<br>Ofrecer <strong style='color:#fff'>línea adicional con 25% de descuento</strong> por lealtad.","34,197,94"))
            if es_veterano:
                ofertas.append(card("🏆","Premio por lealtad",
                    f"Con <strong style='color:#fff'>{tenure} meses</strong> es un cliente valioso.<br>Invitar al <strong style='color:#fff'>programa VIP</strong> con beneficios exclusivos.","34,197,94"))
            if not tiene_familia:
                ofertas.append(card("👨‍👩‍👧","Plan familiar",
                    "Proponer <strong style='color:#fff'>plan familiar</strong> con descuento al agregar dependientes.<br>Aumenta ingresos y refuerza el vínculo con la empresa.","34,197,94"))

            grid = "".join(ofertas)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(34,197,94,0.08),rgba(22,163,74,0.04));
                border:1px solid rgba(34,197,94,0.25);border-radius:14px;padding:22px 26px;margin-top:8px">
              <div style="font-size:13px;font-weight:700;color:#86EFAC;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px">
                🟡 RIESGO BAJO — Oportunidad de crecimiento ({prob:.1%})
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{grid}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # FIEL < 10%
            ofertas = []
            ofertas.append(card("🏆","Programa VIP",
                f"<strong style='color:#fff'>{nombre_show}</strong> es un cliente fiel con {tenure} meses.<br>Acceso exclusivo al <strong style='color:#fff'>programa de lealtad premium</strong>.","34,197,94"))
            ofertas.append(card("📱","Segunda línea premium",
                f"Ofrecer <strong style='color:#fff'>segunda línea con 30% de descuento</strong> permanente.<br>Beneficio por ser cliente destacado.","34,197,94"))
            if tiene_dsl or sin_internet:
                ofertas.append(card("🚀","Fibra óptica exclusiva",
                    "Upgrade a <strong style='color:#fff'>fibra óptica con instalación gratuita</strong>.<br>Velocidad premium como reconocimiento a su lealtad.","34,197,94"))
            ofertas.append(card("🎁","Beneficio sorpresa",
                f"Cargo actual: <strong style='color:#fff'>${monthly_charges}/mes</strong>.<br>Agregar <strong style='color:#fff'>servicio de TV streaming gratis</strong> por 3 meses.","34,197,94"))

            grid = "".join(ofertas)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(6,182,212,0.08),rgba(14,165,233,0.04));
                border:1px solid rgba(6,182,212,0.3);border-radius:14px;padding:22px 26px;margin-top:8px">
              <div style="font-size:13px;font-weight:700;color:#67E8F9;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px">
                💎 CLIENTE FIEL — Máxima fidelización ({prob:.1%})
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{grid}</div>
            </div>
            """, unsafe_allow_html=True)

        registro = {
            "Nombre": nombre_show, "Contrato": contract, "Internet": internet_service,
            "Cargo": f"${monthly_charges}", "Antigüedad": f"{tenure}m",
            "Probabilidad": f"{prob:.1%}", "Resultado": "🔴 RIESGO" if es_riesgo else "🟢 ESTABLE",
            "Hora": datetime.datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state["historial"].append(registro)

    if st.session_state["historial"]:
        st.markdown('<div class="sec-title" style="margin-top:32px">Historial de evaluaciones</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state["historial"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        c1, c2 = st.columns([1,4])
        with c1:
            if st.button("🗑️ Limpiar"):
                st.session_state["historial"] = []
                st.rerun()
        with c2:
            csv_hist = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Exportar CSV", csv_hist, "historial.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════
#  PÁGINA: RESULTADOS
# ══════════════════════════════════════════════════════════════════
elif selected == "📊  Resultados":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">📊 Comparación de <span>Modelos</span></div>
      <div class="page-desc">Rendimiento comparativo de todos los algoritmos implementados</div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Evaluando modelos…"):
        ml_results, dl_results = train_models_cached(df)

    comparison = [
        {"Modelo":"KNN",              "Tipo":"Clasificación", "Accuracy":f"{ml_results['knn']['accuracy']:.4f}",    "Notas":"Bueno para fronteras no lineales"},
        {"Modelo":"Random Forest",    "Tipo":"Clasificación", "Accuracy":f"{ml_results['rf']['accuracy']:.4f}",     "Notas":"Robusto, maneja overfitting"},
        {"Modelo":"ID3 (Árbol)",      "Tipo":"Clasificación", "Accuracy":f"{ml_results['id3']['accuracy']:.4f}",    "Notas":"Interpretable, entropía"},
        {"Modelo":"Red Neuronal DL",  "Tipo":"Deep Learning", "Accuracy":f"{dl_results['accuracy']:.4f}",           "Notas":"PyTorch · capas ocultas"},
        {"Modelo":"Keras DL",         "Tipo":"Deep Learning", "Accuracy":f"{dl_results['keras_accuracy']:.4f}",     "Notas":"TensorFlow/Keras paralelo"},
        {"Modelo":"KMeans",           "Tipo":"Clustering",    "Accuracy":f"Sil: {ml_results['kmeans']['silhouette_score']:.3f}", "Notas":"Segmentación no supervisada"},
        {"Modelo":"Regresión Lineal", "Tipo":"Regresión",     "Accuracy":f"R²: {ml_results['linear_reg']['r2']:.3f}", "Notas":"Predice MonthlyCharges"},
    ]
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    models_acc = {
        "KNN":          ml_results["knn"]["accuracy"],
        "Random Forest":ml_results["rf"]["accuracy"],
        "ID3":          ml_results["id3"]["accuracy"],
        "PyTorch DL":   dl_results["accuracy"],
        "Keras DL":     dl_results["keras_accuracy"],
    }
    best_model = max(models_acc, key=models_acc.get)
    st.success(f"🏆 **Mejor modelo: {best_model}** — Accuracy: {models_acc[best_model]:.4f}")

    st.markdown('<div class="sec-title">Comparación de precisión</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bar_colors = [AZ, AZ2, "#7C3AED", AM, RD]
    bars = ax.bar(models_acc.keys(), models_acc.values(), color=bar_colors, width=0.55)
    ax.set_ylim(0.65, max(models_acc.values())*1.08)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy comparativo — todos los modelos de clasificación", fontsize=13)
    for bar, val in zip(bars, models_acc.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f"{val:.4f}",
                ha="center", color=TX, fontsize=11, fontweight="700")
    fig, ax = fig_style(fig, ax)
    ax.grid(axis="x", visible=False)
    st.pyplot(fig)

    csv = pd.DataFrame(comparison).to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar resultados CSV", csv, "resultados_modelos.csv", "text/csv")