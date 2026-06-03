import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db_setup import get_data_from_db, setup_database
from models.ml_models import run_all_ml_models
from models.dl_model import train_dl_model, get_model_summary, predict_churn, preprocess_data
from visualizations.nn_visualizer import draw_neural_network

st.set_page_config(
    page_title="TelecomAI - Predicción de Churn",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .main-header { color: #4fc3f7; font-size: 2.5rem; font-weight: bold; text-align: center; margin-bottom: 1rem; }
    .sub-header { color: #ffffff; font-size: 1.2rem; text-align: center; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 2px solid #4fc3f7;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(79, 195, 247, 0.1);
    }
    .metric-value { font-size: 2.2rem; font-weight: bold; color: #4fc3f7; }
    .metric-label { font-size: 0.9rem; color: #cccccc; margin-top: 5px; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #4fc3f7;
    }
    .sidebar-header { color: #4fc3f7; font-size: 1.5rem; font-weight: bold; text-align: center; padding: 1rem 0; }
    .stButton>button {
        background: linear-gradient(135deg, #4fc3f7, #2196F3);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
    }
    .stButton>button:hover { opacity: 0.9; }
    .risk-high { color: #ff4444; font-size: 1.5rem; font-weight: bold; }
    .risk-low { color: #00C853; font-size: 1.5rem; font-weight: bold; }
    h1, h2, h3 { color: #4fc3f7 !important; }
    .stDataFrame { border: 1px solid #4fc3f7; border-radius: 5px; }
    div[data-testid="stTab"] button { color: #4fc3f7; }
</style>
""", unsafe_allow_html=True)


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

sidebar_logo = st.sidebar.markdown('<div class="sidebar-header">📡 TelecomAI</div>', unsafe_allow_html=True)

nav_options = ["🏠 Inicio", "📊 Datos SQL", "🤖 Modelos ML", "🧠 Red Neuronal DL", "🔮 Predicción", "📈 Resultados"]
selected = st.sidebar.radio("Navegación", nav_options, index=0)

if selected == "🏠 Inicio":
    st.markdown('<div class="main-header">📡 TelecomAI - Predicción de Churn</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Sistema inteligente de predicción de abandono de clientes '
        'usando Machine Learning y Deep Learning</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">7,043</div>
            <div class="metric-label">Total Clientes</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        churn_rate = (df["Churn"].value_counts().get("Yes", 0) / len(df) * 100) if len(df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Tasa de Churn</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        at_risk_revenue = df[df["Churn"] == "Yes"]["MonthlyCharges"].sum() if len(df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${at_risk_revenue:,.0f}</div>
            <div class="metric-label">Ingresos en Riesgo (Mensual)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🧠 Arquitectura de la Red Neuronal")
    nn_fig = draw_neural_network()
    st.pyplot(nn_fig)

elif selected == "📊 Datos SQL":
    st.markdown('<div class="main-header">📊 Datos desde SQLite</div>', unsafe_allow_html=True)

    query = """SELECT customerID, gender, SeniorCitizen, tenure, PhoneService,
       InternetService, Contract, MonthlyCharges, TotalCharges, Churn
FROM customers
WHERE TotalCharges != ' '"""

    st.subheader("Consulta SQL")
    st.code(query, language="sql")

    st.subheader("Vista previa de datos (primeras 100 filas)")
    st.dataframe(df.head(100), use_container_width=True)

    st.subheader("Estadísticas descriptivas")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Distribución de Churn")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts = df["Churn"].value_counts()
    axes[0].bar(churn_counts.index, churn_counts.values, color=["#4fc3f7", "#ff4444"])
    axes[0].set_title("Conteo de Churn", color="white")
    axes[0].set_xlabel("Churn", color="white")
    axes[0].set_ylabel("Clientes", color="white")
    axes[0].tick_params(colors="white")
    axes[0].set_facecolor("#1a1a2e")
    wedges, texts, autotexts = axes[1].pie(
        churn_counts.values, labels=churn_counts.index, autopct="%1.1f%%",
        colors=["#4fc3f7", "#ff4444"], textprops={"color": "white"}
    )
    axes[1].set_title("Proporción de Churn", color="white")
    fig.patch.set_facecolor("#0e1117")
    st.pyplot(fig)

elif selected == "🤖 Modelos ML":
    st.markdown('<div class="main-header">🤖 Modelos de Machine Learning</div>', unsafe_allow_html=True)

    with st.spinner("Entrenando modelos de Machine Learning..."):
        ml_results, _ = train_models_cached(df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["KNN", "KMeans", "Random Forest", "ID3 (Árbol)", "Regresión Lineal", "Apriori"])

    with tab1:
        st.subheader("K-Nearest Neighbors (KNN)")
        st.metric("Precisión", f"{ml_results['knn']['accuracy']:.4f}")
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(ml_results["knn"]["confusion"]).plot(ax=ax, cmap="Blues")
        ax.set_title("Matriz de Confusión - KNN", color="white")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#0e1117")
        st.pyplot(fig)

    with tab2:
        st.subheader("K-Means Clustering (Segmentación)")
        st.metric("Puntaje Silhouette", f"{ml_results['kmeans']['silhouette_score']:.4f}")
        st.info("KMeans agrupa clientes en 2 clusters: los que probablemente se quedan vs. los que se van.")

        # ── Scatter plot de clusters ──────────────────────────
        st.subheader("Gráfico de Dispersión — Clientes por Cluster")
        scatter_df = df.copy()
        scatter_df["TotalCharges"] = pd.to_numeric(scatter_df["TotalCharges"], errors="coerce")
        scatter_df.dropna(subset=["TotalCharges"], inplace=True)
        labels = ml_results["kmeans"]["labels"]
        scatter_df = scatter_df.iloc[:len(labels)].copy()
        scatter_df["Cluster"] = labels

        colores_cluster = {0: "#4fc3f7", 1: "#ff4444"}
        nombres_cluster = {0: "Cluster 0 — Perfil Estable", 1: "Cluster 1 — Perfil en Riesgo"}

        fig_s, ax_s = plt.subplots(figsize=(10, 5))
        for cl in [0, 1]:
            subset = scatter_df[scatter_df["Cluster"] == cl]
            ax_s.scatter(
                subset["tenure"], subset["MonthlyCharges"],
                c=colores_cluster[cl], label=nombres_cluster[cl],
                alpha=0.45, s=18, edgecolors="none"
            )
        ax_s.set_xlabel("Antigüedad (meses)", color="white")
        ax_s.set_ylabel("Cargo Mensual ($)", color="white")
        ax_s.set_title("Dispersión de Clientes por Cluster", color="white")
        ax_s.tick_params(colors="white")
        ax_s.set_facecolor("#1a1a2e")
        ax_s.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
        fig_s.patch.set_facecolor("#0e1117")
        st.pyplot(fig_s)
        st.caption("Cluster 0 (azul) = mayor antigüedad y cargos bajos → menor riesgo.  Cluster 1 (rojo) = contratos cortos y cargos altos → mayor riesgo.")

        # ── Agrupaciones: conteo por cluster ─────────────────
        st.subheader("Agrupación de Clientes por Cluster")
        conteo = scatter_df["Cluster"].value_counts().reset_index()
        conteo.columns = ["Cluster", "Cantidad de Clientes"]
        conteo["Perfil"] = conteo["Cluster"].map({0: "Perfil Estable", 1: "Perfil en Riesgo"})
        st.dataframe(conteo, use_container_width=True)

        fig_b, ax_b = plt.subplots(figsize=(6, 3))
        ax_b.bar(conteo["Perfil"], conteo["Cantidad de Clientes"], color=["#4fc3f7", "#ff4444"])
        ax_b.set_title("Clientes por Cluster", color="white")
        ax_b.set_ylabel("Cantidad", color="white")
        ax_b.tick_params(colors="white")
        ax_b.set_facecolor("#1a1a2e")
        fig_b.patch.set_facecolor("#0e1117")
        for bar, val in zip(ax_b.patches, conteo["Cantidad de Clientes"]):
            ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                      str(val), ha="center", color="white", fontweight="bold")
        st.pyplot(fig_b)

    with tab3:
        st.subheader("Random Forest")
        st.metric("Precisión", f"{ml_results['rf']['accuracy']:.4f}")
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(ml_results["rf"]["confusion"]).plot(ax=ax, cmap="Blues")
        ax.set_title("Matriz de Confusión - Random Forest", color="white")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#0e1117")
        st.pyplot(fig)

        st.subheader("Importancia de Características (Top 10)")
        fi = ml_results["rf"]["feature_importance"]
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        fi_df = pd.DataFrame(fi_sorted, columns=["Característica", "Importancia"])
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.barh(fi_df["Característica"], fi_df["Importancia"], color="#4fc3f7")
        ax2.set_title("Importancia de Características", color="white")
        ax2.set_xlabel("Importancia", color="white")
        ax2.tick_params(colors="white")
        ax2.set_facecolor("#1a1a2e")
        fig2.patch.set_facecolor("#0e1117")
        st.pyplot(fig2)

    with tab4:
        st.subheader("ID3 - Árbol de Decisión")
        st.metric("Precisión", f"{ml_results['id3']['accuracy']:.4f}")
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(ml_results["id3"]["confusion"]).plot(ax=ax, cmap="Blues")
        ax.set_title("Matriz de Confusión - ID3", color="white")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#0e1117")
        st.pyplot(fig)

    with tab5:
        st.subheader("Regresión Lineal (predice MonthlyCharges)")
        st.metric("ECM (Error Cuadrático Medio)", f"{ml_results['linear_reg']['mse']:.4f}")
        st.metric("Puntaje R²", f"{ml_results['linear_reg']['r2']:.4f}")

    with tab6:
        st.subheader("Reglas de Asociación (Apriori)")
        rules_df = ml_results["apriori"]
        if len(rules_df) > 0:
            st.dataframe(rules_df, use_container_width=True)
        else:
            st.warning("No se encontraron reglas de asociación significativas.")

elif selected == "🧠 Red Neuronal DL":
    st.markdown('<div class="main-header">🧠 Red Neuronal (Deep Learning)</div>', unsafe_allow_html=True)

    with st.spinner("Entrenando red neuronal PyTorch..."):
        _, dl_results = train_models_cached(df)

    model = dl_results["model"]
    input_size = dl_results["input_size"]

    st.subheader("Arquitectura del Modelo")
    layers_data = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            act = "ReLU" if module.out_features > 1 else "Sigmoide"
            layers_data.append({
                "Capa": name,
                "Neuronas": module.out_features,
                "Activación": act,
                "Parámetros": sum(p.numel() for p in module.parameters())
            })
    arch_df = pd.DataFrame(layers_data)
    st.dataframe(arch_df, use_container_width=True)

    st.subheader("Evolución del Entrenamiento")

    col_a, col_b = st.columns(2)
    with col_a:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(range(1, 51), dl_results["train_losses"], color="#4fc3f7", linewidth=2)
        ax1.set_title("Pérdida por Época (BCE)", color="white")
        ax1.set_xlabel("Época", color="white")
        ax1.set_ylabel("Pérdida", color="white")
        ax1.tick_params(colors="white")
        ax1.set_facecolor("#1a1a2e")
        fig1.patch.set_facecolor("#0e1117")
        st.pyplot(fig1)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(range(1, 51), dl_results["mse_per_epoch"], color="#ff8c00", linewidth=2)
        ax2.set_title("MSE por Época", color="white")
        ax2.set_xlabel("Época", color="white")
        ax2.set_ylabel("MSE", color="white")
        ax2.tick_params(colors="white")
        ax2.set_facecolor("#1a1a2e")
        fig2.patch.set_facecolor("#0e1117")
        st.pyplot(fig2)

    st.subheader("Precisión Final")
    st.metric("Precisión en Prueba", f"{dl_results['accuracy']:.4f}")

    st.subheader("Diagrama de la Red Neuronal")
    nn_fig = draw_neural_network()
    st.pyplot(nn_fig)

elif selected == "🔮 Predicción":
    st.markdown('<div class="main-header">🔮 Predicción de Churn</div>', unsafe_allow_html=True)

    # ── Inicializar historial en session_state ────────────────────
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
        nombre_cliente = st.text_input("Nombre del cliente (opcional)", placeholder="Ej: Juan Pérez García")
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
            payment_method = st.selectbox("Método de Pago", [
                "Cheque electrónico", "Cheque por correo", "Transferencia bancaria", "Tarjeta de crédito"
            ])
            partner = st.selectbox("Pareja", ["Sí (tiene)", "No (tiene)"])
            dependents = st.selectbox("Dependientes", ["Sí (tiene)", "No (tiene)"])

        submitted = st.form_submit_button("Predecir Churn 🚀")

    if submitted:
        raw_input = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "SeniorCitizen": senior_citizen,
            "gender": value_map[gender],
            "PhoneService": value_map[phone_service],
            "InternetService": value_map[internet_service],
            "Contract": value_map[contract],
            "PaymentMethod": value_map[payment_method],
            "Partner": value_map[partner],
            "Dependents": value_map[dependents],
            "TotalCharges": tenure * monthly_charges,
            "Churn": "No"
        }

        input_df = pd.DataFrame([raw_input])
        X_input, _, _, input_cols = preprocess_data(input_df)

        # Alinear columnas del input con las columnas del entrenamiento
        input_encoded = pd.DataFrame(X_input, columns=input_cols)
        input_aligned = input_encoded.reindex(columns=feature_cols, fill_value=0)
        X_input_aligned = input_aligned.values.astype(np.float32)

        X_input_scaled = scaler.transform(X_input_aligned)
        prob = predict_churn(model, scaler, feature_cols, X_input_scaled[0])

        st.markdown("---")
        st.markdown("### Resultado de la Predicción")

        gauge_val = prob * 100
        if prob >= 0.5:
            st.markdown(f'<div class="risk-high">🔴 CLIENTE EN RIESGO</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">🟢 CLIENTE ESTABLE</div>', unsafe_allow_html=True)

        st.markdown(f"**Probabilidad de Churn:** {prob:.2%}")

        # ── Guardar en historial ──────────────────────────────
        import datetime
        registro = {
            "Nombre": nombre_cliente if nombre_cliente.strip() else "Cliente sin nombre",
            "Contrato": contract,
            "Internet": internet_service,
            "Cargo Mensual": f"${monthly_charges}",
            "Antigüedad": f"{tenure} meses",
            "Probabilidad": f"{prob:.2%}",
            "Resultado": "🔴 EN RIESGO" if prob >= 0.5 else "🟢 ESTABLE",
            "Hora": datetime.datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state["historial"].append(registro)

        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh([0], [gauge_val], color="#ff4444" if prob >= 0.5 else "#00C853",
                height=0.5)
        ax.barh([0], [100 - gauge_val], left=[gauge_val],
                color="#333333", height=0.5)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        st.pyplot(fig)

    # ── Historial de predicciones ────────────────────────────────
    if st.session_state["historial"]:
        st.markdown("---")
        st.markdown("### 📋 Historial de Evaluaciones")
        hist_df = pd.DataFrame(st.session_state["historial"])
        st.dataframe(hist_df, use_container_width=True)
        if st.button("🗑️ Limpiar historial"):
            st.session_state["historial"] = []
            st.rerun()
        csv_hist = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Descargar historial CSV",
            data=csv_hist,
            file_name="historial_predicciones.csv",
            mime="text/csv"
        )

elif selected == "📈 Resultados":
    st.markdown('<div class="main-header">📈 Comparación de Modelos</div>', unsafe_allow_html=True)

    with st.spinner("Evaluando modelos..."):
        ml_results, dl_results = train_models_cached(df)

    comparison = [
        {"Modelo": "KNN", "Tipo": "Clasificación", "Accuracy": f"{ml_results['knn']['accuracy']:.4f}", "Notas": "Bueno para datos con fronteras no lineales"},
        {"Modelo": "Random Forest", "Tipo": "Clasificación", "Accuracy": f"{ml_results['rf']['accuracy']:.4f}", "Notas": "Robusto, maneja overfitting bien"},
        {"Modelo": "ID3 (Árbol)", "Tipo": "Clasificación", "Accuracy": f"{ml_results['id3']['accuracy']:.4f}", "Notas": "Interpretable, propenso a overfitting"},
        {"Modelo": "Red Neuronal (DL)", "Tipo": "Clasificación", "Accuracy": f"{dl_results['accuracy']:.4f}", "Notas": "Captura relaciones complejas"},
        {"Modelo": "KMeans", "Tipo": "Clustering", "Accuracy": f"Silhouette: {ml_results['kmeans']['silhouette_score']:.3f}", "Notas": "Segmentación no supervisada"},
        {"Modelo": "Regresión Lineal", "Tipo": "Regresión", "Accuracy": f"R²: {ml_results['linear_reg']['r2']:.3f}", "Notas": "Predice MonthlyCharges"}
    ]

    comparison_df = pd.DataFrame(comparison)
    st.dataframe(comparison_df, use_container_width=True)

    models_acc = {
        "KNN": ml_results["knn"]["accuracy"],
        "Random Forest": ml_results["rf"]["accuracy"],
        "ID3": ml_results["id3"]["accuracy"],
        "Red Neuronal": dl_results["accuracy"]
    }
    best_model = max(models_acc, key=models_acc.get)
    st.success(f"🏆 **Mejor modelo recomendado: {best_model}** con precisión de {models_acc[best_model]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models_acc.keys(), models_acc.values(), color=["#4fc3f7", "#2196F3", "#1976D2", "#0D47A1"])
    ax.set_title("Comparación de Precisión entre Modelos", color="white", fontsize=14)
    ax.set_ylabel("Precisión", color="white")
    ax.tick_params(colors="white")
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0e1117")
    for bar, val in zip(bars, models_acc.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", color="white", fontweight="bold")
    st.pyplot(fig)

    csv = comparison_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar Resultados CSV",
        data=csv,
        file_name="comparacion_modelos.csv",
        mime="text/csv"
    )