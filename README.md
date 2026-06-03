# 📡 TelecomAI - Churn Prediction System

Sistema inteligente de predicción de abandono de clientes (Churn) para una empresa de telecomunicaciones, utilizando Machine Learning y Deep Learning.

## 🚀 Características

- **Múltiples modelos ML**: KNN, Random Forest, ID3, K-Means, Regresión Lineal, Apriori
- **Red Neuronal**: PyTorch con 3 capas ocultas
- **Visualización**: Diagrama de arquitectura de red neuronal
- **Dashboard interactivo**: Streamlit con 6 páginas
- **Base de datos**: SQLite para almacenamiento de datos

## 📋 Requisitos

```bash
pip install -r requirements.txt
```

## 📦 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/telecom-churn.git
cd telecom-churn
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Descarga los datos:
```bash
curl -o data/telco_churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

4. Inicializa la base de datos:
```bash
python database/db_setup.py
```

5. Ejecuta la aplicación:
```bash
streamlit run app.py
```

## 🌐 Despliegue en Streamlit Cloud

1. Sube el proyecto a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Configura:
   - Main file: `app.py`
   - Python version: 3.9+
5. Despliega

## 📁 Estructura del Proyecto

```
telecom_churn/
├── data/
│   └── telco_churn.csv
├── database/
│   └── db_setup.py
├── models/
│   ├── ml_models.py
│   └── dl_model.py
├── visualizations/
│   └── nn_visualizer.py
├── app.py
├── requirements.txt
└── README.md
```

## 🔬 Modelos Implementados

| Modelo | Tipo | Librería |
|--------|------|----------|
| KNN | Clasificación | scikit-learn |
| Random Forest | Clasificación | scikit-learn |
| ID3 (Decision Tree) | Clasificación | scikit-learn |
| Red Neuronal | Clasificación | PyTorch |
| K-Means | Clustering | scikit-learn |
| Regresión Lineal | Regresión | scikit-learn |
| Apriori | Asociación | mlxtend |

## 📊 Páginas de la App

1. **Inicio** - KPIs y diagrama de red neuronal
2. **Datos SQL** - Exploración de datos desde SQLite
3. **Modelos ML** - Resultados de todos los modelos
4. **Red Neuronal DL** - Entrenamiento y métricas de PyTorch
5. **Predicción** - Formulario interactivo de predicción
6. **Resultados** - Comparación y descarga de resultados
