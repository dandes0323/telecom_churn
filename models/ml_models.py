import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings("ignore")


def run_all_ml_models(df):
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)

    df["Churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0})

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["customerID", "Churn", "Churn_binary"]]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    feature_cols = [c for c in df_encoded.columns if c not in ["customerID", "Churn", "Churn_binary"]]

    X = df_encoded[feature_cols].values
    y = df["Churn_binary"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    results["knn"] = {
        "accuracy": accuracy_score(y_test, y_pred_knn),
        "report": classification_report(y_test, y_pred_knn, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred_knn)
    }

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    sil_score = silhouette_score(X, kmeans_labels)
    results["kmeans"] = {
        "silhouette_score": sil_score,
        "labels": kmeans_labels,
        "centers": kmeans.cluster_centers_
    }

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["rf"] = {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "report": classification_report(y_test, y_pred_rf, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred_rf),
        "feature_importance": dict(zip(feature_cols, rf.feature_importances_))
    }

    id3 = DecisionTreeClassifier(criterion="entropy", random_state=42)
    id3.fit(X_train, y_train)
    y_pred_id3 = id3.predict(X_test)
    results["id3"] = {
        "accuracy": accuracy_score(y_test, y_pred_id3),
        "report": classification_report(y_test, y_pred_id3, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred_id3)
    }

    lr_cols = [c for c in feature_cols if c != "MonthlyCharges"]
    X_lr = df_encoded[lr_cols].values
    y_lr = df["MonthlyCharges"].values
    X_lr_train, X_lr_test, y_lr_train, y_lr_test = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
    lin_reg = LinearRegression()
    lin_reg.fit(X_lr_train, y_lr_train)
    y_pred_lr = lin_reg.predict(X_lr_test)
    results["linear_reg"] = {
        "mse": mean_squared_error(y_lr_test, y_pred_lr),
        "r2": r2_score(y_lr_test, y_pred_lr)
    }

    service_cols = ["PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    service_cols = [c for c in service_cols if c in df.columns]
    df_services = df[service_cols].copy()
    for col in df_services.columns:
        df_services[col] = df_services[col].apply(
            lambda x: True if str(x).strip() not in ["No", "0", "0.0", ""] else False
        )
    df_services = df_services.astype(bool)

    frequent_itemsets = apriori(df_services, min_support=0.1, use_colnames=True)
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
        rules = rules.sort_values("lift", ascending=False)
        results["apriori"] = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10)
    else:
        results["apriori"] = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    return results
