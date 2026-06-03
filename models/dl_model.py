import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ─────────────────────────────────────────────
#  MODELO PYTORCH
# ─────────────────────────────────────────────
class ChurnNN(nn.Module):
    def __init__(self, input_size):
        super(ChurnNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


def get_model_summary(model, input_size):
    print(f"{'Capa':<20} {'Neuronas':<12} {'Activación':<15} {'Parámetros':<12}")
    print("-" * 60)
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n_params = sum(p.numel() for p in module.parameters())
            total_params += n_params
            act = "ReLU"
            if module.out_features == 1:
                act = "Sigmoide"
            print(f"{name:<20} {module.out_features:<12} {act:<15} {n_params:<12}")
    print("-" * 60)
    print(f"{'Total':<20} {'':<12} {'':<15} {total_params:<12}")
    return total_params


# ─────────────────────────────────────────────
#  MODELO TENSORFLOW / KERAS
# ─────────────────────────────────────────────
def build_keras_model(input_size):
    """
    Red neuronal equivalente construida con TensorFlow/Keras.
    Misma arquitectura que ChurnNN: 64 → 32 → 16 → 1 (Sigmoid).
    Optimizador: Adam  |  Pérdida: Binary Cross-Entropy  |  Métrica: MSE
    """
    model = keras.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1,  activation="sigmoid"),
    ], name="ChurnNN_Keras")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["mse", "accuracy"],
    )
    return model


def train_keras_model(X_train, y_train, X_test, y_test, input_size, epochs=50):
    """
    Entrena el modelo Keras y devuelve el historial de pérdida y MSE por época.
    """
    model = build_keras_model(input_size)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        verbose=0,
    )

    loss, mse, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return {
        "model":        model,
        "train_losses": history.history["loss"],
        "mse_per_epoch": history.history["mse"],
        "val_losses":   history.history["val_loss"],
        "val_mse":      history.history["val_mse"],
        "accuracy":     accuracy,
        "mse_final":    mse,
    }


# ─────────────────────────────────────────────
#  PREPROCESAMIENTO COMPARTIDO
# ─────────────────────────────────────────────
def preprocess_data(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df["Churn_binary"] = df["Churn"].map({"Yes": 1, "No": 0})

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["customerID", "Churn", "Churn_binary"]]
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    feature_cols = [c for c in df_encoded.columns if c not in ["customerID", "Churn", "Churn_binary"]]
    X = df_encoded[feature_cols].values.astype(np.float32)
    y = df["Churn_binary"].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, feature_cols


# ─────────────────────────────────────────────
#  ENTRENAMIENTO PRINCIPAL  (PyTorch + Keras)
# ─────────────────────────────────────────────
def train_dl_model(df):
    X, y, scaler, feature_cols = preprocess_data(df)
    input_size = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── PyTorch ───────────────────────────────
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)

    pt_model   = ChurnNN(input_size)
    criterion  = nn.BCELoss()
    optimizer  = optim.Adam(pt_model.parameters(), lr=0.001)

    train_losses   = []
    mse_per_epoch  = []

    for epoch in range(50):
        pt_model.train()
        epoch_loss, batch_count = 0.0, 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = pt_model(batch_X)
            loss    = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss  += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)

        pt_model.eval()
        with torch.no_grad():
            all_preds = pt_model(X_test_t)
            mse = nn.functional.mse_loss(all_preds, y_test_t).item()
            mse_per_epoch.append(mse)

        if (epoch + 1) % 10 == 0:
            print(f"[PyTorch]  Época {epoch+1}/50 — Pérdida: {avg_loss:.4f}  MSE: {mse:.4f}")

    pt_model.eval()
    with torch.no_grad():
        y_pred_prob = pt_model(X_test_t).numpy().flatten()
        y_pred      = (y_pred_prob >= 0.5).astype(float)
        pt_accuracy = np.mean(y_pred == y_test)

    print(f"\n[PyTorch]  Precisión final: {pt_accuracy:.4f}")

    # ── TensorFlow / Keras ────────────────────
    print("\n[Keras]    Entrenando modelo TensorFlow/Keras...")
    keras_results = train_keras_model(
        X_train, y_train, X_test, y_test, input_size, epochs=50
    )
    print(f"[Keras]    Precisión final: {keras_results['accuracy']:.4f}  "
          f"MSE final: {keras_results['mse_final']:.4f}")

    return {
        # PyTorch
        "model":          pt_model,
        "train_losses":   train_losses,
        "mse_per_epoch":  mse_per_epoch,
        "accuracy":       pt_accuracy,
        "input_size":     input_size,
        "scaler":         scaler,
        "feature_cols":   feature_cols,
        # TensorFlow / Keras
        "keras_model":         keras_results["model"],
        "keras_train_losses":  keras_results["train_losses"],
        "keras_mse_per_epoch": keras_results["mse_per_epoch"],
        "keras_val_losses":    keras_results["val_losses"],
        "keras_accuracy":      keras_results["accuracy"],
        "keras_mse_final":     keras_results["mse_final"],
    }


# ─────────────────────────────────────────────
#  PREDICCIÓN
# ─────────────────────────────────────────────
def predict_churn(model, scaler, feature_cols, customer_data):
    """Predicción con el modelo PyTorch (usado en el formulario de la app)."""
    model.eval()
    input_tensor = torch.tensor(customer_data, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        prob = model(input_tensor).item()
    return prob


def predict_churn_keras(keras_model, customer_data):
    """Predicción con el modelo TensorFlow/Keras."""
    input_array = np.array(customer_data, dtype=np.float32).reshape(1, -1)
    prob = float(keras_model.predict(input_array, verbose=0)[0][0])
    return prob