import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================
# Load Dataset
# =========================
df = pd.read_csv("dataset_preprocessing.csv")

# =========================
# Feature dan Target
# =========================
X = df.drop(columns=["label"])
y = df["label"]

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Training Model
# =========================
n_estimators = 200

model = RandomForestClassifier(
    n_estimators=n_estimators,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# Prediksi
# =========================
y_pred = model.predict(X_test)

# =========================
# Evaluasi
# =========================
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# =========================
# MLflow Logging
# =========================
mlflow.set_experiment("Crop_Recommendation")

with mlflow.start_run():

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "test_size",
        0.2
    )

    mlflow.log_metric(
        "accuracy",
        acc
    )

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model"
    )

# =========================
# Simpan Model
# =========================
joblib.dump(
    model,
    "crop_model.pkl"
)

print("Model berhasil disimpan")