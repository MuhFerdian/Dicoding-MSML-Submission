import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Crop_Recommendation")

# Load Dataset
df = pd.read_csv("dataset_preprocessing.csv")

# Feature dan Target
X = df.drop(columns=["label"])
y = df["label"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

with mlflow.start_run():

    # Parameter
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("random_state", 42)

    # Training
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    # Metric
    mlflow.log_metric(
        "accuracy",
        float(acc)
    )

    # Artifact model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

# Simpan model lokal
joblib.dump(
    model,
    "crop_model.pkl"
)

print("Model berhasil disimpan")