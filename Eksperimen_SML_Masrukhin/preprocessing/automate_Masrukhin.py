import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def preprocess_data(
    input_path="../dataset_raw/Crop_recommendation.csv",
    output_path="dataset_preprocessing.csv"
):
    # Load Dataset
    df = pd.read_csv(input_path)

    # Duplicate Check
    df = df.drop_duplicates()

    # Label Encoding
    encoder = LabelEncoder()
    df["label"] = encoder.fit_transform(df["label"])

    # Feature Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        df.drop(columns=["label"])
    )

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=df.drop(columns=["label"]).columns
    )

    # Gabungkan kembali
    dataset_final = X_scaled.copy()
    dataset_final["label"] = df["label"]

    # Simpan hasil preprocessing
    dataset_final.to_csv(
        output_path,
        index=False
    )

    print("Dataset preprocessing berhasil disimpan")

    return dataset_final


if __name__ == "__main__":
    preprocess_data()