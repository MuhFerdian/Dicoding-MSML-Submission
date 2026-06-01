# Crop Recommendation ML

Proyek machine learning untuk merekomendasikan jenis tanaman berdasarkan:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- pH
- Rainfall

## Struktur

### Eksperimen_SML_Masrukhin
Eksperimen preprocessing dataset.

### Membangun_model
Training model Random Forest dan penyimpanan model.

### Workflow-CI
Implementasi CI/CD menggunakan GitHub Actions.

## Model

Algoritma:
- Random Forest Classifier

## CI/CD
Folder: Workflow-CI
Workflow otomatis berjalan saat push ke branch main dan menghasilkan artifact model (.pkl).

## Dataset
Crop Recommendation Dataset

## Eksperimen
Folder: Eksperimen_SML_Masrukhin

## Training Model
Folder: Membangun_model

Algoritma:
- Random Forest

GitHub Actions digunakan untuk:
- install dependencies
- menjalankan training
- menghasilkan artifact model