# PPG AF Detection

Deteksi Atrial Fibrillation dari sinyal PPG menggunakan 1D-CNN, di-deploy ke Grove Vision AI V2 (Himax EPII CM55M + Ethos-U55).

## Fitur
- Dua mode operasi: Golden (data referensi) dan Sensor (real-time dari MAX30102)
- Dashboard GUI untuk monitoring dan kontrol interaktif
- Model INT8 terkuantisasi untuk inference di edge device

## Struktur Proyek

```
PPG_AF_Project/
├── firmware/ # Kode C untuk board (Grove Vision AI V2)
├── training/ # Training & preprocessing (Python)
├── dashboard/ # GUI dashboard (Python/Tkinter)
├── models/ # Model TFLite (.tflite) dan Keras (.keras)
└── docs/ # Dokumentasi (laporan, dll.)
```

## Dataset

Proyek ini menggunakan dua dataset:

### 1. Dataset Training (Long-term ECG + PPG)
Dataset dari Vilnius University dengan 45 pasien, sampling rate PPG 100 Hz dan ECG 500 Hz.

**Download:** [Zenodo - Long-term ECG and PPG recordings](https://zenodo.org/records/11242869)

### 2. Dataset Golden Mode (MIMIC PERform AF)
Dataset MIMIC PERform AF untuk mode golden di dashboard.

**Download:** [Zenodo - MIMIC PERform Datasets](https://zenodo.org/records/15906524)

## Model

Arsitektur 1D-CNN:

- Conv1D(64, 3) -> MaxPool(2)
- Conv1D(128, 3) -> MaxPool(2)
- Flatten -> Dense(128) -> Dropout(0.5) -> Dense(2) (softmax)
- Kuantisasi INT8 (PTQ) menggunakan TensorFlow Lite
