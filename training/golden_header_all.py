import numpy as np
import os
import glob
import tensorflow as tf

# ============================================================
# KONFIGURASI
# ============================================================
OUTPUT_FOLDER = "npy_preprocessed"          # folder hasil preprocess_mimic_to_npy.py
HEADER_FILE = "golden_all.h"
MODEL_PATH = "af_cnn_int8_v2.tflite"        # model TFLite (untuk ambil scale & zero_point)
MAX_WINDOWS_PER_PATIENT = 4

# ============================================================
# LOAD MODEL & DAPATKAN PARAMETER KUANTISASI
# ============================================================
interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
input_scale = inp['quantization'][0]
input_zero_point = inp['quantization'][1]

print(f"Input scale: {input_scale}, zero_point: {input_zero_point}")

# ============================================================
# KUANTISASI (sama seperti di golden_reference_all.py)
# ============================================================
def quantize_window(window_float):
    """
    window_float: array float normalized (shape: 1000)
    return: int8 array (shape: 1000)
    """
    # Ubah ke shape (1, 1000, 1) sesuai input model
    sample = window_float.reshape(1, -1, 1).astype(np.float32)
    q = np.round(sample / input_scale + input_zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q.flatten()

# ============================================================
# PROSES SEMUA FILE .npy
# ============================================================
x_files = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "*_X.npy")))
if not x_files:
    print(f"Tidak ada file X ditemukan di folder '{OUTPUT_FOLDER}'.")
    print("Jalankan preprocess_mimic_to_npy.py dulu.")
    exit()

print(f"Ditemukan {len(x_files)} file X")

all_data = []
all_labels = []
patient_ids = []
max_windows = 0

for x_file in x_files:
    base = os.path.basename(x_file).replace("_X.npy", "")
    y_file = os.path.join(OUTPUT_FOLDER, f"{base}_Y.npy")
    
    if not os.path.exists(y_file):
        print(f"Warning: {y_file} tidak ditemukan, skip {base}")
        continue
    
    X = np.load(x_file)          # shape (N, 1000) float
    Y = np.load(y_file)          # shape (N,) int8

    X = X[:MAX_WINDOWS_PER_PATIENT] 
    Y = Y[:MAX_WINDOWS_PER_PATIENT]
    
    # Cek NaN dan ganti dengan 0
    if np.any(np.isnan(X)):
        print(f"Warning: NaN ditemukan di {base}, mengganti dengan 0")
        X = np.nan_to_num(X, nan=0.0)
    
    # Kuantisasi setiap window
    X_quant = np.array([quantize_window(window) for window in X], dtype=np.int8)
    
    all_data.append(X_quant)
    all_labels.append(Y)
    patient_ids.append(base)
    if X_quant.shape[0] > max_windows:
        max_windows = X_quant.shape[0]

num_patients = len(all_data)
print(f"Total patients: {num_patients}, max windows: {max_windows}")

# Buat array 3D (num_patients x max_windows x 1000)
X_all = np.zeros((num_patients, max_windows, 1000), dtype=np.int8)
Y_all = np.full((num_patients, max_windows), -1, dtype=np.int8)  # -1 = padding (tidak dipakai)

for i, (X, Y) in enumerate(zip(all_data, all_labels)):
    n = X.shape[0]
    X_all[i, :n, :] = X
    Y_all[i, :n] = Y

# ============================================================
# GENERATE HEADER C
# ============================================================
def array3d_to_c(arr, name, dtype="int8_t"):
    """
    Ubah array 3D numpy menjadi string array C 3D.
    """
    lines = []
    for patient in arr:
        patient_lines = []
        for window in patient:
            values = ", ".join(str(int(v)) for v in window)
            patient_lines.append(f"        {{{values}}}")
        lines.append("    {\n" + ",\n".join(patient_lines) + "\n    }")
    return f"static const {dtype} {name}[{arr.shape[0]}][{arr.shape[1]}][{arr.shape[2]}] = {{\n" + ",\n".join(lines) + "\n};"

def array2d_to_c(arr, name, dtype="int8_t"):
    """
    Ubah array 2D numpy menjadi string array C 2D.
    """
    lines = []
    for patient in arr:
        values = ", ".join(str(int(v)) for v in patient)
        lines.append(f"    {{{values}}}")
    return f"static const {dtype} {name}[{arr.shape[0]}][{arr.shape[1]}] = {{\n" + ",\n".join(lines) + "\n};"

# Tulis header
with open(HEADER_FILE, "w") as f:
    f.write("#ifndef GOLDEN_ALL_H\n#define GOLDEN_ALL_H\n#include <stdint.h>\n\n")
    f.write(array3d_to_c(X_all, "golden_input_all"))
    f.write("\n\n")
    f.write(array2d_to_c(Y_all, "golden_label_all"))
    f.write("\n\n")
    f.write(f"#define TOTAL_PATIENTS {num_patients}\n")
    f.write(f"#define MAX_WINDOWS {max_windows}\n")
    f.write("#define WINDOW_SIZE 1000\n")
    f.write("#endif\n")

# ============================================================
# STATISTIK & VERIFIKASI
# ============================================================
print(f"\nHeader file '{HEADER_FILE}' generated successfully.")
print(f"Total data size:")
print(f"  Input  : {X_all.nbytes / 1024:.2f} KiB")
print(f"  Labels : {Y_all.nbytes / 1024:.2f} KiB")
print(f"  Total  : {(X_all.nbytes + Y_all.nbytes) / 1024:.2f} KiB")

# Cek beberapa sample pertama
print("\nSample data (patient 0, window 0, first 10 values):")
print(X_all[0, 0, :10])
