import numpy as np
import pandas as pd
import os
from scipy.signal import butter, sosfiltfilt, resample
import re

# ============================================================
# KONFIGURASI
# ============================================================
FS_ORIG = 125          # sampling rate asli MIMIC
FS_TARGET = 100        # sampling rate target (sesuai model)
WINDOW_SEC = 10
WINDOW_SAMPLES = FS_TARGET * WINDOW_SEC  # 1000
LOWCUT = 0.5
HIGHCUT = 8.0
ORDER = 4

AF_FOLDER = "datasets/mimic_perform_af_csv"
NON_AF_FOLDER = "datasets/mimic_perform_non_af_csv"
OUTPUT_FOLDER = "npy_preprocessed"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================
# FILTER BANDPASS (sama seperti firmware & golden_reference)
# ============================================================
def bandpass_filter(signal, fs=FS_TARGET):
    nyquist = fs / 2
    sos = butter(ORDER, [LOWCUT/nyquist, HIGHCUT/nyquist], btype='band', output='sos')
    return sosfiltfilt(sos, signal)

# ============================================================
# PREPROCESS SATU FILE CSV -> WINDOW (filter + normalisasi)
# ============================================================
def preprocess_windows(ppg_raw):
    # 1. Resample 125 → 100 Hz
    num_samples = int(len(ppg_raw) * FS_TARGET / FS_ORIG)
    ppg_resampled = resample(ppg_raw, num_samples)
    
    # 2. Bandpass filter
    filtered = bandpass_filter(ppg_resampled, fs=FS_TARGET)
    
    # 3. Potong menjadi window 10 detik (non‑overlap)
    windows = []
    n = len(filtered)
    for start in range(0, n - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
        window = filtered[start:start + WINDOW_SAMPLES]
        if len(window) == WINDOW_SAMPLES:
            # 4. Normalisasi Z‑score per window
            mean = np.mean(window)
            std = np.std(window) + 1e-8
            norm = (window - mean) / std
            # Cek jika std = 0 (sinyal konstan) -> skip
            if np.std(window) < 1e-8:
                continue
            windows.append(norm)
    return np.array(windows, dtype=np.float32)

# ============================================================
# PROSES SEMUA FILE DI SATU FOLDER
# ============================================================
def process_folder(folder_path, label, prefix):
    """
    folder_path: path ke folder CSV
    label: 1 untuk AF, 0 untuk non-AF
    prefix: 'af' atau 'non' untuk nama file output
    """
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('_data.csv')])
    results = []
    for filename in files:
        # Ambil ID (angka) dari nama file
        match = re.search(r'(\d+)', filename)
        if match:
            subject_id = match.group(1)
        else:
            subject_id = filename.split('_')[-2]
        
        print(f"Processing {prefix}_{subject_id}...")
        
        df = pd.read_csv(os.path.join(folder_path, filename))
        ppg = df['PPG'].values.astype(np.float32)
        windows = preprocess_windows(ppg)
        
        if len(windows) == 0:
            print(f"  No windows for {subject_id}")
            continue
        
        # Simpan X dan Y
        x_file = os.path.join(OUTPUT_FOLDER, f"{prefix}_{subject_id}_X.npy")
        y_file = os.path.join(OUTPUT_FOLDER, f"{prefix}_{subject_id}_Y.npy")
        np.save(x_file, windows)
        np.save(y_file, np.full(len(windows), label, dtype=np.int8))
        
        print(f"  Saved {len(windows)} windows")
        results.append((subject_id, len(windows), label))
    return results

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("MIMIC CSV -> Preprocessed NPY")
    print("Filter: 0.5–8 Hz, Normalisasi Z-score")
    print("="*60)
    
    print("\n[1] Processing AF subjects...")
    af_results = process_folder(AF_FOLDER, label=1, prefix="af")
    
    print("\n[2] Processing non-AF subjects...")
    non_results = process_folder(NON_AF_FOLDER, label=0, prefix="non")
    
    # Ringkasan
    print("\n" + "="*60)
    print("RINGKASAN")
    print("="*60)
    total = 0
    for r in af_results + non_results:
        print(f"  {r[0]}: {r[1]} windows, label={r[2]}")
        total += r[1]
    print(f"\nTotal windows: {total}")
    print(f"\nOutput saved to: {OUTPUT_FOLDER}/")
