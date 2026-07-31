#Pipeline training + kuantisasi INT8 untuk AF CNN.

import numpy as np
import h5py
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import butter, sosfiltfilt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------------------------------------------------------
# BANDPASS FILTER
# ---------------------------------------------------------------
def bandpass_filter(signal, fs, lowcut=0.5, highcut=8.0, order=4):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal)

# ---------------------------------------------------------------
# KONFIGURASI
# ---------------------------------------------------------------
DATA_DIR = Path("datasets/zenodo_training")
ECG_FS = 500
PPG_FS = 100
WINDOW_SEC = 10
WINDOW_SAMPLES = PPG_FS * WINDOW_SEC
STEP_SAMPLES = WINDOW_SAMPLES              # step normal (non-overlap) -- dipakai test & non-AF
FINE_STEP_SAMPLES = WINDOW_SAMPLES // 4    # step rapat (overlap 75%) -- HANYA training, HANYA label AF

SUBJECT_IDS = [f"{i:03d}" for i in range(1, 13)]
BEST_SEED = 0                 # seed test split (dari pencarian sebelumnya)
VAL_SEED = 1                  # seed validation split
TEST_SUBJECT_IDS = {"005", "007", "012"}   # deterministik untuk BEST_SEED=0, hasil GroupShuffleSplit

# ---------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------
def ascii_to_str(arr):
    return "".join(chr(int(x)) for x in np.array(arr).ravel())

def parse_day(arr):
    return int(ascii_to_str(arr))

def parse_time_sec(arr):
    h, m, s = map(int, ascii_to_str(arr).split(":"))
    return h * 3600 + m * 60 + s

def dereference_cell(f, name):
    return [np.array(f[ref][:]).squeeze() for ref in f[name][:, 0]]

def load_ecg(subject_id):
    path = DATA_DIR / f"{subject_id}_ECG.mat"
    with h5py.File(path, "r") as f:
        ecg_day = parse_day(f["recording_startday"][:])
        ecg_time = parse_time_sec(f["recording_starttime"][:])
        qrs_idx = f["QRSindex"][:].ravel()
        af = f["AF_annotation"][:].ravel()
    beat_times = qrs_idx / ECG_FS
    interval_start = beat_times[:-1]
    interval_end = beat_times[1:]
    return ecg_day, ecg_time, interval_start, interval_end, af

def load_ppg_segments(subject_id):
    path = DATA_DIR / f"{subject_id}_PPG.mat"
    with h5py.File(path, "r") as f:
        signals = dereference_cell(f, "PPG_GREEN")
        startdays = dereference_cell(f, "recording_startday")
        starttimes = dereference_cell(f, "recording_starttime")
    segments = []
    for i, (sig, day_arr, time_arr) in enumerate(zip(signals, startdays, starttimes)):
        try:
            day = parse_day(day_arr)
            time_sec = parse_time_sec(time_arr)
        except ValueError:
            print(f"    [skip segmen {i}] subject {subject_id}: metadata timestamp tidak valid")
            continue
        if len(sig) == 0:
            continue
        segments.append((sig, day, time_sec))
    return segments


def segment_and_label(subject_id):
    """
    Pasien TEST (TEST_SUBJECT_IDS)  -> selalu step normal, semua label.
    Pasien TRAINING                 -> label AF pakai step rapat (overlap),
                                        label non-AF tetap step normal.
    """
    is_test_subject = subject_id in TEST_SUBJECT_IDS

    ecg_day, ecg_time, interval_start, interval_end, af = load_ecg(subject_id)
    ppg_segments = load_ppg_segments(subject_id)

    windows, labels = [], []
    for sig, seg_day, seg_time in ppg_segments:
        offset = (seg_day - ecg_day) * 86400 + (seg_time - ecg_time)

        if len(sig) > 100:
            sig = bandpass_filter(sig, fs=PPG_FS, lowcut=0.5, highcut=8.0)

        n = len(sig)
        scan_step = STEP_SAMPLES if is_test_subject else FINE_STEP_SAMPLES

        for start in range(0, n - WINDOW_SAMPLES + 1, scan_step):
            end = start + WINDOW_SAMPLES
            t0 = offset + start / PPG_FS
            t1 = offset + end / PPG_FS

            overlap = (interval_start < t1) & (interval_end > t0)
            beats_af = af[overlap]

            if len(beats_af) == 0:
                continue

            label = 1 if beats_af.mean() > 0.5 else 0

            # non-AF (dan SEMUA window pasien test) cuma diambil di step normal
            if (label == 0 or is_test_subject) and start % STEP_SAMPLES != 0:
                continue

            windows.append(sig[start:end].astype(np.float32))
            labels.append(label)

    return np.array(windows, dtype=np.float32), np.array(labels)


def build_dataset(subject_ids):
    all_X, all_y, all_groups = [], [], []
    for sid in subject_ids:
        try:
            X_sub, y_sub = segment_and_label(sid)
        except FileNotFoundError:
            print(f"  [skip] file untuk subject {sid} tidak ditemukan")
            continue
        if len(X_sub) == 0:
            continue
        all_X.append(X_sub)
        all_y.append(y_sub)
        all_groups.append(np.full(len(y_sub), sid))
        print(f"  subject {sid}: {len(X_sub)} window, {int(y_sub.sum())} AF / {len(y_sub) - int(y_sub.sum())} non-AF")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    return X, y, groups


# ---------------------------------------------------------------
# KUANTISASI INT8 (forward-pass only, bukan training)
# ---------------------------------------------------------------
def make_representative_dataset(X_train, groups_train, n_samples_per_subject=30):
    unique_subjects = np.unique(groups_train)
    selected_indices = []
    rng = np.random.default_rng(42)
    for subj in unique_subjects:
        subj_idx = np.where(groups_train == subj)[0]
        n_pick = min(n_samples_per_subject, len(subj_idx))
        picked = rng.choice(subj_idx, size=n_pick, replace=False)
        selected_indices.extend(picked)
    selected_indices = np.array(selected_indices)
    print(f"Representative dataset: {len(selected_indices)} sample dari {len(unique_subjects)} pasien")

    def representative_dataset():
        for idx in selected_indices:
            yield [X_train[idx:idx + 1].astype(np.float32)]

    return representative_dataset


def quantize_model_int8(model, X_train, groups_train, output_path="af_cnn_int8.tflite"):
    rep_dataset_fn = make_representative_dataset(X_train, groups_train)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset_fn
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model terkuantisasi disimpan: {output_path}")
    return output_path


def evaluate_tflite_model(tflite_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]

    y_pred = []
    for i in range(len(X_test)):
        sample = X_test[i:i + 1].astype(np.float32)
        if input_scale != 0:
            sample_q = np.round(sample / input_scale + input_zero_point)
            sample_q = np.clip(sample_q, -128, 127).astype(input_details["dtype"])
        else:
            sample_q = sample.astype(input_details["dtype"])

        interpreter.set_tensor(input_details["index"], sample_q)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        y_pred.append(np.argmax(output))
        if (i + 1) % 10000 == 0:
            print(f"  Evaluasi... {i + 1}/{len(X_test)}")

    y_pred = np.array(y_pred)
    print("\n=== Hasil model INT8 (.tflite) ===")
    print(classification_report(y_test, y_pred, target_names=["non-AF", "AF"]))
    print(confusion_matrix(y_test, y_pred))
    return y_pred


# ---------------------------------------------------------------
# GPU CONFIG
# ---------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Konfigurasi GPU Memory Growth Berhasil diaktifkan!")
    except RuntimeError as e:
        print(f"Gagal mengaktifkan Memory Growth: {e}")

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("Membangun dataset (overlapping HANYA untuk pasien training)...")
    X, y, groups = build_dataset(SUBJECT_IDS)
    print(f"\nTotal window: {len(X)}  |  AF: {int(y.sum())}  |  non-AF: {len(y) - int(y.sum())}")

    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    X = X.reshape(X.shape[0], X.shape[1], 1).astype(np.float32)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=BEST_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    print(f"Train: {len(X_train)} window dari {len(set(groups_train))} pasien")
    print(f"Test:  {len(X_test)} window dari {len(set(groups[test_idx]))} pasien")
    assert set(groups[test_idx]) == TEST_SUBJECT_IDS, "Pasien test tidak sesuai asumsi -- cek ulang seed!"

    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=VAL_SEED)
    tr_idx, val_idx = next(gss_val.split(X_train, y_train, groups=groups_train))
    X_tr, X_val = X_train[tr_idx], X_train[val_idx]
    y_tr, y_val = y_train[tr_idx], y_train[val_idx]

    print(f"  -> Sub-train: {len(X_tr)} window, Validation: {len(X_val)} window")

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"Class weights: {class_weight_dict}")

    # --- Model ---
    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=(X_tr.shape[1], 1)),
        MaxPooling1D(2),
        Conv1D(128, 3, activation="relu"),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_delta=1e-4, verbose=1)
    checkpoint = ModelCheckpoint("af_cnn_best_model_v2.keras", monitor="val_loss", save_best_only=True, verbose=1)

    history = model.fit(
        X_tr, y_tr,
        epochs=30, batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr, checkpoint],
    )

    model.save("af_cnn_final_v2.keras")
    print("Model tersimpan sebagai af_cnn_final_v2.keras")

    # --- Evaluasi TEST SET (bersih, non-overlap) ---
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\n=== Classification Report (Test Set, non-overlap) ===")
    print(classification_report(y_test, y_pred, target_names=["non-AF", "AF"]))
    print(confusion_matrix(y_test, y_pred))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="training loss")
    axes[0].plot(history.history["val_loss"], label="validation loss")
    axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(history.history["accuracy"], label="training accuracy")
    axes[1].plot(history.history["val_accuracy"], label="validation accuracy")
    axes[1].legend(); axes[1].set_title("Accuracy")
    plt.tight_layout()
    plt.savefig("training_history_v2.png")

    # ---------------------------------------------------------------
    # LANJUT KE KUANTISASI INT8 (forward-pass only, TIDAK training ulang)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KUANTISASI INT8")
    print("=" * 60)

    tflite_path = quantize_model_int8(model, X_train, groups_train, output_path="af_cnn_int8_v2.tflite")
    evaluate_tflite_model(tflite_path, X_test, y_test)
