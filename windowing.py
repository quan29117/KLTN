import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

FS = 15.5                       
WINDOW_SEC = 60            

ML_STEP_SIZE_SEC = 30              
ML_THRESHOLD = 0.80        
ML_WINDOW_SAMPLES = int(WINDOW_SEC * FS)
ML_STEP_SAMPLES = int(ML_STEP_SIZE_SEC * FS)

DL_STEP_SIZE_SEC = 30 
DL_THRESHOLD = 0.80
LABEL_MAP = {'relax': 0, 'medium': 1, 'high': 2}
DL_CHANNELS = ['ECG-mV', 'hand_GSR-mV', 'foot_GSR-mV', 'RESP-mV', 'EMG-mV']
DL_WINDOW_SAMPLES = int(WINDOW_SEC * FS)
DL_STEP_SAMPLES = int(DL_STEP_SIZE_SEC * FS)

def extract_ecg_features(ecg_signal, fs):
    peaks, _ = find_peaks(ecg_signal, distance=5, prominence=0.1)
    num_peaks = len(peaks)
    
    if num_peaks > 1:
        rr_intervals = np.diff(peaks) / fs
        mean_rr = np.mean(rr_intervals)
    else:
        mean_rr = np.nan

    return {
        "ECG_peaks":num_peaks,
        "mean_rr_interval": mean_rr
    }

def extract_gsr_features(gsr_signal, prefix="handGSR", fs=15.5, window_sec=60):
    if len(gsr_signal) < 5:
        return {
            f"{prefix}_peaks": np.nan,
            f"{prefix}_mean_peak_interval": np.nan,
            f"{prefix}_mean_peak_diff": np.nan,
            f"{prefix}_has_peaks": 0
        }

    peaks, _ = find_peaks(gsr_signal, distance=int(1.0 * fs), prominence=0.05)
    num_peaks = len(peaks)

    if num_peaks >= 2:
        peak_intervals = np.diff(peaks) / fs
        peak_values = gsr_signal[peaks]
        peak_diffs = np.abs(np.diff(peak_values))

        mean_peak_interval = np.mean(peak_intervals)
        mean_peak_diff = np.mean(peak_diffs)
        has_peaks = 1
    else:
        mean_peak_interval = float(window_sec)
        mean_peak_diff = 0.0
        has_peaks = 0

    return {
        f"{prefix}_peaks": num_peaks,
        f"{prefix}_mean_peak_interval": mean_peak_interval,
        f"{prefix}_mean_peak_diff": mean_peak_diff,
        f"{prefix}_has_peaks": has_peaks
    }

def extract_resp_features(resp_signal, fs=15.5):
    if len(resp_signal) < 5:
        return {
            "RESP_mean_rr": np.nan
        }

    peaks, _ = find_peaks(resp_signal, distance=5, prominence=0.1)

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / fs
        mean_rr_resp = np.mean(rr_intervals)
    else:
        mean_rr_resp = np.nan

    return {
        "RESP_mean_rr": mean_rr_resp
    }
def extract_emg_features(emg_signal, fs=15.5):
    if len(emg_signal) < 5:
        return {
            "EMG_rms": np.nan
        }

    rms_emg = np.sqrt(np.mean(emg_signal**2))

    return {
        "EMG_rms": np.round(rms_emg, 6)
    }

def extract_hr_features(hr_signal):
    if len(hr_signal) < 2 or np.all(np.isnan(hr_signal)):
        return {
            "HR_mean": np.nan,
            "HR_std": np.nan,
            "HR_rmssd": np.nan,
            "HR_min": np.nan,
            "HR_max": np.nan
        }

    diffs = np.diff(hr_signal)
    rmssd = np.sqrt(np.mean(diffs**2))

    return {
        "HR_mean": np.mean(hr_signal),
        "HR_std": np.std(hr_signal),
        "HR_rmssd": rmssd,
        "HR_min": np.min(hr_signal),
        "HR_max": np.max(hr_signal)
    }

def ml_compute_features(window_df, fs=15.5, window_sec=60):
    features = {}
        
    if 'ECG-mV' in window_df.columns:
        ecg_signal = window_df['ECG-mV'].dropna().values
        features.update(extract_ecg_features(ecg_signal, fs=fs))
        
    if 'hand_GSR-mV' in window_df.columns:
        hgsr_signal = window_df['hand_GSR-mV'].dropna().values
        features.update(extract_gsr_features(hgsr_signal, prefix="handGSR", fs=fs, window_sec=window_sec))
        
    if 'foot_GSR-mV' in window_df.columns:
        fgsr_signal = window_df['foot_GSR-mV'].dropna().values
        features.update(extract_gsr_features(fgsr_signal, prefix="footGSR", fs=fs, window_sec=window_sec))
        
    if 'RESP-mV' in window_df.columns:
        resp_signal = window_df['RESP-mV'].dropna().values
        features.update(extract_resp_features(resp_signal, fs=fs))
        
    if 'EMG-mV' in window_df.columns:
        emg_signal = window_df['EMG-mV'].dropna().values
        features.update(extract_emg_features(emg_signal, fs=fs))
        
    if 'HR-bpm' in window_df.columns:
        hr_signal = window_df['HR-bpm'].dropna().values
        features.update(extract_hr_features(hr_signal))

    return features

def ml_process_drive(df, drive_name):
    total_samples = len(df)
    features_list = []
    
    stats = {'total': 0, 'rejected_validity': 0, 'rejected_purity': 0, 'accepted': 0}
    
    # Slide the window across the dataframe
    for start_idx in range(0, total_samples - ML_WINDOW_SAMPLES + 1, ML_STEP_SAMPLES):
        end_idx = start_idx + ML_WINDOW_SAMPLES
        window = df.iloc[start_idx:end_idx]
        stats['total'] += 1
        
        # RULE 1: Validity Check (Reject if window contains any artifact/interpolated data)
        if 'is_valid' in window.columns and not window['is_valid'].all():
            stats['rejected_validity'] += 1
            continue
            
        # RULE 2: Binary Label Mapping & Purity Check
        labels_raw = window['Stress'].dropna()
        if len(labels_raw) == 0:
            stats['rejected_purity'] += 1
            continue
            
        binary_labels = labels_raw.map({'relax': 'no', 'medium': 'stressed', 'high': 'stressed'})
        
        binary_labels = binary_labels.dropna()
        if len(binary_labels) == 0:
            stats['rejected_purity'] += 1
            continue
            
        label_counts = binary_labels.value_counts()
        dominant_label = str(label_counts.index[0])
        purity_ratio = label_counts.iloc[0] / len(binary_labels)
        
        if purity_ratio < float(ML_THRESHOLD):
            stats['rejected_purity'] += 1
            continue
            
        # RULE 3: Feature Extraction (Window is accepted)
        window_features = ml_compute_features(window)

        # Append metadata
        window_features['Stress_Binary'] = dominant_label
        window_features['Drive'] = drive_name
        window_features['Window_Start'] = start_idx
        
        features_list.append(window_features)
        stats['accepted'] += 1
        
    print(f"    [{drive_name}] Accepted: {stats['accepted']} | "
          f"Rejected (Validity): {stats['rejected_validity']} | "
          f"Rejected (Purity): {stats['rejected_purity']}")
          
    return pd.DataFrame(features_list)

def dl_normalize_drive(df, channels):
    df_norm = df.copy()
    
    # Tìm giá trị trung bình của giai đoạn 'relax' để làm mốc (Baseline)
    baseline_df = df[df['Stress'] == 'relax']
    
    for col in channels:
        if col in df_norm.columns:
            if not baseline_df.empty:
                # Trừ đi trung bình lúc nghỉ để lấy độ lệch stress
                mean_val = baseline_df[col].mean()
                std_val = df[col].std() # Dùng std toàn cục để giữ scale
            else:
                mean_val = df[col].mean()
                std_val = df[col].std()
                
            if std_val > 0:
                df_norm[col] = (df_norm[col] - mean_val) / std_val
    return df_norm

def dl_process_drive(df, drive_name):
    df_norm = dl_normalize_drive(df, DL_CHANNELS)
    total_samples = len(df_norm)
    tensors, labels, groups = [], [], []
    
    stats = {'total': 0, 'rejected_validity': 0, 'rejected_relax': 0, 'rejected_purity': 0, 'accepted': 0}

    for start_idx in range(0, total_samples - DL_WINDOW_SAMPLES + 1, DL_STEP_SAMPLES):
        end_idx = start_idx + DL_WINDOW_SAMPLES
        window = df_norm.iloc[start_idx:end_idx]
        stats['total'] += 1

        if 'is_valid' in window.columns and not window['is_valid'].all():
            stats['rejected_validity'] += 1
            continue

        labels_raw = window['Stress'].dropna()
        binary_mapped = labels_raw.map({'medium': 0, 'high': 1}).dropna()
        
        if len(binary_mapped) < (DL_WINDOW_SAMPLES * DL_THRESHOLD):
            stats['rejected_relax'] += 1
            continue

        label_counts = binary_mapped.value_counts()
        dominant_label = int(label_counts.index[0])
        purity_ratio = label_counts.iloc[0] / len(binary_mapped)

        if purity_ratio < float(DL_THRESHOLD):
            stats['rejected_purity'] += 1
            continue

        sequence = window[DL_CHANNELS].fillna(0).values
        tensors.append(sequence)
        labels.append(dominant_label)
        groups.append(drive_name)
        stats['accepted'] += 1

    print(f"    [DL - {drive_name}] Accepted: {stats['accepted']} | "
          f"Rejected(Valid): {stats['rejected_validity']} | "
          f"Rejected(Relax): {stats['rejected_relax']} | "
          f"Rejected(Purity): {stats['rejected_purity']}")
    
    return tensors, labels, groups

def save_ml_results(all_features, output_path):
    if not all_features:
        print("[Warning] No ML windows extracted.")
        return
    final_df = pd.concat(all_features, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    print(f"\n[Success] ML Feature matrix saved to {output_path.name}")
    print(f"Total ML samples: {len(final_df)}")
    print(final_df['Stress_Binary'].value_counts().to_string())

def save_dl_results(X_list, y_list, g_list, output_path):
    if not X_list:
        print("[Warning] No DL tensors extracted.")
        return
    X_final = np.array(X_list)
    y_final = np.array(y_list)
    groups_final = np.array(g_list)
    
    np.savez_compressed(output_path, X=X_final, y=y_final, groups=groups_final)
    
    print(f"\n[Success] DL Tensor dataset saved to {output_path.name}")
    print(f"Shape: {X_final.shape} (Samples, Time, Channels)")
    unique, counts = np.unique(y_final, return_counts=True)
    print(f"Distribution: {dict(zip(unique, counts))}")

def batch_extract_features(input_dir, output_dir_ml, output_dir_dl):
    in_path = Path(input_dir)
    csv_files = sorted(in_path.glob("*_labeled.csv"))
    
    all_ml_features = []
    all_dl_X, all_dl_y, all_dl_groups = [], [], []
    
    for file in csv_files:
        drive_name = file.stem.replace("_labeled", "")
        print(f"Processing: {drive_name}...")
        
        try:
            df = pd.read_csv(file)
            
            df_ml = ml_process_drive(df, drive_name)
            if not df_ml.empty:
                all_ml_features.append(df_ml)
                
            X_dl, y_dl, g_dl = dl_process_drive(df, drive_name)
            if X_dl:
                all_dl_X.extend(X_dl)
                all_dl_y.extend(y_dl)
                all_dl_groups.extend(g_dl)
                
        except Exception as e:
            print(f"    [Error] {drive_name}: {e}")

    Path(output_dir_ml).mkdir(parents=True, exist_ok=True)
    Path(output_dir_dl).mkdir(parents=True, exist_ok=True)
    
    save_ml_results(all_ml_features, Path(output_dir_ml) / "final_feature_matrix.csv")
    save_dl_results(all_dl_X, all_dl_y, all_dl_groups, Path(output_dir_dl) / "dl_tensor_dataset.npz")
    
# def dl_normalize_drive(df, channels):
#     df_norm = df.copy()
    
#     # Tìm giá trị trung bình của giai đoạn 'relax' để làm mốc (Baseline)
#     baseline_df = df[df['Stress'] == 'relax']
    
#     for col in channels:
#         if col in df_norm.columns:
#             if not baseline_df.empty:
#                 # Trừ đi trung bình lúc nghỉ để lấy độ lệch stress
#                 mean_val = baseline_df[col].mean()
#                 std_val = df[col].std() # Dùng std toàn cục để giữ scale
#             else:
#                 # Nếu không có đoạn relax (hiếm gặp), dùng trung bình toàn cục
#                 mean_val = df[col].mean()
#                 std_val = df[col].std()
                
#             if std_val > 0:
#                 df_norm[col] = (df_norm[col] - mean_val) / std_val
#     return df_norm

# def dl_process_drive(df, drive_name):
#     # 1. Chuẩn hóa dữ liệu theo Baseline của chính chuyến đi đó
#     df_norm = dl_normalize_drive(df, DL_CHANNELS)
#     total_samples = len(df_norm)
#     tensors, labels, groups = [], [], []
    
#     # Thống kê để theo dõi
#     stats = {0: 0, 1: 0, 2: 0, 'rejected': 0}

#     # 2. Quét cửa sổ trên toàn bộ dữ liệu (Không lọc bỏ 3-9 phút)
#     for start_idx in range(0, total_samples - DL_WINDOW_SAMPLES + 1, DL_STEP_SAMPLES):
#         end_idx = start_idx + DL_WINDOW_SAMPLES
#         window = df_norm.iloc[start_idx:end_idx]

#         # Kiểm tra tính hợp lệ của tín hiệu (is_valid)
#         if 'is_valid' in window.columns and not window['is_valid'].all():
#             stats['rejected'] += 1
#             continue

#         # Lấy nhãn Stress và ánh xạ sang 3 lớp
#         labels_raw = window['Stress'].dropna()
#         # Ánh xạ: relax->0, medium->1, high->2
#         mapped = labels_raw.map(LABEL_MAP).dropna()
        
#         # Kiểm tra độ lấp đầy (Purity/Threshold)
#         if len(mapped) < (DL_WINDOW_SAMPLES * DL_THRESHOLD):
#             stats['rejected'] += 1
#             continue

#         # Xác định nhãn chiếm đa số trong cửa sổ
#         label_counts = mapped.value_counts()
#         dominant_label = int(label_counts.index[0])
#         purity_ratio = label_counts.iloc[0] / len(mapped)

#         # Kiểm tra độ thuần khiết của nhãn trong cửa sổ
#         if purity_ratio < float(DL_THRESHOLD):
#             stats['rejected'] += 1
#             continue

#         # Trích xuất Tensor và lưu trữ
#         sequence = window[DL_CHANNELS].fillna(0).values
#         tensors.append(sequence)
#         labels.append(dominant_label)
#         groups.append(drive_name)
        
#         stats[dominant_label] += 1

#     print(f"    [DL 3-Class - {drive_name}] Accepted: {len(tensors)} windows")
#     print(f"    Distribution: Relax(0): {stats[0]} | Medium(1): {stats[1]} | High(2): {stats[2]} | Rejected: {stats['rejected']}")
    
#     return tensors, labels, groups
def dl_ml_process_hybrid(df, drive_name):
    # 1. Chuẩn hóa Baseline trước (giống cách DL làm)
    df_norm = dl_normalize_drive(df, DL_CHANNELS)
    total_samples = len(df_norm)
    
    hybrid_X_dl = []    # Chứa tín hiệu thô
    hybrid_X_meta = []  # Chứa đặc trưng thủ công
    hybrid_y = []
    hybrid_groups = []
    
    # Sử dụng bước nhảy chung (ví dụ 30s)
    for start_idx in range(0, total_samples - DL_WINDOW_SAMPLES + 1, DL_STEP_SAMPLES):
        end_idx = start_idx + DL_WINDOW_SAMPLES
        window = df_norm.iloc[start_idx:end_idx]
        
        # --- KIỂM TRA ĐIỀU KIỆN (Giống logic cũ của bạn) ---
        if 'is_valid' in window.columns and not window['is_valid'].all():
            continue

        labels_raw = window['Stress'].dropna()
        # Ánh xạ về 2 lớp: Medium(0) và High(1)
        binary_mapped = labels_raw.map({'medium': 0, 'high': 1}).dropna()
        
        if len(binary_mapped) < (DL_WINDOW_SAMPLES * DL_THRESHOLD):
            continue

        label_counts = binary_mapped.value_counts()
        dominant_label = int(label_counts.index[0])
        purity_ratio = label_counts.iloc[0] / len(binary_mapped)

        if purity_ratio < float(DL_THRESHOLD):
            continue

        # --- TRÍCH XUẤT SONG SONG ---
        # A. Tín hiệu thô cho DL (CNN-LSTM)
        sequence = window[DL_CHANNELS].fillna(0).values
        
        # B. Đặc trưng thủ công cho ML (Nhánh Dense)
        # Lưu ý: ml_compute_features trả về một dictionary
        ml_features_dict = ml_compute_features(window, fs=FS, window_sec=WINDOW_SEC)
        # Chuyển dict thành vector (list) theo thứ tự cố định
        ml_vector = list(ml_features_dict.values())

        hybrid_X_dl.append(sequence)
        hybrid_X_meta.append(ml_vector)
        hybrid_y.append(dominant_label)
        hybrid_groups.append(drive_name)
        
    return hybrid_X_dl, hybrid_X_meta, hybrid_y, hybrid_groups

def batch_extract_hybrid(input_dir, output_dir):
    in_path = Path(input_dir)
    csv_files = sorted(in_path.glob("*_labeled.csv"))
    
    all_X_dl, all_X_meta, all_y, all_groups = [], [], [], []
    
    for file in csv_files:
        drive_name = file.stem.replace("_labeled", "")
        print(f"Hybrid Processing: {drive_name}...")
        
        try:
            df = pd.read_csv(file)
            X_dl, X_meta, y, g = dl_ml_process_hybrid(df, drive_name)
            
            if X_dl:
                all_X_dl.extend(X_dl)
                all_X_meta.extend(X_meta)
                all_y.extend(y)
                all_groups.extend(g)
                
        except Exception as e:
            print(f"    [Error] {drive_name}: {e}")

    # Lưu file NPZ nén
    output_path = Path(output_dir) / "hybrid_dataset.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path, 
        X_dl=np.array(all_X_dl), 
        X_meta=np.array(all_X_meta), 
        y=np.array(all_y), 
        groups=np.array(all_groups)
    )
    
    print(f"\n[Success] Hybrid dataset saved to {output_path}")
    print(f"Total Samples: {len(all_y)}")
    
batch_extract_features("./data/labeled_samples", "./data/extracted_features", "./data/dl_tensors")
batch_extract_hybrid("./data/labeled_samples", "./hybrid")