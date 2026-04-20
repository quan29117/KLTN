import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

FS = 15.5                       
WINDOW_SEC = 60            
STEP_SIZE_SEC = 15              
PURITY_THRESHOLD = 0.80        

WINDOW_SAMPLES = int(WINDOW_SEC * FS)
STEP_SAMPLES = int(STEP_SIZE_SEC * FS)

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

def compute_window_features(window_df, fs=15.5, window_sec=60):
    features = {}
        
    if 'ECG-mV' in window_df.columns:
        valid_ecg = window_df['ECG-mV'].dropna().values
        features.update(extract_ecg_features(valid_ecg, fs=fs))
        
    if 'hand_GSR-mV' in window_df.columns:
        valid_hgsr = window_df['hand_GSR-mV'].dropna().values
        features.update(extract_gsr_features(valid_hgsr, prefix="handGSR", fs=fs, window_sec=window_sec))
        
    if 'foot_GSR-mV' in window_df.columns:
        valid_fgsr = window_df['foot_GSR-mV'].dropna().values
        features.update(extract_gsr_features(valid_fgsr, prefix="footGSR", fs=fs, window_sec=window_sec))
        
    if 'RESP-mV' in window_df.columns:
        valid_resp = window_df['RESP-mV'].dropna().values
        features.update(extract_resp_features(valid_resp, fs=fs))
        
    if 'EMG-mV' in window_df.columns:
        valid_emg = window_df['EMG-mV'].dropna().values
        features.update(extract_emg_features(valid_emg, fs=fs))
        
    if 'HR-bpm' in window_df.columns:
        features.update(extract_hr_features(window_df['HR-bpm'].values))

    return features

def extract_windows_from_drive(df, drive_name):
    total_samples = len(df)
    features_list = []
    
    stats = {'total': 0, 'rejected_validity': 0, 'rejected_purity': 0, 'accepted': 0}
    
    # Slide the window across the dataframe
    for start_idx in range(0, total_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES):
        end_idx = start_idx + WINDOW_SAMPLES
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
        
        if purity_ratio < float(PURITY_THRESHOLD):
            stats['rejected_purity'] += 1
            continue
            
        # RULE 3: Feature Extraction (Window is accepted)
        window_features = compute_window_features(window)

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

def batch_extract_features(input_dir, output_dir):
    """
    Processes all labeled sample files and compiles the final feature matrix.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(in_path.glob("*_labeled.csv"))
    print(f"Starting feature extraction for {len(csv_files)} drives...")
    print(f"Parameters: Window={WINDOW_SEC}s, Step={STEP_SIZE_SEC}s, Purity>={PURITY_THRESHOLD*100}%")
    
    all_drives_features = []
    
    for file in csv_files:
        drive_name = file.stem.replace("_labeled", "")
        
        try:
            df = pd.read_csv(file)
            
            df_features = extract_windows_from_drive(df, drive_name)
            
            if not df_features.empty:
                all_drives_features.append(df_features)
                
        except Exception as e:
            print(f"    [Error] Failed processing {drive_name}: {e}")

    if all_drives_features:
        final_dataset = pd.concat(all_drives_features, ignore_index=True)
        save_path = out_path / "final_feature_matrix.csv"
        final_dataset.to_csv(save_path, index=False)
        print(f"\n[Success] Feature matrix saved to {save_path.name}")
        print(f"Total training samples (windows): {len(final_dataset)}")
        print("\nClass Distribution:")
        print(final_dataset['Stress_Binary'].value_counts())
    else:
        print("\n[Warning] No valid windows extracted from any drive.")
        
batch_extract_features("./data/labeled_samples", "./data/extracted_features")