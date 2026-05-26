import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from tqdm import tqdm

def extract_ecg_features(ecg_signal, fs):
    peaks, _ = find_peaks(ecg_signal, distance=int(0.5 * fs), prominence=0.05)
    num_peaks = len(peaks)
    
    if num_peaks > 1:
        rr_intervals = np.diff(peaks) / fs
        mean_rr = np.mean(rr_intervals)
    else:
        mean_rr = np.nan

    return {
        "ECG_peaks": num_peaks,
        "ECG_mean_rr_interval": mean_rr
    }

def extract_gsr_features(gsr_signal, prefix, fs, window_sec=60):
    if len(gsr_signal) < int(5 * fs/15.5):
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

def extract_resp_features(resp_signal, fs):
    peaks, _ = find_peaks(resp_signal, distance=int(0.35 * fs), prominence=0.1)

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / fs
        mean_rr_resp = np.mean(rr_intervals)
    else:
        mean_rr_resp = np.nan

    return {
        "RESP_mean_rr": mean_rr_resp
    }

def extract_emg_features(emg_signal):
    if len(emg_signal) < 5:
        return {"EMG_rms": np.nan}
    rms_emg = np.sqrt(np.mean(emg_signal**2))
    return {"EMG_rms": np.round(rms_emg, 6)}

def extract_hr_features(hr_signal):
    if len(hr_signal) < 2 or np.all(np.isnan(hr_signal)):
        return {
            "HR_mean": np.nan, "HR_std": np.nan,
            "HR_rmssd": np.nan, "HR_min": np.nan, "HR_max": np.nan
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

def get_dynamic_multipliers(h5_file_obj, base_fs=15.5):
    meta_info = {}
    if 'meta' in h5_file_obj and 'fs' in h5_file_obj['meta']:
        fs_group = h5_file_obj['meta']['fs']
        for sig_name in fs_group.keys():
            current_fs = fs_group[sig_name][0]
            k_factor = current_fs / base_fs
            meta_info[sig_name] = {'fs': current_fs, 'K': k_factor}
    return meta_info

def extract_features(map_csv_path, h5_dir, output_csv_path, window_sec=60):
    print(f"\n{'='*60}")
    print("D. Extract features from signals")
    
    df_map = pd.read_csv(map_csv_path)
    all_features = []
    
    drives = df_map['drive_id'].unique()
    h5_path_obj = Path(h5_dir)
    
    for drive_id in tqdm(drives, desc="Processing Drives"):
        h5_file = h5_path_obj / f"{drive_id}.h5"
        
        if not h5_file.exists():
            print(f"File {h5_file.name} cannot found.")
            continue
            
        with h5py.File(h5_file, 'r') as f:
            meta_map = get_dynamic_multipliers(f)
            signals = f['signals']
            
            drive_windows = df_map[df_map['drive_id'] == drive_id]
            
            for _, row in drive_windows.iterrows():
                original_label = row['label']
                binary_label = 0 if original_label == 0 else 1
                
                feat_dict = {
                    'window_id': row['window_id'],
                    'drive_id': drive_id,
                    'label': binary_label,
                    'stage': row['stage']
                }
                
                s_base = row['start_idx_15_5Hz']
                e_base = row['end_idx_15_5Hz']
                
                try:
                    if 'ECG' in signals and 'ECG' in meta_map:
                        K = meta_map['ECG']['K']
                        fs = meta_map['ECG']['fs']
                        seg = signals['ECG'][int(s_base*K) : int(e_base*K)]
                        ecg_feats = extract_ecg_features(seg, fs)
                        
                        if ecg_feats['ECG_peaks'] < 40:
                            continue
                        
                        feat_dict.update(extract_ecg_features(seg, fs))
                        
                    if 'hand_GSR' in signals and 'hand_GSR' in meta_map:
                        K = meta_map['hand_GSR']['K']
                        fs = meta_map['hand_GSR']['fs']
                        seg = signals['hand_GSR'][int(s_base*K) : int(e_base*K)]
                        feat_dict.update(extract_gsr_features(seg, "handGSR", fs, window_sec))
                        
                    if 'foot_GSR' in signals and 'foot_GSR' in meta_map:
                        K = meta_map['foot_GSR']['K']
                        fs = meta_map['foot_GSR']['fs']
                        seg = signals['foot_GSR'][int(s_base*K) : int(e_base*K)]
                        feat_dict.update(extract_gsr_features(seg, "footGSR", fs, window_sec))
                        
                    if 'RESP' in signals and 'RESP' in meta_map:
                        K = meta_map['RESP']['K']
                        fs = meta_map['RESP']['fs']
                        seg = signals['RESP'][int(s_base*K) : int(e_base*K)]
                        feat_dict.update(extract_resp_features(seg, fs))
                        
                    if 'EMG' in signals:
                        seg = signals['EMG'][s_base : e_base]
                        feat_dict.update(extract_emg_features(seg))
                        
                    if 'HR' in signals:
                        seg = signals['HR'][s_base : e_base]
                        feat_dict.update(extract_hr_features(seg))

                except Exception as e:
                    print(f"\n[Error] {row['window_id']}: {e}")
                    continue
                
                all_features.append(feat_dict)

    df_final = pd.DataFrame(all_features)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_csv_path, index=False)
    print(f"\nSuccess: {output_csv_path}")
    print(f"Total: {len(df_final)}")
    print(df_final['label'].value_counts().rename(index={0: 'No Stress (0)', 1: 'Stress (1)'}))
    
def run():
    extract_features(map_csv_path="./data/label/window_map.csv",
                     h5_dir="./data/preprocessed_h5",
                     output_csv_path="./data/extracted_features/ml_features_dataset.csv")

if __name__ == "__main__":  
    extract_features(map_csv_path="./data/label/window_map.csv",
                     h5_dir="./data/preprocessed_h5",
                     output_csv_path="./data/extracted_features/ml_features_dataset.csv")