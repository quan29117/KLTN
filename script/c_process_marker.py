import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from scipy.signal import find_peaks

stages = ['Start', 'Rest1', 'City1', 'Hwy1', 'Return', 'Hwy2', 'City2', 'Rest2']

def generate_marker_info(input_dir, output_file):
    base_path = Path(input_dir).resolve()
    h5_files = sorted(list(base_path.glob("*.h5")))
    
    marker_results = []

    for h5_f in h5_files:
        try:
            with h5py.File(h5_f, 'r') as f:
                marker_signal = f['signals/marker'][:].flatten()

            peaks, _ = find_peaks(
                marker_signal, 
                distance=4000,
                prominence=1
            )

            drive_info = {'Driver': h5_f.stem}
            
            for i in range(len(stages)):
                if i < len(peaks):
                    drive_info[stages[i]] = peaks[i]
                else:
                    drive_info[stages[i]] = np.nan
            
            if drive_info['Driver'] == 'drive16':
                drive_info['City2'] = 60418

            marker_results.append(drive_info)

        except Exception as e:
            print(f"Error file: {f.name}: {e}")

    marker_df = pd.DataFrame(marker_results)
    marker_df = marker_df[['Driver'] + stages]
    marker_df = clean_invalid_drives(marker_df, input_dir)
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    marker_df.to_csv(output_file, index=False)
    
    print(f"\nSuccess: {output_file}")
    return marker_df

def clean_invalid_drives(marker_df, input_dir):
    """
    Identifies drives with insufficient markers (< 5) and deletes 
    their corresponding CSV files to ensure data consistency.
    """
    stage_cols = [col for col in marker_df.columns if col != 'Driver']
    
    invalid_rows = marker_df[marker_df[stage_cols].count(axis=1) < 5]
    invalid_drives = invalid_rows['Driver'].tolist()
    
    if not invalid_drives:
        print("All drives meet the minimum marker requirement.")
        return marker_df

    print(f"Removing {len(invalid_drives)} invalid drives: {invalid_drives}")

    for drive in invalid_drives:
        file_to_delete = Path(input_dir) / f"{drive}.h5"
        try:
            if file_to_delete.exists():
                file_to_delete.unlink()
                print(f"Deleted: {file_to_delete.name}")
        except Exception as e:
            print(f"Error deleting {drive}: {e}")

    return marker_df[marker_df[stage_cols].count(axis=1) >= 5].copy()

def generate_window_map(marker_csv_path, h5_clean_dir, output_csv_path, window_sec=60, overlap_sec=30, fs=15.5, purity_threshold=0.8):
    marker_df = pd.read_csv(marker_csv_path)
    window_size = int(window_sec * fs)
    step_size = int((window_sec - overlap_sec) * fs)
    
    window_list = []
    time_cols = list(marker_df.columns[1:])
    
    print(f"\n{'='*60}")
    print("C. Process marker")

    for _, row in marker_df.iterrows():
        driver_id = row['Driver']
        
        if pd.isna(row['Rest2']):
            h5_path = Path(h5_clean_dir) / f"{driver_id}.h5"
            if h5_path.exists():
                with h5py.File(h5_path, 'r') as f:
                    if 'marker' in f['signals']:
                        actual_total_samples = len(f['signals']['marker'])
                    else:
                        fs_ecg = f['meta']['fs']['ECG'][0]
                        actual_total_samples = int(len(f['signals']['ECG']) / (fs_ecg / fs))
                    
                    row['Rest2'] = actual_total_samples
            else:
                print(f"[Warning] {driver_id}: Lack of Rest2 & no .h5 file")
                continue

        total_samples = int(row['Rest2'])
        
        drive_labels = np.full(total_samples, -1)
        drive_stages = np.full(total_samples, "Unknown", dtype=object)
        
        for i in range(1, len(time_cols)):
            start_col = time_cols[i-1]
            end_col = time_cols[i]
            
            if pd.isna(row[start_col]) or pd.isna(row[end_col]):
                continue
                
            start_idx = int(row[start_col])
            end_idx = int(row[end_col])
            
            stage_name = end_col
            label = 0 if 'rest' in stage_name.lower() else (2 if 'city' in stage_name.lower() else 1)
            
            drive_labels[start_idx:end_idx] = label
            drive_stages[start_idx:end_idx] = stage_name
            
        # windowing
        for w_start in range(0, total_samples - window_size + 1, step_size):
            w_end = w_start + window_size
            w_labels = drive_labels[w_start:w_end]
            
            valid_labels = w_labels[w_labels != -1]
            if len(valid_labels) < (window_size * purity_threshold):
                continue
                
            unique_labels, counts = np.unique(valid_labels, return_counts=True)
            dominant_idx = np.argmax(counts)
            purity_ratio = counts[dominant_idx] / window_size
            
            if purity_ratio >= purity_threshold:
                dominant_label = unique_labels[dominant_idx]
                w_stages = drive_stages[w_start:w_end]
                valid_stages = w_stages[w_stages != "Unknown"]
                u_stages, s_counts = np.unique(valid_stages, return_counts=True)
                dominant_stage = u_stages[np.argmax(s_counts)]
                
                window_list.append({
                    'window_id': f"{driver_id}_{w_start}",
                    'drive_id': driver_id,
                    'start_idx_15_5Hz': w_start,
                    'end_idx_15_5Hz': w_end,
                    'label': dominant_label,
                    'stage': dominant_stage,
                    # 'purity': round(purity_ratio, 3)
                })
                
    df_windows = pd.DataFrame(window_list)
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_windows.to_csv(output_csv_path, index=False)
    print(f"Get window_map.csv: {len(df_windows)} windows.")
    
def run():
    marker_info_df = generate_marker_info(input_dir='./data/parsed_h5', output_file='./data/label/marker_info.csv')
    generate_window_map('./data/label/marker_info.csv', './data/preprocessed_h5', './data/label/window_map.csv')
    
if __name__ == "__main__":
    marker_info_df = generate_marker_info(input_dir='./data/parsed_h5', output_file='./data/label/marker_info.csv')
    generate_window_map('./data/label/marker_info.csv', './data/preprocessed_h5', './data/label/window_map.csv')