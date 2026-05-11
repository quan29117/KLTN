import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def create_binary_stress_map(window_map_csv, output_csv=None):
    df = pd.read_csv(window_map_csv)
    
    df = df[df["label"].isin([1, 2])].copy()
    df["binary_label"] = df["label"].map({
        1: 0,
        2: 1
    })

    output_df = df[
        [
            "window_id",
            "drive_id",
            "start_idx_15_5Hz",
            "end_idx_15_5Hz",
            "binary_label",
            "stage"
        ]
    ].reset_index(drop=True)
    
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")
        
    print(f"Total windows: {len(output_df)}")
    print("\nLabel distribution:")
    print(output_df["binary_label"].value_counts())

    return output_df

def prepare_multibranch_dataset(window_map, h5_dir, output_file):
    print(f"\n{'='*60}")
    print("F. Second layer - High / Medium Stress")
    
    df = create_binary_stress_map(window_map)
    
    sensor_config = {
        'ECG': {'target_len': 29760, 'base_fs': 496.0},
        'hand_GSR': {'target_len': 1860, 'base_fs': 31.0},
        'foot_GSR': {'target_len': 1860, 'base_fs': 31.0},
        'RESP': {'target_len': 1860, 'base_fs': 31.0},
        'HR': {'target_len': 930, 'base_fs': 15.5},
        'EMG': {'target_len': 930, 'base_fs': 15.5}
    }
    
    branch_data = {sensor: [] for sensor in sensor_config.keys()}
    labels = []

    for i, row in df.iterrows():
        drive_id = row['drive_id']
        start_15_5 = int(row['start_idx_15_5Hz'])
        labels.append(row['binary_label'])
        
        h5_f = Path(h5_dir) / f"{drive_id}.h5"

        with h5py.File(h5_f, 'r') as f:
            for sensor, config in sensor_config.items():
                fs_actual = f[f'meta/fs/{sensor}'][0]
                
                scale_factor = fs_actual / 15.5
                actual_start = int(start_15_5 * scale_factor)
                expected_samples = int(60 * fs_actual)
                
                raw_sig = f[f'signals/{sensor}'][actual_start : actual_start + expected_samples].flatten()
                
                if len(raw_sig) > 0:
                    mean_val = np.mean(raw_sig)
                    std_val = np.std(raw_sig) + 1e-8
                    norm_sig = (raw_sig - mean_val) / std_val
                else:
                    norm_sig = np.zeros((expected_samples,))

                padded_sig = np.zeros(config['target_len'], dtype=np.float32)
                
                actual_len = min(len(norm_sig), config['target_len'])
                padded_sig[:actual_len] = norm_sig[:actual_len]
                
                branch_data[sensor].append(padded_sig.reshape(-1, 1))

    final_output = {sensor: np.array(data) for sensor, data in branch_data.items()}
    final_output['labels'] = np.array(labels, dtype=np.int8)
        
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **final_output)
    
    print(f"\nSuccess: {output_file}")

def run():
    prepare_multibranch_dataset('./data/label/window_map.csv', './data/preprocessed_h5', './data/dl_data/dl_data.npz')

prepare_multibranch_dataset('./data/label/window_map.csv', './data/preprocessed_h5', './data/dl_data/dl_data.npz')