import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

stages = ['Start', 'Rest1', 'City1', 'Hwy1', 'Return', 'Hwy2', 'City2', 'Rest2']

def generate_marker_info(input_dir="./data/drive_csv", output_file="./data/marker_info.csv"):
    base_path = Path(input_dir).resolve()
    csv_files = sorted(list(base_path.glob("*.csv")))
    
    marker_results = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            marker_signal = df['marker-mV'].values

            peaks, _ = find_peaks(
                marker_signal, 
                distance=4000,
                prominence=1
            )

            drive_info = {'Driver': csv_file.stem}
            
            for i in range(len(stages)):
                if i < len(peaks):
                    drive_info[stages[i]] = peaks[i]
                else:
                    drive_info[stages[i]] = np.nan

            marker_results.append(drive_info)

        except Exception as e:
            print(f"Lỗi tại file {csv_file.name}: {e}")

    marker_df = pd.DataFrame(marker_results)
    marker_df = marker_df[['Driver'] + stages]
    marker_df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully generated {output_file}")
    return marker_df

marker_info_df = generate_marker_info()

# --- LABEL SAMPLE ---
def label_samples(df, marker_row):    
    label_map = {
        'Rest1': 'relax', 'Rest2': 'relax',
        'Hwy1': 'medium', 'Hwy2': 'medium', 'Return': 'medium',
        'City1': 'high', 'City2': 'high'
    }
    
    df['Stress'] = np.nan
    for i in range(len(stages) - 1):
        s, e = marker_row[stages[i]], marker_row[stages[i+1]]
        
        if pd.isna(s) or pd.isna(e): 
            continue
            
        current_label = label_map.get(stages[i+1])
        df.loc[int(s)+1 : int(e), 'Stress'] = current_label
        
    return df.dropna(subset=['Stress']).copy()

def run_sample_labeling(input_dir, marker_csv, output_dir):
    markers = pd.read_csv(marker_csv)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for _, row in markers.iterrows():
        f_path = Path(input_dir) / f"{row['Driver']}.csv"
        
        if f_path.exists():
            df = pd.read_csv(f_path)
            df_labeled = label_samples(df, row)
            
            save_path = out_path / f"{row['Driver']}_labeled.csv"
            df_labeled.to_csv(save_path, index=False)
            print(f"Success: {row['Driver']}")