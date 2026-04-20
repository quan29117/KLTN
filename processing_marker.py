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
            print(f"Error file: {csv_file.name}: {e}")

    marker_df = pd.DataFrame(marker_results)
    marker_df = marker_df[['Driver'] + stages]
    marker_df = clean_invalid_drives(marker_df, input_dir)
    marker_df.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully generated {output_file}")
    return marker_df

def clean_invalid_drives(marker_df, input_dir):
    """
    Identifies drives with insufficient markers (< 5) and deletes 
    their corresponding CSV files to ensure data consistency.
    """
    stage_cols = [col for col in marker_df.columns if col != 'Driver']
    
    invalid_rows = marker_df[marker_df[stage_cols].count(axis=1) < 5]
    invalid_drivers = invalid_rows['Driver'].tolist()
    
    if not invalid_drivers:
        print("All drives meet the minimum marker requirement.")
        return marker_df

    print(f"Removing {len(invalid_drivers)} invalid drives: {invalid_drivers}")

    for driver in invalid_drivers:
        file_to_delete = Path(input_dir) / f"{driver}.csv"
        try:
            if file_to_delete.exists():
                file_to_delete.unlink()
                print(f"Deleted: {file_to_delete.name}")
        except Exception as e:
            print(f"Error deleting {driver}: {e}")

    return marker_df[marker_df[stage_cols].count(axis=1) >= 5].copy()

marker_info_df = generate_marker_info()