import pandas as pd
import numpy as np
from pathlib import Path

stages = ['Start', 'Rest1', 'City1', 'Hwy1', 'Return', 'Hwy2', 'City2', 'Rest2']

# --- LABEL SAMPLE ---
def label_samples(df, marker_row):    
    label_map = {
        'Rest1': 'relax', 'Rest2': 'relax',
        'Hwy1': 'medium', 'Hwy2': 'medium', 'Return': 'medium',
        'City1': 'high', 'City2': 'high'
    }
    
    df['Stress'] = None
    for i in range(len(stages) - 1):
        start, end = marker_row[stages[i]], marker_row[stages[i+1]]
        
        if pd.isna(start) or pd.isna(end): 
            continue
            
        current_label = label_map.get(stages[i+1])
        df.loc[int(start)+1 : int(end), 'Stress'] = current_label
        
    return df.dropna(subset=['Stress']).copy()

def batch_sample_labeling(input_dir, marker_csv, output_dir):
    markers = pd.read_csv(marker_csv)
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting sample-level labeling for {len(markers)} drives...")
    
    for _, row in markers.iterrows():
        driver_id = row['Driver']
        file_name = f"{driver_id}_filtered.csv"
        file_path = in_path / file_name
        
        if not file_path.exists():
            print(f"    [Skip] File not found: {file_name}")
            continue
            
        try:
            df = pd.read_csv(file_path)
            
            # Apply labeling logic
            df_labeled = label_samples(df, row)
            
            # Save labeled data
            save_path = out_path / f"{driver_id}_labeled.csv"
            df_labeled.to_csv(save_path, index=False)
            
            # Log distribution for validation
            label_counts = df_labeled['Stress'].value_counts().to_dict()
            print(f"    [Done] {driver_id} | Samples: {len(df_labeled)} | Labels: {label_counts}")
            
        except Exception as e:
            print(f"    [Error] Failed to label {driver_id}: {e}")

    print("Sample-level labeling completed.")
            
batch_sample_labeling(
    input_dir="./data/filtered_data",
    marker_csv="./data/marker_info.csv",
    output_dir="./data/labeled_samples"
)