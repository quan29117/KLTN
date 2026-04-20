import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, filtfilt
from pathlib import Path

def repair_signal_gaps(df, column_name='HR-bpm'):
    """
    Automatically detects segments of zeros in a specific column 
    and repairs them using linear interpolation.
    """
    if column_name not in df.columns:
        print(f"Column {column_name} not found.")
        return df

    is_zero = (df[column_name] <= 0)
    
    zero_count = is_zero.sum()
    if zero_count == 0:
        print(f"No gaps detected in {column_name}.")
        return df

    df['is_valid'] = True
    df.loc[is_zero, 'is_valid'] = False
    df_repaired = df.copy()
    df_repaired.loc[is_zero, column_name] = np.nan
    
    df_repaired[column_name] = df_repaired[column_name].interpolate(method='linear', limit_direction='both')

    df_repaired[column_name] = df_repaired[column_name].ffill().bfill()

    print(f"Automatically detected and repaired {zero_count} samples in '{column_name}'.")
    
    return df_repaired

def fft_filter(signal, fs=15.5, cutoff=0.5):
    N = len(signal)
    freqs = fftfreq(N, d=1.0/fs)
    X_k = fft(signal)
    X_k[np.abs(freqs) <= cutoff] = 0
    filtered = np.real(ifft(X_k))
    return filtered

def resp_filter(signal, fs):
    nyq = fs / 2
    Wn = 0.05 / nyq
    b, a = butter(4, Wn, btype='highpass')

    return filtfilt(b, a, signal)

def clean_signals(df, fs=15.5):
    """
    Applies interpolation and filters to a single dataframe.
    """
    df = repair_signal_gaps(df, column_name='HR-bpm')
    
    if 'ECG-mV' in df.columns:
        df['ECG-mV'] = fft_filter(df['ECG-mV'].values, fs=fs, cutoff=0.5)

    if 'hand_GSR-mV' in df.columns:
        df['hand_GSR-mV'] = fft_filter(df['hand_GSR-mV'].values, fs=fs, cutoff=0.0) 
        
    if 'foot_GSR-mV' in df.columns:
        df['foot_GSR-mV'] = fft_filter(df['foot_GSR-mV'].values, fs=fs, cutoff=0.0) 

    if 'RESP-mV' in df.columns:
        df['RESP-mV'] = resp_filter(df['RESP-mV'].values, fs=fs)

    return df

def batch_clean(input_dir, output_dir, fs=15.5):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(in_path.glob("*.csv"))
    
    for file in csv_files:
        drive_name = file.stem
        
        try:
            df = pd.read_csv(file)
            
            df_clean = clean_signals(df, fs=fs)
            
            save_path = out_path / f"{drive_name}_filtered.csv"
            df_clean.to_csv(save_path, index=False)
            print(f"Saved: {save_path.name} \n")
            
        except Exception as e:
            print(f"Failed to process {drive_name}: {e}")
            
    print("Batch filter cleaning completed.")
    
batch_clean(input_dir="./data/drive_csv", output_dir="./data/filtered_data")