import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, firwin
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from pathlib import Path

def butter_bandpass(signal, fs, low=5, high=15, order=4):
    """
    Adapted from SciPy Cookbook:
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    
    nyquist = 0.5 * fs
    low = low / nyquist 
    high = high / nyquist 
    b_butter, a_butter = butter(order, [low, high], btype='band')
    
    return filtfilt(b_butter, a_butter, signal)

def butter_lowpass(signal, fs, cutoff=4.0, order=4):
    """
    Adapted from SciPy Cookbook:
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    """
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b_butter, a_butter = butter(order, normal_cutoff, btype='low', analog=False)
    
    return filtfilt(b_butter, a_butter, signal)

def fir_bandpass(signal, fs, low=1.5, high=150, numtaps=101):
    nyquist = 0.5 * fs
    low = low / nyquist 
    high = high / nyquist
    taps = firwin(numtaps, [low, high], pass_zero=False)
    
    return filtfilt(taps, [1.0], signal)

def min_max_normalize(signal):
    min_signal = np.min(signal)
    max_signal = np.max(signal)
    
    return (signal - min_signal) / (max_signal - min_signal)

def max_normalize(signal):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    
    return signal / max_val

def filter_ecg(ecg_signal, fs):
    """
    1. Butterworth bandpass filter
       Adapted from:
       https://doi.org/10.1145/3186585

    2. FIR bandpass filter & Min-max normalization
       Adapted from:
       https://doi.org/10.3390/diagnostics13111897
    """
    
    ecg_signal = butter_bandpass(ecg_signal, fs, low=5, high=15)
    ecg_signal = fir_bandpass(ecg_signal, fs, low=1.5, high=150)
    ecg_signal = min_max_normalize(ecg_signal)

    return ecg_signal

def filter_gsr(gsr_signal, fs):
    """
    1. Butterworth lowpass & Gaussian filter
       Adapted from:
       https://doi.org/10.1016/j.bspc.2021.103203
    """
    
    gsr_signal = butter_lowpass(gsr_signal, fs, cutoff=4.0)
    gsr_signal = gaussian_filter1d(gsr_signal, sigma=2)
    gsr_signal = max_normalize(gsr_signal)
    
    return gsr_signal

def filter_resp(resp_signal, fs):
    """
    1. Butterworth bandpass filter
       Adapted from:
       https://doi.org/10.3390/diagnostics13111897       
    """
    resp_signal = butter_bandpass(resp_signal, fs, low=0.05, high=0.70)
    
    return resp_signal

def preprocess_single_drive(input_path, output_path):
    with h5py.File(input_path, 'r') as h5_in, h5py.File(output_path, 'w') as h5_out:
        signals_group = h5_out.create_group("signals")
        meta_group = h5_out.create_group("meta")
        fs_group = meta_group.create_group("fs")
        
        if "record_name" in h5_in["meta"]:
            meta_group.create_dataset("record_name", data=h5_in["meta"]["record_name"][()])
        if "original_signal_names" in h5_in["meta"]:
            meta_group.create_dataset("original_signal_names", data=h5_in["meta"]["original_signal_names"][()])

        for sig_name in h5_in['signals'].keys():
            raw_data = h5_in['signals'][sig_name][:]
            fs = h5_in['meta']['fs'][sig_name][0]
            
            preprocess_signal = raw_data.copy()
            name_lower = sig_name.lower()

            if 'ecg' in name_lower:
                preprocess_signal = filter_ecg(raw_data, fs)
            elif 'gsr' in name_lower:
                preprocess_signal = filter_gsr(raw_data, fs)
            elif 'resp' in name_lower:
                preprocess_signal = filter_resp(raw_data, fs)

            signals_group.create_dataset(
                sig_name, 
                data=preprocess_signal.astype(np.float32)
            )
            
            fs_group.create_dataset(
                sig_name, 
                data=np.array([fs], dtype=np.float32)
            )

    return True

def preprocess_all_drives(input_dir="parsed_h5", output_dir="preprocessed_h5"):
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    h5_files = sorted(in_path.glob("*.h5"))
    
    print(f"\n{'='*60}")
    print("B. Preprocess data")

    success = 0
    
    for h5_file in tqdm(h5_files, desc="Processing Drives"):
        output_file_path = out_path / h5_file.name
        
        try:
            preprocess_single_drive(h5_file, output_file_path)
            success += 1
        except Exception as e:
            print(f"\n[Error] {h5_file}: {e}")

    print(f"Success: {success}/{len(h5_files)}")
    
def run():
    preprocess_all_drives(input_dir="./data/drive_h5", output_dir="./data/preprocessed_h5")

if __name__ == "__main__":
    preprocess_all_drives(input_dir="./data/drive_h5", output_dir="./data/preprocessed_h5")