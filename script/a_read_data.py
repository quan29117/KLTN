import numpy as np
import re
import h5py
from pathlib import Path
from tqdm import tqdm

REQUIRED_SIGNALS = ['ECG', 'EMG', 'foot GSR', 'hand GSR', 'HR', 'RESP', 'marker']

def normalize_column_names(sig_names):
    normalized = []
    for sig in sig_names:
        sig_lower = sig.lower()
        matched = sig
        for req in REQUIRED_SIGNALS:
            if req.lower() in sig_lower:
                matched = req
                break
        normalized.append(matched.replace(" ", "_").replace(",", "_").replace("/", "_"))
        
    return normalized

def validate_drive(available_signals):
    available_lower = [s.lower() for s in available_signals]
    missing = [req for req in REQUIRED_SIGNALS if not any(req.lower() in s for s in available_lower)]
    
    return len(missing) == 0

def parse_header(hea_path):
    with open(hea_path, "r") as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    first = lines[0].split()
    record_name, base_fs, n_frames = first[0], float(first[2]), int(first[3])
    signals = []
    
    for line in lines[1:]:
        parts = line.split()
        
        gain_str = parts[2]
        format_str = parts[1]
        multi = 1
        
        if "x" in format_str:
            match = re.search(r"x(\d+)", format_str)
            if match:
                multi = int(match.group(1))
                
        signal_name = " ".join(parts[8:]) if len(parts) > 8 else parts[-1]
        
        signals.append({
            "gain": gain_str, 
            "name": signal_name,
            "spf": multi, 
            "fs": base_fs * multi
        })
        
    return {"record": record_name, "base_fs": base_fs, "n_frames": n_frames, "signals": signals}

def read_dat(dat_path, header):
    total_spf = sum(s["spf"] for s in header["signals"])
    raw = np.fromfile(dat_path, dtype="<i2").reshape(header["n_frames"], total_spf)
    output, idx = {}, 0
    
    for s in header["signals"]:
        sig = raw[:, idx:idx+s["spf"]].reshape(-1).astype(np.float32)
        try:
            gain_val = float(s["gain"])
            if gain_val != 0.0:
                sig /= gain_val
        except ValueError:
            pass
        
        output[s["name"]] = {"fs": s["fs"], "signal": sig}
        idx += s["spf"]
        
    return output

def save_to_hdf5(output_path, record_name, normalized_map, parsed_signals):
    with h5py.File(output_path, "w") as f:
        g_sig, g_fs = f.create_group("signals"), f.create_group("meta/fs")
        
        for norm, orig in normalized_map.items():
            g_sig.create_dataset(norm, data=parsed_signals[orig]["signal"], compression="gzip", chunks=True)
            g_fs.create_dataset(norm, data=np.array([parsed_signals[orig]["fs"]], dtype=np.float32))
            
        f.create_dataset("meta/record_name", data=record_name)

def process_single_drive(hea_path, dat_path, output_path):
    h = parse_header(hea_path)
    sig_names = [s["name"] for s in h["signals"]]
    
    if not validate_drive(sig_names):
        return False
    
    normalized_names = normalize_column_names(sig_names)
    signal_mapping = dict(zip(normalized_names, sig_names))
    parsed_data = read_dat(dat_path, h)

    save_to_hdf5(output_path, h["record"], signal_mapping, parsed_data)
    
    return True

def process_all_drives(raw_dir, output_dir):
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    hea_files = sorted(raw_path.glob("*.hea"))
    success = 0
    
    print(f"\n{'='*60}")
    print("A. Read data")
    for hea_f in tqdm(hea_files):
        record = hea_f.stem
        if process_single_drive(raw_path / f"{record}.hea", raw_path / f"{record}.dat", output_path / f"{record}.h5"):
            success += 1
            
    print(f"\nSummary: Success {success}, Failed {len(hea_files)-success}")
    
def run():
    process_all_drives(raw_dir="./data/raw", output_dir="./data/drive_h5")

if __name__ == "__main__":
    process_all_drives(raw_dir="./data/raw", output_dir= "./data/drive_h5")