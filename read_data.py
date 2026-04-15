import wfdb
import pandas as pd
from pathlib import Path

REQUIRED_SIGNALS = ['ECG', 'EMG', 'foot GSR', 'hand GSR', 'HR', 'RESP', 'marker']

def normalize_column_names(sig_names, units):
    normalized = []
    for sig, unit in zip(sig_names, units):
        sig_lower = sig.lower()
        matched_name = sig
        
        for req in REQUIRED_SIGNALS:
            if req.lower() in sig_lower:
                matched_name = req
                break

        col_name = f"{matched_name}-{unit}".replace(" ", "_").replace(",", "_")
        normalized.append(col_name)
    return normalized

def validate_drive(available_signals):
    available_lower = [s.lower() for s in available_signals]
    missing = []

    for req in REQUIRED_SIGNALS:
        found = any(req.lower() in s for s in available_lower)
        if not found:
            missing.append(req)

    return len(missing) == 0, missing

def read_data(input_path="./DRIVEDB"):
    base_path = Path(input_path).resolve()

    output_dir = Path('./data/drive_csv')
    output_dir.mkdir(parents=True, exist_ok=True)

    dat_files = sorted(list(base_path.glob("*.dat")))
    
    for dat_file in dat_files:
        try:
            record_path = str(dat_file.with_suffix(''))
            signals, fields = wfdb.rdsamp(record_path)
            sig_names = fields['sig_name']

            is_valid, missing_sigs = validate_drive(sig_names)
            if not is_valid:
                print(f" Skipped {dat_file.name}: Missing {missing_sigs}")
                continue

            units = fields['units']
            col_names = normalize_column_names(sig_names, units)

            df = pd.DataFrame(signals, columns=col_names)

            save_path = output_dir / f"{dat_file.stem}.csv"
            df.to_csv(save_path, index=False)

            print(f"{dat_file.name} -> {save_path.name}")

        except Exception as e:
            print(f"Error processing {dat_file.name}: {e}")

read_data(input_path="./data/raw")