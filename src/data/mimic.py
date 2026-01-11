import json
import os
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

try:
    import wfdb
except ImportError:
    wfdb = None

# Standard 12 leads order used in the project
STANDARD_12_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
]

def load_mimic_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load the MIMIC manifest (JSONL format)."""
    manifest = []
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                manifest.append(json.loads(line))
    return manifest

def load_wfdb_record(record_path: str, target_fs: Optional[int] = None) -> Tuple[np.ndarray, int, List[str]]:
    """Load a single WFDB record and return (data, fs, sig_names).
    
    Data shape: [num_leads, num_samples]
    """
    if wfdb is None:
        raise ImportError("wfdb package is required to load raw MIMIC data. Run: pip install wfdb")
    
    # record_path should not include extension
    record = wfdb.rdrecord(record_path)
    data = record.p_signal.T # [C, T]
    fs = record.fs
    sig_names = record.sig_name
    
    return data, fs, sig_names

def preprocess_mimic_record(
    data: np.ndarray, 
    fs_in: int, 
    sig_names: List[str],
    fs_out: int = 100,
    low: float = 0.67,
    high: float = 40.0
) -> np.ndarray:
    """Preprocess raw ECG data: reorder leads, resample, and bandpass filter."""
    # 1. Reorder to standard 12 leads
    name_to_idx = {name.upper(): i for i, name in enumerate(sig_names)}
    selected = []
    for name in STANDARD_12_LEADS:
        # Try both upper and exact match
        idx = name_to_idx.get(name.upper())
        if idx is not None:
            selected.append(data[idx])
        else:
            # Pad with zeros if lead is missing
            selected.append(np.zeros_like(data[0]))
    data = np.stack(selected, axis=0) # [12, T]

    # 2. Resample
    if fs_in != fs_out:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(fs_in, fs_out)
        data = resample_poly(data, fs_out // g, fs_in // g, axis=-1)

    # 3. Bandpass filter
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * fs_out
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    data = filtfilt(b, a, data, axis=-1)

    # 4. Normalize (zero mean, unit std)
    mu = np.nanmean(data, axis=-1, keepdims=True)
    std = np.nanstd(data, axis=-1, keepdims=True) + 1e-6
    data = (data - mu) / std
    
    # 5. Handle NaNs: fill with 0 after normalization
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data.astype(np.float32)

class MimicLoader:
    """Convenience class to iterate over MIMIC records from a manifest."""
    
    def __init__(self, manifest_path: str, fs_out: int = 100):
        self.manifest = load_mimic_manifest(manifest_path)
        self.fs_out = fs_out

    def __len__(self) -> int:
        return len(self.manifest)

    def get_record(self, idx: int) -> Dict[str, Any]:
        item = self.manifest[idx]
        path = item['ecg_path']
        
        raw_data, fs_in, sig_names = load_wfdb_record(path)
        processed_data = preprocess_mimic_record(raw_data, fs_in, sig_names, fs_out=self.fs_out)
        
        # Extract label from assistant message if present
        label_text = ""
        for msg in item.get('messages', []):
            if msg.get('role') == 'assistant':
                label_text = msg.get('content', "")
                break
        
        return {
            "ecg": processed_data, # [12, T]
            "id": item['id'],
            "label_text": label_text,
            "fs": self.fs_out
        }

