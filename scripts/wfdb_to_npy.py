#!/usr/bin/env python3
"""
Convert WFDB records to standardized .npy samples for ecggen.

Outputs:
    dst/{train,val}/<record>_<idx>.npy with a dict:
        {
            "ecg": np.ndarray [L, T] float32 (12 leads, target fs and time_len),
            "label": None,
            "id": str
        }

Dependencies:
    - wfdb (optional, required to read raw records)
    - scipy (optional, for bandpass/resample; if missing, simple fallbacks are used)
"""
import argparse
import os
import random
from typing import List, Tuple

import numpy as np


def _maybe_import_signal_tools():
    try:
        import wfdb  # type: ignore
    except Exception as e:
        raise RuntimeError("wfdb is required for this script. pip install wfdb") from e
    try:
        from scipy.signal import butter, filtfilt, resample_poly  # type: ignore
    except Exception:
        butter = filtfilt = resample_poly = None
    return wfdb, butter, filtfilt, resample_poly


STANDARD_12_LEADS = [
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
]


def _bandpass(x: np.ndarray, fs: int, low: float, high: float, butter, filtfilt) -> np.ndarray:
    """Apply bandpass; fall back to simple highpass+lowpass via moving average if scipy not available."""
    if butter is not None and filtfilt is not None:
        b, a = butter(3, [low / (fs / 2.0), high / (fs / 2.0)], btype="band")
        return filtfilt(b, a, x, axis=-1)
    # Fallback: detrend + moving average low-pass
    y = x - x.mean(axis=-1, keepdims=True)
    win = max(1, int(0.1 * fs))
    kernel = np.ones(win, dtype=np.float64) / float(win)
    # Convolve per lead
    lp = np.apply_along_axis(lambda s: np.convolve(s, kernel, mode="same"), -1, y)
    return lp.astype(np.float64)


def _resample(x: np.ndarray, fs_in: int, fs_out: int, resample_poly) -> np.ndarray:
    if fs_in == fs_out:
        return x
    if resample_poly is not None:
        # Rational approximation
        from math import gcd
        g = gcd(fs_in, fs_out)
        up = fs_out // g
        down = fs_in // g
        return resample_poly(x, up, down, axis=-1)
    # Fallback: linear interpolation
    T_in = x.shape[-1]
    T_out = int(round(T_in * fs_out / fs_in))
    t_in = np.linspace(0.0, 1.0, T_in, endpoint=False)
    t_out = np.linspace(0.0, 1.0, T_out, endpoint=False)
    out = np.zeros((x.shape[0], T_out), dtype=np.float64)
    for i in range(x.shape[0]):
        out[i] = np.interp(t_out, t_in, x[i])
    return out


def _reorder_leads(data: np.ndarray, sig_names: List[str]) -> np.ndarray:
    name_to_idx = {name: i for i, name in enumerate(sig_names)}
    selected = []
    for name in STANDARD_12_LEADS:
        if name in name_to_idx:
            selected.append(data[name_to_idx[name]])
        else:
            # If missing, pad zeros for that lead to keep shape consistent
            selected.append(np.zeros_like(data[0]))
    return np.stack(selected, axis=0)


def find_records(src: str) -> List[str]:
    recs = []
    for fn in os.listdir(src):
        if fn.endswith(".hea"):
            recs.append(os.path.splitext(fn)[0])
    recs.sort()
    return recs


def save_windows(ecg: np.ndarray, rid: str, dst_split: str, time_len: int, stride: int) -> int:
    os.makedirs(dst_split, exist_ok=True)
    L, T = ecg.shape
    count = 0
    for start in range(0, max(1, T - time_len + 1), stride):
        end = start + time_len
        if end > T:
            # pad last window
            pad = end - T
            win = np.pad(ecg, ((0, 0), (0, pad)), mode="constant")
        else:
            win = ecg[:, start:end]
        out = {"ecg": win.astype(np.float32), "label": None, "id": f"{rid}_{start}"}
        np.save(os.path.join(dst_split, f"{rid}_{start}.npy"), out, allow_pickle=True)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="WFDB -> npy converter for ecggen")
    parser.add_argument("--src", type=str, required=True, help="Directory containing WFDB records (.hea/.dat pairs).")
    parser.add_argument("--dst", type=str, required=True, help="Output root directory.")
    parser.add_argument("--fs_out", type=int, default=100, help="Target sampling rate.")
    parser.add_argument("--time_len", type=int, default=512, help="[L,T] target T length per sample.")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    wfdb, butter, filtfilt, resample_poly = _maybe_import_signal_tools()
    random.seed(args.seed)
    np.random.seed(args.seed)

    records = find_records(args.src)
    if not records:
        raise RuntimeError("No WFDB .hea records found in src")

    os.makedirs(args.dst, exist_ok=True)
    dst_train = os.path.join(args.dst, "train")
    dst_val = os.path.join(args.dst, "val")
    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_val, exist_ok=True)

    for rec in records:
        rec_path = os.path.join(args.src, rec)
        sig, meta = wfdb.rdsamp(rec_path)
        fs_in = int(meta["fs"])
        sig_names = meta["sig_name"]
        data = sig.T  # [C, T]
        data = _reorder_leads(data, sig_names)  # [12, T] (missing leads zeroed)
        data = _resample(data, fs_in, args.fs_out, resample_poly)  # [12, T_out]
        data = _bandpass(data, args.fs_out, low=0.67, high=40.0, butter=butter, filtfilt=filtfilt)
        # Normalize per lead (zero mean, unit std, with epsilon)
        mu = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True) + 1e-6
        data = (data - mu) / std

        # choose split
        dst_split = dst_train if random.random() < args.train_ratio else dst_val
        # save overlapped windows (50% overlap)
        stride = args.time_len // 2
        save_windows(data, rid=rec, dst_split=dst_split, time_len=args.time_len, stride=stride)


if __name__ == "__main__":
    main()


