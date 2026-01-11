from dataclasses import dataclass
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


from .mimic import MimicLoader

class ECGDataset(Dataset):
    # ... existing ECGDataset implementation ...
    def __init__(self, root: str, split: str, num_leads: int = 12, time_len: Optional[int] = None):
        super().__init__()
        self.root = root
        self.split = split
        self.num_leads = num_leads
        self.time_len = time_len

        split_dir = os.path.join(root, split)
        self.files: List[str] = []
        if os.path.isdir(split_dir):
            for fn in sorted(os.listdir(split_dir)):
                if fn.endswith(".npy"):
                    self.files.append(os.path.join(split_dir, fn))

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy files found in {split_dir}. Dataset cannot be empty.")
        
        self.length = len(self.files)

    def __len__(self) -> int:
        return self.length

    def _load_file(self, path: str) -> Dict[str, Any]:
        arr = np.load(path, allow_pickle=True).item()
        ecg = arr["ecg"]  # expected shape [L, T]
        assert ecg.ndim == 2, "ECG must be [L, T]"
        assert ecg.shape[0] == self.num_leads, "Lead count mismatch"
        if self.time_len is not None and ecg.shape[1] != self.time_len:
            # Center crop/pad to match time_len for batching simplicity
            T_src = ecg.shape[1]
            T_tgt = self.time_len
            if T_src > T_tgt:
                start = (T_src - T_tgt) // 2
                ecg = ecg[:, start:start + T_tgt]
            else:
                pad_left = (T_tgt - T_src) // 2
                pad_right = T_tgt - T_src - pad_left
                ecg = np.pad(ecg, ((0, 0), (pad_left, pad_right)))
        label = arr.get("label", None)
        id_str = arr.get("id", os.path.basename(path))
        return {"ecg": ecg.astype(np.float32), "label": label, "id": id_str}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        return self._load_file(path)


class MimicRawDataset(Dataset):
    """Dataset for raw MIMIC-IV WFDB files using a manifest."""
    
    def __init__(self, manifest_path: str, fs_out: int = 100, time_len: Optional[int] = None, num_leads: int = 12):
        super().__init__()
        self.loader = MimicLoader(manifest_path, fs_out=fs_out)
        self.time_len = time_len
        self.num_leads = num_leads

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.loader.get_record(idx)
        ecg = item["ecg"]
        
        # Center crop/pad to match time_len
        if self.time_len is not None:
            T_src = ecg.shape[1]
            T_tgt = self.time_len
            if T_src > T_tgt:
                start = (T_src - T_tgt) // 2
                ecg = ecg[:, start:start + T_tgt]
            elif T_src < T_tgt:
                pad_left = (T_tgt - T_src) // 2
                pad_right = T_tgt - T_src - pad_left
                ecg = np.pad(ecg, ((0, 0), (pad_left, pad_right)))
            
        return {
            "ecg": ecg.astype(np.float32),
            "label": None, # MIMIC raw text label is in label_text
            "id": item["id"]
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # ... existing _collate implementation ...
    ecgs = [torch.from_numpy(x["ecg"]) if isinstance(x["ecg"], np.ndarray) else x["ecg"] for x in batch]
    labels = [x["label"] for x in batch]
    ids = [x["id"] for x in batch]

    T_list = [e.shape[1] for e in ecgs]
    assert len(set(T_list)) == 1, "All items must share the same T"
    L_list = [e.shape[0] for e in ecgs]
    assert len(set(L_list)) == 1, "All items must share the same lead count"

    ecg_tensor = torch.stack(ecgs, dim=0)  # [B, L, T]
    label_tensor = None
    if any(lbl is not None for lbl in labels):
        label_vals = [(-1 if lbl is None else int(lbl)) for lbl in labels]
        label_tensor = torch.tensor(label_vals, dtype=torch.long)
    return {"ecg": ecg_tensor, "label": label_tensor, "id": ids}


def make_dataloader(cfg: Dict[str, Any], split: str) -> DataLoader:
    """Construct DataLoader for the given split.

    Config keys (data section):
        - dataset_type: 'npy' (default) or 'mimic_raw'
        - meta_root: base directory for .npy or path to manifest for mimic_raw
        - fs: target sampling rate (default 100)
        - num_leads: expected lead count (default 12)
        - time_len: target time length for [L, T]
    """
    data_cfg = cfg.get("data", {})
    dtype = data_cfg.get("dataset_type", "npy")
    root = data_cfg.get("meta_root", "./data")
    fs = int(data_cfg.get("fs", 100))
    num_leads = int(data_cfg.get("num_leads", 12))
    time_len = data_cfg.get("time_len", None)
    if time_len is not None:
        time_len = int(time_len)
    batch_size = int(cfg.get("train", {}).get("batch_size", 4))
    num_workers = int(cfg.get("train", {}).get("num_workers", 0))

    if dtype == "mimic_raw":
        # For mimic_raw, root is expected to be the path to the manifest JSONL
        ds = MimicRawDataset(manifest_path=root, fs_out=fs, time_len=time_len, num_leads=num_leads)
    else:
        ds = ECGDataset(root=root, split=split, num_leads=num_leads, time_len=time_len)
        
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, collate_fn=_collate)


