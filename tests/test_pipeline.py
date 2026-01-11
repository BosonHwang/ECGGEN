import os
import numpy as np
import torch
import tempfile

from ecggen.src.data.pipeline import make_dataloader
from ecggen.src.utils.config import Config


def test_dataloader_shapes():
    # Create a temporary directory with a dummy .npy file
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train")
        os.makedirs(train_dir)
        
        # Create a dummy .npy file
        dummy_data = {
            "ecg": np.random.randn(12, 512).astype(np.float32),
            "label": 1,
            "id": "dummy_1"
        }
        np.save(os.path.join(train_dir, "dummy.npy"), dummy_data, allow_pickle=True)
        
        cfg = Config(raw={
            "data": {"meta_root": tmpdir, "num_leads": 12, "time_len": 512},
            "train": {"batch_size": 2, "num_workers": 0}
        })
        dl = make_dataloader(cfg=cfg.raw, split="train")
        batch = next(iter(dl))
        ecg = batch["ecg"]
        assert ecg.shape[1] == 12 and ecg.shape[2] == 512


