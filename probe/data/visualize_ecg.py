import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

# Set project root to allow importing from ecggen
# Current file is at ecggen/probe/data/visualize_ecg.py
# Root is ecggen/
# So we need to go up two levels to reach the parent of ecggen, 
# but wait, the project structure is:
# /home/gbsguest/Research/boson/BIO/ecggen/
#   src/
#   configs/
#   probe/
#     data/
#       visualize_ecg.py
#
# If we want to import `ecggen.src.data`, we need /home/gbsguest/Research/boson/BIO to be in sys.path.
# OR we can add /home/gbsguest/Research/boson/BIO/ecggen to sys.path and import from src.

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is /home/gbsguest/Research/boson/BIO/ecggen
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
# PARENT_DIR is /home/gbsguest/Research/boson/BIO
PARENT_DIR = os.path.dirname(PROJECT_ROOT)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from ecggen.src.utils.config import load_config
from ecggen.src.data.pipeline import make_dataloader
from ecggen.src.data.mimic import STANDARD_12_LEADS
from ecggen.src.data.tokenizer import ECGTokenizer

def visualize_beat_diffs(batch, fs=100, beat_len=128, num_samples=3):
    """Visualize the difference between consecutive beats."""
    ecgs = batch["ecg"] # [B, L, T]
    ids = batch["id"]
    B, L, T = ecgs.shape
    num_samples = min(num_samples, B)
    
    tokenizer = ECGTokenizer(beat_len=beat_len, fs=fs)
    
    for i in range(num_samples):
        ecg_tensor = ecgs[i:i+1] # [1, L, T]
        sample_id = ids[i]
        
        # 1. 使用 tokenizer 的逻辑提取并重采样所有 beat
        # _forward_rr 返回 (X, meta), 其中 X 是 [B, L, N, d], 我们需要的是重采样后的波形
        # 这里我们手动提取重采样后的波形以便可视化
        ref = ecg_tensor[0, 1] # Use Lead II as reference
        peaks = tokenizer._detect_r_peaks(ref, fs)
        
        if len(peaks) < 3:
            print(f"Skipping ID {sample_id}: not enough beats for diff (found {len(peaks)})")
            continue
            
        mids = [(peaks[j] + peaks[j + 1]) // 2 for j in range(len(peaks) - 1)]
        starts = [max(0, peaks[0] - (mids[0] - peaks[0]))] + mids[:-1]
        ends = mids + [min(T, peaks[-1] + (peaks[-1] - mids[-1]))]
        
        beats = []
        for start, end in zip(starts, ends):
            seg = ecg_tensor[0, :, start:end] # [L, seg_len]
            seg_rs = torch.nn.functional.interpolate(
                seg.unsqueeze(1), size=beat_len, mode="linear", align_corners=True
            ).squeeze(1) # [L, beat_len]
            beats.append(seg_rs.cpu().numpy())
        
        # 2. 计算差值 (beat[n] - beat[n-1])
        diffs = []
        for j in range(1, len(beats)):
            diffs.append(beats[j] - beats[j-1])
        
        # 3. 将所有差值片段在时间轴上展开，拼接成一条长曲线
        full_diff = np.concatenate(diffs, axis=-1) # [L, (N-1) * beat_len]
            
        # 4. 画图
        fig, axes = plt.subplots(L, 1, figsize=(12, 2 * L), sharex=True)
        if L == 1: axes = [axes]
        
        for l in range(L):
            ax = axes[l]
            ax.plot(full_diff[l], color='blue', linewidth=0.8)
            
            lead_name = STANDARD_12_LEADS[l] if l < len(STANDARD_12_LEADS) else f"Lead {l+1}"
            ax.set_ylabel(lead_name, rotation=0, labelpad=20, verticalalignment='center')
            ax.axhline(0, color='red', linestyle=':', alpha=0.5)
            # 画出每个 beat 的分界线
            for b_idx in range(1, len(diffs)):
                ax.axvline(x=b_idx * beat_len, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            
            ax.grid(True, which='both', linestyle='--', alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        axes[-1].set_xlabel("Time (Concatenated Beat Index * beat_len)")
        plt.suptitle(f"Sequential Beat-to-Beat Waveform Differences\nID: {sample_id} (Each segment is Beat[n] - Beat[n-1])", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        safe_id = str(sample_id).replace('/', '_').replace(':', '_')
        save_path = os.path.join(CURRENT_DIR, "plots", f"ecg_diff_{safe_id}.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved beat diff visualization to {save_path}")
        plt.close()

def visualize_ecg(batch, fs=100, num_samples=3):
    ecgs = batch["ecg"] # [B, L, T]
    ids = batch["id"]
    
    B, L, T = ecgs.shape
    num_samples = min(num_samples, B)
    
    # Initialize tokenizer to use its R-peak detection logic
    tokenizer = ECGTokenizer(fs=fs)
    
    os.makedirs(os.path.join(CURRENT_DIR, "plots"), exist_ok=True)
    
    for i in range(num_samples):
        # Create a figure with 12 subplots in a 6x2 or 4x3 grid? 
        # For 12 leads, 6 rows, 2 columns is often readable.
        # Or just 12 rows for maximum detail. Let's do 12 rows.
        fig, axes = plt.subplots(L, 1, figsize=(12, 2 * L), sharex=True)
        if L == 1:
            axes = [axes]
        
        ecg = ecgs[i].cpu().numpy()
        sample_id = ids[i]
        
        # Detect R-peaks using Lead II (index 1) as reference
        # We need to pass a torch tensor to _detect_r_peaks
        lead_ii_tensor = torch.from_numpy(ecg[1])
        r_peaks = tokenizer._detect_r_peaks(lead_ii_tensor, fs)
        
        for l in range(L):
            ax = axes[l]
            ax.plot(ecg[l], color='black', linewidth=0.8)
            
            # Draw red lines for R-peaks
            for peak in r_peaks:
                ax.axvline(x=peak, color='red', linestyle='--', linewidth=0.8, alpha=0.7)
            
            lead_name = STANDARD_12_LEADS[l] if l < len(STANDARD_12_LEADS) else f"Lead {l+1}"
            ax.set_ylabel(lead_name, rotation=0, labelpad=20, verticalalignment='center')
            ax.grid(True, which='both', linestyle='--', alpha=0.5)
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
        axes[-1].set_xlabel("Time (samples)")
        plt.suptitle(f"12-Lead ECG Visualization with R-Peaks\nID: {sample_id}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Replace / with _ for filename safety
        safe_id = str(sample_id).replace('/', '_').replace(':', '_')
        save_path = os.path.join(CURRENT_DIR, "plots", f"ecg_rr_{safe_id}.png")
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
        plt.close()

def main():
    config_path = os.path.join(PROJECT_ROOT, "configs/train/v1.yaml")
    if not os.path.exists(config_path):
        print(f"Config not found at {config_path}")
        return
        
    print(f"Loading config from {config_path}...")
    cfg = load_config(config_path)
    
    # Update config for visualization
    cfg.raw["train"]["batch_size"] = 5
    cfg.raw["train"]["num_workers"] = 0
    fs = int(cfg.data.get("fs", 100))
    beat_len = int(cfg.model.get("beat_len", 128))
    
    print(f"Initializing dataloader (type: {cfg.data.get('dataset_type')})...")
    try:
        loader = make_dataloader(cfg=cfg.raw, split="train")
        
        print("Fetching first batch...")
        batch = next(iter(loader))
        print(f"Loaded batch with shape: {batch['ecg'].shape}")
        
        visualize_ecg(batch, fs=fs)
        visualize_beat_diffs(batch, fs=fs, beat_len=beat_len)
        print("\nVisualization complete! Check the 'plots' directory.")

        
    except Exception as e:
        print(f"Error during data loading or visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

