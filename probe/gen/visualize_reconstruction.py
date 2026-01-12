"""
ECG Reconstruction Visualization Script.

This script:
1. Loads a trained ECGGenModel checkpoint
2. Randomly selects 2 ECGs from the validation set
3. Runs inference with random 3-lead masking
4. Visualizes original vs reconstructed ECG (12 subplots per image)
5. Saves the plots to the current folder

Usage:
    cd /home/gbsguest/Research/boson/BIO
    conda activate torch2.5
    python -m ecggen.probe.gen.visualize_reconstruction
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from ecggen.src.utils.config import load_config
from ecggen.src.utils.trainer import build_model
from ecggen.src.data.pipeline import make_dataloader
from ecggen.src.models.utils.loss import random_lead_mask


# Configuration
CONFIG_PATH = "/home/gbsguest/Research/boson/BIO/ecggen/configs/train/v1.yaml"
CHECKPOINT_DIR = "/home/gbsguest/Research/boson/BIO/ecggen/checkpoints/m1s1k1"
OUTPUT_DIR = Path(__file__).parent
NUM_SAMPLES = 2

# Lead names in MIMIC order
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def find_best_checkpoint(ckpt_dir: str) -> str:
    """Find the best checkpoint in the directory."""
    ckpt_dir = Path(ckpt_dir)
    
    # Try to find best.json
    best_json = ckpt_dir / "best.json"
    if best_json.exists():
        import json
        with open(best_json) as f:
            info = json.load(f)
            best_path = info.get("best_path")
            if best_path and os.path.exists(best_path):
                return best_path
    
    # Fallback: find any .pt file
    pt_files = list(ckpt_dir.glob("*.pt"))
    if pt_files:
        # Sort by modification time, get newest
        pt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(pt_files[0])
    
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def load_model(cfg, checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    model = build_model(cfg)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


@torch.no_grad()
def reconstruct_ecg(model, ecg: torch.Tensor, num_visible: int = 3, device: torch.device = None):
    """
    Reconstruct ECG with random masking.
    
    Args:
        model: ECGGenModel
        ecg: [1, 12, T] single ECG
        num_visible: Number of visible leads
        device: Target device
    
    Returns:
        dict with 'recon', 'visible_indices', 'mask'
    """
    if device is None:
        device = next(model.parameters()).device
    
    ecg = ecg.to(device)
    B = ecg.shape[0]
    
    # Generate random mask
    visible_indices, mask = random_lead_mask(
        batch_size=B,
        num_leads=12,
        num_visible=num_visible,
        device=device
    )
    
    # Forward pass
    outputs = model(ecg, visible_indices)
    
    return {
        'recon': outputs['recon'].cpu(),
        'original': ecg.cpu(),
        'visible_indices': visible_indices.cpu(),
        'mask': mask.cpu(),
        'VCG': outputs.get('VCG', None),
        'geom_proj': outputs.get('geom_proj', None)
    }


def plot_reconstruction(
    original: np.ndarray,
    recon: np.ndarray,
    visible_indices: np.ndarray,
    sample_id: str,
    output_path: str,
    fs: int = 100
):
    """
    Plot original vs reconstructed ECG.
    
    Args:
        original: [12, T] original ECG
        recon: [12, T] reconstructed ECG
        visible_indices: [3] indices of visible leads
        sample_id: Sample identifier
        output_path: Path to save the figure
        fs: Sampling frequency
    """
    num_leads = 12
    T = original.shape[1]
    time = np.arange(T) / fs
    
    # Create figure with 12 subplots (4 rows x 3 cols)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    visible_set = set(visible_indices.tolist())
    
    for i in range(num_leads):
        ax = axes[i]
        
        # Plot original
        ax.plot(time, original[i], 'b-', linewidth=0.8, alpha=0.7, label='Original')
        
        # Plot reconstruction
        ax.plot(time, recon[i], 'r-', linewidth=0.8, alpha=0.7, label='Reconstructed')
        
        # Indicate if this lead was visible or masked
        if i in visible_set:
            lead_label = f"{LEAD_NAMES[i]} (visible)"
            ax.set_facecolor('#e6ffe6')  # Light green for visible
        else:
            lead_label = f"{LEAD_NAMES[i]} (masked)"
            ax.set_facecolor('#ffe6e6')  # Light red for masked
        
        ax.set_title(lead_label, fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        
        # Only show legend on first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=7)
    
    # Compute reconstruction error on masked leads
    mask = np.array([i not in visible_set for i in range(num_leads)])
    if mask.any():
        mse_masked = np.mean((original[mask] - recon[mask]) ** 2)
        corr_masked = np.corrcoef(original[mask].flatten(), recon[mask].flatten())[0, 1]
    else:
        mse_masked = 0
        corr_masked = 1
    
    fig.suptitle(
        f"ECG Reconstruction: {sample_id}\n"
        f"Visible leads: {[LEAD_NAMES[i] for i in visible_indices]} | "
        f"Masked MSE: {mse_masked:.4f} | Masked Corr: {corr_masked:.4f}",
        fontsize=12, fontweight='bold'
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    print(f"  Visible: {[LEAD_NAMES[i] for i in visible_indices]}")
    print(f"  Masked MSE: {mse_masked:.4f}, Corr: {corr_masked:.4f}")


def main():
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg = load_config(CONFIG_PATH)
    
    # Find and load checkpoint
    try:
        ckpt_path = find_best_checkpoint(CHECKPOINT_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure a trained model exists.")
        return
    
    model = load_model(cfg, ckpt_path, device)
    
    # Load validation data
    print("Loading validation data...")
    val_loader = make_dataloader(cfg.raw, split="val")
    
    # Collect all validation samples
    all_samples = []
    for batch in val_loader:
        for i in range(batch['ecg'].shape[0]):
            all_samples.append({
                'ecg': batch['ecg'][i:i+1],  # Keep batch dim
                'id': batch['id'][i] if 'id' in batch else f"sample_{len(all_samples)}"
            })
        if len(all_samples) >= 100:  # Only collect first 100 for efficiency
            break
    
    print(f"Collected {len(all_samples)} validation samples")
    
    # Randomly select samples
    selected = random.sample(all_samples, min(NUM_SAMPLES, len(all_samples)))
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each sample
    for idx, sample in enumerate(selected):
        ecg = sample['ecg']
        sample_id = sample['id']
        
        print(f"\nProcessing sample {idx+1}/{len(selected)}: {sample_id}")
        
        # Reconstruct
        result = reconstruct_ecg(model, ecg, num_visible=3, device=device)
        
        # Extract numpy arrays
        original = result['original'][0].numpy()  # [12, T]
        recon = result['recon'][0].numpy()        # [12, T]
        visible_indices = result['visible_indices'][0].numpy()  # [3]
        
        # Plot and save
        output_path = OUTPUT_DIR / f"recon_{sample_id}.png"
        plot_reconstruction(
            original=original,
            recon=recon,
            visible_indices=visible_indices,
            sample_id=sample_id,
            output_path=str(output_path),
            fs=cfg.data.get('fs', 100)
        )
    
    print(f"\nDone! Saved {len(selected)} visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

