"""
Loss functions for ECG multi-lead reconstruction.

This module provides:
- masked_reconstruction_loss: MSE loss computed only on masked leads
- random_lead_mask: Generate random mask for visible/masked leads
- apply_lead_mask: Apply mask to ECG signals

Key Design Principles:
- Only supervise masked (invisible) leads during training
- Visible leads are used for VCG reconstruction but not supervised
"""

from typing import Tuple

import torch


def random_lead_mask(
    batch_size: int,
    num_leads: int = 12,
    num_visible: int = 3,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random lead mask for ECG reconstruction.
    
    Pipeline Role: Create random selection of visible leads for each sample
                   in the batch. Used for self-supervised reconstruction training.
    
    Output Semantics:
        - visible_indices: Which leads are visible (input to VCG recovery)
        - mask: Boolean mask where True = masked (not visible)
    
    Args:
        batch_size: Number of samples in batch
        num_leads: Total number of leads (typically 12)
        num_visible: Number of visible leads (typically 3)
        device: Target device for tensors
    
    Returns:
        visible_indices: [B, num_visible] - Indices of visible leads
        mask: [B, num_leads] - Boolean mask (True = masked, False = visible)
    
    Shape Contracts:
        visible_indices: [B, 3]
        mask: [B, 12] where exactly 9 values are True per row
    """
    if device is None:
        device = torch.device('cpu')
    
    visible_indices_list = []
    for _ in range(batch_size):
        # Randomly select num_visible leads without replacement
        indices = torch.randperm(num_leads, device=device)[:num_visible]
        visible_indices_list.append(indices)
    
    # Stack to tensor: [B, num_visible]
    visible_indices = torch.stack(visible_indices_list)
    
    # Generate mask: True = masked (invisible)
    # Start with all True (all masked)
    mask = torch.ones(batch_size, num_leads, dtype=torch.bool, device=device)
    
    # Set visible leads to False
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_visible)
    mask[batch_idx, visible_indices] = False
    
    return visible_indices, mask


def apply_lead_mask(
    ecg: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply mask to ECG signals by zeroing masked leads.
    
    Pipeline Role: Prepare input for model by zeroing out masked leads.
                   Model only sees visible leads; masked leads are supervision targets.
    
    Input Semantics:
        - ecg: Original 12-lead ECG signals
        - mask: Boolean mask (True = should be masked/zeroed)
    
    Output Semantics:
        - masked_ecg: ECG with masked leads set to zero
    
    Args:
        ecg: ECG signals [B, 12, T]
        mask: Boolean mask [B, 12] where True = masked
    
    Returns:
        masked_ecg: [B, 12, T] with masked leads zeroed
    
    Shape Contract:
        Input: [B, 12, T] + [B, 12]
        Output: [B, 12, T]
    """
    # Clone to avoid modifying original
    masked_ecg = ecg.clone()
    
    # Expand mask: [B, 12] -> [B, 12, 1] for broadcasting
    mask_expanded = mask.unsqueeze(-1)  # [B, 12, 1]
    
    # Zero out masked leads
    masked_ecg = masked_ecg.masked_fill(mask_expanded, 0.0)
    
    return masked_ecg


def masked_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute MSE loss only on masked (invisible) leads.
    
    Pipeline Role: Supervision signal for reconstruction training.
                   Only penalizes errors on leads that were masked during input.
    
    Input Semantics:
        - recon: Model's reconstructed ECG
        - target: Ground truth ECG
        - mask: Boolean mask (True = was masked, should be supervised)
    
    Output Semantics:
        - loss: Scalar MSE loss computed only on masked leads
    
    Args:
        recon: Reconstructed ECG [B, 12, T]
        target: Ground truth ECG [B, 12, T]
        mask: Boolean mask [B, 12] where True = masked (supervise these)
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: Reconstruction loss scalar (if reduction='mean' or 'sum')
              or per-sample loss [B] (if reduction='none')
    
    Shape Contract:
        Input: [B, 12, T] x2 + [B, 12]
        Output: scalar (or [B] if reduction='none')
    """
    # Compute element-wise squared error: [B, 12, T]
    mse = (recon - target) ** 2
    
    # Expand mask: [B, 12] -> [B, 12, 1] for broadcasting
    mask_float = mask.unsqueeze(-1).float()  # [B, 12, 1]
    
    # Apply mask: only keep errors from masked leads
    masked_mse = mse * mask_float  # [B, 12, T]
    
    if reduction == 'none':
        # Sum over leads and time, keep batch dimension
        return masked_mse.sum(dim=(1, 2))  # [B]
    elif reduction == 'sum':
        return masked_mse.sum()
    else:  # 'mean'
        # Compute number of masked samples
        # mask.sum() gives total masked leads across batch
        # Multiply by T to get total masked time points
        T = recon.shape[-1]
        num_masked = mask.sum() * T  # Total masked samples
        
        # Avoid division by zero
        loss = masked_mse.sum() / (num_masked + 1e-8)
        return loss


def full_reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute MSE loss on all leads (no masking).
    
    Useful for evaluation or when supervising all leads.
    
    Args:
        recon: Reconstructed ECG [B, 12, T]
        target: Ground truth ECG [B, 12, T]
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: Reconstruction loss
    """
    mse = (recon - target) ** 2
    
    if reduction == 'none':
        return mse.sum(dim=(1, 2))  # [B]
    elif reduction == 'sum':
        return mse.sum()
    else:  # 'mean'
        return mse.mean()


def vcg_reconstruction_loss(
    recon_vcg: torch.Tensor,
    target_vcg: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute MSE loss on VCG trajectories.
    
    Args:
        recon_vcg: Reconstructed VCG [B, 3, T]
        target_vcg: Ground truth VCG [B, 3, T]
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        loss: VCG reconstruction loss
    """
    mse = (recon_vcg - target_vcg) ** 2
    
    if reduction == 'none':
        return mse.sum(dim=(1, 2))  # [B]
    elif reduction == 'sum':
        return mse.sum()
    else:  # 'mean'
        return mse.mean()


class ReconstructionLoss(torch.nn.Module):
    """
    Module wrapper for masked reconstruction loss.
    
    Convenient for use in training loops where loss needs to be a nn.Module.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            recon: [B, 12, T]
            target: [B, 12, T]
            mask: [B, 12]
        
        Returns:
            loss: Scalar or [B]
        """
        return masked_reconstruction_loss(recon, target, mask, self.reduction)

