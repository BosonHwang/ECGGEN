"""
VCG (Vectorcardiogram) modules for ECG multi-lead reconstruction.

This module provides:
- VCGPseudoInverse: Recover VCG from visible ECG leads via geometric pseudo-inverse
- GeometricLeadProjection: Project VCG to 12-lead ECG via geometric projection

Key Design Principles:
- No learnable parameters in VCG recovery/projection (pure geometry)
- Numerical stability via regularization in pseudo-inverse
- Lead-agnostic: works with any subset of visible leads
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..data.angle import compute_lead_directions, LEAD_ANGLES_PTBXL_ORDER


class VCGPseudoInverse(nn.Module):
    """
    VCG recovery via pseudo-inverse from visible ECG leads.
    
    Pipeline Role: Convert K visible ECG leads to 3D VCG trajectory
                   using geometric pseudo-inverse (no learnable parameters).
    
    Input Semantics:
        - S: K visible ECG lead signals [B, K, T]
        - theta: Elevation angles for K leads [B, K]
        - phi: Azimuth angles for K leads [B, K]
    
    Output Semantics:
        - VCG: 3D heart vector trajectory [B, 3, T]
    
    Mathematical Formulation:
        Each ECG lead is a linear projection: s_i(t) = u_i^T * v(t)
        Given K leads S and direction matrix U (K x 3):
            VCG = U^+ @ S
        where U^+ = (U^T U + eps*I)^{-1} U^T is the regularized pseudo-inverse.
    
    Key Design:
        - No learnable parameters (pure geometric computation)
        - Regularization (eps) for numerical stability
        - Works with any K >= 3 visible leads
    """
    
    def __init__(self, eps: float = 1e-6):
        """
        Args:
            eps: Regularization coefficient for pseudo-inverse stability
        """
        super().__init__()
        self.eps = eps
    
    def forward(
        self,
        S: torch.Tensor,
        theta: torch.Tensor,
        phi: torch.Tensor
    ) -> torch.Tensor:
        """
        Recover VCG from visible leads via pseudo-inverse.
        
        Args:
            S: Visible lead signals [B, K, T] where K is number of visible leads
            theta: Elevation angles [B, K]
            phi: Azimuth angles [B, K]
        
        Returns:
            VCG: 3D trajectory [B, 3, T]
        
        Shape Flow:
            S: [B, K, T]
            theta, phi: [B, K]
            U: [B, K, 3] (direction vectors)
            U^+: [B, 3, K] (pseudo-inverse)
            VCG = U^+ @ S: [B, 3, T]
        """
        B, K, T = S.shape
        device = S.device
        dtype = S.dtype
        
        assert theta.shape == (B, K), f"theta shape mismatch: expected {(B, K)}, got {theta.shape}"
        assert phi.shape == (B, K), f"phi shape mismatch: expected {(B, K)}, got {phi.shape}"
        
        # Compute direction vectors U: [B, K, 3]
        # u = [cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta)]
        U = compute_lead_directions(theta, phi)  # [B, K, 3]
        
        # Compute pseudo-inverse U_pinv: [B, 3, K]
        # U_pinv = (U^T U + eps*I)^{-1} U^T
        Ut = U.transpose(1, 2)           # [B, 3, K]
        UtU = Ut @ U                     # [B, 3, 3]
        
        # Add regularization for numerical stability
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)  # [1, 3, 3]
        UtU_reg = UtU + self.eps * eye   # [B, 3, 3]
        
        # Compute inverse
        UtU_inv = torch.linalg.inv(UtU_reg)  # [B, 3, 3]
        U_pinv = UtU_inv @ Ut            # [B, 3, K]
        
        # Recover VCG: [B, 3, K] @ [B, K, T] -> [B, 3, T]
        VCG = U_pinv @ S
        
        return VCG


class GeometricLeadProjection(nn.Module):
    """
    Project VCG to multi-lead ECG via geometric projection.
    
    Pipeline Role: Convert 3D VCG trajectory to L-lead ECG signals
                   using fixed lead direction vectors (no learnable parameters).
    
    Input Semantics:
        - VCG: 3D heart vector trajectory [B, 3, T]
    
    Output Semantics:
        - ECG: L-lead ECG signals [B, L, T]
    
    Mathematical Formulation:
        For each lead l: ecg_l(t) = u_l^T @ vcg(t)
        where u_l is the unit direction vector for lead l.
    
    Key Design:
        - No learnable parameters (pure geometric projection)
        - Lead directions are pre-computed and stored as buffer
        - Supports any number of leads with provided angles
    """
    
    def __init__(self, lead_angles: torch.Tensor):
        """
        Args:
            lead_angles: [L, 2] tensor with [theta, phi] for each lead
        """
        super().__init__()
        
        # lead_angles: [L, 2] where [:, 0] is theta, [:, 1] is phi
        assert lead_angles.ndim == 2 and lead_angles.shape[1] == 2, \
            f"lead_angles must be [L, 2], got {lead_angles.shape}"
        
        theta = lead_angles[:, 0]  # [L]
        phi = lead_angles[:, 1]    # [L]
        
        # Compute direction vectors: [L, 3]
        U = compute_lead_directions(theta, phi)
        
        # Register as buffer (not learnable, but moves with model)
        self.register_buffer('U', U)  # [L, 3]
        self.num_leads = lead_angles.shape[0]
    
    def forward(self, VCG: torch.Tensor) -> torch.Tensor:
        """
        Project VCG to multi-lead ECG.
        
        Args:
            VCG: 3D trajectory [B, 3, T]
        
        Returns:
            ECG: L-lead signals [B, L, T]
        
        Shape Flow:
            VCG: [B, 3, T]
            U: [L, 3]
            ECG = einsum('lc,bct->blt'): [B, L, T]
        """
        assert VCG.ndim == 3 and VCG.shape[1] == 3, \
            f"VCG must be [B, 3, T], got {VCG.shape}"
        
        # U: [L, 3], VCG: [B, 3, T]
        # ECG_l(t) = sum_c U[l, c] * VCG[:, c, t]
        ECG = torch.einsum('lc,bct->blt', self.U, VCG)  # [B, L, T]
        
        return ECG


# =============================================================================
# Legacy VCG Generator (kept for backward compatibility)
# =============================================================================

class VCGGenerator(nn.Module):
    """
    Generate latent VCG trajectories V [B, 3, T'] from state W [B, D].
    
    DEPRECATED: This is the old TTT-based VCG generator.
    For the new reconstruction task, use VCGPseudoInverse + GeometricLeadProjection.
    
    1) Role: render a shared latent source (not ECG) from W.
    2) I/O: W [B, D] -> V [B, 3, T'] via low-DOF basis A(W) @ B.
    3) Pipeline: feeds LeadProjection; regularizers constrain geometry/dynamics.
    """

    def __init__(self, state_dim: int, basis_k: int = 32, time_len: int = 256):
        super().__init__()
        self.state_dim = int(state_dim)
        self.basis_k = int(basis_k)
        self.time_len = int(time_len)
        hidden = max(64, state_dim * 2)
        self.A_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3 * basis_k),
        )
        # Shared temporal basis across all samples
        self.B = nn.Parameter(torch.randn(basis_k, time_len) * 0.1)

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        assert W.ndim == 2, "W must be [B, D]"
        assert W.shape[1] == self.state_dim, "state_dim mismatch"
        B = W.shape[0]
        A = self.A_net(W).reshape(B, 3, self.basis_k)  # [B, 3, K]
        V = torch.matmul(A, self.B)  # [B, 3, T']
        return V

    @staticmethod
    def regularizer_smoothness(V: torch.Tensor) -> torch.Tensor:
        """Penalize curvature via second differences along time."""
        d1 = V[..., 1:] - V[..., :-1]
        d2 = d1[..., 1:] - d1[..., :-1]
        return (d2 ** 2).mean()

    @staticmethod
    def regularizer_energy(V: torch.Tensor) -> torch.Tensor:
        """Penalize overall energy to prevent scale drift."""
        return (V ** 2).mean()

    @staticmethod
    def regularizer_loop_closure(V: torch.Tensor) -> torch.Tensor:
        """Encourage start ≈ end if applicable (optional)."""
        start = V[..., 0]
        end = V[..., -1]
        return ((start - end) ** 2).mean()
