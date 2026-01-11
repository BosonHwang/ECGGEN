from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class VCGGenerator(nn.Module):
    """Generate latent VCG trajectories V [B, 3, T'] from state W [B, D].

    1) Role: render a shared latent source (not ECG) from W.
    2) I/O: W [B, D] -> V [B, 3, T’] via low-DOF basis A(W) @ B.
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


