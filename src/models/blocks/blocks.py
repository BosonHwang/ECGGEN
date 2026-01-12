from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEncoder(nn.Module):
    """Token-level interaction encoder (no temporal modeling).

    1) Role: mixes tokens across leads/beats, not time rollout.
    2) I/O: X [B, L, N, d] -> H [B, L*N, d_model].
    3) Pipeline: consumes tokenizer output; feeds TTT/VCG; no recurrence.
    """

    def __init__(self, token_dim: int, d_model: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(token_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert X.ndim == 4, "X must be [B, L, N, d]"
        B, L, N, d = X.shape
        assert d == self.proj.in_features, "token_dim mismatch"
        Y = X.reshape(B, L * N, d)  # [B, L*N, d]
        H_in = self.proj(Y)  # [B, L*N, d_model]
        H = self.encoder(H_in)  # [B, L*N, d_model]
        return H


class LeadProjection(nn.Module):
    """Project latent VCG V [B,3,T] to multi-lead ECG [B,L,T].

    Geometry:
        - Each ECG lead is a linear observation of the 3D latent source (VCG).
        - For lead ℓ we learn a unit direction vector l_ℓ ∈ R^3, a gain a_ℓ, and a bias b_ℓ.
        - The output for lead ℓ is ê_ℓ(t) = a_ℓ * (l_ℓᵀ V(t)) + b_ℓ
        - lead vectors are physical directions (not embeddings).
        - gain/bias account for observation scale and baseline offset.
    """

    def __init__(self, num_leads: int = 12):
        super().__init__()
        self.num_leads = num_leads
        self.lead_vectors = nn.Parameter(torch.randn(num_leads, 3))
        self.gain = nn.Parameter(torch.ones(num_leads))
        self.bias = nn.Parameter(torch.zeros(num_leads))

    def forward(self, V: torch.Tensor) -> torch.Tensor:
        assert V.ndim == 3 and V.shape[1] == 3, "V must be [B, 3, T]"
        B, _, T = V.shape
        # Normalize lead vectors to unit length for geometric clarity
        lv = F.normalize(self.lead_vectors, dim=-1)  # [L, 3]
        # Project: per-lead dot product with V(t)
        # Compute [L, B, T]: for each lead, take l^T @ V
        # We implement via batch matmul
        # V: [B, 3, T] -> [B, T, 3] for matmul convenience
        V_bt3 = V.permute(0, 2, 1)  # [B, T, 3]
        # lv: [L, 3] -> [L, 3, 1]
        lv_31 = lv.unsqueeze(-1)  # [L, 3, 1]
        # Dot for each lead separately
        # Compute [L, B, T] by (lv[ℓ]^T V_b(t)) over b,t
        E_hat_list = []
        for lidx in range(self.num_leads):
            proj = torch.matmul(V_bt3, lv_31[lidx]).squeeze(-1)  # [B, T]
            e = self.gain[lidx] * proj + self.bias[lidx]
            E_hat_list.append(e)
        E_hat = torch.stack(E_hat_list, dim=1)  # [B, L, T]
        return E_hat


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise + pointwise Conv1D with small capacity."""

    def __init__(self, channels: int, kernel_size: int = 7, expansion: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels * expansion, kernel_size=1)
        self.proj = nn.Conv1d(channels * expansion, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(y)
        y = self.act(y)
        y = self.proj(y)
        return y


class ResidualHead(nn.Module):
    """Small residual head to compensate projection imperfections.

    Role:
        - Input is base ECG [B, L, T] from lead projection
        - Output is residual of the same shape to correct small mismatches
        - Capacity is deliberately small to preserve the bottleneck semantics:
          residual should not dominate reconstruction.
    """

    def __init__(self, num_leads: int, kernel_size: int = 7):
        super().__init__()
        self.block = DepthwiseSeparableConv1d(num_leads, kernel_size=kernel_size, expansion=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, "x must be [B, L, T]"
        y = self.block(x)
        return y


class AngleCalib(nn.Module):
    """Skeleton for angle calibration (not implemented).

    Purpose (as required):
        - Angle/lead geometry is a nuisance variable, not representation.
        - Use only a short prefix of beats (no long context) to estimate a small SO(3) rotation.
        - Correction applies to lead directions or VCG frame; must not alter or leak into W learning.
        - Should carry a small-rotation prior; same record shares angle.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Angle calibration is intentionally left as a skeleton.")


# =========================
# SO(3) rotation utilities
# =========================

def axis_angle_to_matrix(r: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle r[... ,3] to rotation matrix R[..., 3, 3].

    Geometric meaning:
        - Direction of r is the rotation axis.
        - Magnitude of r is the rotation angle (radians).
        - Useful for small orientation adjustments in observation geometry.
    """
    assert r.shape[-1] == 3
    theta = torch.linalg.norm(r, dim=-1, keepdim=True)  # [..., 1]
    small = (theta < 1e-6).float()
    k = torch.where(theta > 0, r / torch.clamp(theta, min=1e-6), torch.zeros_like(r))  # unit axis
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]

    # Rodrigues formula
    ct = torch.cos(theta)[..., 0]
    st = torch.sin(theta)[..., 0]
    vt = 1 - ct

    R = torch.zeros(r.shape[:-1] + (3, 3), dtype=r.dtype, device=r.device)
    R[..., 0, 0] = ct + kx * kx * vt
    R[..., 0, 1] = kx * ky * vt - kz * st
    R[..., 0, 2] = kx * kz * vt + ky * st
    R[..., 1, 0] = ky * kx * vt + kz * st
    R[..., 1, 1] = ct + ky * ky * vt
    R[..., 1, 2] = ky * kz * vt - kx * st
    R[..., 2, 0] = kz * kx * vt - ky * st
    R[..., 2, 1] = kz * ky * vt + kx * st
    R[..., 2, 2] = ct + kz * kz * vt
    return R


def apply_rotation(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v by rotation matrix R.

    - R: [..., 3, 3]
    - v: [..., 3] or [..., 3, T]
    Preserves norms; used to rotate VCG trajectories or lead vectors.
    """
    assert R.shape[-2:] == (3, 3)
    if v.ndim >= 2 and v.shape[-2:] == (3,):
        # [..., 3]
        return torch.matmul(R, v.unsqueeze(-1)).squeeze(-1)
    elif v.ndim >= 3 and v.shape[-2] == 3:
        # [..., 3, T]
        return torch.matmul(R, v)
    else:
        raise ValueError("v must be [...,3] or [...,3,T]")


def rotation_magnitude(R: torch.Tensor) -> torch.Tensor:
    """Return a scalar measure of rotation magnitude for regularization.

    One simple proxy is the angle from the trace: trace(R) = 1 + 2 cos(theta).
    """
    assert R.shape[-2:] == (3, 3)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    # Clamp to valid range to avoid NaNs from numerical noise
    cos_theta = torch.clamp((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = torch.arccos(cos_theta)
    return theta


