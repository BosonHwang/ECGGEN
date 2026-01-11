# VCG from 3-Lead ECG with Known Angles (Compact)

Given three ECG leads and their azimuth/elevation angles, compute the Vectorcardiogram (VCG) and synthesize ECG signals at arbitrary orientations.

## Notation
ECG signals: S ∈ ℝ^{B×3×T}  
Azimuth angles: φ ∈ ℝ^{B×3}  
Elevation angles: θ ∈ ℝ^{B×3}

## Procedure

Each ECG lead is a linear projection of the VCG:
s(t) = U v(t)

where each lead direction is
u_i = [cosθ_i cosφ_i, cosθ_i sinφ_i, sinθ_i].

```python
# Build lead direction matrix U
u_x = torch.cos(theta) * torch.cos(phi)
u_y = torch.cos(theta) * torch.sin(phi)
u_z = torch.sin(theta)
U = torch.stack([u_x, u_y, u_z], dim=-1)           # [B, 3, 3]

# Compute batched pseudo-inverse of U
Ut = U.transpose(1, 2)
UtU = Ut @ U
UtU_inv = torch.linalg.inv(
    UtU + 1e-6 * torch.eye(3, device=U.device)
)
U_pinv = UtU_inv @ Ut                               # [B, 3, 3]

# Recover VCG
VCG = U_pinv @ S                                   # [B, 3, T]

# Synthesize ECG at arbitrary angle (θ_q, φ_q)
u_q = torch.stack([
    torch.cos(theta_q) * torch.cos(phi_q),
    torch.cos(theta_q) * torch.sin(phi_q),
    torch.sin(theta_q)
], dim=1)                                          # [B, 3]

ECG_q = torch.einsum("bc,bct->bt", u_q, VCG)        # [B, T]