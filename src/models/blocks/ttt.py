from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


class FastState(nn.Module):
    """Quasi-static heart state W (state estimation, not recurrence).

    1) Role: store/update W during test-time training.
    2) I/O: forward returns W [B, D] (or MLP(W) in mlp mode).
    3) Pipeline: used after TokenEncoder; updated by TTTUpdater only, encoder/generator frozen.

    Modes:
        - 'vector': W is a learnable vector [B, D] per batch instance
        - 'mlp': W parameterizes a tiny MLP; still optimized as a state vector
    """

    def __init__(self, mode: str, batch_size: int, dim: int):
        super().__init__()
        self.mode = mode
        self.dim = dim
        if mode == "vector":
            self.W = nn.Parameter(torch.zeros(batch_size, dim))
        elif mode == "mlp":
            # Skeleton: a very small MLP whose parameters are treated as the state.
            # In practice, we recommend the 'vector' mode for simplicity and identifiability.
            hidden = max(16, dim // 2)
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )
            # Represent the state as a vector input to the MLP; we still optimize this vector.
            self.W = nn.Parameter(torch.zeros(batch_size, dim))
        else:
            raise ValueError("mode must be 'vector' or 'mlp'")

    def forward(self) -> torch.Tensor:
        if self.mode == "vector":
            return self.W  # [B, D]
        else:
            return self.mlp(self.W)  # [B, D]


class TTTUpdater:
    """Online state estimation via test-time updates (beat-chunked).

    1) Role: refine W using observed beats; no forecasting/rollout.
    2) I/O: H_beats [B, N, d], returns W_final [B, D].
    3) Pipeline: consumes encoder output aggregated per beat; only updates W, keeps encoder/generator frozen.

    Smoothness: ||W_k - W_{k-1}||^2 across beat chunks to enforce slow drift.
    """

    def __init__(self, step_size: float = 1e-2, smooth_lambda: float = 1e-2, chunk_size: int = 4):
        self.step_size = float(step_size)
        self.smooth_lambda = float(smooth_lambda)
        self.chunk_size = int(chunk_size)

    def update(self, state: FastState, H_beats: torch.Tensor, loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> torch.Tensor:
        assert H_beats.ndim == 3, "H_beats must be [B, N, d]"
        B, N, _ = H_beats.shape
        W_prev: Optional[torch.Tensor] = None
        for start in range(0, N, self.chunk_size):
            end = min(N, start + self.chunk_size)
            Hc = H_beats[:, start:end, :]  # [B, c, d]

            W = state()  # [B, D]
            W = W.requires_grad_(True)
            if loss_fn is None:
                # Placeholder self-supervised proxy: encourage W to align with mean chunk features
                target = Hc.mean(dim=(1, 2))  # [B]
                target = target.unsqueeze(-1).repeat(1, W.shape[-1]) if target.ndim == 1 else target
                loss_data = ((W - target[:, : W.shape[-1]]) ** 2).mean()
            else:
                loss_data = loss_fn(W, Hc)

            loss_smooth = torch.tensor(0.0, device=H_beats.device)
            if W_prev is not None and self.smooth_lambda > 0.0:
                loss_smooth = self.smooth_lambda * ((W - W_prev.detach()) ** 2).mean()
            loss = loss_data + loss_smooth

            grad_W = torch.autograd.grad(loss, W, retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                state.W.copy_(state.W - self.step_size * grad_W)
                W_prev = state.W.detach().clone()
        return state()


def _sanity_check() -> None:
    B, N, d = 2, 8, 64
    D = 16
    H = torch.randn(B, N, d)
    state = FastState(mode="vector", batch_size=B, dim=D)
    updater = TTTUpdater(step_size=1e-1, smooth_lambda=1e-2, chunk_size=4)

    def dummy_loss_fn(W: torch.Tensor, Hc: torch.Tensor) -> torch.Tensor:
        target = Hc.mean(dim=(1, 2))  # [B, d]
        target = target[:, : D]
        return ((W - target) ** 2).mean()

    W_final = updater.update(state, H, loss_fn=dummy_loss_fn)
    assert W_final.shape == (B, D)


if __name__ == "__main__":
    _sanity_check()


