from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..data.tokenizer import ECGTokenizer
from .blocks import TokenEncoder, LeadProjection, ResidualHead
from .ttt import FastState, TTTUpdater
from .vcg import VCGGenerator


class ECGGenModel(nn.Module):
    """End-to-end ECG rendering model with TTT-based state estimation.

    Pipeline:
        1) ECG [B, L, T]
        2) Tokenizer -> X [B, L, N, d]
        3) TokenEncoder -> H [B, L*N, d_model]
        4) TTT -> W_final [B, D]
        5) VCG -> V [B, 3, T’]
        6) LeadProjection -> E_hat’ [B, L, T’]
        7) Optional Residual correction -> E_hat

    Returns:
        dict with intermediate tensors for analysis, not losses.
    """

    def __init__(
        self,
        num_leads: int = 12,
        token_dim: int = 64,
        beat_len: int = 128,
        tokenizer_mode: str = "equidistant",
        tokenizer_fs: int = 100,
        d_model: int = 128,
        state_dim: int = 64,
        vcg_basis_k: int = 32,
        vcg_time_len: int = 256,
        residual_enabled: bool = True,
        ttt_step_size: float = 1e-2,
        ttt_smooth_lambda: float = 1e-2,
        ttt_chunk_size: int = 4,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.state_dim = state_dim

        self.tokenizer = ECGTokenizer(beat_len=beat_len, token_dim=token_dim, mode=tokenizer_mode, fs=tokenizer_fs)
        self.encoder = TokenEncoder(token_dim=token_dim, d_model=d_model, num_layers=2, num_heads=4, dropout=0.0)
        self.ttt_updater = TTTUpdater(step_size=ttt_step_size, smooth_lambda=ttt_smooth_lambda, chunk_size=ttt_chunk_size)
        self.vcg = VCGGenerator(state_dim=state_dim, basis_k=vcg_basis_k, time_len=vcg_time_len)
        self.proj = LeadProjection(num_leads=num_leads)
        self.residual = ResidualHead(num_leads=num_leads) if residual_enabled else None

    def forward_gen(self, ecg: torch.Tensor, reg_cfg: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        assert ecg.ndim == 3 and ecg.shape[1] == self.num_leads, "ecg must be [B, L, T]"
        B = ecg.shape[0]
        X, meta = self.tokenizer(ecg)  # [B, L, N, d]
        assert X.ndim == 4, "tokenizer must output [B, L, N, d]"
        N = X.shape[2]
        H = self.encoder(X)  # [B, L*N, d_model]
        d_model = H.shape[-1]
        assert H.shape[1] == self.num_leads * N, "encoder output shape mismatch with L*N"
        # Reshape to beat-major for TTT and aggregate over leads
        H_beats = H.view(B, self.num_leads, N, d_model).mean(dim=1)  # [B, N, d_model]

        # Initialize state for this batch
        state = FastState(mode="vector", batch_size=B, dim=self.state_dim).to(H.device)
        W_final = self.ttt_updater.update(state, H_beats, loss_fn=None)  # [B, D]

        V = self.vcg(W_final)  # [B, 3, T’]
        E_base = self.proj(V)  # [B, L, T’]
        E_hat = E_base
        if self.residual is not None:
            E_hat = E_base + self.residual(E_base)

        reg_terms: Dict[str, torch.Tensor] = {}
        if reg_cfg:
            if reg_cfg.get("smoothness", 0.0) > 0:
                reg_terms["smoothness"] = reg_cfg["smoothness"] * self.vcg.regularizer_smoothness(V)
            if reg_cfg.get("energy", 0.0) > 0:
                reg_terms["energy"] = reg_cfg["energy"] * self.vcg.regularizer_energy(V)
            if reg_cfg.get("loop_closure", 0.0) > 0:
                reg_terms["loop_closure"] = reg_cfg["loop_closure"] * self.vcg.regularizer_loop_closure(V)

        return {
            "X": X,
            "meta": meta,
            "H": H,
            "H_beats": H_beats,
            "W": W_final,
            "V": V,
            "E_base": E_base,
            "E_hat": E_hat,
            "reg_terms": reg_terms,
        }

    def forward_cls(self, ecg: torch.Tensor) -> torch.Tensor:
        """Return only the representation W for classification heads."""
        assert ecg.ndim == 3 and ecg.shape[1] == self.num_leads
        B = ecg.shape[0]
        X, _ = self.tokenizer(ecg)
        H = self.encoder(X)
        N = X.shape[2]
        d_model = H.shape[-1]
        H_beats = H.view(B, self.num_leads, N, d_model).mean(dim=1)
        state = FastState(mode="vector", batch_size=B, dim=self.state_dim).to(H.device)
        W_final = self.ttt_updater.update(state, H_beats, loss_fn=None)
        return W_final


