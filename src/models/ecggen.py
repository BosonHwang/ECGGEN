"""
ECG Multi-Lead Reconstruction Model.

This module provides:
- ECGGenModel: Main model for VCG-based multi-lead ECG reconstruction
- ECGGenModelLegacy: Previous TTT-based model (kept for backward compatibility)

Architecture Overview (ECGGenModel):
    1) Input: 12-lead ECG [B, 12, T] + visible_indices [B, 3]
    2) Extract Visible Leads: [B, 12, T] -> [B, 3, T]
    3) Patch Embedding: [B, 3, T] -> [B, 3, N, d]
    4) Temporal Transformer: [B, 3, N, d] -> [B, 3, N, d']
    5) Linear Unpatch: [B, 3, N, d'] -> [B, 3, T] (refined visible)
    6) VCG Pseudo-Inverse: [B, 3, T] -> [B, 3, T] (VCG)
    7) Geometric Projection: [B, 3, T] -> [B, 12, T]
    8) Decoder Refinement: [B, 12, T] -> [B, 12, T]

Key Design Constraints:
    - Encoder ONLY processes visible leads (no zeros pollution)
    - Lead-independent encoder (shared parameters, no cross-lead attention)
    - VCG has no learnable parameters (pure geometry)
    - Time alignment: output T == input T
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..data.angle import LEAD_ANGLES_PTBXL_ORDER, get_lead_angles
from .blocks.patch import PatchEmbedding, LinearUnpatch
from .blocks.transformer import TemporalTransformer
from .blocks.decoder import ECGRefinementDecoder
from .vcg import VCGPseudoInverse, GeometricLeadProjection


class ECGGenModel(nn.Module):
    """
    ECG Multi-Lead Reconstruction Model via VCG.
    
    Pipeline Overview:
        1) Input 12-lead ECG + visible_indices
        2) Extract 3 visible leads only
        3) Patch Embedding -> Token sequence (3 leads only)
        4) Temporal Transformer -> Temporal modeling (3 leads only)
        5) Linear Unpatch -> Refined visible signals
        6) VCG Pseudo-Inverse -> Recover 3D VCG from 3 leads
        7) Geometric Projection -> 12 leads
        8) Decoder -> Refined output
    
    Key Constraints:
        - Encoder ONLY processes visible leads (no zeros pollution)
        - Lead-independent: Encoder parameters shared, no cross-lead attention
        - VCG is parameter-free: Pure geometric computation
        - Time alignment: Output T == Input T
    """
    
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 128,
        num_leads: int = 12,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        decoder_hidden: int = 64,
        decoder_layers: int = 2,
        dropout: float = 0.0,
        lead_order: str = 'ptbxl'
    ):
        """
        Args:
            patch_size: Samples per patch for tokenization
            embed_dim: Token embedding dimension
            num_leads: Number of ECG leads (typically 12)
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            dim_feedforward: FFN hidden dimension
            decoder_hidden: Decoder hidden channels
            decoder_layers: Number of decoder conv layers
            dropout: Dropout rate
            lead_order: Lead angle order ('ptbxl' or 'mimic')
        """
        super().__init__()
        
        # Store config
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_leads = num_leads
        
        # Patch Embedding: [B, L, T] -> [B, L, N, d]
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            input_dim=1,
            embed_dim=embed_dim
        )
        
        # Temporal Transformer: [B, L, N, d] -> [B, L, N, d]
        self.transformer = TemporalTransformer(
            d_model=embed_dim,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Linear Unpatch: [B, L, N, d] -> [B, L, T]
        self.unpatch = LinearUnpatch(
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        
        # VCG Pseudo-Inverse: [B, 3, T] -> [B, 3, T]
        self.vcg_inverse = VCGPseudoInverse(eps=1e-6)
        
        # Load 12-lead angles and create geometric projection
        lead_angles = get_lead_angles(order=lead_order, as_tensor=True)
        self.lead_projection = GeometricLeadProjection(lead_angles)
        
        # Decoder: [B, 12, T] -> [B, 12, T]
        self.decoder = ECGRefinementDecoder(
            num_leads=num_leads,
            hidden_dim=decoder_hidden,
            num_layers=decoder_layers
        )
        
        # Register lead angles as buffer
        self.register_buffer('lead_angles', lead_angles)  # [12, 2]
    
    def forward(
        self,
        ecg: torch.Tensor,
        visible_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ECG reconstruction.
        
        Args:
            ecg: Input ECG [B, 12, T] (full ECG, not masked)
            visible_indices: Indices of visible leads [B, 3]
        
        Returns:
            Dictionary containing:
                - 'recon': Reconstructed 12-lead ECG [B, 12, T]
                - 'VCG': Recovered VCG trajectory [B, 3, T]
                - 'geom_proj': Geometric projection before decoder [B, 12, T]
                - 'visible_refined': Refined visible leads [B, 3, T]
        
        Shape Flow:
            ecg: [B, 12, T]
            -> extract visible: [B, 3, T]
            -> patch_embed: [B, 3, N, d]
            -> transformer: [B, 3, N, d]
            -> unpatch: [B, 3, T] (refined visible)
            -> vcg_inverse: [B, 3, T] (VCG)
            -> lead_projection: [B, 12, T]
            -> decoder: [B, 12, T]
        """
        B, L, T = ecg.shape
        assert L == 12, f"Expected 12 leads, got {L}"
        assert T % self.patch_size == 0, \
            f"T={T} must be divisible by patch_size={self.patch_size}"
        
        device = ecg.device
        num_visible = visible_indices.shape[1]  # Typically 3
        
        # 1) Extract visible leads FIRST: [B, 12, T] -> [B, 3, T]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_visible)
        visible_ecg_input = ecg[batch_idx, visible_indices]  # [B, 3, T]
        
        # 2) Patch Embedding (only visible leads): [B, 3, T] -> [B, 3, N, d]
        tokens = self.patch_embed(visible_ecg_input)
        
        # 3) Temporal Transformer (only visible leads): [B, 3, N, d] -> [B, 3, N, d]
        tokens = self.transformer(tokens)
        
        # 4) Linear Unpatch (only visible leads): [B, 3, N, d] -> [B, 3, T]
        visible_refined = self.unpatch(tokens)  # Refined visible leads
        
        # 5) Get visible lead angles for VCG recovery
        # lead_angles: [12, 2], visible_indices: [B, 3]
        visible_angles = self.lead_angles[visible_indices]  # [B, 3, 2]
        visible_theta = visible_angles[..., 0]  # [B, 3]
        visible_phi = visible_angles[..., 1]    # [B, 3]
        
        # 6) VCG Pseudo-Inverse: [B, 3, T] -> [B, 3, T]
        VCG = self.vcg_inverse(visible_refined, visible_theta, visible_phi)
        
        # 7) Geometric Projection: [B, 3, T] -> [B, 12, T]
        geom_proj = self.lead_projection(VCG)
        
        # 8) Decoder Refinement: [B, 12, T] -> [B, 12, T]
        recon = self.decoder(geom_proj)
        
        return {
            'recon': recon,
            'VCG': VCG,
            'geom_proj': geom_proj,
            'visible_refined': visible_refined
        }
    
    def forward_with_intermediates(
        self,
        ecg: torch.Tensor,
        visible_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all intermediate outputs for debugging/analysis.
        
        Returns additional keys:
            - 'visible_ecg_input': Extracted visible leads [B, 3, T]
            - 'tokens': Patch embeddings [B, 3, N, d]
            - 'tokens_out': Transformer output [B, 3, N, d]
            - 'visible_refined': Refined visible leads [B, 3, T]
        """
        B, L, T = ecg.shape
        device = ecg.device
        num_visible = visible_indices.shape[1]
        
        # Extract visible leads first
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_visible)
        visible_ecg_input = ecg[batch_idx, visible_indices]  # [B, 3, T]
        
        # Encoder (only visible leads)
        tokens = self.patch_embed(visible_ecg_input)
        tokens_out = self.transformer(tokens)
        visible_refined = self.unpatch(tokens_out)
        
        # VCG recovery
        visible_angles = self.lead_angles[visible_indices]
        visible_theta = visible_angles[..., 0]
        visible_phi = visible_angles[..., 1]
        
        VCG = self.vcg_inverse(visible_refined, visible_theta, visible_phi)
        geom_proj = self.lead_projection(VCG)
        recon = self.decoder(geom_proj)
        
        return {
            'recon': recon,
            'VCG': VCG,
            'geom_proj': geom_proj,
            'visible_ecg_input': visible_ecg_input,
            'tokens': tokens,
            'tokens_out': tokens_out,
            'visible_refined': visible_refined
        }


# =============================================================================
# Legacy Model (for backward compatibility)
# =============================================================================

class ECGGenModelLegacy(nn.Module):
    """
    Legacy ECG rendering model with TTT-based state estimation.
    
    DEPRECATED: Use ECGGenModel for the new reconstruction task.
    Kept for backward compatibility with existing checkpoints.

    Pipeline:
        1) ECG [B, L, T]
        2) Tokenizer -> X [B, L, N, d]
        3) TokenEncoder -> H [B, L*N, d_model]
        4) TTT -> W_final [B, D]
        5) VCG -> V [B, 3, T']
        6) LeadProjection -> E_hat' [B, L, T']
        7) Optional Residual correction -> E_hat
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
        
        # Import here to avoid circular imports
        from ..data.tokenizer import ECGTokenizer
        from .blocks.blocks import TokenEncoder, LeadProjection, ResidualHead
        from .blocks.ttt import FastState, TTTUpdater
        from .vcg import VCGGenerator
        
        self.num_leads = num_leads
        self.state_dim = state_dim

        self.tokenizer = ECGTokenizer(beat_len=beat_len, token_dim=token_dim, mode=tokenizer_mode, fs=tokenizer_fs)
        self.encoder = TokenEncoder(token_dim=token_dim, d_model=d_model, num_layers=2, num_heads=4, dropout=0.0)
        self.ttt_updater = TTTUpdater(step_size=ttt_step_size, smooth_lambda=ttt_smooth_lambda, chunk_size=ttt_chunk_size)
        self.vcg = VCGGenerator(state_dim=state_dim, basis_k=vcg_basis_k, time_len=vcg_time_len)
        self.proj = LeadProjection(num_leads=num_leads)
        self.residual = ResidualHead(num_leads=num_leads) if residual_enabled else None

    def forward_gen(self, ecg: torch.Tensor, reg_cfg: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        from .blocks.ttt import FastState
        
        assert ecg.ndim == 3 and ecg.shape[1] == self.num_leads, "ecg must be [B, L, T]"
        B = ecg.shape[0]
        X, meta = self.tokenizer(ecg)
        assert X.ndim == 4, "tokenizer must output [B, L, N, d]"
        N = X.shape[2]
        H = self.encoder(X)
        d_model = H.shape[-1]
        assert H.shape[1] == self.num_leads * N, "encoder output shape mismatch with L*N"
        H_beats = H.view(B, self.num_leads, N, d_model).mean(dim=1)

        state = FastState(mode="vector", batch_size=B, dim=self.state_dim).to(H.device)
        W_final = self.ttt_updater.update(state, H_beats, loss_fn=None)

        V = self.vcg(W_final)
        E_base = self.proj(V)
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
        from .blocks.ttt import FastState
        
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
