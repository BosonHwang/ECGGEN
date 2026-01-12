"""
Temporal Transformer module for ECG signal processing.

This module provides:
- TemporalTransformer: Self-attention based temporal modeling for token sequences

Key Design Principles:
- Lead-independent: Each lead's token sequence is processed independently
- Parameter sharing: All leads share the same Transformer weights
- Temporal only: No cross-lead attention (preserves lead independence)
"""

from typing import Optional

import torch
import torch.nn as nn


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer Encoder for ECG token sequences.
    
    Pipeline Role: Model temporal dependencies within each lead's token sequence.
                   All leads share parameters but are processed independently.
    
    Input Semantics:
        - x: Token sequence [B, L, N, d]
          - B: batch size
          - L: number of leads
          - N: number of tokens (patches)
          - d: token dimension
    
    Output Semantics:
        - out: Enhanced token sequence [B, L, N, d']
          - d' = d_model (may equal d if dimensions match)
    
    Key Design:
        - Lead-independent: No cross-lead attention
        - Parameter sharing: All 12 leads use the same Transformer
        - Only temporal modeling: Attention is over time (N dimension)
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = 'gelu'
    ):
        """
        Args:
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        self.d_model = d_model
        
        # Standard PyTorch TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # Input: [B, N, d]
        )
        
        # Stack of encoder layers
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply temporal self-attention to each lead independently.
        
        Args:
            x: Token sequence [B, L, N, d]
            src_key_padding_mask: Optional mask [B*L, N] for padded positions
        
        Returns:
            out: Enhanced tokens [B, L, N, d']
        
        Shape Flow:
            x: [B, L, N, d]
            -> reshape: [B*L, N, d]
            -> transformer: [B*L, N, d']
            -> reshape: [B, L, N, d']
        """
        B, L, N, d = x.shape
        
        assert d == self.d_model, \
            f"Input dim {d} doesn't match d_model {self.d_model}"
        
        # Reshape: [B, L, N, d] -> [B*L, N, d]
        # Each lead becomes an independent sequence
        x = x.view(B * L, N, d)
        
        # Apply transformer (temporal self-attention)
        # [B*L, N, d] -> [B*L, N, d']
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Reshape back: [B*L, N, d'] -> [B, L, N, d']
        d_out = out.shape[-1]
        out = out.view(B, L, N, d_out)
        
        return out


class TemporalTransformerWithProj(nn.Module):
    """
    Temporal Transformer with input/output projection layers.
    
    Extended version that handles dimension mismatches between
    input token dim and transformer d_model.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim: Input token dimension
            d_model: Transformer hidden dimension
            output_dim: Output token dimension
            nhead: Number of attention heads
            num_layers: Number of layers
            dim_feedforward: FFN hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input projection if needed
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        
        # Transformer
        self.transformer = TemporalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output projection if needed
        self.output_proj = nn.Linear(d_model, output_dim) if d_model != output_dim else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, N, input_dim]
        
        Returns:
            out: [B, L, N, output_dim]
        """
        # Project input
        x = self.input_proj(x)  # [B, L, N, d_model]
        
        # Transformer
        x = self.transformer(x)  # [B, L, N, d_model]
        
        # Project output
        out = self.output_proj(x)  # [B, L, N, output_dim]
        
        return out


class LeadAwareTemporalTransformer(nn.Module):
    """
    Temporal Transformer with optional lead embedding.
    
    Adds learnable lead embeddings to distinguish between leads
    while still processing them independently.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        num_leads: int = 12,
        use_lead_embed: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            d_model: Transformer dimension
            nhead: Number of heads
            num_layers: Number of layers
            dim_feedforward: FFN dimension
            num_leads: Number of leads (for lead embedding)
            use_lead_embed: Whether to add lead embeddings
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.num_leads = num_leads
        self.use_lead_embed = use_lead_embed
        
        if use_lead_embed:
            # Learnable lead embedding: [num_leads, d_model]
            self.lead_embed = nn.Parameter(torch.randn(num_leads, d_model) * 0.02)
        else:
            self.lead_embed = None
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, N, d]
        
        Returns:
            out: [B, L, N, d]
        """
        B, L, N, d = x.shape
        assert d == self.d_model
        
        # Add lead embedding if enabled
        if self.use_lead_embed and self.lead_embed is not None:
            assert L <= self.num_leads
            # lead_embed[:L]: [L, d] -> [1, L, 1, d] for broadcasting
            x = x + self.lead_embed[:L].unsqueeze(0).unsqueeze(2)
        
        # Reshape and process
        x = x.view(B * L, N, d)
        out = self.transformer(x)
        out = out.view(B, L, N, d)
        
        return out

