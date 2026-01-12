"""
Patch Embedding and Unpatch modules for ECG signal processing.

This module provides:
- PatchEmbedding: Split ECG signals into patches and project to token space
- LinearUnpatch: Reconstruct continuous signals from token sequences

Key Design Principles:
- Lead-independent: Each lead is processed independently with shared parameters
- Time-preserving: Output T = Input T (no temporal compression/expansion)
- Simple linear projections without position encoding (added elsewhere if needed)
"""

from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Patch embedding for ECG signals.
    
    Pipeline Role: Convert raw ECG signals [B, L, T] to token sequences [B, L, N, d]
                   where N = T // patch_size.
    
    Input Semantics:
        - x: Raw ECG waveform [B, L, T]
          - B: batch size
          - L: number of leads (typically 12)
          - T: time length (must be divisible by patch_size)
    
    Output Semantics:
        - tokens: Token sequence [B, L, N, d]
          - N: number of patches = T // patch_size
          - d: embedding dimension
          - Each token represents one temporal window
    
    Key Design:
        - Simple linear projection: patch [patch_size] -> token [d]
        - No position encoding (can be added by caller)
        - Lead-independent with shared parameters
    """
    
    def __init__(self, patch_size: int, input_dim: int, embed_dim: int):
        """
        Args:
            patch_size: Number of samples per patch
            input_dim: Input channels (typically 1 for univariate ECG per lead)
            embed_dim: Token embedding dimension d
        """
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Linear projection: [patch_size] -> [embed_dim]
        self.proj = nn.Linear(patch_size, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed ECG signals into patch tokens.
        
        Args:
            x: ECG signals [B, L, T]
        
        Returns:
            tokens: Token sequence [B, L, N, d] where N = T // patch_size
        
        Shape Flow:
            x: [B, L, T]
            -> reshape: [B, L, N, patch_size]
            -> proj: [B, L, N, d]
        """
        B, L, T = x.shape
        
        assert T % self.patch_size == 0, \
            f"T={T} must be divisible by patch_size={self.patch_size}"
        
        N = T // self.patch_size
        
        # Reshape to patches: [B, L, T] -> [B, L, N, patch_size]
        x = x.view(B, L, N, self.patch_size)
        
        # Project to embedding space: [B, L, N, patch_size] -> [B, L, N, d]
        tokens = self.proj(x)
        
        return tokens


class LinearUnpatch(nn.Module):
    """
    Linear unpatch to reconstruct continuous signals from tokens.
    
    Pipeline Role: Convert token sequences [B, L, N, d'] back to continuous
                   ECG signals [B, L, T] where T = N * patch_size.
    
    Input Semantics:
        - x: Token sequence [B, L, N, d']
          - d': may differ from original embed_dim if transformer changed it
    
    Output Semantics:
        - out: Reconstructed ECG [B, L, T]
          - T = N * patch_size (time dimension restored)
    
    Key Design:
        - Linear projection: token [d'] -> patch [patch_size]
        - Time alignment: ensures output T matches expected length
        - Lead-independent with shared parameters
    """
    
    def __init__(self, embed_dim: int, patch_size: int):
        """
        Args:
            embed_dim: Input token dimension d'
            patch_size: Output patch size (samples per patch)
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Linear projection: [embed_dim] -> [patch_size]
        self.proj = nn.Linear(embed_dim, patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct continuous signals from tokens.
        
        Args:
            x: Token sequence [B, L, N, d']
        
        Returns:
            out: Continuous signal [B, L, T] where T = N * patch_size
        
        Shape Flow:
            x: [B, L, N, d']
            -> proj: [B, L, N, patch_size]
            -> reshape: [B, L, T]
        """
        B, L, N, d = x.shape
        
        # Project to patch space: [B, L, N, d'] -> [B, L, N, patch_size]
        x = self.proj(x)
        
        # Reshape to continuous: [B, L, N, patch_size] -> [B, L, T]
        T = N * self.patch_size
        out = x.view(B, L, T)
        
        return out


class PatchEmbeddingWithPosEnc(nn.Module):
    """
    Patch embedding with optional learnable position encoding.
    
    Extended version of PatchEmbedding that adds position information
    to help the model understand temporal order.
    
    Input/Output: Same as PatchEmbedding
    """
    
    def __init__(
        self,
        patch_size: int,
        input_dim: int,
        embed_dim: int,
        max_patches: int = 128,
        use_pos_enc: bool = True
    ):
        """
        Args:
            patch_size: Number of samples per patch
            input_dim: Input channels
            embed_dim: Token embedding dimension
            max_patches: Maximum number of patches (for position encoding)
            use_pos_enc: Whether to add position encoding
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_pos_enc = use_pos_enc
        
        self.proj = nn.Linear(patch_size, embed_dim)
        
        if use_pos_enc:
            # Learnable position encoding: [max_patches, embed_dim]
            self.pos_enc = nn.Parameter(torch.randn(max_patches, embed_dim) * 0.02)
        else:
            self.pos_enc = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, T]
        
        Returns:
            tokens: [B, L, N, d]
        """
        B, L, T = x.shape
        assert T % self.patch_size == 0
        N = T // self.patch_size
        
        # Reshape and project
        x = x.view(B, L, N, self.patch_size)  # [B, L, N, patch_size]
        tokens = self.proj(x)                  # [B, L, N, d]
        
        # Add position encoding if enabled
        if self.use_pos_enc and self.pos_enc is not None:
            assert N <= self.pos_enc.shape[0], \
                f"N={N} exceeds max_patches={self.pos_enc.shape[0]}"
            # pos_enc[:N]: [N, d] -> broadcast to [B, L, N, d]
            tokens = tokens + self.pos_enc[:N].unsqueeze(0).unsqueeze(0)
        
        return tokens

