"""
Neural network building blocks for ECG models.

Modules:
- blocks: Legacy TokenEncoder, LeadProjection, ResidualHead
- patch: PatchEmbedding, LinearUnpatch
- transformer: TemporalTransformer
- decoder: ECGRefinementDecoder
- ttt: TTT (Test-Time Training) modules
"""

from .blocks import TokenEncoder, LeadProjection, ResidualHead
from .patch import PatchEmbedding, LinearUnpatch, PatchEmbeddingWithPosEnc
from .transformer import TemporalTransformer, TemporalTransformerWithProj
from .decoder import ECGRefinementDecoder, DepthwiseConvDecoder


__all__ = [
    # Legacy
    'TokenEncoder',
    'LeadProjection',
    'ResidualHead',
    # New patch modules
    'PatchEmbedding',
    'LinearUnpatch',
    'PatchEmbeddingWithPosEnc',
    # Transformer
    'TemporalTransformer',
    'TemporalTransformerWithProj',
    # Decoder
    'ECGRefinementDecoder',
    'DepthwiseConvDecoder',
]

