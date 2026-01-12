"""
ECG models package.

Main modules:
- ecggen: Main ECGGenModel for multi-lead reconstruction
- vcg: VCG pseudo-inverse and geometric projection
- blocks: Neural network building blocks
- heads: Classification heads
"""

from .ecggen import ECGGenModel, ECGGenModelLegacy
from .vcg import VCGPseudoInverse, GeometricLeadProjection, VCGGenerator
from .heads import ClassificationHead

__all__ = [
    'ECGGenModel',
    'ECGGenModelLegacy',
    'VCGPseudoInverse',
    'GeometricLeadProjection',
    'VCGGenerator',
    'ClassificationHead',
]
