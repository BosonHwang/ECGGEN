"""
Lead angle definitions and reorder utilities for ECG multi-lead reconstruction.

This module provides:
- Lead angle definitions (theta, phi) for 12-lead ECG in PTBXL order
- Lead direction vector computation from angles
- Dataset-specific lead reordering functions (MIMIC <-> PTBXL)

Coordinate System:
- theta: elevation angle (from xy-plane)
- phi: azimuth angle (in xy-plane, from x-axis)
- Direction vector: u = [cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta)]
"""

from typing import Dict, List, Union
import numpy as np
import torch


# =============================================================================
# Lead Angle Definitions
# =============================================================================

# Lead angles in PTBXL order: [I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF]
# Format: [theta, phi] where theta is elevation, phi is azimuth
LEAD_ANGLES_PTBXL_ORDER = np.array([
    [np.pi / 2, np.pi / 2],           # I
    [np.pi * 5 / 6, np.pi / 2],       # II
    [np.pi / 2, -np.pi / 18],         # V1
    [np.pi / 2, np.pi / 18],          # V2
    [np.pi * (19 / 36), np.pi / 12],  # V3
    [np.pi * (11 / 20), np.pi / 6],   # V4
    [np.pi * (16 / 30), np.pi / 3],   # V5
    [np.pi * (16 / 30), np.pi / 2],   # V6
    [np.pi * (5 / 6), -np.pi / 2],    # III
    [np.pi * (1 / 3), -np.pi / 2],    # aVR
    [np.pi * (1 / 3), np.pi / 2],     # aVL
    [np.pi * 1, np.pi / 2],           # aVF
], dtype=np.float32)

# Lead names for each order
LEAD_NAMES_PTBXL = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'III', 'aVR', 'aVL', 'aVF']
LEAD_NAMES_MIMIC = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# =============================================================================
# Lead Order Mappings
# =============================================================================

# MIMIC standard order: [I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6]
# PTBXL order:          [I, II, V1, V2, V3, V4, V5, V6, III, aVR, aVL, aVF]

# Index mapping: MIMIC[i] -> PTBXL[MIMIC_TO_PTBXL_ORDER[i]]
# To convert MIMIC-ordered data to PTBXL order: new_data = data[MIMIC_TO_PTBXL_ORDER]
MIMIC_TO_PTBXL_ORDER = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]

# Inverse: PTBXL[i] -> MIMIC[PTBXL_TO_MIMIC_ORDER[i]]
PTBXL_TO_MIMIC_ORDER = [0, 1, 8, 9, 10, 11, 2, 3, 4, 5, 6, 7]

# Mapping dictionary for flexible reordering
ORDER_MAPPINGS: Dict[str, Dict[str, List[int]]] = {
    'mimic': {
        'ptbxl': MIMIC_TO_PTBXL_ORDER,
    },
    'ptbxl': {
        'mimic': PTBXL_TO_MIMIC_ORDER,
    }
}


# =============================================================================
# Direction Vector Computation
# =============================================================================

def compute_lead_directions(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Compute 3D direction vectors from lead angles.
    
    Pipeline Role: Convert angular representation to Cartesian direction vectors
                   for VCG projection/reconstruction.
    
    Input Semantics:
        - theta: elevation angle [L] or [B, L] - angle from xy-plane
        - phi: azimuth angle [L] or [B, L] - angle in xy-plane from x-axis
    
    Output Semantics:
        - Direction vectors u [L, 3] or [B, L, 3]
        - u = [cos(theta)*cos(phi), cos(theta)*sin(phi), sin(theta)]
    
    Args:
        theta: Elevation angles, shape [L] or [B, L]
        phi: Azimuth angles, shape [L] or [B, L]
    
    Returns:
        Direction vectors, shape [L, 3] or [B, L, 3]
    """
    # u_x = cos(theta) * cos(phi)  [L] or [B, L]
    u_x = torch.cos(theta) * torch.cos(phi)
    # u_y = cos(theta) * sin(phi)  [L] or [B, L]
    u_y = torch.cos(theta) * torch.sin(phi)
    # u_z = sin(theta)  [L] or [B, L]
    u_z = torch.sin(theta)
    # Stack to get [L, 3] or [B, L, 3]
    return torch.stack([u_x, u_y, u_z], dim=-1)


def compute_lead_directions_np(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    NumPy version of compute_lead_directions.
    
    Args:
        theta: Elevation angles, shape [L] or [B, L]
        phi: Azimuth angles, shape [L] or [B, L]
    
    Returns:
        Direction vectors, shape [L, 3] or [B, L, 3]
    """
    u_x = np.cos(theta) * np.cos(phi)
    u_y = np.cos(theta) * np.sin(phi)
    u_z = np.sin(theta)
    return np.stack([u_x, u_y, u_z], axis=-1)


# =============================================================================
# Lead Reordering Functions
# =============================================================================

def reorder_leads(
    ecg: Union[np.ndarray, torch.Tensor],
    source: str,
    target: str
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reorder ECG leads from source dataset order to target order.
    
    Pipeline Role: Standardize lead ordering across different datasets
                   before model processing.
    
    Input Semantics:
        - ecg: ECG signal with leads in source dataset order
        - source: Source dataset name ('mimic', 'ptbxl')
        - target: Target order name ('mimic', 'ptbxl')
    
    Output Semantics:
        - ECG signal with leads reordered to target order
    
    Args:
        ecg: ECG signal, shape [B, 12, T] or [12, T]
        source: Source dataset name (e.g., 'mimic', 'ptbxl')
        target: Target order name (e.g., 'ptbxl', 'mimic')
    
    Returns:
        Reordered ECG with same shape as input
    
    Raises:
        ValueError: If source or target is not recognized
    """
    if source == target:
        return ecg
    
    source_lower = source.lower()
    target_lower = target.lower()
    
    if source_lower not in ORDER_MAPPINGS:
        raise ValueError(f"Unknown source order: {source}. Available: {list(ORDER_MAPPINGS.keys())}")
    if target_lower not in ORDER_MAPPINGS[source_lower]:
        raise ValueError(f"No mapping from {source} to {target}")
    
    order = ORDER_MAPPINGS[source_lower][target_lower]
    
    # Handle both numpy and torch
    if isinstance(ecg, np.ndarray):
        if ecg.ndim == 2:
            # [12, T] -> [12, T]
            return ecg[order]
        elif ecg.ndim == 3:
            # [B, 12, T] -> [B, 12, T]
            return ecg[:, order, :]
        else:
            raise ValueError(f"ECG must be 2D or 3D, got {ecg.ndim}D")
    else:
        # torch.Tensor
        if ecg.ndim == 2:
            return ecg[order]
        elif ecg.ndim == 3:
            return ecg[:, order, :]
        else:
            raise ValueError(f"ECG must be 2D or 3D, got {ecg.ndim}D")


def reorder_angles(
    angles: Union[np.ndarray, torch.Tensor],
    source: str,
    target: str
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reorder lead angles from source order to target order.
    
    Args:
        angles: Lead angles, shape [12, 2] or [B, 12, 2]
        source: Source order name
        target: Target order name
    
    Returns:
        Reordered angles with same shape as input
    """
    if source == target:
        return angles
    
    source_lower = source.lower()
    target_lower = target.lower()
    
    if source_lower not in ORDER_MAPPINGS:
        raise ValueError(f"Unknown source order: {source}")
    if target_lower not in ORDER_MAPPINGS[source_lower]:
        raise ValueError(f"No mapping from {source} to {target}")
    
    order = ORDER_MAPPINGS[source_lower][target_lower]
    
    if isinstance(angles, np.ndarray):
        if angles.ndim == 2:
            return angles[order]
        elif angles.ndim == 3:
            return angles[:, order, :]
        else:
            raise ValueError(f"Angles must be 2D or 3D, got {angles.ndim}D")
    else:
        if angles.ndim == 2:
            return angles[order]
        elif angles.ndim == 3:
            return angles[:, order, :]
        else:
            raise ValueError(f"Angles must be 2D or 3D, got {angles.ndim}D")


# =============================================================================
# Convenience Functions
# =============================================================================

def get_lead_angles(order: str = 'ptbxl', as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Get lead angles in specified order.
    
    Args:
        order: Lead order ('ptbxl' or 'mimic')
        as_tensor: If True, return torch.Tensor; else np.ndarray
    
    Returns:
        Lead angles, shape [12, 2] with [theta, phi] per lead
    """
    if order.lower() == 'ptbxl':
        angles = LEAD_ANGLES_PTBXL_ORDER.copy()
    elif order.lower() == 'mimic':
        angles = reorder_angles(LEAD_ANGLES_PTBXL_ORDER, 'ptbxl', 'mimic')
    else:
        raise ValueError(f"Unknown order: {order}")
    
    if as_tensor:
        return torch.from_numpy(angles)
    return angles


def get_lead_directions(order: str = 'ptbxl', as_tensor: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """
    Get lead direction vectors in specified order.
    
    Args:
        order: Lead order ('ptbxl' or 'mimic')
        as_tensor: If True, return torch.Tensor; else np.ndarray
    
    Returns:
        Lead direction vectors, shape [12, 3]
    """
    angles = get_lead_angles(order, as_tensor=False)
    theta = angles[:, 0]  # [12]
    phi = angles[:, 1]    # [12]
    directions = compute_lead_directions_np(theta, phi)  # [12, 3]
    
    if as_tensor:
        return torch.from_numpy(directions.astype(np.float32))
    return directions.astype(np.float32)


def get_visible_lead_angles(
    lead_angles: torch.Tensor,
    visible_indices: torch.Tensor
) -> torch.Tensor:
    """
    Extract angles for visible leads given indices.
    
    Args:
        lead_angles: All lead angles [12, 2]
        visible_indices: Indices of visible leads [B, K]
    
    Returns:
        Visible lead angles [B, K, 2]
    """
    # lead_angles: [12, 2]
    # visible_indices: [B, K]
    # Output: [B, K, 2]
    return lead_angles[visible_indices]  # Advanced indexing

