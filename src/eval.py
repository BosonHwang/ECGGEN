from typing import Optional

import torch


def reconstruction_error(E_hat: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Mean squared error between reconstructed ECG and target."""
    # If time lengths mismatch, center-crop/pad E_hat to match E
    if E_hat.shape[-1] != E.shape[-1]:
        T_src = E_hat.shape[-1]
        T_tgt = E.shape[-1]
        if T_src > T_tgt:
            start = (T_src - T_tgt) // 2
            E_hat = E_hat[..., start:start + T_tgt]
        else:
            pad_left = (T_tgt - T_src) // 2
            pad_right = T_tgt - T_src - pad_left
            E_hat = torch.nn.functional.pad(E_hat, (pad_left, pad_right))
    return torch.mean((E_hat - E) ** 2)


def missing_lead_robustness(model, E: torch.Tensor, drop_prob: float = 0.3) -> torch.Tensor:
    """Evaluate robustness by randomly zeroing a subset of leads at input."""
    mask = (torch.rand_like(E[:, :, :1]) > drop_prob).float()
    E_masked = E * mask
    out = model.forward_gen(E_masked)
    return reconstruction_error(out["E_hat"], E)


def linear_probe_accuracy(head, W: torch.Tensor, y: torch.Tensor) -> float:
    """Compute accuracy of a linear/MLP probe on W; y is [B] integer labels."""
    with torch.no_grad():
        logits = head(W)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == y).float().mean().item()
    return float(acc)


