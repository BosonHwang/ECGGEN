"""
Training utilities for ECG multi-lead reconstruction.

This module provides:
- Training loop for the new ECGGenModel (VCG-based reconstruction)
- Legacy training support for ECGGenModelLegacy (TTT-based)
- Model building, checkpoint management, and logging

Key Training Logic:
1. Random mask: Select 3 visible leads, mask 9 others
2. Apply mask: Zero out masked leads in input
3. Forward: Reconstruct all 12 leads from 3 visible
4. Loss: MSE only on masked leads (not visible ones)
5. Best model: Determined by validation loss (not training loss)
"""

import argparse
import json
import logging
import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecggen.src.utils.config import Config, add_cli_overrides, apply_overrides, load_config
from ecggen.src.utils.run_id import RunIdSpec, ensure_run_dirs
from ecggen.src.data.pipeline import make_dataloader, make_dataloaders
from ecggen.src.models.ecggen import ECGGenModel, ECGGenModelLegacy
from ecggen.src.models.heads import ClassificationHead
from ecggen.src.models.utils.loss import random_lead_mask, masked_reconstruction_loss
from ecggen.src.eval import reconstruction_error


def _setup_logger(log_dir: str) -> logging.Logger:
    """Create a logger that writes to log_dir/train.log and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("ecggen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, "train.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _count_parameters(model: torch.nn.Module) -> int:
    """Count total parameters (all, including non-trainable)."""
    return sum(p.numel() for p in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ecggen trainer entry.")
    parser = add_cli_overrides(parser)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "train_legacy", "eval"])
    return parser.parse_args()


def build_run_id(cfg: Config) -> str:
    mapping = cfg.run.get("id_map", {"m": {}, "s": {}, "k": {}})
    codes = (cfg.run.get("m", "m0"), cfg.run.get("s", "s0"), cfg.run.get("k", "k0"))
    spec = RunIdSpec(mapping=mapping, codes=codes)
    return spec.build()


def build_model(cfg: Config) -> ECGGenModel:
    """Build new ECGGenModel for VCG-based reconstruction."""
    model_cfg = cfg.model
    
    model = ECGGenModel(
        patch_size=int(model_cfg.get("patch_size", 16)),
        embed_dim=int(model_cfg.get("embed_dim", 128)),
        num_leads=int(model_cfg.get("num_leads", 12)),
        nhead=int(model_cfg.get("nhead", 8)),
        num_encoder_layers=int(model_cfg.get("num_encoder_layers", 4)),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
        decoder_hidden=int(model_cfg.get("decoder_hidden", 64)),
        decoder_layers=int(model_cfg.get("decoder_layers", 2)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        lead_order=str(model_cfg.get("lead_order", "ptbxl"))
    )
    return model


def build_model_legacy(cfg: Config) -> ECGGenModelLegacy:
    """Build legacy ECGGenModelLegacy for TTT-based training."""
    model_cfg = cfg.model
    data_cfg = cfg.data
    
    model = ECGGenModelLegacy(
        num_leads=int(model_cfg.get("num_leads", 12)),
        token_dim=int(model_cfg.get("token_dim", 64)),
        beat_len=int(model_cfg.get("beat_len", 128)),
        tokenizer_mode=str(model_cfg.get("tokenizer_mode", "equidistant")),
        tokenizer_fs=int(data_cfg.get("fs", 100)),
        d_model=int(model_cfg.get("d_model", 128)),
        state_dim=int(model_cfg.get("state_dim", 64)),
        vcg_basis_k=int(model_cfg.get("vcg_basis_k", 32)),
        vcg_time_len=int(model_cfg.get("vcg_time_len", 256)),
        residual_enabled=bool(model_cfg.get("residual_enabled", True)),
        ttt_step_size=float(model_cfg.get("ttt_step_size", 1e-2)),
        ttt_smooth_lambda=float(model_cfg.get("ttt_smooth_lambda", 1e-2)),
        ttt_chunk_size=int(model_cfg.get("ttt_chunk_size", 4)),
    )
    return model


def maybe_build_head(cfg: Config, state_dim: int) -> Optional[ClassificationHead]:
    head_cfg = cfg.model.get("head", None)
    if not head_cfg:
        return None
    head_type = head_cfg.get("type", "linear")
    num_classes = int(head_cfg.get("num_classes", 2))
    hidden = int(head_cfg.get("hidden", 128))
    return ClassificationHead(kind=head_type, in_dim=state_dim, num_classes=num_classes, hidden=hidden)


def prepare_dirs(cfg: Config, run_id: str) -> Dict[str, str]:
    ckpt_root = cfg.run.get("checkpoint_root", "./checkpoints")
    log_root = cfg.run.get("log_root", "./logs")
    ckpt_run = ensure_run_dirs(ckpt_root, run_id)
    log_run = ensure_run_dirs(log_root, run_id)
    return {"ckpt_run": ckpt_run, "log_run": log_run}


def train_step(
    model: ECGGenModel,
    ecg: torch.Tensor,
    num_visible: int = 3,
    device: torch.device = None
) -> Dict[str, torch.Tensor]:
    """
    Single training step for ECG reconstruction.
    
    Pipeline:
        1) Generate random mask (select 3 visible leads)
        2) Forward pass: model extracts visible leads internally
        3) Compute loss only on masked leads
    
    Note: The model extracts visible leads internally from the ORIGINAL ecg.
    This avoids polluting the encoder with zeros from masked leads.
    
    Args:
        model: ECGGenModel instance
        ecg: Original ECG [B, 12, T] (full, not masked)
        num_visible: Number of visible leads (default 3)
        device: Target device
    
    Returns:
        Dictionary with:
            - 'loss': Masked reconstruction loss
            - 'recon': Reconstructed ECG [B, 12, T]
            - 'mask': Boolean mask [B, 12]
            - 'visible_indices': [B, 3]
    """
    if device is None:
        device = ecg.device
    
    B = ecg.shape[0]
    
    # 1) Generate random mask (for loss computation)
    visible_indices, mask = random_lead_mask(
        batch_size=B,
        num_leads=12,
        num_visible=num_visible,
        device=device
    )
    
    # 2) Forward pass with ORIGINAL ecg (model extracts visible leads internally)
    # No need to apply mask - model only processes visible leads
    outputs = model(ecg, visible_indices)
    recon = outputs['recon']
    
    # 3) Compute loss only on masked leads
    loss = masked_reconstruction_loss(recon, ecg, mask)
    
    return {
        'loss': loss,
        'recon': recon,
        'mask': mask,
        'visible_indices': visible_indices,
        'VCG': outputs.get('VCG'),
        'geom_proj': outputs.get('geom_proj'),
        'visible_refined': outputs.get('visible_refined')
    }


@torch.no_grad()
def validate(
    model: ECGGenModel,
    val_loader: DataLoader,
    num_visible: int,
    device: torch.device
) -> float:
    """
    Evaluate model on full validation set.
    
    Args:
        model: ECGGenModel instance
        val_loader: Validation DataLoader
        num_visible: Number of visible leads
        device: Target device
    
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in val_loader:
        ecg = batch["ecg"].to(device)
        step_out = train_step(model, ecg, num_visible=num_visible, device=device)
        total_loss += step_out['loss'].item()
        num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def mini_validate(
    model: ECGGenModel,
    val_loader: DataLoader,
    num_visible: int,
    device: torch.device,
    mini_ratio: float = 0.2
) -> float:
    """
    Evaluate model on a random subset (mini) of validation set.
    
    This is faster than full validation and useful for frequent checkpointing.
    
    Args:
        model: ECGGenModel instance
        val_loader: Validation DataLoader
        num_visible: Number of visible leads
        device: Target device
        mini_ratio: Fraction of validation batches to use (default 0.2 = 20%)
    
    Returns:
        Average validation loss on the mini subset
    """
    import random
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Calculate how many batches to sample
    total_batches = len(val_loader)
    num_sample_batches = max(1, int(total_batches * mini_ratio))
    
    # Randomly sample batch indices
    all_indices = list(range(total_batches))
    random.shuffle(all_indices)
    selected_indices = set(all_indices[:num_sample_batches])
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx not in selected_indices:
            continue
            
        ecg = batch["ecg"].to(device)
        step_out = train_step(model, ecg, num_visible=num_visible, device=device)
        total_loss += step_out['loss'].item()
        num_batches += 1
        
        # Early exit once we've processed enough batches
        if num_batches >= num_sample_batches:
            break
    
    model.train()
    return total_loss / max(num_batches, 1)


def run_train(cfg: Config) -> None:
    """
    Main training loop for ECGGenModel (VCG-based reconstruction).
    
    Training Strategy:
        - Random mask: Each batch, randomly select 3 visible leads
        - Zero masked leads in input
        - Supervise only masked leads in loss
        - Best model determined by validation loss (evaluated each epoch)
    """
    force_cuda = bool(cfg.run.get("force_cuda", False))
    if force_cuda and not torch.cuda.is_available():
        raise RuntimeError("force_cuda is True but CUDA is not available.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        torch.cuda.set_device(idx)
        device_name = torch.cuda.get_device_name(idx)
    else:
        device_name = "cpu"

    run_id = build_run_id(cfg)
    dirs = prepare_dirs(cfg, run_id)
    logger = _setup_logger(dirs["log_run"])
    logger.info(f"Using device: {device} ({device_name})")
    logger.info(f"run_id={run_id} | ckpt_dir={dirs['ckpt_run']} | log_dir={dirs['log_run']}")

    # Build model
    model = build_model(cfg).to(device)
    
    # DataLoaders (train and val)
    train_loader, val_loader = make_dataloaders(cfg=cfg.raw)
    
    # Training config
    max_steps = int(cfg.train.get("max_steps", 100000))
    epochs = int(cfg.train.get("epochs", 100))
    lr = float(cfg.train.get("lr", 1e-3))
    num_visible = int(cfg.model.get("num_visible_leads", 3))
    log_every = int(cfg.run.get("log_every", 100))
    val_every = int(cfg.run.get("val_every", 1000))
    mini_val_ratio = float(cfg.run.get("mini_val_ratio", 0.2))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Tracking
    best_val_loss = float("inf")
    best_path: Optional[str] = None
    best_step = -1
    best_epoch = -1
    
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    steps_per_epoch = len(train_loader)
    
    logger.info(
        f"params_total={_count_parameters(model)} "
        f"train_size={train_size} val_size={val_size} "
        f"batch_size={cfg.train.get('batch_size', None)} "
        f"epochs={epochs} max_steps={max_steps} log_every={log_every} "
        f"val_every={val_every} mini_val_ratio={mini_val_ratio} "
        f"steps_per_epoch={steps_per_epoch} num_visible={num_visible}"
    )

    global_step = 0
    model.train()
    
    # Outer loop: Epochs
    for epoch in range(epochs):
        epoch_loss_sum = 0.0
        epoch_steps = 0
        
        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, 
                    desc=f"Epoch {epoch}/{epochs}", leave=True)
        
        for step_in_epoch, batch in pbar:
            if global_step >= max_steps:
                break

            ecg = batch["ecg"].to(device)  # [B, 12, T]
            
            # Single training step
            step_out = train_step(model, ecg, num_visible=num_visible, device=device)
            loss = step_out['loss']

            if torch.isnan(loss):
                logger.error(f"NaN loss at epoch={epoch} step={step_in_epoch} global_step={global_step}")
                raise ValueError("NaN loss encountered")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss_sum += loss.item()
            epoch_steps += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "best_val": f"{best_val_loss:.4f}"})

            # Periodic logging
            if (global_step % log_every == 0):
                logger.info(f"Epoch [{epoch}/{epochs}] GlobalStep [{global_step}] TrainLoss: {loss.item():.6f}")

            # Mini validation every val_every steps (for frequent checkpointing)
            if (global_step > 0) and (global_step % val_every == 0):
                mini_val_loss = mini_validate(model, val_loader, num_visible, device, mini_val_ratio)
                logger.info(
                    f"Epoch [{epoch}/{epochs}] GlobalStep [{global_step}] "
                    f"MiniValLoss: {mini_val_loss:.6f} (best: {best_val_loss:.6f})"
                )
                
                # Check if this is the best model (based on mini validation loss)
                if mini_val_loss < best_val_loss:
                    # Delete previous best model if exists
                    if best_path is not None and os.path.exists(best_path):
                        try:
                            os.remove(best_path)
                            logger.info(f"Deleted previous best: {os.path.basename(best_path)}")
                        except OSError as e:
                            logger.warning(f"Failed to delete previous best: {e}")
                    
                    best_val_loss = mini_val_loss
                    best_step = global_step
                    best_epoch = epoch
                    best_path = os.path.join(dirs["ckpt_run"], f"best_{run_id}_step{global_step}.pt")
                    
                    # Save new best model
                    torch.save(model.state_dict(), best_path)
                    logger.info(f"New best at step {global_step}: val_loss={best_val_loss:.6f} -> {os.path.basename(best_path)}")
                    
                    # Update best.json metadata
                    with open(os.path.join(dirs["ckpt_run"], "best.json"), "w", encoding="utf-8") as f:
                        json.dump({
                            "run_id": run_id, 
                            "best_step": best_step,
                            "best_epoch": best_epoch,
                            "best_val_loss": best_val_loss,
                            "best_path": best_path
                        }, f)
                
                # Ensure model is back in train mode
                model.train()

            global_step += 1
            
        if global_step >= max_steps:
            break
        
        # End of epoch: Validate and check for best model
        avg_train_loss = epoch_loss_sum / max(epoch_steps, 1)
        val_loss = validate(model, val_loader, num_visible, device)
        
        logger.info(
            f"Epoch [{epoch}/{epochs}] completed | "
            f"TrainLoss: {avg_train_loss:.6f} | ValLoss: {val_loss:.6f} | "
            f"BestValLoss: {best_val_loss:.6f}"
        )
        
        # Check if this is the best model (based on validation loss)
        if val_loss < best_val_loss:
            # Delete previous best model if exists
            if best_path is not None and os.path.exists(best_path):
                try:
                    os.remove(best_path)
                    logger.info(f"Deleted previous best: {os.path.basename(best_path)}")
                except OSError as e:
                    logger.warning(f"Failed to delete previous best: {e}")
            
            best_val_loss = val_loss
            best_step = global_step
            best_epoch = epoch
            best_path = os.path.join(dirs["ckpt_run"], f"best_{run_id}_epoch{epoch}.pt")
            
            # Save new best model
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best at epoch {epoch}: val_loss={best_val_loss:.6f} -> {os.path.basename(best_path)}")
            
            # Update best.json metadata
            with open(os.path.join(dirs["ckpt_run"], "best.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": run_id, 
                    "best_step": best_step,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "best_train_loss": avg_train_loss,
                    "best_path": best_path
                }, f)

    logger.info(f"Training completed. Total global_steps: {global_step}")
    logger.info(f"Best model: epoch={best_epoch}, val_loss={best_val_loss:.6f}")


def run_train_legacy(cfg: Config) -> None:
    """
    Legacy training loop for ECGGenModelLegacy (TTT-based).
    
    Kept for backward compatibility.
    """
    force_cuda = bool(cfg.run.get("force_cuda", False))
    if force_cuda and not torch.cuda.is_available():
        raise RuntimeError("force_cuda is True but CUDA is not available.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        torch.cuda.set_device(idx)
        device_name = torch.cuda.get_device_name(idx)
    else:
        device_name = "cpu"

    run_id = build_run_id(cfg)
    dirs = prepare_dirs(cfg, run_id)
    logger = _setup_logger(dirs["log_run"])
    logger.info(f"Using device: {device} ({device_name})")
    logger.info(f"run_id={run_id} | ckpt_dir={dirs['ckpt_run']} | log_dir={dirs['log_run']}")

    model = build_model_legacy(cfg).to(device)
    head = maybe_build_head(cfg, model.state_dim)
    if head is not None:
        head = head.to(device)
    train_loader = make_dataloader(cfg=cfg.raw, split="train")
    max_steps = int(cfg.train.get("max_steps", 5))
    epochs = int(cfg.train.get("epochs", 5))
    lr = float(cfg.train.get("lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reg_cfg = cfg.model.get("reg", {})
    log_every = int(cfg.run.get("log_every", 100))
    best_loss = float("inf")
    best_path: Optional[str] = None
    best_step = -1
    train_size = len(train_loader.dataset)
    steps_per_epoch = len(train_loader)
    
    logger.info(
        f"params_total={_count_parameters(model)} "
        f"train_size={train_size} batch_size={cfg.train.get('batch_size', None)} "
        f"epochs={epochs} max_steps={max_steps} log_every={log_every} steps_per_epoch={steps_per_epoch}"
    )

    global_step = 0
    model.train()
    
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for step_in_epoch, batch in pbar:
            if global_step >= max_steps:
                break

            ecg = batch["ecg"].to(device)
            out = model.forward_gen(ecg, reg_cfg=reg_cfg)
            recon = out["E_hat"]
            target = ecg
            
            # Match time lengths if needed
            if recon.shape[-1] != target.shape[-1]:
                T_src, T_tgt = recon.shape[-1], target.shape[-1]
                if T_src > T_tgt:
                    start = (T_src - T_tgt) // 2
                    recon_use = recon[..., start:start + T_tgt]
                else:
                    pad_left = (T_tgt - T_src) // 2
                    pad_right = T_tgt - T_src - pad_left
                    recon_use = torch.nn.functional.pad(recon, (pad_left, pad_right))
            else:
                recon_use = recon

            loss_recon = torch.mean((recon_use - target) ** 2)
            loss = loss_recon
            if out["reg_terms"]:
                loss = loss + sum(out["reg_terms"].values())

            if torch.isnan(loss):
                logger.error(f"NaN loss detected at epoch={epoch} step={step_in_epoch} global_step={global_step}")
                for k, v in out.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        logger.error(f"  NaN found in output: {k}")
                if torch.isnan(ecg).any():
                    logger.error("  NaN found in input ECG")
                raise ValueError("NaN loss encountered")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": f"{loss_recon.item():.4f}", "best": f"{best_loss:.4f}"})

            is_final_step = (global_step == max_steps - 1) or (epoch == epochs - 1 and step_in_epoch == steps_per_epoch - 1)
            if (global_step % log_every == 0) or is_final_step:
                logger.info(f"Epoch [{epoch}/{epochs}] GlobalStep [{global_step}] Loss: {loss_recon.item():.6f}")

            if loss_recon.item() < best_loss:
                # Delete previous best model if exists
                if best_path is not None and os.path.exists(best_path):
                    try:
                        os.remove(best_path)
                        logger.info(f"Deleted previous best: {os.path.basename(best_path)}")
                    except OSError as e:
                        logger.warning(f"Failed to delete previous best: {e}")
                
                best_loss = float(loss_recon.item())
                best_step = global_step
                best_path = os.path.join(dirs["ckpt_run"], f"best_{run_id}_step{best_step}.pt")
                
                torch.save(model.state_dict(), best_path)
                logger.info(f"New best at step {best_step}: loss={best_loss:.6f} -> {os.path.basename(best_path)}")
                
                with open(os.path.join(dirs["ckpt_run"], "best.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "run_id": run_id, 
                        "best_step": best_step, 
                        "best_epoch": epoch,
                        "best_loss_recon": best_loss, 
                        "best_path": best_path
                    }, f)

            global_step += 1
            
        if global_step >= max_steps:
            break

    logger.info(f"Training completed. Total global_steps: {global_step}")


def run_eval(cfg: Config) -> None:
    """Evaluation mode for the new model."""
    force_cuda = bool(cfg.run.get("force_cuda", False))
    if force_cuda and not torch.cuda.is_available():
        raise RuntimeError("force_cuda is True but CUDA is not available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_model(cfg).to(device)
    loader = make_dataloader(cfg=cfg.raw, split="val")
    num_visible = int(cfg.model.get("num_visible_leads", 3))
    
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            ecg = batch["ecg"].to(device)
            step_out = train_step(model, ecg, num_visible=num_visible, device=device)
            total_loss += step_out['loss'].item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    print(f"[eval] avg_masked_recon_mse={avg_loss:.6f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    
    if args.mode == "train":
        run_train(cfg)
    elif args.mode == "train_legacy":
        run_train_legacy(cfg)
    else:
        run_eval(cfg)


if __name__ == "__main__":
    main()
