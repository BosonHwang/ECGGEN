import argparse
import json
import math
import logging
import os
from typing import Optional, Dict

import torch
from tqdm import tqdm

from ecggen.src.utils.config import Config, add_cli_overrides, apply_overrides, load_config
from ecggen.src.utils.run_id import RunIdSpec, ensure_run_dirs, step_dir
from ecggen.src.data.pipeline import make_dataloader
from ecggen.src.models.ecggen import ECGGenModel
from ecggen.src.models.heads import ClassificationHead
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
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    return parser.parse_args()


def build_run_id(cfg: Config) -> str:
    mapping = cfg.run.get("id_map", {"m": {}, "s": {}, "k": {}})
    codes = (cfg.run.get("m", "m0"), cfg.run.get("s", "s0"), cfg.run.get("k", "k0"))
    spec = RunIdSpec(mapping=mapping, codes=codes)
    return spec.build()


def build_model(cfg: Config) -> ECGGenModel:
    model_cfg = cfg.model
    data_cfg = cfg.data
    model = ECGGenModel(
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
    runs_root = cfg.run.get("runs_root", "./runs")
    ckpt_run = ensure_run_dirs(ckpt_root, run_id)
    log_run = ensure_run_dirs(log_root, run_id)
    runs_run = ensure_run_dirs(runs_root, run_id)
    return {"ckpt_run": ckpt_run, "log_run": log_run, "runs_run": runs_run}


def run_train(cfg: Config) -> None:
    force_cuda = bool(cfg.run.get("force_cuda", False))
    if force_cuda and not torch.cuda.is_available():
        raise RuntimeError("force_cuda is True but CUDA is not available.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Use provided index or default to 0
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

    model = build_model(cfg).to(device)
    head = maybe_build_head(cfg, model.state_dim)
    if head is not None:
        head = head.to(device)
    train_loader = make_dataloader(cfg=cfg.raw, split="train")
    max_steps = int(cfg.train.get("max_steps", 5))
    epochs = int(cfg.train.get("epochs", 5))
    lr = float(cfg.train.get("lr", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reg_cfg = cfg.model.get("reg", {})
    save_every = int(cfg.run.get("save_every", 10))
    best_loss = float("inf")
    best_path = os.path.join(dirs["ckpt_run"], "best.pt")
    best_step = -1
    train_size = len(train_loader.dataset)
    steps_per_epoch = len(train_loader)
    
    logger.info(
        f"params_total={_count_parameters(model)} "
        f"train_size={train_size} batch_size={cfg.train.get('batch_size', None)} "
        f"epochs={epochs} max_steps={max_steps} save_every={save_every} steps_per_epoch={steps_per_epoch}"
    )

    global_step = 0
    model.train()
    
    # Outer loop: Epochs
    for epoch in range(epochs):
        # Inner loop: Steps within the epoch
        pbar = tqdm(enumerate(train_loader), total=steps_per_epoch, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for step_in_epoch, batch in pbar:
            if global_step >= max_steps:
                break

            ecg = batch["ecg"].to(device)  # [B,L,T]
            out = model.forward_gen(ecg, reg_cfg=reg_cfg)
            recon = out["E_hat"]  # [B,L,T’]
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
                # Debugging info: check where NaN comes from
                for k, v in out.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        logger.error(f"  NaN found in output: {k}")
                if torch.isnan(ecg).any():
                    logger.error("  NaN found in input ECG")
                raise ValueError("NaN loss encountered")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss_recon.item():.4f}", "best": f"{best_loss:.4f}"})

            # Periodic saving & logging strictly on global_step interval
            # Or at the very last step of the entire run
            is_final_step = (global_step == max_steps - 1) or (epoch == epochs - 1 and step_in_epoch == steps_per_epoch - 1)
            if (global_step % save_every == 0) or is_final_step:
                logger.info(f"Epoch [{epoch}/{epochs}] GlobalStep [{global_step}] Loss: {loss_recon.item():.6f}")
                
                # Save to runs directory
                sd_runs = step_dir(dirs["runs_run"], global_step)
                torch.save(model.state_dict(), os.path.join(sd_runs, "model.pt"))
                with open(os.path.join(sd_runs, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump({"loss_recon": float(loss_recon.item()), "epoch": epoch, "step": step_in_epoch}, f)

            # Best model tracking (checkpoints directory)
            if loss_recon.item() < best_loss:
                best_loss = float(loss_recon.item())
                best_step = global_step
                best_named = os.path.join(dirs["ckpt_run"], f"best_{run_id}_step{best_step}.pt")
                # Save the named best and the 'best.pt' symlink-like copy
                torch.save(model.state_dict(), best_named)
                torch.save(model.state_dict(), best_path)
                with open(os.path.join(dirs["ckpt_run"], "best.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "run_id": run_id, 
                        "best_step": best_step, 
                        "best_epoch": epoch,
                        "best_loss_recon": best_loss, 
                        "best_path": best_named
                    }, f)

            global_step += 1
            
        if global_step >= max_steps:
            break

    logger.info(f"Training completed. Total global_steps: {global_step}")


def run_eval(cfg: Config) -> None:
    force_cuda = bool(cfg.run.get("force_cuda", False))
    if force_cuda and not torch.cuda.is_available():
        raise RuntimeError("force_cuda is True but CUDA is not available.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    loader = make_dataloader(cfg=cfg.raw, split="val")
    reg_cfg = cfg.model.get("reg", {})
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ecg = batch["ecg"].to(device)
            out = model.forward_gen(ecg, reg_cfg=reg_cfg)
            err = reconstruction_error(out["E_hat"], ecg)
            print(f"[eval] recon_mse={float(err):.6f}")
            break


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    if args.mode == "train":
        run_train(cfg)
    else:
        run_eval(cfg)


