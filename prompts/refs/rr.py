
class AttentionalPooler(nn.Module):
    def __init__(self, d_model: int, context_dim: int, n_head: int, n_queries: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model) * (d_model ** -0.5))
        self.in_proj = nn.Linear(context_dim, d_model) if context_dim != d_model else nn.Identity()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, tokens, _ = x.shape
        context = self.in_proj(x)
        q = self.query.unsqueeze(0).expand(bsz, -1, -1)
        out, _ = self.attn(q, context, context)
        return self.ln(out)


# ---------- ECG segmentation with optional RR encoding ----------

def _interp_to_length(seg: np.ndarray, target_len: int) -> np.ndarray:
    x_old = np.linspace(0, 1, len(seg), dtype=np.float32)
    x_new = np.linspace(0, 1, target_len, dtype=np.float32)
    return np.interp(x_new, x_old, seg).astype(np.float32)


def extract_rr_segmented_ecg(
    ecg_batch: torch.Tensor,
    frame_len: int = 1000,
    num_frames: int = 10,
    sampling_rate: int = 500,
    rr_max_sec: float = 2.0,
    encode_rr: bool = True,
    ablation: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Segment ECG per lead into frames and extract RR intervals.

    If neurokit2 is available and `ablation != "no_qrs_seg"`, uses R-peak based
    segmentation; otherwise falls back to equal segmentation.
    """
    try:
        import neurokit2 as nk  # type: ignore
        has_nk = True
    except Exception:
        nk = None  # type: ignore
        has_nk = False

    bsz, leads, seq_len = ecg_batch.shape
    device = ecg_batch.device
    frames = torch.zeros((bsz, leads, num_frames, frame_len), dtype=torch.float32, device=device)
    rr_vals = torch.zeros((bsz, leads, num_frames), dtype=torch.float32, device=device)

    ecg_np = ecg_batch.detach().cpu().numpy()

    for b in range(bsz):
        for c in range(leads):
            signal = ecg_np[b, c].astype(np.float32)
            segments: list[np.ndarray] = []
            rrs: list[float] = []
            if has_nk and ablation != "no_qrs_seg":
                cleaned = nk.ecg_clean(signal, sampling_rate=sampling_rate)
                _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
                r_locs = rpeaks.get("ECG_R_Peaks", [])
                if r_locs is None or len(r_locs) < 2:
                    r_locs = [0, seq_len]
                for i in range(len(r_locs) - 1):
                    start, end = int(r_locs[i]), int(r_locs[i + 1])
                    if end <= start + 1:
                        continue
                    rr_sec = (end - start) / float(sampling_rate)
                    seg = _interp_to_length(signal[start:end], frame_len)
                    segments.append(seg)
                    rrs.append(min(rr_sec, rr_max_sec))
            else:
                # Equal segmentation fallback
                seg_len = max(1, seq_len // num_frames)
                for i in range(num_frames):
                    start = i * seg_len
                    end = seq_len if i == num_frames - 1 else (i + 1) * seg_len
                    seg = _interp_to_length(signal[start:end], frame_len)
                    segments.append(seg)
                    rrs.append(0.5)  # dummy RR

            # Pad/trim
            if len(segments) < num_frames:
                pad = num_frames - len(segments)
                segments.extend([np.zeros(frame_len, dtype=np.float32)] * pad)
                rrs.extend([0.0] * pad)
            elif len(segments) > num_frames:
                segments = segments[:num_frames]
                rrs = rrs[:num_frames]

            frames[b, c] = torch.from_numpy(np.stack(segments)).to(device)
            rr_vals[b, c] = torch.tensor(rrs, dtype=torch.float32, device=device) / rr_max_sec

    if encode_rr:
        rr = rr_vals  # [B, C, T]
        rr2 = rr ** 2
        sin_rr = torch.sin(math.pi * rr)
        cos_rr = torch.cos(math.pi * rr)
        rr_feat = torch.stack([rr, rr2, sin_rr, cos_rr], dim=-1)  # [B, C, T, 4]
        bsz, leads, t_steps, d4 = rr_feat.shape
        rr_block = max(1, frame_len // t_steps) if t_steps > 0 else frame_len
        rr_encoded = torch.zeros((bsz, leads, frame_len), device=rr_feat.device)
        for t in range(t_steps):
            block = rr_feat[:, :, t, :]  # [B, C, 4]
            repeat = max(1, math.ceil(rr_block / d4))
            repeated = block.unsqueeze(-1).repeat(1, 1, 1, repeat)  # [B, C, 4, repeat]
            flat = repeated.flatten(-2)  # [B, C, 4*repeat]
            # Ensure we have at least rr_block length then slice
            if flat.shape[-1] < rr_block:
                pad = rr_block - flat.shape[-1]
                flat = F.pad(flat, (0, pad))
            block_flat = flat[:, :, :rr_block]  # [B, C, rr_block]
            start = t * rr_block
            end = min(frame_len, start + rr_block)
            take = end - start
            rr_encoded[:, :, start:end] = block_flat[:, :, :take]
        rr_frame = rr_encoded.unsqueeze(2)  # [B, C, 1, L]
        frames = torch.cat([frames, rr_frame], dim=2)  # [B, C, T+1, L]

    return frames, rr_vals