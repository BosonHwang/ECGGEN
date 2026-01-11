from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGTokenizer(nn.Module):
    """Beat-wise tokenizer for multi-lead ECG.

    1) Role: converts raw ECG [B, L, T] into beat tokens [B, L, N, d] where token index
       equals beat index, not time index. Provides beat metadata for downstream modules.
    2) Inputs/outputs: see shape contract; X [B, L, N, d], meta with intervals/boundaries.
    3) Place in pipeline: first step after raw ECG; feeds TokenEncoder.

    Modes:
        - equidistant (default): fixed non-overlapping windows of length beat_len
        - rr: lightweight R-peak detection on lead 0, beats defined by midpoints, resampled to beat_len
    """

    def __init__(self, beat_len: int = 128, token_dim: int = 64, mode: str = "equidistant", fs: int = 100):
        super().__init__()
        self.beat_len = int(beat_len)
        self.token_dim = int(token_dim)
        self.mode = str(mode)
        self.fs = int(fs)
        # A simple linear mapper from flattened beat waveform to token_dim
        # For each lead, we encode its beat waveform independently, keeping leads separate in the tensor layout.
        self.linear = nn.Linear(self.beat_len, self.token_dim)

    def forward(self, ecg: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        assert ecg.ndim == 3, "ecg must be [B, L, T]"
        B, L, T = ecg.shape
        if self.mode == "rr":
            return self._forward_rr(ecg)
        else:
            return self._forward_equidistant(ecg)

    def _forward_equidistant(self, ecg: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, T = ecg.shape
        beat_boundaries: List[List[Tuple[int, int]]] = []
        approx_num_beats = max(1, T // self.beat_len)
        N = approx_num_beats
        beat_intervals = torch.full((B, N), float(self.beat_len), dtype=torch.float32, device=ecg.device)
        X = torch.zeros((B, L, N, self.token_dim), dtype=ecg.dtype, device=ecg.device)
        for b in range(B):
            bb_list: List[Tuple[int, int]] = []
            for n in range(N):
                start = n * self.beat_len
                end = min(start + self.beat_len, T)
                if end - start < self.beat_len:
                    seg = torch.zeros((L, self.beat_len), dtype=ecg.dtype, device=ecg.device)
                    seg[:, : end - start] = ecg[b, :, start:end]
                else:
                    seg = ecg[b, :, start:end]  # [L, beat_len]
                x = self.linear(seg)  # [L, d]
                X[b, :, n, :] = x
                bb_list.append((start, start + self.beat_len))
            beat_boundaries.append(bb_list)
        meta = {"beat_intervals": beat_intervals, "beat_boundaries": beat_boundaries, "beat_len": self.beat_len}
        return X, meta

    def _detect_r_peaks(self, signal: torch.Tensor, fs: int) -> List[int]:
        """Very simple R-peak detector on a 1D tensor [T].

        Steps:
            - Differentiate -> abs -> moving average
            - Threshold by mean + k*std
            - Local maxima with refractory period ~ 0.25s
        This is a minimal placeholder; replace with a proper detector if needed.
        """
        assert signal.ndim == 1, "signal must be 1D"
        T = signal.shape[0]
        x = signal - signal.mean()
        # derivative-like
        dx = torch.zeros_like(x)
        dx[1:] = x[1:] - x[:-1]
        e = dx.abs()
        # moving average with window ~0.1s
        win = max(1, int(0.1 * fs))
        kernel = torch.ones(1, 1, win, device=signal.device, dtype=signal.dtype) / float(win)
        e2 = e.view(1, 1, T)
        ma = F.conv1d(e2, kernel, padding=win // 2).view(-1)[:T]
        thr = ma.mean() + 0.5 * ma.std()
        candidates = (ma > thr).nonzero(as_tuple=True)[0]
        peaks: List[int] = []
        refractory = int(0.25 * fs)
        last = -refractory
        # Keep local maxima among neighbors within a small window
        for idx in candidates.tolist():
            if idx - last < refractory:
                # keep the higher one
                if peaks and ma[idx] > ma[peaks[-1]]:
                    peaks[-1] = idx
                continue
            # local peak check
            left = max(0, idx - 1)
            right = min(T - 1, idx + 1)
            if ma[idx] >= ma[left] and ma[idx] >= ma[right]:
                peaks.append(idx)
                last = idx
        return peaks

    def _forward_rr(self, ecg: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        B, L, T = ecg.shape
        device = ecg.device
        boundaries_per_b: List[List[Tuple[int, int]]] = []
        intervals_per_b: List[torch.Tensor] = []

        # First pass: find R-peaks on a reference lead (Lead II, index 1) and compute beat boundaries
        for b in range(B):
            ref = ecg[b, 1]  # [T] - Use Lead II as reference
            peaks = self._detect_r_peaks(ref, self.fs)
            # Convert peaks to boundaries using midpoints between peaks
            if len(peaks) < 2:
                # Fallback: treat whole signal as one beat
                boundaries = [(0, T)]
                intervals = torch.tensor([float(T)], device=device)
            else:
                mids = [(peaks[i] + peaks[i + 1]) // 2 for i in range(len(peaks) - 1)]
                starts = [max(0, peaks[0] - (mids[0] - peaks[0]))] + mids[:-1]
                ends = mids + [min(T, peaks[-1] + (peaks[-1] - mids[-1]))]
                boundaries = list(zip(starts, ends))
                # intervals as differences between consecutive peaks (RR)
                rr = torch.tensor([float(peaks[i + 1] - peaks[i]) for i in range(len(peaks) - 1)], device=device)
                # align intervals with beats count; use min length
                n_beats = min(len(boundaries), rr.numel())
                boundaries = boundaries[:n_beats]
                intervals = rr[:n_beats]
            boundaries_per_b.append(boundaries)
            intervals_per_b.append(intervals)

        # Determine common N across batch; fallback to equidistant if N==0
        Ns = [len(bb) for bb in boundaries_per_b]
        if min(Ns) == 0:
            return self._forward_equidistant(ecg)
        N = min(Ns)
        beat_intervals = torch.zeros((B, N), dtype=torch.float32, device=device)
        X = torch.zeros((B, L, N, self.token_dim), dtype=ecg.dtype, device=device)
        beat_boundaries: List[List[Tuple[int, int]]] = []

        # Second pass: slice beats, resample each to beat_len, encode
        for b in range(B):
            bb = boundaries_per_b[b][:N]
            beat_boundaries.append(bb)
            # intervals: if provided beats count differs, pad/truncate
            ints = intervals_per_b[b]
            if ints.numel() < N:
                pad = torch.full((N - ints.numel(),), float(self.beat_len), device=device)
                bi = torch.cat([ints, pad], dim=0)
            else:
                bi = ints[:N]
            beat_intervals[b] = bi
            for n, (start, end) in enumerate(bb):
                seg_len = max(1, end - start)
                seg = ecg[b, :, start:end]  # [L, seg_len]
                # Resample to beat_len using linear interpolation
                seg_ = seg.unsqueeze(1)  # [L, 1, Tseg]
                seg_rs = F.interpolate(seg_, size=self.beat_len, mode="linear", align_corners=True).squeeze(1)  # [L, beat_len]
                x = self.linear(seg_rs)  # [L, d]
                X[b, :, n, :] = x

        meta = {"beat_intervals": beat_intervals, "beat_boundaries": beat_boundaries, "beat_len": self.beat_len}
        return X, meta


