ecggen - Object-centric ECG rendering with quasi-static heart state

Overview
This research codebase models ECG as a rendering of a slowly varying latent heart state W, not a sequence to forecast. Pipeline:
- Beat-wise tokenization (equidistant or RR-based) of multi-lead ECG
- Token-level Transformer encoding (no temporal rollout)
- Test-Time Training (TTT) to estimate a quasi-static state W
- VCG generator renders a latent 3D source
- Lead projection maps VCG to ECG leads; optional light residual head

Strict constraints
- No recurrence; no autoregressive/forecasting
- Only PyTorch + NumPy + Python standard library (core training)
- Clear tensor shapes with asserts; type hints; concise explanatory comments

Config format
- YAML/JSON supported; YAML requires PyYAML. Example: `ecggen/configs/train/v1.yaml`.

Project layout
- ecggen/main.py: entry point outside src
- ecggen/trainer.py: train/eval loop; checkpoint/log writing
- src/utils/config.py: JSON/YAML config loader and CLI overrides
- src/utils/run_id.py: run_id builder and step directories
- src/data/pipeline.py: ECGDataset and DataLoader
- src/data/tokenizer.py: Beat-wise tokenizer (equidistant or RR)
- src/models/blocks.py: TokenEncoder, LeadProjection, ResidualHead, AngleCalib skeleton, SO(3) utils
- src/models/ttt.py: FastState (vector/MLP skeleton) + TTTUpdater
- src/models/vcg.py: VCGGenerator (basis form) + regularizers
- src/models/ecggen.py: end-to-end assembly
- src/models/heads.py: linear/MLP classification heads
- src/eval.py: reconstruction / missing-lead / linear probe utilities
- configs/train/v1.yaml: example config with run_id mapping and reg/TTT
- scripts/wfdb_to_npy.py: optional WFDB → .npy converter (requires wfdb; scipy optional)
- tests/: simple shape/pipeline sanity tests

Data expectations
- Training expects standardized tensors [B, L, T] (default L=12, T=512) in .npy files.
- For raw WFDB, use `scripts/wfdb_to_npy.py` to resample (default 100 Hz), bandpass (0.67–40 Hz), reorder to standard 12 leads, normalize, and window to `[L,T]`.
  - Dependencies: `wfdb` required; `scipy` optional (falls back to simple filters/resample if absent).

Tokenizer modes
- Equidistant (default): fixed windows of `beat_len`.
- RR-based: lightweight R-peak detector (derivative + MA + threshold + refractory) on lead 0; beats are resampled to `beat_len`. Set `model.tokenizer_mode` to `"rr"` and ensure `data.fs` matches the sampling rate.

Quick start
- Train (synthetic fallback if no data found):
  ```bash
  python -m ecggen.main --config ecggen/configs/train/v1.yaml --mode train
  ```
- Convert WFDB to npy:
  ```bash
  python ecggen/scripts/wfdb_to_npy.py --src /path/to/wfdb --dst /path/to/npy --fs_out 100 --time_len 512
  ```
  then point `data.meta_root` in the config to that `dst`.

License
Research use only. No warranty.