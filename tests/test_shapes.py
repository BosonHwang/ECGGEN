import torch

from ecggen.src.models.ecggen import ECGGenModel


def test_forward_gen_shapes():
    B, L, T = 2, 12, 512
    ecg = torch.randn(B, L, T)
    model = ECGGenModel(
        num_leads=L,
        token_dim=32,
        beat_len=64,
        tokenizer_mode="equidistant",
        d_model=64,
        state_dim=16,
        vcg_basis_k=8,
        vcg_time_len=128,
        residual_enabled=True,
    )
    out = model.forward_gen(ecg, reg_cfg={"smoothness": 0.1, "energy": 0.1})
    assert out["W"].shape == (B, 16)
    assert out["V"].shape == (B, 3, 128)
    assert out["E_hat"].shape == (B, L, 128)
    assert "reg_terms" in out and isinstance(out["reg_terms"], dict)


