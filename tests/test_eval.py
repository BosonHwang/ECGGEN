import torch

from ecggen.src.eval import reconstruction_error, missing_lead_robustness, linear_probe_accuracy
from ecggen.src.models.ecggen import ECGGenModel
from ecggen.src.models.heads import ClassificationHead


def test_missing_lead_robustness_runs():
    B, L, T = 2, 12, 256
    ecg = torch.randn(B, L, T)
    model = ECGGenModel(num_leads=L, token_dim=32, beat_len=64, d_model=64, state_dim=16, vcg_basis_k=8, vcg_time_len=128)
    val = missing_lead_robustness(model, ecg, drop_prob=0.5)
    assert isinstance(val, torch.Tensor)


def test_linear_probe_accuracy():
    B, D, C = 4, 8, 3
    W = torch.randn(B, D)
    y = torch.tensor([0, 1, 2, 1])
    head = ClassificationHead(kind="linear", in_dim=D, num_classes=C)
    acc = linear_probe_accuracy(head, W, y)
    assert 0.0 <= acc <= 1.0


