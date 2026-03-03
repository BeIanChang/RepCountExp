from __future__ import annotations

import torch

from transrac_replication.models.transrac_model import TransRACConfig, TransRACModel


def test_forward_shapes() -> None:
    cfg = TransRACConfig(num_frames=64, embedding_dim=512, num_heads=4, num_scales=3)
    model = TransRACModel(cfg)

    b, n, c, h, w = 2, 64, 3, 32, 32
    inputs = {
        "v1": torch.randn(b, n, 1, c, h, w),
        "v4": torch.randn(b, n, 4, c, h, w),
        "v8": torch.randn(b, n, 8, c, h, w),
    }

    density, corr = model(inputs)
    assert density.shape == (b, n)
    assert corr.shape == (b, n, n, 12)


def test_backbone_frozen() -> None:
    cfg = TransRACConfig()
    model = TransRACModel(cfg)
    backbone_params = list(model.encoder.parameters())
    assert backbone_params
    assert all(not p.requires_grad for p in backbone_params)


def main() -> None:
    test_forward_shapes()
    test_backbone_frozen()
    print("test_model: PASS")


if __name__ == "__main__":
    main()
