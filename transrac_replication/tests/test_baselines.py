from __future__ import annotations

import torch

from transrac_replication.models.baselines import BaselineConfig, build_baseline


def _fake_embeddings(batch: int = 2, frames: int = 64, dim: int = 512):
    return {
        "v1": torch.randn(batch, frames, dim),
        "v4": torch.randn(batch, frames, dim),
        "v8": torch.randn(batch, frames, dim),
    }


def test_repnet_like() -> None:
    model = build_baseline("repnet", BaselineConfig())
    density, corr = model.forward_from_embeddings(_fake_embeddings())
    assert density.shape == (2, 64)
    assert corr.shape[0] == 2 and corr.shape[1] == 64 and corr.shape[2] == 64


def test_zhang_like() -> None:
    model = build_baseline("zhang", BaselineConfig())
    density, corr = model.forward_from_embeddings(_fake_embeddings())
    assert density.shape == (2, 64)
    assert corr.shape[0] == 2 and corr.shape[1] == 64 and corr.shape[2] == 64


def main() -> None:
    test_repnet_like()
    test_zhang_like()
    print("test_baselines: PASS")


if __name__ == "__main__":
    main()
