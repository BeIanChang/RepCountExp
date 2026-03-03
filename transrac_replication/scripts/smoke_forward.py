from __future__ import annotations

import torch

from transrac_replication.models.transrac_model import TransRACConfig, TransRACModel


def main() -> None:
    cfg = TransRACConfig()
    model = TransRACModel(cfg)

    b, n, c, h, w = 2, 64, 3, 224, 224
    inputs = {
        "v1": torch.randn(b, n, 1, c, h, w),
        "v4": torch.randn(b, n, 4, c, h, w),
        "v8": torch.randn(b, n, 8, c, h, w),
    }
    density, corr = model(inputs)
    print("density", tuple(density.shape))
    print("corr", tuple(corr.shape))


if __name__ == "__main__":
    main()
