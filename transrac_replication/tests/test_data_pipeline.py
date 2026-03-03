from __future__ import annotations

import numpy as np

from transrac_replication.datasets.multiscale import ScaleSpec, build_multiscale_sequences
from transrac_replication.datasets.sampling import sample_video_frames, uniform_sample_indices


def test_uniform_indices() -> None:
    idx = uniform_sample_indices(total_frames=10, num_samples=64)
    assert idx.shape == (64,)
    assert idx.min() >= 0
    assert idx.max() <= 9
    assert np.all(idx[:-1] <= idx[1:])


def test_sampling_and_multiscale() -> None:
    frames = np.random.rand(37, 3, 224, 224).astype(np.float32)
    sampled, idx = sample_video_frames(frames, num_samples=64)
    assert sampled.shape == (64, 3, 224, 224)
    assert idx.shape == (64,)

    scales = {
        "v1": ScaleSpec(window=1, stride=1),
        "v4": ScaleSpec(window=4, stride=2),
        "v8": ScaleSpec(window=8, stride=4),
    }
    out = build_multiscale_sequences(sampled, scales=scales, target_length=64)
    assert out["v1"].shape == (64, 1, 3, 224, 224)
    assert out["v4"].shape == (64, 4, 3, 224, 224)
    assert out["v8"].shape == (64, 8, 3, 224, 224)


def main() -> None:
    test_uniform_indices()
    test_sampling_and_multiscale()
    print("test_data_pipeline: PASS")


if __name__ == "__main__":
    main()
