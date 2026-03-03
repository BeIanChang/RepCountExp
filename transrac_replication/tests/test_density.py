from __future__ import annotations

import numpy as np

from transrac_replication.datasets.density import (
    build_density_from_original_cycles,
    build_density_map,
    map_cycles_to_sample_bins,
)


def test_density_sum_approx_count() -> None:
    cycles = [(10, 20), (30, 46), (60, 80)]
    density = build_density_from_original_cycles(cycles, total_frames=100, num_bins=64)
    assert density.shape == (64,)
    assert float(density.sum()) > 2.0
    assert float(density.sum()) < 4.0


def test_mapping_monotonic() -> None:
    cycles = [(0, 31), (32, 63)]
    mapped = map_cycles_to_sample_bins(cycles, total_frames=64, num_bins=64)
    assert mapped[0][0] <= mapped[0][1]
    assert mapped[1][0] <= mapped[1][1]


def test_empty_cycles() -> None:
    density = build_density_map([], num_bins=64)
    assert density.shape == (64,)
    assert np.allclose(density, 0.0)


def main() -> None:
    test_density_sum_approx_count()
    test_mapping_monotonic()
    test_empty_cycles()
    print("test_density: PASS")


if __name__ == "__main__":
    main()
