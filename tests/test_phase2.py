"""Deterministic, minimal checks for the Phase 2 experiment wrappers."""

from __future__ import annotations

import json

import ksvd


def test_hessian_mode_wrapper_matches_predictions() -> None:
    result = ksvd.hessian_mode_isolation(smoke=True)
    assert result["metadata"]["dtype"] == "torch.float64"
    families = {row["family"] for row in result["modes"]}
    assert families == {"symmetric", "cross", "tangent"}
    assert max(row["relative_error"] for row in result["modes"]) < 1e-7
    zero_modes = [row for row in result["modes"] if row["predicted"] == 0.0]
    assert zero_modes and all(row["error_kind"] == "absolute" for row in zero_modes)


def test_geometry_wrapper_and_raw_serialization(tmp_path) -> None:
    result = ksvd.geometry_of_kc(smoke=True)
    assert result["part_a"]["grid_size"] == 31
    assert result["part_c"]["certificate"]["definition_version"] == "manuscript_v1"
    path = ksvd.save_raw_result(result, tmp_path / "geometry.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["smoke"] is True
    assert loaded["part_c"]["label"].startswith("empirical lower bounds")


def test_all_eight_wrappers_are_exported() -> None:
    expected = {
        "predicted_local_rates", "hessian_mode_isolation", "boundary_gap_scaling",
        "tied_eigenvalues", "geometry_of_kc", "saddle_escape",
        "step_size_phase_diagram", "initialization_ablation",
    }
    from ksvd.phase2 import EXPERIMENTS
    assert set(EXPERIMENTS) == expected
