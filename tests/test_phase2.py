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
    assert len(result["part_a"]["contour_levels"]) == 3
    assert result["part_a"]["trajectories"]
    assert result["part_c"]["levels"][0]["certificate"]["definition_version"] == "manuscript_v1"
    assert "hessian_operator_norm" in result["part_c"]["levels"][0]["samples"][0]
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


def test_trajectory_classifies_numerically_singular_weighted_gram() -> None:
    """A condition-1e8 ablation must serialize rank loss, not raise."""
    import torch
    from ksvd.phase2 import Phase2Config, _trajectory

    problem = ksvd.spectral_problem([2.0, 1.5, 1.2, 1.0, 0.8])
    y = torch.eye(5, dtype=torch.float64)[:, :4]
    y = y @ torch.diag(torch.tensor([1.0, 1e-2, 1e-4, 1e-8], dtype=torch.float64))
    result = _trajectory(problem, y, 4, 0.5, Phase2Config(max_steps=1))

    assert result["termination"] == "lost_rank"
    assert "numerically rank deficient" in result["termination_detail"]


def test_fit_serializes_exact_window_and_invalid_reason() -> None:
    from ksvd.phase2 import _fit

    records = [{"iteration": index, "error": 1e-3 * .9**index} for index in range(12)]
    fit = _fit(records, "error")
    assert fit["valid"] is True
    assert fit["window_first"] == 0
    assert fit["window_last"] == 11
    invalid = _fit(records[:3], "error")
    assert invalid["valid"] is False
    assert invalid["invalid_reason"] == "fewer_than_ten_usable_points"


def test_tiny_step_is_underflow_not_cycle() -> None:
    import torch
    from ksvd.phase2 import Phase2Config, _trajectory

    problem = ksvd.spectral_problem([2.0, 1.0, 0.5])
    y = torch.tensor([[1.0], [0.1], [0.0]], dtype=torch.float64)
    result = _trajectory(problem, y, 1, 1e-100, Phase2Config(max_steps=3))
    assert result["termination"] == "underflow"
    assert result["warnings"][0]["kind"] == "update_underflow"
    assert result["elapsed_seconds"] >= 0
    assert result["raw_history_bytes"] > 0


def test_phase2_experiment_specific_completeness() -> None:
    tied = ksvd.tied_eigenvalues(smoke=True)
    assert all("canonical_projector_error" in case and "rate_fit" in case for case in tied["cases"])
    assert all("final_tied_block_principal_angles" in case for case in tied["cases"])

    phase = ksvd.step_size_phase_diagram(smoke=True)
    assert phase["grid_duplicate_relative_tolerance"] == 1e-12
    assert phase["cutoff_summaries"]
    assert all("late_rate_fit" in case for case in phase["cases"])

    ablation = ksvd.initialization_ablation(smoke=True)
    assert len(ablation["certified_subset"]) == 3
    assert ablation["family_summaries"]
    assert ablation["regression"]["label"] == "exploratory, not theoretical"
    assert all("rate_fit" in case and "total_iterations" in case for case in ablation["cases"])


def test_boundary_parallelism_preserves_case_order_and_values() -> None:
    serial = ksvd.boundary_gap_scaling(smoke=True, workers=1)
    parallel = ksvd.boundary_gap_scaling(smoke=True, workers=2)
    assert parallel["metadata"]["workers"] == 2
    for left, right in zip(serial["cases"], parallel["cases"], strict=True):
        assert (left["k"], left["delta"], left["family"], left["seed"]) == (
            right["k"], right["delta"], right["family"], right["seed"]
        )
        assert left["termination"] == right["termination"]
        assert left["final_y"] == right["final_y"]
