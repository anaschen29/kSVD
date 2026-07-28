"""Static checks for full-sweep launchers; these tests never run a sweep."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "experiments" / "full"


def test_full_sweep_shell_scripts_have_valid_bash_syntax() -> None:
    scripts = [FULL / "_common.sh", *(FULL / f"run_stage{stage}.sh" for stage in (1, 2, 3))]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


def test_full_sweep_stages_match_the_approved_order() -> None:
    expected = {
        1: ["predicted_local_rates", "hessian_mode_isolation", "saddle_escape"],
        2: ["boundary_gap_scaling", "tied_eigenvalues", "initialization_ablation"],
        3: ["geometry_of_kc", "step_size_phase_diagram"],
    }
    for stage, experiments in expected.items():
        text = (FULL / f"run_stage{stage}.sh").read_text(encoding="utf-8")
        positions = [text.index(f"full_run_experiment {experiment}") for experiment in experiments]
        assert positions == sorted(positions)
        assert text.count("full_run_experiment ") == len(experiments)


def test_full_sweep_common_script_preserves_raw_results_and_logs() -> None:
    text = (FULL / "_common.sh").read_text(encoding="utf-8")
    assert '"experiments/$experiment.py" --full --output "$raw"' in text
    assert "Refusing to overwrite existing artifact" in text
    assert "full_validate_raw" in text
    assert "sha256sum" in text
    assert "manifest.tsv" in text
