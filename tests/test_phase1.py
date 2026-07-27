"""Required Phase 1 validations; all tolerances target torch.float64."""

from __future__ import annotations

import math

import pytest
import torch

import ksvd


DTYPE = torch.float64
ATOL = 2e-11  # Algebraic identities after small dense linear solves.
FD_TOL = 2e-5  # Centered finite differences with step 1e-5.


def problem() -> ksvd.SpectralProblem:
    return ksvd.spectral_problem([5.0, 3.0, 1.0])


def test_ambient_and_reduced_coordinate_update_equivalence() -> None:
    p = problem()
    y = ksvd.random_y(p, 2, seed=7)
    xbar, eta = ksvd.y_to_xbar(p, y), 0.2
    x = ksvd.xbar_to_x(p, xbar)
    y_next = ksvd.reduced_y_update(p, y, eta)
    xb_next = ksvd.reduced_xbar_update(p, xbar, eta)
    x_next = ksvd.ambient_update(p.matrix, x, eta)
    assert torch.allclose(ksvd.y_to_xbar(p, y_next), xb_next, atol=ATOL, rtol=ATOL)
    assert torch.allclose(ksvd.xbar_to_x(p, xb_next), x_next, atol=ATOL, rtol=ATOL)


def test_potential_coordinate_equivalence() -> None:
    p = problem()
    y = ksvd.random_y(p, 2, seed=9)
    xb = ksvd.y_to_xbar(p, y)
    inverse_weighted_norm = torch.sum(xb.square() / p.eigenvalues[:, None])
    sign, logdet = torch.linalg.slogdet(xb.T @ xb)
    transformed = 0.5 * inverse_weighted_norm - 0.5 * logdet
    assert sign > 0
    assert torch.allclose(ksvd.potential(p, y), transformed, atol=ATOL, rtol=ATOL)
    assert torch.allclose(ksvd.xbar_to_y(p, xb), y, atol=ATOL, rtol=ATOL)


def test_dense_and_spectral_objective_equivalence() -> None:
    p = problem()
    xb = ksvd.y_to_xbar(p, ksvd.random_y(p, 2, seed=11))
    assert torch.allclose(
        ksvd.dense_objective(p.matrix, ksvd.xbar_to_x(p, xb)),
        ksvd.spectral_objective(p, xb), atol=ATOL, rtol=ATOL,
    )


def test_full_rank_preservation_for_admissible_step() -> None:
    p = problem()
    y = ksvd.random_y(p, 2, seed=13)
    constants = ksvd.certified_step_quantities(p, 2, float(ksvd.potential(p, y)) + 0.1)
    for _ in range(10):
        y = ksvd.reduced_y_update(p, y, constants.eta_C)
        assert int(torch.linalg.matrix_rank(y)) == 2
        assert torch.isfinite(y).all()


def test_gradient_and_hessian_finite_differences() -> None:
    p = problem()
    x = ksvd.y_to_xbar(p, ksvd.random_y(p, 2, seed=17))
    h = torch.randn(x.shape, generator=torch.Generator().manual_seed(18), dtype=DTYPE)
    eps = 1e-5
    directional = (ksvd.dense_objective(p.matrix, x + eps * h) - ksvd.dense_objective(p.matrix, x - eps * h)) / (2 * eps)
    assert abs(float(directional - torch.sum(ksvd.objective_gradient(p.matrix, x) * h))) < FD_TOL
    gradient_difference = (ksvd.objective_gradient(p.matrix, x + eps * h) - ksvd.objective_gradient(p.matrix, x - eps * h)) / (2 * eps)
    assert torch.allclose(gradient_difference, ksvd.objective_hessian_action(p.matrix, x, h), atol=FD_TOL, rtol=FD_TOL)


def test_hessian_spectrum_at_global_minimizer() -> None:
    p = problem()
    k = 2
    x = ksvd.y_to_xbar(p, ksvd.minimizer_y(p, k)).requires_grad_(True)
    hessian = torch.autograd.functional.hessian(lambda z: ksvd.dense_objective(p.matrix, z), x)
    # Autograd returns output-index dimensions followed by input-index
    # dimensions, so a direct row-major reshape is the matrix representation.
    matrix = hessian.reshape(x.numel(), x.numel())
    actual = torch.linalg.eigvalsh(matrix)
    expected = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=DTYPE)
    assert torch.allclose(actual, expected, atol=ATOL, rtol=ATOL)


def test_rank_one_heron_iteration_at_half_step() -> None:
    p = ksvd.spectral_problem([4.0])
    y = torch.tensor([[3.0]], dtype=DTYPE)
    updated = ksvd.reduced_y_update(p, y, 0.5)
    assert torch.allclose(updated, 0.5 * (y + 1 / y), atol=ATOL, rtol=ATOL)


def test_isotropic_spectrum_preserves_column_space() -> None:
    p = ksvd.spectral_problem([2.0, 2.0, 2.0, 2.0])
    y = ksvd.random_y(p, 2, seed=23)
    updated = ksvd.reduced_y_update(p, y, 0.3)
    assert float(ksvd.subspace_distance(y, updated)) < ATOL


def test_exact_ambient_null_space_decay_for_singular_matrix() -> None:
    lam = torch.tensor([4.0, 2.0], dtype=DTYPE)
    u = torch.eye(4, dtype=DTYPE)[:, :2]
    p = ksvd.SpectralProblem(lam, u)
    x = torch.tensor([[1.0], [0.5], [0.3], [-0.2]], dtype=DTYPE)
    eta = 0.27
    updated = ksvd.ambient_update(p.matrix, x, eta)
    assert torch.allclose(updated[2:], (1 - eta) * x[2:], atol=ATOL, rtol=ATOL)


def test_orthogonal_procrustes_metric_invariance_and_orientation() -> None:
    target = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=DTYPE)
    theta = 0.4
    q = torch.tensor([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]], dtype=DTYPE)
    assert float(ksvd.procrustes_distance(target @ q, target)) < ATOL
    assert torch.allclose(
        ksvd.procrustes_distance(target + 0.1, target),
        ksvd.procrustes_distance((target + 0.1) @ q, target), atol=ATOL, rtol=ATOL,
    )


def test_tied_optimal_family_metric() -> None:
    p = ksvd.spectral_problem([5.0, 3.0, 3.0, 3.0, 1.0])
    theta = 0.37
    optimal = torch.zeros(5, 2, dtype=DTYPE)
    optimal[0, 0] = 1
    optimal[1, 1], optimal[2, 1] = math.cos(theta), math.sin(theta)
    assert float(ksvd.tied_optimal_family_distance(p, optimal, 2)) < ATOL
    wrong = torch.eye(5, dtype=DTYPE)[:, [0, 4]]
    assert torch.allclose(ksvd.tied_optimal_family_distance(p, wrong, 2), torch.tensor(1.0, dtype=DTYPE), atol=ATOL)


def test_certified_descent_below_threshold() -> None:
    p = problem()
    y = ksvd.random_y(p, 2, seed=29)
    before = ksvd.potential(p, y)
    constants = ksvd.certified_step_quantities(p, 2, float(before) + 0.05)
    after = ksvd.potential(p, ksvd.reduced_y_update(p, y, 0.9 * constants.eta_C))
    assert after < before
    assert 0 < constants.eta_C <= 1 / constants.L_C


def test_runner_metadata_serialization_and_guards() -> None:
    p = problem()
    result = ksvd.run_deterministic(p, ksvd.random_y(p, 2, seed=31), ksvd.RunConfig(eta=0.1, max_steps=3, seed=31))
    assert result.termination == "max_steps"
    assert len(result.states) == 4
    assert result.metadata["seed"] == 31 and result.metadata["dtype"] == "torch.float64"
    assert '"torch_version"' in result.to_json()
    deficient = torch.ones(3, 2, dtype=DTYPE)
    guarded = ksvd.run_deterministic(p, deficient, ksvd.RunConfig(eta=0.1, max_steps=1, seed=0))
    assert guarded.termination == "rank_loss"


def test_rate_indexing_and_full_rank_case_validation() -> None:
    p = problem()
    mu, rho = ksvd.predicted_local_rate(p, 2, 0.25)
    assert mu == pytest.approx((3 - 1) / 3)
    assert rho == pytest.approx(max(0.5, 1 - 0.25 * 2 / 3))
    with pytest.raises(ValueError):
        ksvd.predicted_local_rate(p, 3, 0.25)


def test_numerical_rank_loss_rejected_before_solve() -> None:
    p = problem()
    y = torch.tensor([[1.0, 1.0], [0.0, 1e-20], [0.0, 0.0]], dtype=DTYPE)
    with pytest.raises(ValueError, match="numerically rank deficient"):
        ksvd.reduced_y_update(p, y, 0.1)
