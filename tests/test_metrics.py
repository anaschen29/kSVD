import math
import pytest
import torch
from ksvd.problem import SpectralProblem, generator
from ksvd.metrics import trace_deficit, objective_gap, manifold_residual, procrustes_distance, basis, estimate_rate


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("k", [1,2,4])
def test_stable_metrics_equal_dense_formulas(seed,k):
    p=SpectralProblem.make([3.,2.,1.,.5],n=6,seed=3)
    x=torch.randn(6,k,generator=generator(seed),dtype=torch.float64)
    q=basis(x)
    direct=float(p.lam[:k].sum()-torch.trace(q.T@p.apply(q)))
    assert trace_deficit(p,x)==pytest.approx(direct,abs=1e-12)
    gap=float((p.dense()-x@x.T).square().sum())/4-p.optimal_value(k)
    assert objective_gap(p,x)==pytest.approx(gap,abs=1e-11,rel=1e-12)


def test_cancellation_resistant_metrics_near_optimum():
    p=SpectralProblem.make([3.,2.,1.])
    x=p.target(2); x[2,1]=1e-10
    assert 0 < trace_deficit(p,x) < 1e-18
    assert 0 < objective_gap(p,x) < 1e-18


def test_trace_zero_does_not_mean_factor_optimal():
    p=SpectralProblem.make([3.,2.,1.])
    x=4*p.target(2)
    assert trace_deficit(p,x) == 0
    assert objective_gap(p,x) > 1


def test_right_invariance_and_procrustes():
    p=SpectralProblem.make([3.,2.,1.],n=5)
    x=torch.randn(5,2,generator=generator(9),dtype=torch.float64)
    v=torch.tensor([[2.,1.],[0.,.5]],dtype=x.dtype)
    assert trace_deficit(p,x@v)==pytest.approx(trace_deficit(p,x),abs=1e-12)
    rot=torch.tensor([[.6,-.8],[.8,.6]],dtype=x.dtype)
    assert procrustes_distance(x@rot,x)<1e-12


def test_tied_family_not_one_reference_orbit():
    p=SpectralProblem.make([3.,2.,2.,1.])
    y=p.frame(2); y[:,1]=torch.tensor([0.,0.,1.,0.],dtype=y.dtype)
    assert manifold_residual(p,y)<1e-14
    assert procrustes_distance(y,p.frame(2))>1
    assert objective_gap(p,p.to_x(y))<1e-25
    y[3,1]=.1
    assert manifold_residual(p,y)>0


def test_rate_estimation_and_no_fake_fit():
    rows=[{"t":t,"error":.005*.9**t} for t in range(150)]
    fit=estimate_rate(rows,"error")
    assert fit["rho_fit"]==pytest.approx(.9,abs=1e-12)
    assert estimate_rate(rows[:2],"error")["status"]=="insufficient_window"
