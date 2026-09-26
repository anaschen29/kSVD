import math
import pytest
import torch
from ksvd.problem import SpectralProblem
from ksvd.theory import positive_mixing, normal_rate, optimal_step, trace_certificate, sufficient_power, power_log_bound, rank_one_annulus
from ksvd.initialization import controlled_trace_start
from ksvd.dynamics import ambient_step
from ksvd.metrics import trace_deficit


def test_rates_ties_and_full_rank():
    p=SpectralProblem.make([3.,2.,2.,1.])
    assert normal_rate(p,2,.5)==pytest.approx(5/6)
    assert positive_mixing(p,4)==[]
    assert optimal_step(p,4)==(.5,0.)
    assert normal_rate(p,4,.7)==pytest.approx(.4)
    tied=SpectralProblem.make([2.,2.,2.])
    assert optimal_step(tied,1)==(.5,0.)


@pytest.mark.parametrize("mu",[.02,.2,.8])
def test_optimal_damping(mu):
    p=SpectralProblem.make([1.,1-mu])
    eta,rho=optimal_step(p,1)
    assert eta==pytest.approx(2/(2+mu))
    assert rho==pytest.approx(normal_rate(p,1,eta))
    assert all(normal_rate(p,1,v)>=rho-1e-14 for v in [.1,.5,2/3,.9,.99])


@pytest.mark.parametrize("eta",[.1,.5,.9])
def test_trace_certificate_bounds_a_trajectory(eta):
    p=SpectralProblem.make([3.,2.,1.,.5],n=6,seed=5)
    x0=controlled_trace_start(p,2,.6,null_mix=.5,condition=50,seed=3)
    cert=trace_certificate(p,x0,eta)
    assert cert.beta == pytest.approx(p.gap(2)**2/3*(1-cert.e0/p.gap(2)))
    x=x0
    for t in range(1,60):
        x=ambient_step(p,x,eta)
        assert trace_deficit(p,x) <= cert.envelope(t)+1e-12
        gram=torch.linalg.eigvalsh(x.T@x)
        assert float(gram[0]) >= cert.d-1e-11
        assert float(gram[-1]) <= cert.u+1e-11


def test_zero_error_and_outside_band():
    p=SpectralProblem.make([3.,2.,1.])
    cert=trace_certificate(p,p.target(2),.5)
    assert cert.e0==0 and cert.envelope(10)==0
    x=torch.tensor([[0.],[1.],[0.]],dtype=torch.float64)
    with pytest.raises(ValueError,match="outside"):
        trace_certificate(p,x,.5)
    with pytest.raises(ValueError): normal_rate(p,1,1.)


def test_sufficient_q_and_log_formula():
    p=SpectralProblem.make([3.,2.,1.,.5])
    k,q,delta=2,7,.1
    bound=16*3*k**5*(p.r-k)/(math.pi*delta**3)*(.5)**(2*q)
    assert math.exp(power_log_bound(p,k,q,delta))==pytest.approx(bound)
    q=sufficient_power(p,k,.4,delta)
    assert power_log_bound(p,k,q,delta) <= math.log(.4*p.gap(k))+1e-12
    if q>1:
        assert power_log_bound(p,k,q-1,delta)>math.log(.4*p.gap(k))


def test_annulus_formula_and_scale_dependence():
    p=SpectralProblem.make([1.,.5,.25])
    x=torch.tensor([[1.],[0.],[0.]],dtype=torch.float64)
    a=rank_one_annulus(p,x)
    assert a["a_js"]==1 and a["b_js"]==2 and a["tau_js"]==1
    assert rank_one_annulus(p,100*x)["tau_js"] > a["tau_js"]
