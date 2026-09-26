import pytest
import torch
from ksvd.problem import SpectralProblem
from ksvd.initialization import controlled_trace_start, gaussian, power_sketch
from ksvd.metrics import trace_deficit, basis
from ksvd.theory import sufficient_power, sample_power_log_bound


@pytest.mark.parametrize("null_mix",[0.,.5,1.])
@pytest.mark.parametrize("theta",[.1,.5,.9])
def test_controlled_ambient_deficit(null_mix,theta):
    p=SpectralProblem.make([4.,3.,2.,1.],n=7,seed=8)
    x=controlled_trace_start(p,2,theta,null_mix=null_mix,condition=100,seed=4)
    assert trace_deficit(p,x)==pytest.approx(theta*p.gap(2),rel=1e-11,abs=1e-12)
    s=torch.linalg.svdvals(x)
    assert float((s[0]/s[-1])**2)==pytest.approx(100.,rel=1e-11)
    if null_mix:
        assert (x-p.U@(p.U.T@x)).norm()>0.01


def test_power_qr_has_same_subspace_not_same_factor():
    p=SpectralProblem.make([3.,2.,1.,.5],n=7,seed=8)
    omega=gaussian(p,2,4)
    raw=power_sketch(p,omega,4)
    stable=power_sketch(p,omega,4,basis_only=True)
    qr=basis(raw)
    torch.testing.assert_close(qr@qr.T,stable@stable.T,rtol=1e-11,atol=1e-11)
    assert not torch.allclose(raw.T@raw,stable.T@stable)
    assert torch.linalg.matrix_rank(raw)==2
    assert (raw-p.U@(p.U.T@raw)).norm()<1e-10


def test_quantitative_power_entry_fixed_sample():
    p=SpectralProblem.make([1.,.8,.4,.2],n=6)
    q=sufficient_power(p,1,.5,.1)
    omega=gaussian(p,1,0)
    x=power_sketch(p,omega,q,basis_only=True)
    assert trace_deficit(p,x) <= .5*p.gap(1)
    import math
    assert trace_deficit(p,x) <= math.exp(sample_power_log_bound(p,omega,1,q))+1e-28


def test_invalid_tilt_and_power():
    p=SpectralProblem.make([2.,1.])
    with pytest.raises(ValueError): controlled_trace_start(p,1,.5,null_mix=.5)
    with pytest.raises(ValueError): power_sketch(p,gaussian(p,1,0),0)
