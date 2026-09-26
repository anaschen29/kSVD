import pytest
import torch
from ksvd.problem import SpectralProblem, generator
from ksvd.dynamics import ambient_step, reduced_step, gradient_f, hessian_f, potential_f, quotient, right_solve
from ksvd.metrics import basis, polar_frame


def sample(seed=1):
    p = SpectralProblem.make([4., 2., 1., .3], n=7, seed=2)
    y = torch.randn(4, 2, generator=generator(seed), dtype=torch.float64)
    return p, y


@pytest.mark.parametrize("eta", [.01, .25, .5, .9, .99])
def test_coordinate_equivalence_and_gram_floor(eta):
    p, y = sample()
    yn = reduced_step(p, y, eta)
    torch.testing.assert_close(ambient_step(p, p.to_x(y), eta), p.to_x(yn), rtol=2e-12, atol=2e-12)
    assert torch.linalg.eigvalsh(yn.T @ yn)[0] >= 4*eta*(1-eta)-1e-12
    assert torch.linalg.matrix_rank(yn) == 2


@pytest.mark.parametrize("seed", range(4))
def test_analytic_derivatives(seed):
    p, y = sample(seed)
    h = torch.randn(y.shape, generator=generator(seed+10), dtype=y.dtype)
    eps = 1e-6
    directional = (potential_f(p, y+eps*h)-potential_f(p, y-eps*h))/(2*eps)
    torch.testing.assert_close(directional, (gradient_f(p,y)*h).sum(), rtol=1e-6, atol=1e-7)
    fd = (gradient_f(p,y+eps*h)-gradient_f(p,y-eps*h))/(2*eps)
    torch.testing.assert_close(fd, hessian_f(p,y,h), rtol=2e-6, atol=2e-7)


def test_hessian_modes_with_tie():
    p = SpectralProblem.make([3., 2., 2., 1.])
    y = p.frame(2)
    modes = [(0,0,2.), (2,1,0.), (3,1,.5), (2,0,1/3)]
    for i,j,value in modes:
        h = torch.zeros_like(y); h[i,j] = 1.
        torch.testing.assert_close(hessian_f(p,y,h), value*h, rtol=1e-13, atol=1e-13)
    h = torch.zeros_like(y); h[0,1]=1.; h[1,0]=-1.
    torch.testing.assert_close(hessian_f(p,y,h), torch.zeros_like(y), atol=1e-13, rtol=0)


@pytest.mark.parametrize("eta", [.1,.5,.95])
def test_null_component_decays_exactly(eta):
    p=SpectralProblem.make([2.,1.],n=4)
    x=torch.tensor([[1.,.2],[.3,2.],[1.,0.],[0.,.2]],dtype=torch.float64)
    nxt=ambient_step(p,x,eta)
    torch.testing.assert_close(nxt[2:], (1-eta)*x[2:], atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("eta", [.15,.5,.9])
def test_ambient_trace_identity_noncommuting_gram(eta):
    p=SpectralProblem.make([4.,2.,1.,.3],n=6,seed=12)
    x=torch.randn(6,2,generator=generator(7),dtype=torch.float64)
    q=polar_frame(x); g=x.T@x; mq=p.apply(q); c=q.T@mq; r=mq-q@c
    h=(1-eta)*g+eta*c
    z=eta*right_solve(r,h)
    d=z.T@z
    j=torch.linalg.solve(torch.eye(2,dtype=x.dtype)+d,torch.eye(2,dtype=x.dtype))
    qp=basis(ambient_step(p,x,eta)); dp=qp@qp.T-q@q.T
    lhs=torch.trace(qp.T@p.apply(qp))-torch.trace(c)
    rhs=torch.trace(dp@p.dense()@dp)+2*(1-eta)/eta*torch.trace(g@d@j)
    torch.testing.assert_close(lhs,rhs,atol=2e-12,rtol=2e-12)
    assert lhs >= -1e-12


@pytest.mark.parametrize("eta", [.05,.5,.99])
def test_quotient_ascent_and_scalar_heron(eta):
    p,y=sample(8)
    assert float(quotient(p,reduced_step(p,y,eta))-quotient(p,y)) >= -1e-12
    scalar=SpectralProblem.make([7.])
    z=torch.tensor([[2.]],dtype=torch.float64)
    torch.testing.assert_close(reduced_step(scalar,z,eta),(1-eta)*z+eta/z)


def test_isotropic_column_space_preserved():
    p=SpectralProblem.make([2.,2.,2.,2.]); _,y=sample(3)
    q=basis(y); qn=basis(reduced_step(p,y,.7))
    torch.testing.assert_close(q@q.T,qn@qn.T,atol=1e-12,rtol=1e-12)


def test_unsupported_coordinates_and_failed_solve():
    p=SpectralProblem.make([2.,1.],n=3)
    with pytest.raises(ValueError,match="not supported"):
        p.to_y(torch.ones(3,1,dtype=torch.float64))
    with pytest.raises(ArithmeticError,match="Cholesky"):
        right_solve(torch.ones(3,2,dtype=torch.float64),torch.zeros(2,2,dtype=torch.float64))
