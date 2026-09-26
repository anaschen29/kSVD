# Mathematical reference and assumptions

This implementation follows the user's supplied **Preconditioned Gradient
Descent for k-SVD**, manuscript PDF revision (15), September 2026. The paper is
not redistributed here. Equation numbers below refer to that supplied draft.
The code is independent of the deleted historical implementation.

## Coordinates and dynamics

Let `M=U Lambda U.T`, with U of shape (n,r), descending **positive** eigenvalues,
and `1 <= k <= r`. Reduced coordinates satisfy `X=U sqrt(Lambda) Y`.

- Ambient: `X+ = (1-eta) X + eta M X (X.T X)^(-1)` (3.9 / ambient (5.1)).
- Reduced: `Y+ = (1-eta) Y + eta Lambda Y (Y.T Lambda Y)^(-1)`.
- `F(Y)=||Y||_F^2/2 - logdet(Y.T Lambda Y)/2`.
- `Psi(Y)=logdet(Y.T Lambda Y)-logdet(Y.T Y)`.

All inverse actions use Cholesky solves. Failure is an explicit numerical
outcome, not permission to add a ridge or normalize the state. The solver allows
eta >= 1 only for labeled out-of-theorem controls. Certificate functions require
`0 < eta < 1`. Randomized experiments fix eta before sampling.

`basis(X)` uses measurement-only QR. `polar_frame(X)` uses the SVD to return
`Q=X(X.T X)^(-1/2)`. Both project onto the same subspace, but they are NOT
interchangeable in the exact trace identity if G is kept as X.T X.

## Trace certificate (5.9--5.10)

`e0=tau_k-tr(M Pi_X0)`, `Delta=lambda_k-lambda_(k+1)`; use zero for the next
eigenvalue when k=r. The certificate requires `e0 < Delta`.

```text
a = 1-eta
c = lambda_k-e0
d = 4 eta a c
u = max(||X1||_2^2, lambda_1^2/d)
k_minus = eta/(a u + eta lambda_1)
k_plus = eta/(a d + eta c)
alpha = (c+2 a d/eta) k_minus^2 / (1+k_plus^2 lambda_1^2)
beta = Delta^2/lambda_1 * (1-e0/Delta)
zeta = 1-min(alpha beta, 1/2)
e_t <= e0 zeta^(t-1), t >= 1
```

The beta formula is intentionally NOT replaced by the proposed sharper
`Delta-e0` bound. `formula_version=manuscript-v15-original-beta` records this.
`log_zeta=log1p(-min(alpha beta,1/2))` is authoritative for envelope evaluation
when zeta itself rounds to one. No certificate is issued outside the trace band.

A rotated leading direction toward an omitted unit vector v has deficit
`sin(phi)^2 * (lambda_k-v.T M v)`. For mixed positive-tail/nullspace v, the tilt
is chosen using this actual Rayleigh quotient, not incorrectly using Delta.
An SPD right factor sets the Gram condition without changing e0.

## Stable diagnostics (algebraic equivalents derived for this implementation)

Near a minimizer, direct subtraction `tau_k-captured_trace` can lose the entire
signal. With `c=lambda_k`, an orthonormal measurement frame Q, and P=Q Q.T:

```text
e = sum_(i<=k) (lambda_i-c) ||(I-P)u_i||^2
  + sum_(j>k,j<=r) (c-lambda_j) ||Q.T u_j||^2
  + c ||(I-U U.T)Q||_F^2.
```

Every term is nonnegative. The same formula with squared eigenvalues evaluates
`e_(M^2)`. Let `C=Q.T M Q`, `R=M Q-Q C`, and
`S=(Q.T X)(Q.T X).T`. Orthogonal projection and `Q.T M^2 Q=C^2+R.T R` give

```text
4 (g(X)-g*) = e_(M^2)(Q) + ||R||_F^2 + ||S-C||_F^2.
```

This is an exact algebraic identity; it is tested against the dense expression
away from cancellation and tested near known optima. It does not assume that
objective gap equals squared error to one chosen optimal product.

The tied-manifold residual in Y coordinates sums the squares of Gram error,
missing mandatory eigenspace projection, and projection outside the eligible
(above-or-equal-cutoff) eigenspace, then takes the square root. Its zero set is
the full minimizer manifold. We do not claim it is a distance. Procrustes
measures distance to one fixed right-orthogonal orbit only.

## Gaussian power entry (3.3--3.5 / Appendix D)

For k<r and Delta>0, the saved theorem bound is

```text
B_q(delta) = 16 lambda_1 k^5 (r-k)/(pi delta^3)
             * (lambda_(k+1)/lambda_k)^(2q).
```

The sufficient integer q makes this <= theta Delta. Logarithms are stored to
avoid overflow/underflow. A sample-specific bound uses
`lambda_1 k ratio^(2q) ||Z2||_2^2 ||Z1^(-1)||_2^2` with `Z=U.T Omega`;
its inverse norm is evaluated as `1/sigma_min(Z1)`, not a formed inverse.

Reorthogonalizing between multiplications represents the same column space in
exact arithmetic. That representation is used ONLY in the initialization-only
experiment. It is not claimed to produce the same scaled factor or PGD path.
Monte Carlo intervals are pointwise Wilson intervals, not simultaneous bands.

## Supported normal spectrum and rate (6.2--6.9)

Compute ALL positive selected--omitted mixing curvatures
`G={1-lambda_j/lambda_i: i<=k<j<=r, lambda_i>lambda_j}`.

```text
rho_eta = max({abs(1-2 eta)} union {1-eta gamma: gamma in G})
```

The set may be empty even for nonscalar Lambda (when k=r). If empty, the optimum
is eta=1/2 with zero normal derivative and the quadratic regime. Otherwise
`mu=min(G)`, `eta*=2/(2+mu)`, `rho*=(2-mu)/(2+mu)`. At a strict positive-support
cutoff, mu=Delta/lambda_k. Equal eigenvalues are input structural data, not
rounded into ties using a tolerance.

This is an eventual upper factor for general trajectories. Equality requires
an excited slow mode; tests distinguish pure and mixed perturbations. Local
optimal damping does not optimize the entire random-start transient.

## Jedra--Shah comparison

Reference: Yassir Jedra and Devavrat Shah, *k-SVD with Gradient Descent*,
arXiv:2502.00320v2, Lemma 1, equations (12)--(14), and Theorem 1.
For k=1, `x0=M omega` and eta=1/2 are the same unaccelerated iteration.
With `alignment=|u1.T x0|/||x0||`, the norm-annulus quantities are

```text
a_JS = 2 sqrt(eta(1-eta)) max(alignment, sqrt(lambda_r/lambda_1))
b_JS = 2(1-eta) + sqrt(eta/(1-eta))
       * min(1/alignment, sqrt(lambda_1/lambda_r))
tau_JS = max(1, ceil(4/(3 eta) log( ((1-eta)s+eta/s)/2 )))
s = ||x0||/sqrt(lambda_1).
```

Use these with supported nonzero initialization. They concern bounded norms,
not entry into the trace band or the eventual-rate neighborhood. We do not
instantiate unspecified constants in their main theorem or reproduce the
questioned overlap-free prefactor in their Lemma 3. The spectral reference
`rho_(1/2)=1-(lambda_1-lambda_2)/(2 lambda_1)` is the manuscript's rank-one
asymptotic result, not a fitted competing algorithm.
