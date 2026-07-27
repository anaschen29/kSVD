# Mathematical reference

## Spectral reduction

Let

    M = U_r Lambda U_r^T,

where Lambda = diag(lambda_1, ..., lambda_r) is strictly positive.

For X in range(M)^{x k}, write

    X = U_r Xbar,
    Xbar = Lambda^{1/2} Y.

## Reduced update

    A(Y) = Y^T Lambda Y

    Y_{t+1}
      = (1 - eta)Y_t
        + eta Lambda Y_t A(Y_t)^{-1}.

## Potential

    F(Y)
      = 1/2 ||Y||_F^2
        - 1/2 log det(Y^T Lambda Y).

## Objective

    g(Xbar)
      = 1/4 [
          ||Lambda||_F^2
          - 2 tr(Xbar^T Lambda Xbar)
          + ||Xbar^T Xbar||_F^2
        ].

    g_star = 1/4 sum_{i=k+1}^r lambda_i^2.

## Predicted local rate

When k < r and lambda_k > lambda_{k+1},

    mu_k = (lambda_k - lambda_{k+1}) / lambda_k

and

    rho_eta
      = max(
          |1 - 2 eta|,
          1 - eta mu_k
        ).

The objective gap is predicted to contract asymptotically at rho_eta^2 for a
generic trajectory containing the slowest normal mode.

## Locally optimal step

    eta_local_star = 2 / (2 + mu_k)

    rho_local_star = (2 - mu_k) / (2 + mu_k).

## Certified sublevel step

The authoritative definitions and derivation are in
`docs/preconditioned_gd_convergence.tex`, section “A sufficient step-size
bound.” The threshold is sufficient rather than sharp and the theorem requires
the strict inequality `0 < eta < eta_C`.

## Exact parameterization

For k = r, there are no cross-subspace modes. The normal linear factor is

    |1 - 2 eta|.

At eta = 1/2, convergence to the minimizer manifold is locally quadratic.
