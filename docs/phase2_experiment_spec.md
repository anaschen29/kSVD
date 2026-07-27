# Phase 2 Experiment Specification

This document is the authoritative specification for the eight Phase 2 experiments. It fixes the semantics that experiment wrappers must implement. Configuration files may override numerical defaults, but must not change the mathematical definitions or recorded metrics.

## Common conventions

All theory-facing experiments run in reduced coordinates with

\[
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_r),\qquad
Y_{t+1}=(1-\eta)Y_t+\eta\Lambda Y_t(Y_t^\top\Lambda Y_t)^{-1}.
\]

Use `torch.float64`, explicit seeds, and linear solves rather than matrix inverses.

### Common per-iteration records

Every iterative experiment records:

- iteration index;
- potential `F`;
- objective and objective gap;
- factor-manifold distance when the top-\(k\) subspace is unique;
- subspace error;
- tied-optimal-family error when applicable;
- gradient norm;
- step norm;
- minimum and maximum singular values of \(Y_t\);
- condition number of \(Y_t\);
- empirical one-step error ratio when defined;
- whether the potential decreased on that step.

### Common stopping and failure rules

Unless an experiment overrides them:

- `max_steps = 20_000`;
- convergence tolerance is `1e-10` for the relevant geometric error;
- objective-gap tolerance is `1e-14`;
- stop successfully when either the relevant geometric error or objective gap reaches tolerance and remains finite;
- classify as `lost_rank` if the numerical rank is below \(k\) using tolerance `100 * eps * sigma_max`;
- classify as `diverged` if any norm or objective exceeds `1e12`;
- classify as `nan` if any recorded quantity is non-finite;
- detect period-two cycling by comparing \(Y_t\) with \(Y_{t-2}\) after Procrustes alignment.

### Rate fitting

Estimate an asymptotic factor only from iterates satisfying

\[
10^{-9}\le e_t\le 10^{-3},
\]

with at least ten usable points. Report both:

\[
\widehat\rho_{\rm ratio}=\operatorname{median}_t e_{t+1}/e_t
\]

and the least-squares slope of \(\log e_t\) against \(t\). Do not fit after the error reaches floating-point noise.

### Replication

Full sweeps use 20 seeds unless otherwise stated. Smoke tests use two seeds and at most three values of each swept variable.

## Experiment 1: Predicted local rates

### Purpose

Validate

\[
\rho_\eta=\max\left\{|1-2\eta|,\,1-\eta\frac{\lambda_k-\lambda_{k+1}}{\lambda_k}\right\}
\]

and the objective-gap factor \(\rho_\eta^2\).

### Independent variables

- \(k\in\{1,2,4,8\}\);
- \(\eta\in\{0.1,0.25,0.5,0.7,\eta_{\rm loc}^\star\}\), where
  \[
  \eta_{\rm loc}^\star=\frac{2}{2+\mu_k},\qquad
  \mu_k=\frac{\lambda_k-\lambda_{k+1}}{\lambda_k};
  \]
- 20 seeds.

### Fixed construction

Set \(r=\max\{20,2k+2\}\). Use \(\lambda_k=1\), \(\lambda_{k+1}=0.7\), linearly spaced top eigenvalues from \(2\) to \(1\), and a geometric tail beginning at \(0.7\) with ratio \(0.8\).

Initialize locally by

\[
Y_0=E+10^{-2}H,
\]

where \(H\) is a seeded Gaussian perturbation projected onto the normal space of the minimizer manifold and normalized to unit Frobenius norm.

### Comparisons and outputs

- empirical factor-error ratio versus \(\rho_\eta\);
- empirical objective-gap ratio versus \(\rho_\eta^2\);
- log-error curves with theoretical slope overlays;
- table of predicted and fitted rates, absolute error, and relative error.

## Experiment 2: Hessian-mode isolation

### Purpose

Validate every eigenmode of the Jacobian at a global minimizer.

### Fixed construction

Use \(r=8\), \(k=3\),

\[
\lambda=(2.0,1.5,1.0,0.7,0.5,0.35,0.2,0.1),
\]

and \(\eta\in\{0.25,0.5,0.7\}\). Let \(E=[I_k;0]\).

### Mode families

1. Symmetric top-block basis directions, with predicted Jacobian factor \(1-2\eta\).
2. Cross-subspace directions \(H=e_je_i^\top\), \(i\le k<j\), with predicted factor
   \[
   1-\eta\left(1-\frac{\lambda_j}{\lambda_i}\right).
   \]
3. Skew-symmetric tangent directions, with predicted Jacobian factor \(1\).

### Procedure

Use the centred finite-difference Jacobian action

\[
\frac{T_\eta(E+\varepsilon H)-T_\eta(E-\varepsilon H)}{2\varepsilon},
\qquad \varepsilon\in\{10^{-4},10^{-5},10^{-6}\}.
\]

### Outputs

- predicted versus measured factor for every mode;
- relative error versus \(\varepsilon\);
- heat map indexed by \((i,j)\) for cross-subspace modes.

## Experiment 3: Boundary-gap scaling

### Purpose

Test whether the asymptotic iteration count scales as

\[
\frac{\lambda_k}{\lambda_k-\lambda_{k+1}}\log\frac1\epsilon.
\]

### Independent variables

- \(k\in\{1,2,4,8\}\);
- \(\delta\in\{10^{-4},3\cdot10^{-4},10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},10^{-1},0.3,0.5\}\);
- initialization type in `{local_normal, support_gaussian}`;
- 20 seeds for random initialization and 5 seeds for local perturbations.

### Spectrum

Set \(r=\max\{20,2k+2\}\), \(\lambda_k=1\), \(\lambda_{k+1}=1-\delta\), linearly spaced top eigenvalues from \(2\) to \(1\), and a geometric tail below \(1-\delta\) with ratio \(0.8\).

Use \(\eta=1/2\).

### Outputs

- fitted asymptotic rate versus \(1-\delta/2\);
- iterations from entry into the fitted linear regime to tolerance versus \(1/\delta\);
- total iterations for support-Gaussian initialization versus \(1/\delta\);
- separate reporting of transient and linear-regime iterations.

## Experiment 4: Tied eigenvalues

### Purpose

Show that a boundary tie enlarges the optimal family rather than causing failure.

### Independent variable

\[
\delta\in\{0,10^{-4},10^{-3},10^{-2},10^{-1}\}.
\]

### Fixed construction

Use \(r=10\), \(k=4\),

\[
\lambda=(2,1.5,1,1,1-\delta,1-\delta,0.6,0.4,0.2,0.1).
\]

At \(\delta=0\), the mandatory top space is \(\operatorname{span}(e_1,e_2)\), and the tied block is \(\operatorname{span}(e_3,e_4,e_5,e_6)\); the optimizer may choose any two-dimensional subspace from that tied block.

Use support-Gaussian initialization, \(\eta=1/2\), and 50 seeds.

### Outputs

- canonical top-\(k\) projector error;
- tied-optimal-family error;
- objective gap;
- distribution of final principal angles within the tied block;
- rate and iteration count as \(\delta\downarrow0\).

Include an isotropic sanity case \(\Lambda=I_r\), verifying exact column-space preservation and quadratic convergence of singular values for \(\eta=1/2\).

## Experiment 5: Geometry of \(\mathcal K_C\)

### Part A: exact \(r=2,k=1\) visualization

Use \(\Lambda=\operatorname{diag}(2,0.5)\). Evaluate \(F(y)\) on a `601 x 601` grid over \([-3,3]^2\), excluding the origin. Plot contours at

\[
C=C_\star+\{0.05,0.2,0.5,1,2\},
\qquad
C_\star=\frac12-\frac12\log2.
\]

Overlay global minima, nonglobal critical points, and trajectories from a fixed grid of initial points.

### Part B: block slice \(r=3,k=2\)

Use \(\Lambda=\operatorname{diag}(2,1,0.4)\),

\[
Q(\theta)=[e_1,\cos\theta\,e_2+\sin\theta\,e_3],
\]

and evaluate the two slices

\[
Y(\theta,s)=Q(\theta)\operatorname{diag}(e^s,e^{-s}),
\]

\[
Y(\theta,a)=aQ(\theta),
\]

on \(\theta\in[0,\pi/2]\), \(s\in[-3,3]\), and \(a\in[10^{-2},10^2]\).

### Part C: empirical versus certified bounds

For selected levels \(C\), sample points in \(\mathcal K_C\) by rejection sampling from mixtures over scale, condition number, and Haar subspaces. Record empirical extrema of

- \(\|Y\|_F^2\);
- \(\lambda_{\min}(Y^\top\Lambda Y)\);
- \(\|\nabla^2F(Y)\|_{\rm op}\).

Compare them with \(S_C\), \(a_C\), and \(\mathcal L_C\). Report bound-to-empirical ratios. Label these as empirical lower bounds on conservativeness, not exact extrema.

## Experiment 6: Saddle escape

### Fixed construction

Use \(r=8\), \(k=3\),

\[
\lambda=(2,1.5,1.0,0.7,0.5,0.35,0.2,0.1).
\]

Choose the nonglobal invariant subspace indexed by \(I=\{1,2,4\}\). Let \(Y_s\) have columns \((e_1,e_2,e_4)\). The unstable direction replaces selected eigenvalue \(\alpha=\lambda_4\) by omitted eigenvalue \(\beta=\lambda_3\):

\[
H_-=e_3 e_3^\top
\]

where the column index on the right refers to the third column of \(Y_s\). The negative Hessian eigenvalue is \(-\gamma\), with

\[
\gamma=\frac{\beta}{\alpha}-1.
\]

### Independent variables

- \(\varepsilon\in\{10^{-12},10^{-11},\ldots,10^{-2}\}\);
- \(\eta\in\{0.25,0.5,0.75\}\);
- perturbation sign \(\pm\).

Initialize \(Y_0=Y_s\pm\varepsilon H_-\). Define escape time as the first \(t\) such that the Procrustes-aligned distance to \(Y_sO(k)\) exceeds \(c=0.1\).

### Comparison

Compare the measured escape time with

\[
\frac{\log(c/\varepsilon)}{\log(1+\eta\gamma)}.
\]

Also verify that exact initialization at \(Y_s\) remains fixed.

## Experiment 7: Step-size phase diagram

### Purpose

Compare the sufficient certified threshold with empirical monotone-descent and convergence cutoffs.

### Problem families

Use \(r=20,k=4\) and three spectra:

1. well separated: boundary gap \(0.5\), condition number \(20\);
2. small gap: boundary gap \(0.01\), condition number \(20\);
3. ill conditioned: boundary gap \(0.2\), condition number \(10^4\).

For each spectrum use initialization families `{support_gaussian, reduced_gaussian, orthonormal, ill_conditioned}` with five seeds per family.

### Step grid

For every initialization compute \(\eta_C\). Use the sorted union of

- \(\{0.1\eta_C,\eta_C,10\eta_C,100\eta_C\}\);
- 60 logarithmically spaced values from `1e-8` to `1e-1`;
- 141 linearly spaced values from `0.1` to `1.5`;
- \(1/2\), \(1\), and \(\eta_{\rm loc}^\star\).

Remove duplicates within relative tolerance `1e-12`.

### Classifications

- monotone convergence;
- nonmonotone convergence;
- bounded nonconvergence/cycle;
- lost rank;
- divergence/non-finite.

Define the empirical monotone cutoff as the largest tested \(\eta\) for which all steps before convergence satisfy the descent inequality numerically. Define the empirical convergence cutoff as the largest tested \(\eta\) classified as convergent. These are grid-dependent estimates and must be labelled as such.

### Outputs

- phase diagram over \((\eta,\text{initialization})\) and aggregated over \((\eta,C)\);
- overlays of \(\eta_C\), \(1/2\), and \(\eta_{\rm loc}^\star\);
- empirical late-stage rate versus \(\rho_\eta\);
- ratios \(\widehat\eta_{\rm desc}/\eta_C\) and \(\widehat\eta_{\rm conv}/\eta_C\).

## Experiment 8: Initialization ablation

### Fixed problem

Use \(r=20\), \(k=4\), boundary gap \(0.2\), condition number \(20\), and \(\eta=1/2\) for the primary experiment. On a small secondary subset also run \(\eta=0.5\eta_C\) to confirm certified behaviour.

### Groups

1. Support Gaussian: \(\overline X_0=\Lambda Z\).
2. Reduced Gaussian: \(\overline X_0=Z\).
3. Orthonormal: \(\overline X_0=Q\).
4. Scale sweep: \(X_0=sX_{\rm base}\), \(s\in\{10^{-4},10^{-3},\ldots,10^4\}\).
5. Condition sweep: prescribed \(\kappa(X_0)\in\{1,10,10^2,\ldots,10^8\}\) at fixed subspace and Frobenius norm.
6. Overlap sweep: prescribed
   \[
   \alpha_0=\sigma_{\min}(U_k^\top Q_0)\in\{10^{-8},10^{-7},\ldots,1\}.
   \]
7. Near-saddle sweep using the saddle family from Experiment 6.
8. Ambient versus support initialization for singular \(M\), verifying exact null-space decay.

Use 20 seeds for stochastic groups and 10 seeds for controlled constructions.

### Initial descriptors

Record

- \(F(Y_0)\);
- \(\|Y_0\|_F\);
- \(\sigma_{\min}(Y_0)\);
- \(\kappa(Y_0)\);
- \(\alpha_0\);
- distance to the nearest tested nonglobal saddle family;
- \(\eta_C\).

### Transient definition

Let \(e_t\) be factor-manifold distance. Define \(T_{\rm lin}\) as the first index \(T\) for which at least 20 consecutive usable ratios satisfy

\[
\left|e_{t+1}/e_t-\rho_\eta\right|\le 0.1(1-\rho_\eta)
\]

while \(10^{-9}\le e_t\le10^{-3}\). If no such window exists, report `T_lin = null`.

### Outputs

- transient time and total iterations versus \(-\log\alpha_0\), \(\log\kappa(Y_0)\), \(|\log s|\), and \(F(Y_0)\);
- asymptotic fitted rate versus the same descriptors;
- success/failure rate by initialization family;
- descriptive regression for transient time, clearly labelled exploratory rather than theoretical.
