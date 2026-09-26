# Experimental protocol

## Order of work

1. Unit tests for identities, derivatives, coordinates, certificates, and edge cases.
2. Small smoke configurations of each study, followed by figure inspection.
3. Review the saved diagnostics and choose paper-scale budgets explicitly.
4. Freeze configs/seeds before interpreting final results; save all failures.

No full experimental sweep is implied by successful smoke validation. The
current code covers the five planned study families plus the optional damping
stress control. Sequential deflation versus block-PGD runtime benchmarking and
q-plus-PGD total-work comparisons remain separate future experiments, not
claims made by the present suite.

## Trace contraction

Construct e0=theta Delta exactly in real arithmetic. A mixture of a positive-tail
direction and a nullspace direction gives genuinely ambient starts. The
right SPD factor uses random eigenvectors, so the Gram and compression need
not commute. Vary gap, theta, damping, and Gram condition independently.
Compute the theorem envelope from X0 and X1, not by regression. The raw trace
ratios exclude e_t <= 1e-12 from the ratio diagnostic, but raw values are retained.
A theta>=1 override is outside the certificate and must still be geometrically
attainable by the selected one-direction construction.

## Gaussian power sketches

Within a trial reuse exactly the same Omega for all q. Save every trial's e_q,
not just averages. QR between powers stabilizes the subspace-only measurement.
Probability plots show pointwise 95% Wilson intervals. Median, 90th, and 95th
quantiles are descriptive sample quantiles. Bounds are displayed in log space.
If the sufficient q exceeds the smoke q range, annotate that fact rather than
suggesting the finite q range tested the theorem's sufficient condition.

## Normal modes, damping, and quadratic convergence

Use a known canonical minimizer and one perturbed column in Y coordinates.
The radial component has signed linear multiplier 1-2 eta; the slow mixing
component has multiplier 1-eta Delta/lambda_k. Other columns remain fixed in
this controlled construction, so the actual limiting factor is known without
using the final numerical iterate as a surrogate truth.

Fit the last 30 eligible samples with errors between 1e-11 and 1e-2 by default,
requiring at least 8 points. Save the range, point count, slope and R-squared.
There is no fit when the window is too short. The fitting rule does not use the
predicted rho to select a favorable window. Some pure modes need not attain
the worst-case spectral rate. A slope reference in the figure is anchored and
explicitly not a global error envelope.

The damping sweep also uses raw Gaussian starts and a common relative product
error threshold 1e-8. Aggregate iteration plots label reached/attempted counts;
medians are among target-reaching runs only. Max-step outcomes are censored,
not claimed successes. Total timing includes diagnostic work.

The k=r nonscalar example is diagonal in Y and converges to I at eta=1/2.
Optional mpmath scalar Heron trajectories give an independent 100-digit
reference. Decimal errors and log errors are retained. This is not an arbitrary
precision implementation of every experiment.

## Tied minimizers

Use diag(3,2,2,1), k=2, then split the tie to diag(3,2+epsilon,2,1).
Track the true problem's minimizer residual, the unsplit family's residual,
distance to a single reference orbit, and the selected line in coordinates 2--3.
The line angle is modulo pi; early ambiguous angles are flagged. Plotting may
unwrap the angle for continuity but does not change stored values. Endpoints
are observed finite-run endpoints, not assertions that every run has converged
or that their empirical distribution is exactly uniform.

## Rank one and the damping boundary

The Jedra--Shah study has one implementation of the common half-step optimizer.
Separate Gaussian starts from controlled starts with prescribed top overlap.
Controlled tails include both u2 and a lower mode, permitting outside-band
starts while retaining the slow asymptotic mode. Compare norm-annulus entry,
trace-band entry, and late spectral behavior, not two identical solvers' runtime.

Ill-conditioned supported starts stress the open damping interval. The scalar
eta=1 reciprocal two-cycle is explicitly outside the theorem. The cycle detector
requires an exact repeated floating-point state and a nontrivial step. Approximate
recurrence alone could mislabel slowly damped radial oscillations as a cycle.
Small steps alone are never interpreted as global convergence.

## Reproducibility and interpretation

Seeds are explicit, PyTorch CPU uses one thread by default, and the source hash,
versions, device, and full config are saved. Bitwise equality across devices or
library versions is not promised. Use numerical tolerances appropriate to each
identity; finite-difference tests use looser tolerances than exact algebra.
No NaNs/Infinity are silently converted into plausible measurements. Numerical
failures are serialized with status and message. Raw arrays precede plotting.
