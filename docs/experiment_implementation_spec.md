# Experimental implementation specification

This document is the authoritative implementation specification for the
preconditioned-gradient-descent experiments.

Codex must preserve the stated formulas, metrics, initialization definitions,
and experiment semantics. Proposed code snippets are illustrative; they may be
adapted to the repository architecture, but mathematical behavior must not be
changed.

## Delivery phases

### Phase 1: Core numerical library

Implement:

- problem representation;
- coordinate transformations;
- reduced and ambient dynamics;
- objective and potential;
- theory constants;
- metrics;
- initialization constructors;
- generic runner;
- unit tests.

Do not run scientific experiments in this phase.

### Phase 2: Theorem-validation experiments

Implement:

1. predicted local rates;
2. Hessian-mode isolation;
3. boundary-gap scaling;
4. tied eigenvalues;
5. geometry of K_C;
6. saddle escape;
7. step-size phase diagram;
8. initialization ablation.

Only minimal smoke configurations may be run until full sweeps are explicitly
approved.

### Phase 3: Algorithmic comparisons

Implement:

- simultaneous block preconditioned GD;
- sequential k=1 method with deflation;
- block power iteration;
- unpreconditioned gradient descent;
- fair matvec-equivalent and runtime accounting.

### Phase 4: Reproducible experiment execution

Add:

- configuration files;
- result manifests;
- resumable sweeps;
- aggregation;
- publication-quality figures;
- tables and captions.
