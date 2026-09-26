# kSVD PGD experiments

This repository contains a PyTorch implementation of preconditioned gradient
descent for low-rank positive-semidefinite matrix approximation:

```text
g(X) = 1/4 ||M - XX^T||_F^2.
```

The reusable numerical package lives in `src/ksvd`, and the mathematical and
experimental specifications live in `docs/`.

