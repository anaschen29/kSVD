# Rank-one reporting cleanup (no new optimization)

Use this dedicated report for the rank-one comparison's paper tables. It reads
existing `jedra_shah_rank1` case JSON and the manifest; it does not run PGD,
refit convergence slopes, or change plotting. The generic `ksvd-summarize`
command and its original raw-file counts remain unchanged for compatibility.

```bash
python -m pip install -e '.[test]'
ksvd-rankone-report \
  --input results/jedra-collection-v2 \
  --out reports/jedra-reporting-v3
```

The reporter is also a standalone standard-library program:

```bash
python src/ksvd/rankone_reporting.py \
  --input results/jedra-collection-v2 \
  --out reports/jedra-reporting-v3
```

Use a new output directory. Overlapping raw/report directories and nonempty
report directories are rejected. The raw manifest and all case files are
read-only. The ZIP archive should be extracted before invoking this command.

## Outputs

- `hitting_times.csv`: one row per retained analysis case, with norm-annulus,
  strict-band, fixed-margin-band, absolute factor-accuracy, and relative
  product-accuracy first hits side by side. The initial overlap, annulus bounds,
  saved norm-entry estimate, final accuracy, no-later-observed-exit flags,
  termination reason, and saved spectral fit are also retained.
- `events.csv`: full per-event details, including the values and leading
  alignment at first entry, missing observations, observed-tail checks, and
  whether the first-hit prefix contains every iteration starting at zero.
- `duplicate_mapping.csv`: every original ID and SHA-256, its representative
  ID, nominal seed, fingerprint and inclusion flag. No source file is deleted.
- `summary.json`: full provenance, raw versus retained counts, all cases and
  per-configuration aggregates. Medians use hitting cases only and include
  available/hitting denominators. No confidence interval is created for
  deterministic controls.
- `README.md`: a readable first-hit table and interpretation notes.

The analysis records its own source SHA-256, source manifest hash, individual
case hashes, original run metadata/configuration and effective thresholds.

## Deduplication rule

Only explicitly labeled `controlled` cases are candidates. Two cases must
have the same non-seed parameters, complete saved result payload (including
problem, initial state, trajectory measurements, final state/status and theory),
and recorded execution/source identifiers. A hash of that evidence identifies
exact stored replays. The lexicographically first case ID is the representative;
all source IDs and nominal seeds remain linked to it. Wall-clock duration and
the nominal seed label are not part of the equivalence test.

Gaussian cases are **never** deduplicated, even when some diagnostics agree.
Controls with different initial states, observations, problem, solver settings,
precision or source remain separate. Failures and controls lacking the required
initial-state/problem/source evidence also remain separate. This rule is
conservative: differing budgets or records are not combined into a guessed
single trajectory. Equality of saved evidence is not a statement about
unrecorded floating-point states. Retained rows are not automatically
independent stochastic samples.

For the supplied 45-file collection, this yields 21 rows: 15 Gaussian runs
and six deterministic controls. The 24 excluded replay files remain in the
provenance table. This is a data-counting correction, not a discarded failure.

## Event definitions

The default comparisons are made directly against stored diagnostics, without
rounding or introducing a hidden tolerance:

1. Norm annulus: `a_js <= normalized_norm <= b_js` (both endpoints included).
2. Strict trace band: `trace_over_gap < 1` (equality is excluded).
3. Fixed-margin band: `trace_over_gap <= 0.5` (equality is included).
4. Factor accuracy: `factor_error <= 1e-8` (absolute rank-one factor error).
5. Product accuracy: `product_error <= 1e-8` (relative product error).

The annulus constants and `tau_js` are copied from the source's theory fields;
this reporter does not recompute or independently verify their derivation.
The rate fit is also copied and explicitly labeled `saved_rate_fit`.

Overrides are explicit and saved in the report:

```bash
ksvd-rankone-report --input results/jedra-collection-v2 \
  --out reports/jedra-margin-quarter \
  --fixed-margin 0.25 --factor-tolerance 1e-9 --product-tolerance 1e-9
```

A first observed hit is not an infinite-time convergence claim. A trajectory
can leave a region after first entering it; the report separately records the
first hit with no later observed exit and whether the original hit remains
satisfied in the saved tail. Sparse or missing measurements are flagged rather
than interpolated. Missing final diagnostics or a failed terminal evaluation
make final membership unknown, not a successful last-good value. A first hit
at iteration zero is retained as zero; blank CSV fields never mean zero.

The strict trace band may be entered with an arbitrarily small margin and poor
factor accuracy. Keep the fixed-margin and factor/product columns separate.
None of these event times should be silently labeled entry into an unspecified
asymptotic neighborhood.

## Validation

The 35 new reporting-only tests use synthetic saved records, not numerical
optimization. They cover exact replay detection, Gaussian preservation,
differing or missing evidence, failures, strict/inclusive boundaries,
iteration-zero hits, exits after entry, sparse observations, unavailable
terminal metrics, threshold overrides, manifest identity/path checks,
read-only source hashes and standalone execution with optimizer imports
blocked. The supplied collection was also reprocessed without rerunning it.
The existing numerical experiment tests are unchanged.
