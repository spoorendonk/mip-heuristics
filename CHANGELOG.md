# Changelog

Notable changes to options and behaviour. Measurements live in the issues
linked from each entry.

## Unreleased

### Added

- **`mip_heuristic_fj_gpu`** (bool, default `false`; #265, #282, #283): in a
  `MIP_HEURISTICS_CUDA` build, an option-on run races a batch of GPU
  FeasibilityJump climbers, one per streaming multiprocessor, against FJ's
  CPU workers in the main MIP's first FJ presolve dispatch (not in sub-MIPs
  or after a root restart). Both offer into one pool, the
  climbers' effort is charged with FJ's, and the run is not reproducible,
  even at `threads=1`. A build without `MIP_HEURISTICS_CUDA` that runs FJ
  warns once per solve and runs FJ on the CPU only, as a GPU build does when
  its device setup fails. A default run is unchanged.
- **`mip_heuristic_scylla_cuopt_pump`** (bool, default `false`; #290, part of
  #211): Scylla's pump projects with cuOpt's technique instead of Mexi et
  al.'s: an exact L1 distance (auxiliary columns for general integers), an
  objective weight that decays every projection and resets on an
  improvement, a PDLP tolerance read off how integral the last LP point is,
  a warm start from the last rounding, and FPR with repair on every chain.
  A default run, and a Scylla run without the option, are unchanged.

## v0.1.11 (2026-11-12)

No option was added, removed or changed its default, and a default run is
unchanged.

### Changed

- **Linux build and test fixes for HiGHS v1.16.0** (#274, #275): the core
  boundary is checked by `core_boundary` alone, the stripped-archive link
  that v1.16 broke is gone, and the dive-time `fpr_lp` tests set up their
  own LP-iteration envelope.
- **GPU FJ step loop** (#263, GPU build only): a GPU climber takes
  upstream FJ's `solve` steps. Nothing calls it yet.

## v0.1.10 (2026-11-12)

### Changed

- **HiGHS v1.15.1 → v1.16.0** (#267). A default run changes with upstream:
  its presolve, B&B and `trySolution` changed, and B&B now searches with
  up to `threads` workers at the default `parallel=choose` (v1.15.1 ran one
  unless `parallel=on`). `fpr_lp` (off by default) skips every dive while
  B&B has more than one worker, so with it on, a multi-core run gets it only
  during the single-worker ramp-up; `parallel=off` restores every dive.
  FeasibilityJump is now
  upstream's own: v1.16.0 carries both #139 fixes, so the patch no longer
  makes them, and upstream's version of the negative-coefficient fix also
  drops the empty-interval skip and the equality row's `(rhs, rhs)`
  interval. No option was added, removed or changed its default.

## v0.1.9 (2026-11-11)

### Added

All off by default, so default runs are unchanged.

- **Recombination** (#209, #227, #229, #234, #235): a presolve chain stage
  after ruin-and-repair that recombines pool solutions, enabled by
  `mip_heuristic_recombine_effort` above zero, with
  `mip_heuristic_recombine_patience` (both default 0).
- **FJ's LP start persona** (#217): `mip_heuristic_fj_lp_persona` (default
  `false`); with `fpr_lp` on, its last worker runs FJ from the dive's LP
  solution, randomly rounded.
- **FJ's cardinality exchange** (#218): `mip_heuristic_fj_exchange` (default
  `false`), a compound move on uniform binary equality rows.

### Removed

- **Line-segment search** (#213, #246, #247; removed in #254): the
  recombiner's line-segment arm and `fpr_lp`'s line-segment lane, together
  with the option `mip_heuristic_line_segment`. It was off by default, and
  #252 measured no significant gain. Added and removed since v0.1.8, so no
  tagged release accepts the option name.
- **Objective cut** (#208, #222-#225, #231-#233; removed in #255): the
  presolve heuristics' search under `objective <= best - max(rel*|best|, abs)`
  once the pool held a solution, in FJ, LocalMIP, FPR, Scylla's rounding and
  its PDLP LP, together with the options `mip_heuristic_objective_cut`,
  `mip_heuristic_objective_cut_rel` and `mip_heuristic_objective_cut_abs`.
  It was off by default and measured worse on every seed (primal integral
  1.091). Default runs are unchanged; HiGHS now rejects the
  three option names as unknown. Ruin-and-repair's own cutoff row is kept.
- **CDCL lane** (#236, #237; removed in #257): the CDCL SAT solver and
  row encoder for pure-binary models, run as FJ's last presolve worker's
  first attempt while no feasible point was known, together with the
  option `mip_heuristic_fj_cdcl`. It was off by default, and #212
  measured no significant gain (primal integral 0.986 [0.969, 1.003]).
  Added and removed since v0.1.8, so no tagged release accepts the option
  name.
