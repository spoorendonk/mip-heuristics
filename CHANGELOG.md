# Changelog

Notable changes to options and behaviour. Measurements live in the issues
linked from each entry.

## Unreleased

### Added

- **`mip_heuristic_long_lived`** (bool, default `false`; #332, part of
  #311): the skeleton of long-lived mode. With it on, the main MIP's solve
  keeps an owner from setup to `cleanupSolve` whose threads queue
  solutions, and only HiGHS's solving thread installs them, at the
  external-solution poll points and once more at the end. No heuristic runs
  long-lived yet, so an option-on run finds what a default run does; a
  default run allocates nothing and starts no thread. The owner also keeps
  one solution pool per epoch for its producers (#333): seeded from the
  presolve chain's final pool, and given every new HiGHS incumbent at the
  next drain.
- **`mip_heuristic_fj_gpu`** (bool, default `false`; #265, #282, #283): in a
  `MIP_HEURISTICS_CUDA` build, an option-on run races a batch of GPU
  FeasibilityJump climbers, one per streaming multiprocessor, against FJ's
  CPU workers in the main MIP's first FJ presolve dispatch (not in sub-MIPs
  or after a root restart). Both offer into one pool, the
  climbers' effort is charged with FJ's, and the run is not reproducible,
  even at `threads=1`. A build without `MIP_HEURISTICS_CUDA` that runs FJ
  warns once per solve and runs FJ on the CPU only, as a GPU build does when
  its device setup fails. The CUDA context is created on a thread of its
  own from before presolve, with blocking-sync scheduling, and the race
  never waits for it: where CPU FJ ends first, FJ's dispatch ends without
  the GPU. Only the solve's end waits for a creation still running
  (#289), and the race's end stops a related-variable table still being
  built (#304). A default run is unchanged.
- **`mip_heuristic_scylla_cuopt_pump`** (bool, default `false`; #290, part of
  #211): Scylla's pump projects with cuOpt's technique instead of Mexi et
  al.'s: an exact L1 distance (auxiliary columns for general integers), an
  objective weight that decays every projection and resets on an
  improvement, a PDLP tolerance read off how integral the last LP point is,
  a warm start from the last rounding, and FPR with repair on every chain.
  Two cycle detectors, a ring of recent infeasible roundings and a window
  of L1 distances, replace Mexi's three-deep window (#295); on a cycle the
  pump runs FJ from the rounding before it perturbs (#297), and each
  rounding FPR cannot repair earns FJ 0.2 of the round's projection and
  rounding effort as a credit, spent on one FJ run from a failed rounding
  once the credit reaches FJ's callback period (#298). #302 measured it level with Mexi's pump (primal
  integral 0.984 [0.949, 1.020], 25 instances); with Scylla on it was
  2.092 [1.538, 2.844] against main (Mexi's pump: 2.125), and neither pump
  found an improving incumbent. A default run, and a Scylla run without the
  option, are unchanged.
- **`[Heur] gpu_steps=`** (#305): when FJ's GPU race ran, FJ's `[Heur]`
  line ends with how many times the climbers launched; `0` is a race in
  which no climber stepped. Measured with it, the race stepped no climber
  in 25 of 25 runs on 5 instances. A default run is unchanged.
- **`[Heur] stopped_by=`** (#303): every presolve-chain and `fpr_lp`
  `[Heur]` line ends with what stopped the dispatch — `deadline`,
  `interrupt`, `budget`, `patience` or `done` — appended after
  `abandoned_setup`, so older parsers still read the line. The `early_fj`
  line carries none. A default run's search is unchanged.

### Changed

- **No `cudaDeviceReset()` after a GPU PDLP solve** (#299): in a
  `MIP_HEURISTICS_CUDA` build, a Scylla run no longer crashes when RINS
  sub-MIPs run PDLP on several threads at once. Upstream's cuPDLP-C wrapper
  reset the device after every solve, destroying the CUDA context under the
  other threads. A default run, and every CPU build, are unchanged.
- **Early FJ checks its point in the presolved space** (#306): with
  `mip_heuristic_early_fj_effort` on, the point handed to HiGHS as the MIP
  start must also be feasible once crushed into the presolved model, since
  `runSetup` takes the upper bound from the crushed point's objective. One
  that is not is dropped, with a `dropped: infeasible in the presolved
  model` log line, rather than setting a cutoff no presolved point reaches.
  A dropped point is no longer the answer of a run whose search finds
  nothing, and the benchmark parser withdraws its `found at` incumbents.
  A default run is unchanged.
- **Early FJ's reported bound and vector agree** (#343): with
  `mip_heuristic_early_fj_effort` on, a point whose crushed image has
  another objective (presolve fixed a column it sets otherwise, say) is
  handed over as that image's postsolve, logged as `postsolved from its
  presolved image`, so the primal bound `runSetup` takes from the crushed
  point is the reported vector's objective. If the postsolved point does
  not attain it, the point is dropped. A default run is unchanged.
- **Early FJ calls log callbacks on the solving thread only** (#309): its
  `found at` lines still go to the console and the log file as found, and
  reach a user log callback after the worker is joined, ahead of the
  hand-over's lines. A default run is unchanged.
- **One-shot FJ runs hold to their budget** (#309): ruin's repair,
  recombination's local search and the cuOpt pump's cycle escape stop
  within one FJ step past FJ's setup plus their budget, where they could
  overshoot a budget below FJ's 500000-effort callback period (patch
  version 47). All three are off by default; a default run is unchanged.
- **A GPU find counts as FJ's progress** (#309): with
  `mip_heuristic_fj_gpu` and `mip_heuristic_fj_patience` above zero, a
  climber's improving offer resets the runner's patience gate, which
  could otherwise run out while the GPU was still improving. A default
  run is unchanged.
- **FJ's persona starts are charged and stop at the deadline** (#326): with
  `mip_heuristic_fj_personas` or `mip_heuristic_fj_exchange` on, the row
  structure pass counts against FJ's effort, and with personas on each
  persona start is charged to its slot's first attempt and stops at the
  solve's deadline. Solves with either option on may differ at fixed seeds,
  since FJ's budget now runs out sooner. Both are off by default; a default
  run is unchanged.

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
- **`MIP_HEURISTICS_CUDA` also builds the GPU FeasibilityJump** (#210,
  #261, #262): besides HiGHS's cuPDLP backend, a GPU tree compiles the
  `mip_heuristics_gpu` device code into `highs`, its targets set by the
  new `MIP_HEURISTICS_CUDA_ARCHITECTURES` (default `native`), with `[gpu]`
  tests. Nothing calls it in this release; CPU builds are unchanged.

## v0.1.9 (2026-11-11)

### Added

All off by default, so default runs are unchanged.

- **Recombination** (#209, #227, #229, #234, #235): a presolve chain stage
  after ruin-and-repair that recombines pool solutions, enabled by
  `mip_heuristic_recombine_effort` above zero, with
  `mip_heuristic_recombine_patience` (both default 0). Its finds carry the
  solution-source letter `O`. #244 later measured it slightly worse than
  main (primal integral 1.037 [1.004, 1.071], 15 instances).
- **FJ's LP start persona** (#217): `mip_heuristic_fj_lp_persona` (default
  `false`); with `fpr_lp` on, its last worker runs FJ from the dive's LP
  solution, randomly rounded.
- **FJ's cardinality exchange** (#218): `mip_heuristic_fj_exchange` (default
  `false`), a compound move on uniform binary equality rows. #249
  measured no gain (1.008 [0.995, 1.022] with personas on too, against
  main, 7 instances).

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

## v0.1.8 (2026-11-11)

### Added

All off by default, so default runs are unchanged by these.

- **Early FeasibilityJump** (#205, #215): `mip_heuristic_early_fj_effort`
  (double, default 0) runs one FJ worker on the original model while
  HiGHS presolves it, and hands its best feasible point to HiGHS as the
  MIP start when presolve returns. It is not part of the chain. #216 later
  measured it 2.19x better on the primal integral [1.68, 2.86] on 3
  slow-presolve instances, as logged by its `found at` lines (#240); HiGHS
  gets the point only when presolve ends, about when main finds its
  first, and final bounds were unchanged.
- **FJ personas** (#206): `mip_heuristic_fj_personas` (default `false`)
  starts FJ's presolve workers 1 and 2 from a greedy covering start and an
  exact-k start where the model has rows they can act on. #206 measured
  about 1% better on the 11 of 25 instances where a persona fires (not
  significant, p = 0.15), and #249 1.004 [0.991, 1.018] on 7 exact-k
  instances.
- **Ruin-and-repair** (#207, #219): a presolve chain stage after Scylla,
  enabled by `mip_heuristic_ruin_effort` above zero, with
  `mip_heuristic_ruin_patience` (both default 0). It resets a few integer
  columns of the pool's best point and repairs with FJ under an objective
  cutoff. Its finds carry the new solution-source letter `N`. #244 later
  measured no gain (1.005 [0.972, 1.041], 15 instances).
- **Objective cut** (#208, #221, #222): `mip_heuristic_objective_cut`,
  `_rel` and `_abs`. Removed in v0.1.9, which says why.

### Changed

- **The solution pool refuses a twin** (#204): an offer whose integer
  columns match an entry's replaces that entry if it is better, and is
  refused otherwise, where the pool used to admit every offer while
  filling, so copies of one assignment took several slots. **This changes
  a default run.** No A/B of it was posted.
- **macOS and Apple clang build** (#214), and our targets compile with
  `-ffp-contract=off`: on arm64 the compilers fused multiply-adds by
  default and FJ's incremental activities drifted. Baseline x86-64 has no
  FMA, so a Linux run is unchanged.
