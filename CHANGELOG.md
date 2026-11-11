# Changelog

Notable changes to options and behaviour. Measurements live in the issues
linked from each entry.

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
