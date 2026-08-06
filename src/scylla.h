#pragma once

#include <cstddef>

class HighsMipSolver;
class SolutionPool;

namespace scylla {

// Scylla feasibility-pump heuristic (Mexi et al. 2023, Algorithm 1.1):
// N workers sharing a single PDLP instance via a mutex-guarded
// `ContestedPdlp`.  Each worker owns its own warm-start, α_K decay,
// cycle history, RNG, and static FPR rounding strategy
// (`kFprConfigs[w % kNumFprConfigs]`).  Only one PDLP solve is in
// flight at a time, so cuPDLP GPU state is never contended.
//
// Workers run continuously (`run_opportunistic_loop`) with per-worker
// self-termination; a retired chain is rebuilt in place with a fresh
// seed.  Set `threads=1` for a single chain whose behaviour is
// reproducible under a fixed `random_seed`.
//
// `pool` is owned by the caller (mode_dispatch::run_sequential).
// Workers insert feasible pumps with kSolutionSourceScylla; the caller
// flushes the pool once all sequential heuristics have run.
//
// Returns the total effort consumed.  The caller is responsible for
// booking it into `mipdata->heuristic_effort_used` — same contract as
// `local_mip::run_parallel` (issue #79).  This makes mode_dispatch.cpp
// the single point of Scylla effort accounting.
size_t run_parallel(HighsMipSolver &mipsolver, SolutionPool &pool, size_t max_effort);

}  // namespace scylla
