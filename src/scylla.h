#pragma once

#include <cstddef>

class HighsMipSolver;

namespace scylla {

// Scylla feasibility pump: N independent pump chains (Mexi et al. 2023,
// Algorithm 1.1) sharing a single PDLP instance via a mutex-guarded
// `ContestedPdlp`.  Each chain owns its own warm-start, α_K decay,
// cycle history, RNG, and static FPR rounding strategy
// (`kFprConfigs[w % kNumFprConfigs]`).  Only one PDLP solve is in
// flight at a time, so cuPDLP GPU state is never contended.
//
// `opportunistic=false` runs epoch-gated barrier synchronization
// (`run_epoch_loop`); `opportunistic=true` runs continuous parallelism
// (`run_opportunistic_loop`).  Both share the same `PumpWorker` body.
void run_parallel(HighsMipSolver &mipsolver, size_t max_effort, bool opportunistic);

}  // namespace scylla
