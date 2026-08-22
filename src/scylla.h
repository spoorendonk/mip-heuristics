#pragma once

#include <cstddef>

class IncumbentSink;
struct ExecutionContext;
struct HeuristicBudget;
struct ProblemView;

namespace scylla {

// Stall threshold, in effort units (PDLP iterations x nnz) per
// constraint-matrix nonzero (issue #111).  Scylla's counter is in a
// different unit from FPR's and LocalMIP's, so the constants are not
// comparable across heuristics.
//
// Scope: **whole dispatch**, matching `mip_heuristic_scylla_effort`.
//
// 512 reproduces the pre-#111 runner gate: at the default effort 0.0296
// that gate was `heuristic_effort_budget(nnz, 0.0296) / 4` = 606 x nnz,
// and 512 is the neighbouring power of two (0.84x).  Small in absolute
// terms because a single PDLP solve charges `iters x nnz`, so this is a
// handful of unproductive pump rounds rather than hundreds of sweeps.
//
// PROVISIONAL, pending the per-heuristic budget calibration (#106).
// #106 sweeps each heuristic's effort option and will show where each
// one actually stops producing solutions; these values are placeholders
// chosen to reproduce the pre-#111 gate at the shipped default effort,
// not the result of a measurement.
inline constexpr size_t kStallPerNnzScylla = 512;

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
// Implements the uniform runner contract; see heuristic_context.h.
size_t run(const ProblemView& problem, const HeuristicBudget& budget, ExecutionContext& exec,
           IncumbentSink& sink);

}  // namespace scylla
