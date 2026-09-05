#pragma once

#include <cstddef>
#include <cstdint>

class IncumbentSink;
struct DispatchOutcome;
struct ExecutionContext;
struct HeuristicBudget;
struct ProblemView;

namespace local_mip {

// Patience: `mip_heuristic_local_mip_patience`, a multiple of
// `nnz << 10` — the same unit as this heuristic's effort option (issue
// #111, made an option by #106, put on the effort unit by #116).
// LocalMIP's counter is in coefficient accesses.
//
// Scope: **whole dispatch**, matching `mip_heuristic_local_mip_effort`.
// Each worker's share is this divided by the worker count
// (`HeuristicBudget::worker_stale`); the runner-level gate uses the
// value as it stands.
//
// The default 4096 reproduces the pre-#111 runner gate: at the default
// effort 0.1821 that gate was `heuristic_effort_budget(nnz, 0.1821) / 4`
// = 3729 x nnz, and 4096 is the neighbouring power of two (1.10x).  This
// is the heuristic #111's evidence came from — LocalMIP on `fiball`
// found its one solution at 0.6 s and then spent 11 s of a 12 s limit
// finding nothing else, because the gate it was measured against grew
// with the budget.  0 disables the gate entirely.  The default is
// registered in `third_party/highs_patch/apply_patch.cmake` and pinned
// by `tests/test_smoke.cpp`; `docs/PARAMETERS.md` carries the
// calibration notes.

// Compile-time instrumentation switch (R3-1 round-4 review).  Driven
// by the `MIP_HEURISTICS_INSTRUMENT` CMake option, which defaults to
// ON when tests are being built and OFF otherwise.  When false, the
// warm-start counter increments below compile out via
// `if constexpr (kInstrumented)` in `local_mip.cpp` and the API calls
// become no-ops (`reset_*` writes nothing, `warm_start_counters` just
// returns zeros).  The tests then exercise the counters under their
// own translation unit which sees `kInstrumented == true`.  Production
// release builds can opt out with `-DMIP_HEURISTICS_INSTRUMENT=OFF`.
#if defined(MIP_HEURISTICS_INSTRUMENT_ENABLED) && MIP_HEURISTICS_INSTRUMENT_ENABLED
inline constexpr bool kInstrumented = true;
#else
inline constexpr bool kInstrumented = false;
#endif

// Test-only introspection: counts how many times each branch of
// `resolve_worker_start` fired during the current process.  The three
// branches correspond to the three #75 cold-start fallback rungs:
//   - `pool`   : pool.copy_best succeeded (warm-start from #74 pool).
//   - `incumbent` : pool was empty, mipdata->incumbent picked up.
//   - `construction` : pool and incumbent both empty; either the
//                      paper's construction phase ran or its
//                      cold-start cache was reused (a single first-
//                      worker construction can amortise across N
//                      peers via `cold_start_cache`).
// Counters cover the `run` start-resolution paths only; restart-callback
// warm-starts inside the parallel loops are NOT counted (their work
// happens after the initial start has already been resolved).
// Call `reset_warm_start_counters()` before a HiGHS run, then read
// `warm_start_counters()` after to assert which path actually fired.
// Used by R1-8 / R2-7 / R3-3 round-3 review tests to distinguish #74
// (pool warm-start) from #75 (cold-start construction); without this
// the integration tests can't tell those paths apart since both
// produce non-zero effort.
//
// When `kInstrumented == false`, all three counters always read zero
// regardless of how many runs have completed.
struct WarmStartCounters {
    int64_t pool;
    int64_t incumbent;
    int64_t construction;
};

void reset_warm_start_counters();
WarmStartCounters warm_start_counters();

// Test-only introspection on the search loop's wall-clock poll (#162).
// `polls` counts how many times `LocalMipWorker::run_attempt` asked
// `ExecutionContext::past_deadline()`; `effort` is the effort those
// attempts charged.
//
// The pair is the *mechanism* the fix has to be pinned on, and it is what
// a whole-solve assertion cannot see: the cadence is denominated in
// charged work, so the invariant is "a poll happens at least every
// `kTermCheckWork` units", i.e. `polls * kTermCheckWork >= effort`.  Under
// the retired step cadence the poll rate was tied to `step_` and to
// nothing the model could scale, so on any instance whose average step
// charges more than `kTermCheckWork / 1000` units the same run polls too
// rarely and the inequality fails.  Reading it needs no clock, which is
// what makes the check load-safe — the wall-clock overrun it prevents is
// not something `ctest -j$(nproc)` can be trusted to reproduce.
struct DeadlinePollCounters {
    int64_t polls;
    int64_t effort;
};

void reset_deadline_poll_counters();
DeadlinePollCounters deadline_poll_counters();

// Called by `LocalMipWorker::run_attempt`: once per wall-clock poll, and
// once at the end of each attempt with the effort that attempt charged.
void note_deadline_poll();
void note_attempt_effort(int64_t effort_charged);
// Runs N continuous `parallel::for_each` workers with per-worker
// self-termination.  Worker 0 starts from the unperturbed incumbent;
// workers 1..N-1 start from perturbed incumbents.  Stalled workers are
// restarted from the pool's best solution with fresh perturbation.  Set
// `threads=1` for a single worker whose behaviour is reproducible under
// a fixed `random_seed`.
//
// Implements the uniform runner contract; see heuristic_context.h.
//
// The effort returned covers the cold-start construction sweep as well
// as the search itself.
DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, IncumbentSink& sink);
}  // namespace local_mip
