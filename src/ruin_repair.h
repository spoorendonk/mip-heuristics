#pragma once
#include "heuristic_context.h"
#include "rng.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

class SolutionSink;

// Ruin-and-repair LNS with FeasibilityJump as the repair step, guided by the
// incumbent (#207).  After NVIDIA cuOpt's CPU LNS worker
// (`run_cpufj_lns_ruin_repair` in `cpp/src/mip_heuristics/lns/cpufj.cuh` at
// f3fe1a5, Apache-2.0; read for technique, no code copied).
//
// Each attempt of a worker:
//   1. reads the population's best point — the caller's `RestartSource`
//      under `run_until_stopped`, else the sink's pool — and checks it is
//      feasible, falling back to the worker's own best when that point is
//      unusable;
//   2. adopts it when it beats the worker's own best by `kObjectiveEpsilon`
//      (or the worker has none), and then makes the *guidance pool* the
//      integer columns where the worker's old best disagrees with it;
//      otherwise restarts from its own best;
//   3. ruins `ruin_size(...)` non-fixed integer columns, from the shuffled
//      guidance pool first and then uniformly: each takes, on an even coin,
//      the population best's value, or else a uniform integer in its bounds;
//   4. repairs with a fresh `FjWorker` from the ruined point, every column
//      free, under the objective cutoff `best - kCutoffMarginFeastols *
//      max(feastol, |best| * DBL_EPSILON)`, for `repair_budget(...)` FJ
//      effort units; any point FJ reports is strictly better than the
//      worker's best and goes to the sink as FJ finds it.
// A repair that finds nothing counts towards the ruin size's growth.
//
// Deviations from cuOpt, each because HiGHS's FJ or this runner differs:
//   - the repair is bounded by FJ effort, not seconds, so a run is
//     reproducible under `threads=1` (constants in `docs/PARAMETERS.md`);
//   - the objective bound is a hard row in FJ's model, not cuOpt's
//     objective-weighted scoring against its best, which HiGHS's FJ lacks;
//   - cuOpt's bound-aware "geometric" repair for disjunctive models
//     (`configure_cpufj_lns_geometry`) is not carried over;
//   - guidance and the coin's population branch compare against the
//     worker's best, not FJ's current assignment, which a fresh FJ per
//     repair does not keep.  The population's best is usually the worker's
//     own best (its improvements reach the pool first), so both act only
//     after a peer's or the source's better point is adopted; otherwise
//     about half the ruin set keeps its value (see `ruin` in the .cpp).
namespace ruin_repair {

// cuOpt's `OBJECTIVE_EPSILON`: how much a population point has to beat a
// worker's own best by to be adopted.
inline constexpr double kObjectiveEpsilon = 1e-7;

// The objective cutoff sits this many `feastol` below the worker's best
// (this many ulps of the best, if larger).
// FJ calls a row satisfied within `feastol` of its bound, so a margin of
// one would let FJ report a point equal to the best; two make every point
// it reports better by at least `feastol`.
inline constexpr double kCutoffMarginFeastols = 2.0;

// cuOpt's ruin-size schedule: a base of `|I| / kRuinBaseDivisor`, clamped to
// [kRuinBaseMin, kRuinBaseMax], times `1 + min(no_improve /
// kRuinGrowthPeriod, kRuinGrowthMax)`, at most `|I|`.
inline constexpr size_t kRuinBaseDivisor = 200;
inline constexpr size_t kRuinBaseMin = 6;
inline constexpr size_t kRuinBaseMax = 40;
inline constexpr size_t kRuinGrowthPeriod = 6;
inline constexpr size_t kRuinGrowthMax = 6;

// The repair's FJ effort: the FJ model's nonzeros, which FJ charges for
// building its start, plus cuOpt's `min(2 s, 0.15 s + 0.02 s * ruined)` at
// `kRepairEffortPerSecond`, laptop-measured FJ effort per second on one
// thread (see `docs/PARAMETERS.md`).
inline constexpr double kRepairSecondsBase = 0.15;
inline constexpr double kRepairSecondsPerRuined = 0.02;
inline constexpr double kRepairSecondsMax = 2.0;
inline constexpr double kRepairEffortPerSecond = 2.0e8;

// How long a `run_until_stopped` slot sleeps while it has no usable point
// to ruin, as cuOpt's worker does.
inline constexpr int kWaitForSeedMs = 1;

// The number of columns to ruin among `num_integer` non-fixed integer ones
// after `no_improve` repairs in a row that found nothing.
[[nodiscard]] size_t ruin_size(size_t num_integer, size_t no_improve);

// The repair's FJ effort for `ruined` columns on a model of `nnz` nonzeros.
[[nodiscard]] size_t repair_budget(size_t nnz, size_t ruined);

// The columns to ruin, into `out`: the shuffled `guidance` first, then
// uniform draws from `integer_cols`, without repeats, until `size` columns
// or `4 * |integer_cols| + 16` draws.  `chosen` is a zeroed scratch flag per
// column, left zeroed; `guidance` is shuffled in place.
void select_ruin_set(std::vector<HighsInt>& guidance, std::span<const HighsInt> integer_cols,
                     size_t size, Rng& rng, std::vector<uint8_t>& chosen,
                     std::vector<HighsInt>& out);

// Whether `x` satisfies `problem`'s column bounds, integrality and rows,
// each to within `problem.feastol`.
[[nodiscard]] bool is_feasible(const ProblemView& problem, const std::vector<double>& x);

// The presolve chain's runner: `exec.num_workers` workers until the budget,
// the patience or the deadline ends the dispatch.  The population is the
// sink's pool, so a dispatch with an empty pool does nothing.  Implements the
// uniform runner contract; see heuristic_context.h.
DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink);

// One worker on the calling thread until `exec` says stop (#171's contract,
// beside `RestartSource` in heuristic_context.h).  The population best is
// `source`'s point if feasible, else the sink's pool's if feasible, else
// the worker's own best; while there is none the slot waits,
// `kWaitForSeedMs` at a time, polling the stop flag, the deadline and the
// terminator.
[[nodiscard]] size_t run_until_stopped(const ProblemView& problem, const HeuristicBudget& budget,
                                       const ExecutionContext& exec, int worker,
                                       const RestartSource& source, SolutionSink& sink);

}  // namespace ruin_repair
