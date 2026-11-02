#include "ruin_repair.h"

#include "fj_worker.h"
#include "heuristic_common.h"
#include "opportunistic_runner.h"
#include "solution_sink.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <thread>
#include <utility>

namespace ruin_repair {

size_t ruin_size(size_t num_integer, size_t no_improve) {
    const size_t base =
        std::min(kRuinBaseMax, std::max(kRuinBaseMin, num_integer / kRuinBaseDivisor));
    const size_t growth = 1 + std::min(no_improve / kRuinGrowthPeriod, kRuinGrowthMax);
    return std::min(num_integer, base * growth);
}

size_t repair_budget(size_t nnz, size_t ruined) {
    const double seconds =
        std::min(kRepairSecondsMax,
                 kRepairSecondsBase + (kRepairSecondsPerRuined * static_cast<double>(ruined)));
    return nnz + static_cast<size_t>(seconds * kRepairEffortPerSecond);
}

void select_ruin_set(std::vector<HighsInt>& guidance, std::span<const HighsInt> integer_cols,
                     size_t size, Rng& rng, std::vector<uint8_t>& chosen,
                     std::vector<HighsInt>& out) {
    out.clear();
    std::shuffle(guidance.begin(), guidance.end(), rng);
    for (const HighsInt j : guidance) {
        if (out.size() >= size) {
            break;
        }
        if (chosen[j] == 0) {
            chosen[j] = 1;
            out.push_back(j);
        }
    }
    if (!integer_cols.empty()) {
        std::uniform_int_distribution<size_t> pick(0, integer_cols.size() - 1);
        const size_t max_draws = (4 * integer_cols.size()) + 16;
        for (size_t draws = 0; out.size() < size && draws < max_draws; ++draws) {
            const HighsInt j = integer_cols[pick(rng)];
            if (chosen[j] == 0) {
                chosen[j] = 1;
                out.push_back(j);
            }
        }
    }
    for (const HighsInt j : out) {
        chosen[j] = 0;
    }
}

bool is_feasible(const ProblemView& problem, const std::vector<double>& x) {
    if (std::cmp_not_equal(x.size(), problem.ncol)) {
        return false;
    }
    const double tol = problem.feastol;
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        const double v = x[j];
        if (!std::isfinite(v) || v < (*problem.col_lower)[j] - tol ||
            v > (*problem.col_upper)[j] + tol) {
            return false;
        }
        if (is_integer(*problem.integrality, j) && std::abs(v - std::round(v)) > tol) {
            return false;
        }
    }
    const auto& start = *problem.ar_start;
    const auto& index = *problem.ar_index;
    const auto& value = *problem.ar_value;
    for (HighsInt i = 0; i < problem.nrow; ++i) {
        double lhs = 0.0;
        for (HighsInt k = start[i]; k < start[i + 1]; ++k) {
            lhs += value[k] * x[index[k]];
        }
        if (is_row_violated(lhs, (*problem.row_lower)[i], (*problem.row_upper)[i], tol)) {
            return false;
        }
    }
    return true;
}

namespace {

// Whether a repair can search at all: `FjWorker::run_attempt` returns at
// once under 64-bit `HighsInt`, and a slot whose attempts still charge a
// feasibility check would never retire.
#ifdef HIGHSINT64
constexpr bool kRepairSearches = false;
#else
constexpr bool kRepairSearches = true;
#endif

// What every slot of a dispatch shares, built once on the calling thread.
struct Shared {
    // Integer columns whose bounds leave more than `feastol` of room: the
    // only ones worth ruining (cuOpt's `integer_vars`).
    std::vector<HighsInt> integer_cols;
    // Whether any column has a cost: without one no point improves on
    // another, and the LNS has nothing to do.
    bool has_cost = false;
};

Shared make_shared_state(const ProblemView& problem) {
    Shared shared;
    shared.integer_cols.reserve(problem.ncol);
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        if ((*problem.col_cost)[j] != 0.0) {
            shared.has_cost = true;
        }
        if (is_integer(*problem.integrality, j) &&
            (*problem.col_upper)[j] - (*problem.col_lower)[j] > problem.feastol) {
            shared.integer_cols.push_back(j);
        }
    }
    return shared;
}

// One worker: its own best point and the scratch its attempts reuse.
struct Slot {
    std::vector<double> best;
    double best_objective = std::numeric_limits<double>::infinity();
    std::vector<double> population;
    // The last point from each population source (the pool; the caller's
    // `RestartSource`) that failed `is_feasible`, so the same point is not
    // checked again on every attempt (cuOpt's `rejected_seed`).
    std::vector<double> rejected;
    std::vector<double> rejected_source;
    std::vector<double> start;
    std::vector<HighsInt> guidance;
    std::vector<HighsInt> ruin_set;
    std::vector<uint8_t> chosen;
    size_t no_improve = 0;
    WorkerTrace trace;
};

Slot make_slot(const ProblemView& problem, int worker) {
    Slot slot;
    slot.chosen.assign(problem.ncol, 0);
    slot.trace = WorkerTrace{worker, 0};
    return slot;
}

// Into `slot.population`, the point `fetch` gives if it is usable: found
// and feasible.  `rejected` is that source's last infeasible point.
// `effort` is charged `nnz` for a feasibility check.
template <typename Fetch>
bool usable_point(const ProblemView& problem, Slot& slot, Fetch& fetch,
                  std::vector<double>& rejected, size_t& effort) {
    if (!fetch(slot.population) || slot.population == rejected) {
        return false;
    }
    effort += problem.nnz;
    if (!is_feasible(problem, slot.population)) {
        rejected = slot.population;
        return false;
    }
    rejected.clear();
    return true;
}

// Into `slot.population`, the point to ruin from: the caller's source's if
// usable, else the pool's best if usable, else the worker's own best.  An
// infeasible point (the pool admits without checking; a source may hand out
// anything) must not idle a slot when a feasible one is to hand.
template <typename FromSource, typename FromPool>
bool next_start_point(const ProblemView& problem, Slot& slot, FromSource& from_source,
                      FromPool& from_pool, size_t& effort) {
    if (usable_point(problem, slot, from_source, slot.rejected_source, effort) ||
        usable_point(problem, slot, from_pool, slot.rejected, effort)) {
        return true;
    }
    if (slot.best.empty()) {
        return false;
    }
    slot.population = slot.best;
    return true;
}

// The ruined start, into `slot.start`, from the worker's best and the
// population point in `slot.population`.  Adopts the population point first
// if it beats the worker's best.
void ruin(const ProblemView& problem, const Shared& shared, Slot& slot, Rng& rng) {
    const double population_objective = problem.objective(slot.population);
    if (slot.best.empty() || population_objective + kObjectiveEpsilon < slot.best_objective) {
        // cuOpt compares its climber's current assignment (`h_assignment`)
        // here while its comment says "this climber's own best-known point";
        // we read the comment: the current assignment is FJ's internal state
        // at the end of the last repair, which the repair step here does not
        // expose.  On the first adoption there is no best and so no guidance.
        // What that reading costs: every improvement a slot makes is offered
        // to the sink first, so the population's best is usually this slot's
        // own best.  Then there is no adoption, `guidance` keeps what the
        // last adoption left (empty under one worker after the first), and
        // the coin's population branch below rewrites a column to the value
        // it already has, so about half the ruin set stays as it was.  Only
        // a peer's better point (or the source's, in `run_until_stopped`)
        // differs from the best and makes both mechanisms act.
        slot.guidance.clear();
        if (!slot.best.empty()) {
            for (const HighsInt j : shared.integer_cols) {
                if (std::abs(slot.best[j] - slot.population[j]) > problem.feastol) {
                    slot.guidance.push_back(j);
                }
            }
        }
        slot.best = slot.population;
        slot.best_objective = population_objective;
        slot.no_improve = 0;
    }
    slot.start = slot.best;

    select_ruin_set(slot.guidance, shared.integer_cols,
                    ruin_size(shared.integer_cols.size(), slot.no_improve), rng, slot.chosen,
                    slot.ruin_set);

    std::bernoulli_distribution coin(0.5);
    for (const HighsInt j : slot.ruin_set) {
        if (coin(rng)) {
            slot.start[j] = slot.population[j];
            continue;
        }
        const double lo = std::ceil((*problem.col_lower)[j]);
        const double hi = std::floor((*problem.col_upper)[j]);
        if (std::isfinite(lo) && std::isfinite(hi) && hi - lo <= kSafeInt64DoubleRange) {
            const auto span = static_cast<int64_t>(hi - lo);
            slot.start[j] =
                lo + static_cast<double>(std::uniform_int_distribution<int64_t>(0, span)(rng));
        } else {
            const double bound = coin(rng) ? lo : hi;
            if (std::isfinite(bound)) {
                slot.start[j] = bound;
            }
        }
    }
}

// One repair of `slot.start` with a fresh FJ under the worker's cutoff, for
// at most `run_cap` effort.  Keeps what FJ found if it is feasible and beats
// the worker's best.
AttemptResult repair(const ProblemView& problem, const ExecutionContext& exec, SolutionSink& sink,
                     Slot& slot, Rng& rng, size_t run_cap) {
    const size_t budget = std::min(repair_budget(problem.nnz, slot.ruin_set.size()), run_cap);
    // The margin is relative once the objective is large: an absolute
    // `feastol` below the objective's ulp would round away and let FJ
    // repair back to an equal objective.
    const double margin = kCutoffMarginFeastols *
                          std::max(problem.feastol, std::abs(slot.best_objective) *
                                                        std::numeric_limits<double>::epsilon());
    const double cutoff = slot.best_objective - margin;
    FjWorker fj(problem, exec, sink, budget, std::numeric_limits<size_t>::max(),
                static_cast<uint32_t>(rng()), slot.start, slot.trace, cutoff);
    AttemptResult result = fj.run_attempt(budget);
    slot.trace.effort_base = fj.traced_effort();

    const std::vector<double>& found = fj.best_found();
    bool improved = false;
    if (!found.empty() && fj.best_found_objective() + kObjectiveEpsilon < slot.best_objective) {
        result.effort += problem.nnz;
        slot.trace.effort_base += problem.nnz;
        if (is_feasible(problem, found)) {
            // Recomputed rather than FJ's incrementally maintained value, so
            // its drift does not feed the next cutoff.
            const double objective = problem.objective(found);
            if (objective + kObjectiveEpsilon < slot.best_objective) {
                slot.best = found;
                slot.best_objective = objective;
                improved = true;
            }
        }
    }
    slot.no_improve = improved ? 0 : slot.no_improve + 1;
    return result;
}

// A ruin and a repair from `slot.population`, the usable point
// `next_population_point` just fetched; `check_effort` is what fetching it
// charged.
AttemptResult ruin_and_repair(const ProblemView& problem, const Shared& shared,
                              const ExecutionContext& exec, SolutionSink& sink, Slot& slot,
                              Rng& rng, size_t run_cap, size_t check_effort) {
    slot.trace.effort_base += check_effort;
    ruin(problem, shared, slot, rng);
    AttemptResult result = repair(problem, exec, sink, slot, rng, run_cap);
    result.effort += check_effort;
    return result;
}

}  // namespace

DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink) {
    if (problem.degenerate() || budget.disabled()) {
        return {};
    }
    const Shared shared = make_shared_state(problem);
    if (!kRepairSearches || shared.integer_cols.empty() || !shared.has_cost) {
        return {};
    }
    auto no_source = [](std::vector<double>& /*out*/) { return false; };
    auto from_pool = [&sink](std::vector<double>& out) { return sink.copy_best(out); };
    return {.effort = run_opportunistic_loop(
                exec, budget,
                [&problem](int worker_idx, Rng& /*rng*/) { return make_slot(problem, worker_idx); },
                [&](Slot& slot, Rng& rng, size_t run_cap) -> AttemptResult {
                    // No usable point and no best of its own charges at most
                    // one check and then nothing, which retires the slot: a
                    // dispatch whose pool holds no feasible point has nothing
                    // to ruin.
                    size_t effort = 0;
                    if (!next_start_point(problem, slot, no_source, from_pool, effort)) {
                        return {.effort = effort, .found_improvement = false};
                    }
                    return ruin_and_repair(problem, shared, exec, sink, slot, rng, run_cap, effort);
                })};
}

size_t run_until_stopped(const ProblemView& problem, const HeuristicBudget& budget,
                         const ExecutionContext& exec, int worker, const RestartSource& source,
                         SolutionSink& sink) {
    if (problem.degenerate()) {
        return 0;
    }
    const Shared shared = make_shared_state(problem);
    if (!kRepairSearches || shared.integer_cols.empty() || !shared.has_cost) {
        return 0;
    }
    return run_on_caller_thread(
        exec, budget, worker,
        [&problem](int worker_idx, Rng& /*rng*/) { return make_slot(problem, worker_idx); },
        [&](Slot& slot, Rng& rng, size_t run_cap) -> AttemptResult {
            auto from_source = [&](std::vector<double>& out) {
                return take_restart(source, rng, out, problem);
            };
            auto from_pool = [&sink](std::vector<double>& out) { return sink.copy_best(out); };
            // Wait for a usable point, as cuOpt's worker does, rather than
            // return zero effort, which would end the run.  This thread
            // holds the poller seat, so the wait polls the terminator too.
            size_t effort = 0;
            while (!next_start_point(problem, slot, from_source, from_pool, effort)) {
                if (exec.terminated()) {
                    return {.effort = effort, .found_improvement = false};
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(kWaitForSeedMs));
            }
            return ruin_and_repair(problem, shared, exec, sink, slot, rng, run_cap, effort);
        });
}

}  // namespace ruin_repair
