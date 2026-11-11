#include "recombine.h"

#include "fj_worker.h"
#include "heuristic_common.h"
#include "opportunistic_runner.h"
#include "ruin_repair.h"
#include "solution_sink.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <thread>

namespace recombine {

size_t next_max_vars_from_other(size_t current, bool same_as_parent) {
    // cuOpt's rounding, inexact factors and all: `200 * 1.1` is a hair above
    // 220 in double, so 200 grows to 221, as it does there.
    const auto m = static_cast<double>(current);
    if (same_as_parent) {
        return std::min(static_cast<size_t>(std::ceil(m * kBpVarsIncreaseFactor)),
                        kBpMaxVarsFromOther);
    }
    return std::max(static_cast<size_t>(std::floor(m * kBpVarsDecreaseFactor)),
                    kBpMinVarsFromOther);
}

BoundPropRecombiner::BoundPropRecombiner(const ProblemView& problem)
    : problem_(problem),
      restricted_(problem),
      col_lower_(*problem.col_lower),
      col_upper_(*problem.col_upper),
      midpoint_(static_cast<size_t>(problem.ncol), 0.0),
      is_free_(static_cast<size_t>(problem.ncol), 0) {
    restricted_.incumbent.clear();
    restricted_.col_lower = &col_lower_;
    restricted_.col_upper = &col_upper_;
    restricted_.uplocks = &no_locks_;
    restricted_.downlocks = &no_locks_;
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        if (is_integer(*problem.integrality, j)) {
            integer_cols_.push_back(j);
        }
    }
    differing_.reserve(integer_cols_.size());
    if (!problem.degenerate()) {
        fixer_.emplace(problem.ncol, problem.nrow, problem.ar_start->data(),
                       problem.ar_index->data(), problem.ar_value->data(), *problem.csc,
                       col_lower_.data(), col_upper_.data(), problem.row_lower->data(),
                       problem.row_upper->data(), problem.integrality->data(), problem.feastol);
    }
}

size_t BoundPropRecombiner::fix_and_propagate(PropEngine& fixer, const std::vector<double>& guiding,
                                              size_t num_free, const Deadline& deadline) {
    std::ranges::copy(*problem_.col_lower, col_lower_.begin());
    std::ranges::copy(*problem_.col_upper, col_upper_.begin());
    for (size_t k = 0; k < num_free; ++k) {
        is_free_[differing_[k]] = 1;
    }
    for (const HighsInt j : integer_cols_) {
        if (is_free_[j] == 0) {
            const double v = std::clamp(std::round(guiding[j]), col_lower_[j], col_upper_[j]);
            col_lower_[j] = v;
            col_upper_[j] = v;
        }
    }
    fixer.reset();
    for (const HighsInt j : integer_cols_) {
        if (is_free_[j] == 0) {
            fixer.seed_worklist(j);
        }
    }
    for (size_t k = 0; k < num_free; ++k) {
        is_free_[differing_[k]] = 0;
    }
    fixer.set_deadline(deadline);
    // A truncated fixpoint is sound, so its tightenings are kept; only a
    // proven infeasibility leaves the bounds at the bare fixings, for FPR's
    // own search and repair to make what they can of.
    if (fixer.propagate(-1) != PropResult::kInfeasible) {
        for (HighsInt j = 0; j < problem_.ncol; ++j) {
            col_lower_[j] = fixer.var(j).lb;
            col_upper_[j] = fixer.var(j).ub;
        }
    }
    return fixer.effort();
}

Offspring BoundPropRecombiner::recombine(const std::vector<double>& a, double obj_a,
                                         const std::vector<double>& b, double obj_b,
                                         size_t max_effort, const Deadline& deadline, Rng& rng,
                                         std::vector<double>& out) {
    out.clear();
    Offspring offspring;
    // `fixer_` is built for every view that is not degenerate.
    if (!fixer_.has_value()) {
        return offspring;
    }
    const auto ncol = static_cast<size_t>(problem_.ncol);
    offspring.effort = ncol;
    // cuOpt's rule with both parents feasible: `a` guides, whatever the
    // objectives, and the caller's draw decides the order.
    const std::vector<double>& guiding = a;
    const std::vector<double>& other = b;

    differing_.clear();
    for (const HighsInt j : integer_cols_) {
        if (std::round(guiding[j]) != std::round(other[j])) {
            differing_.push_back(j);
        }
    }
    // One shuffle serves both the random subset (step 3) and the random
    // probe order: cuOpt probes in its own order, FPR's `kLR` would probe in
    // formulation order, and a fixed order would bias every recombination
    // of the same pair the same way (a deviation, recorded in the header).
    std::shuffle(differing_.begin(), differing_.end(), rng);
    const size_t num_differing = differing_.size();
    const size_t num_free = std::min(num_differing, max_vars_from_other_);
    const size_t num_fixed = integer_cols_.size() - num_free;
    if (num_free == 0 || num_fixed == 0) {
        return offspring;
    }

    offspring.effort += fix_and_propagate(*fixer_, guiding, num_free, deadline);

    for (size_t j = 0; j < ncol; ++j) {
        midpoint_[j] = 0.5 * (a[j] + b[j]);
    }
    // cuOpt completes the free columns by probing each at one parent's
    // value, with the other's as the fallback, and solves an LP over the
    // continuous ones.  Here FPR's DFS does the probing: `kLp` on the
    // midpoint rounds a column where the parents differ by one to either
    // value on a fair coin, and `compute_alt` gives a binary the other
    // parent's value as its alternative.  A general integer's alternative
    // is the domain's other end rather than the other parent's value, and
    // FPR's fill and WalkSAT repair stand in for the LP: deviations, as is
    // the effort cap in place of cuOpt's two-second limit.  Only the free
    // columns are ranked, so FPR's `ncol + 1` node limit is all theirs; the
    // fixed ones fill at their singleton bounds in Phase 2.5.
    const FprConfig cfg{
        .max_effort = max_effort,
        .cont_fallback = midpoint_.data(),
        .mode = FrameworkMode::kDfsrep,
        .strategy = &kStratLRLp,
        .lp_ref = midpoint_.data(),
        .precomputed_var_order = differing_.data(),
        .precomputed_var_order_size = static_cast<HighsInt>(num_free),
        .scratch = &scratch_,
    };
    HeuristicResult result = fpr_attempt(restricted_, deadline, cfg, rng, 0);
    offspring.effort += result.effort;
    if (result.solution.size() != ncol) {
        return offspring;
    }
    out = std::move(result.solution);

    bool same_as_guiding = true;
    bool same_as_other = true;
    for (const HighsInt j : integer_cols_) {
        const double v = std::round(out[j]);
        same_as_guiding = same_as_guiding && v == std::round(guiding[j]);
        same_as_other = same_as_other && v == std::round(other[j]);
    }
    const bool same = same_as_guiding || same_as_other;
    if (num_differing > max_vars_from_other_) {
        max_vars_from_other_ = next_max_vars_from_other(max_vars_from_other_, same);
    }
    offspring.feasible = result.found_feasible;
    if (offspring.feasible) {
        offspring.objective = result.objective;
    }
    offspring.success =
        (offspring.feasible && offspring.objective < std::min(obj_a, obj_b)) || !same;
    return offspring;
}

namespace {

// A draw in [0, n): multiply-shift on `rng`, not
// `std::uniform_int_distribution`, whose stream differs across standard
// libraries.
size_t draw_below(Rng& rng, size_t n) {
    return static_cast<size_t>((static_cast<unsigned __int128>(rng()) * n) >> 64U);
}

}  // namespace

// Two seconds of FJ, the unit `kRewardNominalEffort` stands for.
static_assert(kRewardNominalEffort == 2.0 * ruin_repair::kRepairEffortPerSecond);

double recombination_reward(double best_of_parents, double best, double offspring, size_t effort) {
    const double margin = std::max(kRewardMarginAbs, kRewardMarginRel * std::abs(best));
    if (offspring + margin < best) {
        return kRewardBeatsBest;
    }
    if (!(offspring + margin < best_of_parents)) {
        return 0.0;
    }
    // cuOpt multiplies the work-normalised value by a factor of 1 near the
    // best and 0 elsewhere.
    if (!(std::abs(offspring - best) / (std::abs(best) + 1.0) < kRewardNearBest)) {
        return 0.0;
    }
    return std::max(kRewardWorkFloor,
                    kRewardWorkCeiling - (static_cast<double>(effort) / kRewardNominalEffort));
}

RecombinerArm RecombinerBandit::select() {
    ++steps_;
    for (size_t i = 0; i < kNumRecombinerArms; ++i) {
        if (arms_[i].pulls == 0) {
            return static_cast<RecombinerArm>(i);
        }
    }
    const double log_t = std::log(static_cast<double>(steps_));
    double max_ucb = -kHighsInf;
    std::array<size_t, kNumRecombinerArms> best{};
    size_t num_best = 0;
    for (size_t i = 0; i < kNumRecombinerArms; ++i) {
        const double ucb =
            arms_[i].q + std::sqrt(2.0 * log_t / static_cast<double>(arms_[i].pulls));
        if (ucb > max_ucb + kBanditTieTolerance) {
            max_ucb = ucb;
            num_best = 0;
            best[num_best++] = i;
        } else if (std::abs(ucb - max_ucb) < kBanditTieTolerance) {
            best[num_best++] = i;
        }
    }
    return static_cast<RecombinerArm>(best[draw_below(rng_, num_best)]);
}

void RecombinerBandit::reward(RecombinerArm arm, double r) {
    Arm& a = arms_[index(arm)];
    ++a.pulls;
    a.q += kBanditAlpha * (r - a.q);
}

RecombineOperator::RecombineOperator(const ProblemView& problem, const ExecutionContext& exec,
                                     SolutionSink& sink, uint64_t seed, WorkerTrace trace)
    : problem_(problem),
      exec_(exec),
      sink_(sink),
      bound_prop_(problem),
      // SplitMix64 expands each seed, so neighbouring seeds give unrelated
      // streams.
      bandit_(seed + 1),
      rng_(seed),
      trace_(trace),
      integer_mask_(static_cast<size_t>(problem.ncol), false) {
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        if ((*problem.integrality)[j] != HighsVarType::kContinuous) {
            integer_mask_[j] = true;
            integer_cols_.push_back(j);
        }
    }
    entries_.reserve(kPoolCapacity);
    offspring_.reserve(static_cast<size_t>(problem.ncol));
}

Offspring RecombineOperator::recombine(RecombinerArm arm, const SolutionPool::EntryRef& first,
                                       const SolutionPool::EntryRef& second,
                                       size_t recombine_effort, const Deadline& deadline) {
    const std::vector<double>& a = *first.solution;
    const std::vector<double>& b = *second.solution;
    if (arm == RecombinerArm::kBoundProp) {
        return bound_prop_.recombine(a, first.objective, b, second.objective, recombine_effort,
                                     deadline, rng_, offspring_);
    }
    if (arm == RecombinerArm::kGuidedCrossover) {
        SolutionPool::guided_crossover(integer_mask_, a, b, rng_, offspring_);
    } else {
        const bool first_better = first.objective <= second.objective;
        SolutionPool::neighborhood_crossover(integer_mask_, first_better ? a : b,
                                             first_better ? b : a, rng_, offspring_);
    }
    Offspring offspring;
    offspring.effort = static_cast<size_t>(problem_.ncol);
    bool same_as_a = true;
    bool same_as_b = true;
    for (const HighsInt j : integer_cols_) {
        const double v = std::round(offspring_[j]);
        same_as_a = same_as_a && v == std::round(a[j]);
        same_as_b = same_as_b && v == std::round(b[j]);
    }
    offspring.success = !same_as_a && !same_as_b;
    return offspring;
}

size_t local_search_budget(size_t nnz, bool feasible) {
    const double seconds = feasible ? kLsSecondsFeasible : kLsSecondsInfeasible;
    return nnz + static_cast<size_t>(seconds * ruin_repair::kRepairEffortPerSecond);
}

size_t RecombineOperator::take_external(const RestartSource& source) {
    if (!take_restart(source, rng_, external_, problem_) || external_ == last_external_) {
        return 0;
    }
    last_external_ = external_;
    trace_.effort_base += problem_.nnz;
    if (ruin_repair::is_feasible(problem_, external_)) {
        // The caller's point joining the population is no improvement of the
        // operator's, so the verdict is not read.
        static_cast<void>(
            sink_.offer(problem_.objective(external_), external_, trace_, trace_.at(0)));
    }
    return problem_.nnz;
}

std::optional<RecombineStep> RecombineOperator::step(size_t recombine_effort, size_t ls_cap,
                                                     const Deadline& deadline) {
    sink_.entry_refs(entries_);
    const size_t n = entries_.size();
    if (n < 2) {
        return std::nullopt;
    }
    // cuOpt's tournament draw, on a pool sorted best first.
    const size_t i = std::min(draw_below(rng_, n), draw_below(rng_, n));
    size_t j = std::min(draw_below(rng_, n - 1), draw_below(rng_, n - 1));
    if (j >= i) {
        ++j;
    }
    const SolutionPool::EntryRef& first = entries_[i];
    const SolutionPool::EntryRef& second = entries_[j];
    const double best = entries_.front().objective;
    const double best_of_parents = std::min(first.objective, second.objective);

    RecombineStep result;
    result.arm = bandit_.select();
    const Offspring offspring = recombine(result.arm, first, second, recombine_effort, deadline);
    result.success = offspring.success && !offspring_.empty();
    result.effort = offspring.effort;
    trace_.effort_base += offspring.effort;
    if (!result.success) {
        bandit_.reward(result.arm, 0.0);
        return result;
    }

    // The crossovers report no feasibility; bound-prop's verdict stands.
    bool feasible = offspring.feasible;
    double objective = offspring.objective;
    if (result.arm == RecombinerArm::kGuidedCrossover ||
        result.arm == RecombinerArm::kNeighborhoodCrossover) {
        result.effort += problem_.nnz;
        trace_.effort_base += problem_.nnz;
        feasible = ruin_repair::is_feasible(problem_, offspring_);
        objective = feasible ? problem_.objective(offspring_) : kHighsInf;
    }
    if (feasible) {
        result.objective = objective;
        // FJ reports a feasible start too, where it runs at all, and is then
        // refused as a twin; this reaches the pool whatever FJ does.
        result.improved =
            sink_.offer(objective, offspring_, trace_, trace_.at(0)).improved_incumbent;
    }

    // cuOpt's local search opens on the same check, with reward 0 and nothing
    // more to do; FJ would otherwise build its whole model before it first
    // polled a clock (#114, #117).
    if (deadline.expired() || exec_.past_deadline()) {
        bandit_.reward(result.arm, 0.0);
        return result;
    }

    const size_t ls_budget = std::min(local_search_budget(problem_.nnz, feasible), ls_cap);
    FjWorker fj(problem_, exec_, sink_, ls_budget, std::numeric_limits<size_t>::max(),
                static_cast<uint32_t>(rng_()), offspring_, trace_);
    const AttemptResult ls = fj.run_attempt(ls_budget);
    result.effort += ls.effort;
    result.improved = result.improved || ls.found_improvement;
    trace_.effort_base = fj.traced_effort();
    if (!fj.best_found().empty()) {
        result.objective = std::min(result.objective, fj.best_found_objective());
    }

    result.reward = recombination_reward(best_of_parents, best, result.objective, offspring.effort);
    bandit_.reward(result.arm, result.reward);
    return result;
}

namespace {

// Whether a recombination can improve anything on `problem`: an objective,
// and an integer column whose bounds leave room.
bool can_improve(const ProblemView& problem) {
    if (problem.degenerate() || !has_objective(problem)) {
        return false;
    }
    for (HighsInt j = 0; j < problem.ncol; ++j) {
        if (is_integer(*problem.integrality, j) &&
            (*problem.col_upper)[j] - (*problem.col_lower)[j] > problem.feastol) {
            return true;
        }
    }
    return false;
}

// The operator's step within one attempt's `run_cap`.
std::optional<RecombineStep> capped_step(RecombineOperator& op, const ExecutionContext& exec,
                                         size_t run_cap) {
    const size_t recombine_effort = std::min(run_cap, static_cast<size_t>(kRewardNominalEffort));
    return op.step(recombine_effort, run_cap, exec.deadline);
}

}  // namespace

DispatchOutcome run(const ProblemView& problem, const HeuristicBudget& budget,
                    ExecutionContext& exec, SolutionSink& sink) {
    if (budget.disabled() || !can_improve(problem)) {
        return {};
    }
    // Before any worker builds an operator: nothing else offers during a
    // chain step, so a pool short of two stays short.
    std::vector<SolutionPool::EntryRef> entries;
    sink.entry_refs(entries);
    if (entries.size() < 2) {
        return {};
    }
    return {.effort = run_opportunistic_loop(
                exec, budget,
                [&](int worker, Rng& rng) {
                    return RecombineOperator(problem, exec, sink, rng(), WorkerTrace{worker, 0});
                },
                [&](RecombineOperator& op, Rng& /*rng*/, size_t run_cap) -> AttemptResult {
                    // No step charges nothing, which retires the slot.
                    const std::optional<RecombineStep> step = capped_step(op, exec, run_cap);
                    return step.has_value() ? AttemptResult{.effort = step->effort,
                                                            .found_improvement = step->improved}
                                            : AttemptResult{};
                })};
}

size_t run_until_stopped(const ProblemView& problem, const HeuristicBudget& budget,
                         const ExecutionContext& exec, int worker, const RestartSource& source,
                         SolutionSink& sink) {
    if (!can_improve(problem)) {
        return 0;
    }
    return run_on_caller_thread(
        exec, budget, worker,
        [&](int worker_idx, Rng& rng) {
            return RecombineOperator(problem, exec, sink, rng(), WorkerTrace{worker_idx, 0});
        },
        [&](RecombineOperator& op, Rng& /*rng*/, size_t run_cap) -> AttemptResult {
            // Wait for two entries, as a ruin slot waits for its point, rather
            // than return zero effort, which would end the run.  This thread
            // holds the poller seat, so the wait polls the terminator too.
            size_t effort = 0;
            while (true) {
                effort += op.take_external(source);
                // `source` or `on_accept` may have set the stop just now, and
                // a recombiner would not see it.
                if (exec.past_deadline()) {
                    return {.effort = effort, .found_improvement = false};
                }
                const std::optional<RecombineStep> step = capped_step(op, exec, run_cap);
                if (step.has_value()) {
                    return {.effort = effort + step->effort, .found_improvement = step->improved};
                }
                if (exec.terminated()) {
                    return {.effort = effort, .found_improvement = false};
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(ruin_repair::kWaitForSeedMs));
            }
        });
}

}  // namespace recombine
