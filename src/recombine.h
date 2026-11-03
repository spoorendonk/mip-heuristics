#pragma once
#include "fpr_core.h"
#include "heuristic_context.h"
#include "prop_engine.h"
#include "rng.h"
#include "solution_pool.h"
#include "worker_base.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

struct Deadline;
struct ExecutionContext;
class SolutionSink;

// Recombination of two solutions (#209, after NVIDIA cuOpt's diversity
// manager at f3fe1a5, Apache-2.0; read for technique, no code copied).
namespace recombine {

// cuOpt's `bp_recombiner_config_t` (`recombiners/recombiner_configs.hpp`):
// at most `max_vars_from_other` of the integer columns where the parents
// differ are left free, starting at the initial value and moved by
// `next_max_vars_from_other` within [min, max].
inline constexpr size_t kBpInitialVarsFromOther = 200;
inline constexpr size_t kBpMinVarsFromOther = 20;
inline constexpr size_t kBpMaxVarsFromOther = 10000;
inline constexpr double kBpVarsIncreaseFactor = 1.1;
inline constexpr double kBpVarsDecreaseFactor = 0.99;

// cuOpt's adaptation, applied only after a recombination whose parents
// differed on more columns than the limit allowed free: an offspring equal
// to a parent on the integer columns grows the limit by
// `kBpVarsIncreaseFactor` (rounded up, at most `kBpMaxVarsFromOther`), any
// other shrinks it by `kBpVarsDecreaseFactor` (rounded down, at least
// `kBpMinVarsFromOther`).
[[nodiscard]] size_t next_max_vars_from_other(size_t current, bool same_as_parent);

// What one recombination produced.  `success` is cuOpt's verdict, the
// offspring is better than both parents or differs from both on the
// integer columns, with "better" meaning feasible and of strictly lower
// objective: cuOpt compares a weighted quality that can rank an infeasible
// offspring above feasible parents, and there are no weights here (a
// deviation).  `objective` is set only when `feasible`.
struct Offspring {
    bool success = false;
    bool feasible = false;
    double objective = kHighsInf;
    size_t effort = 0;
};

// cuOpt's bound-prop recombiner (`bound_prop_recombiner_t::recombine`,
// feasible-guiding branch), with FPR as the completion:
//   1. `a` guides, whatever the objectives, as in cuOpt when both parents
//      are feasible: the caller's draw decides the order.  Both are taken
//      to be feasible, as every pool entry is: cuOpt's branch for an
//      infeasible guiding parent is not carried over.
//   2. Integer columns where the rounded values agree are fixed there.
//   3. Of the `d` columns where they differ, at most
//      `max_vars_from_other()` stay free, a random subset when `d` is
//      larger; the rest are fixed at the guiding parent's value.
//   4. With no column fixed, or none free, there is no offspring: `out` is
//      left empty and `success` false.
//   5. The fixings are propagated once, into the bounds of a restricted
//      view, and FPR completes the free columns on it
//      (`fpr_attempt`, `kDfsrep`): value strategy `kLp` on the parents'
//      midpoint, which for a column where they differ by one rounds to
//      either parent's value on a fair coin, with the other value as the
//      branch's alternative for a binary; continuous zero-cost columns fill
//      at the midpoint.  Every fixed column keeps its value: FPR never
//      leaves the view's bounds.
//   6. `max_vars_from_other()` adapts (see `next_max_vars_from_other`).
//
// Deviations from cuOpt, each recorded beside its code in recombine.cpp:
// FPR's DFS stands in for cuOpt's single-rounding probe (whose second probe
// for a general integer column is the other parent's value, where FPR's is
// the domain's other end) and FPR's fill and WalkSAT repair for cuOpt's LP
// over the continuous columns; the free columns are probed in a random
// order; there is no wall-clock limit, only `max_effort`.
//
// One per worker: it owns its scratch and its adaptive limit (a static in
// cuOpt, which would be a race across threads here).  Not copyable, since
// its restricted view points into its own members.
class BoundPropRecombiner {
public:
    // `problem` must outlive the recombiner.
    explicit BoundPropRecombiner(const ProblemView& problem);
    BoundPropRecombiner(const BoundPropRecombiner&) = delete;
    BoundPropRecombiner& operator=(const BoundPropRecombiner&) = delete;
    BoundPropRecombiner(BoundPropRecombiner&&) = delete;
    BoundPropRecombiner& operator=(BoundPropRecombiner&&) = delete;
    ~BoundPropRecombiner() = default;

    // Recombine `a` and `b` (objectives `obj_a`, `obj_b` in the view's
    // minimisation form) into `out`: an `ncol` point, integral on the
    // integer columns and within the view's bounds, feasible or not, or
    // empty when there is no offspring.  `max_effort` caps FPR's search, in
    // its own unit (coefficient accesses); `effort` reports everything
    // spent, the propagation of the fixings and the O(ncol) setup included.
    [[nodiscard]] Offspring recombine(const std::vector<double>& a, double obj_a,
                                      const std::vector<double>& b, double obj_b, size_t max_effort,
                                      const Deadline& deadline, Rng& rng, std::vector<double>& out);

    [[nodiscard]] size_t max_vars_from_other() const { return max_vars_from_other_; }

private:
    // Fix the columns step 3 leaves fixed, propagate, and write the result
    // into the restricted bounds.  Returns the propagation's effort.
    size_t fix_and_propagate(PropEngine& fixer, const std::vector<double>& guiding, size_t num_free,
                             const Deadline& deadline);

    const ProblemView& problem_;
    ProblemView restricted_;
    // The restricted view's column bounds, and the parents' midpoint.  Sized
    // once, so the pointers the view and the engines hold stay valid.
    std::vector<double> col_lower_;
    std::vector<double> col_upper_;
    std::vector<double> midpoint_;
    std::vector<HighsInt> integer_cols_;
    // The differing integer columns; the first `num_free` are left free.
    std::vector<HighsInt> differing_;
    std::vector<uint8_t> is_free_;
    // Empty lock counts for the restricted view, which switch off FPR's
    // trivially-roundable fixings: they would move a free column to a bound
    // neither parent holds.
    std::vector<HighsInt> no_locks_;
    std::optional<PropEngine> fixer_;
    FprScratch scratch_;
    size_t max_vars_from_other_ = kBpInitialVarsFromOther;
};

// The recombiners a `RecombinerBandit` chooses among: the bound-prop
// recombiner above and the pool's two crossovers.  cuOpt's sub-MIP arm has
// no counterpart in the core (HiGHS's RENS/RINS live outside it).
enum class RecombinerArm : uint8_t { kBoundProp, kGuidedCrossover, kNeighborhoodCrossover };
inline constexpr size_t kNumRecombinerArms = 3;

// cuOpt's `mab_t` with UCB (`diversity/multi_armed_bandit.cu`): the step
// size of the recency-weighted average, an arm's initial estimate, and the
// tolerance within which two UCB values tie.
inline constexpr double kBanditAlpha = 0.05;
inline constexpr double kBanditInitialQ = 0.5;
inline constexpr double kBanditTieTolerance = 1e-9;

// cuOpt's recombiner reward (`mab_t::add_mab_reward`): `kRewardBeatsBest`
// for an offspring better than the best known; for one better than the
// better parent and within `kRewardNearBest` of the best (relative,
// `|off - best| / (|best| + 1)`), the work-normalised
// `max(kRewardWorkFloor, kRewardWorkCeiling - effort / kRewardNominalEffort)`.
// "Better" is by more than `max(kRewardMarginAbs, kRewardMarginRel * |best|)`.
// `effort` is the recombiner's alone, not the local search after it, as
// cuOpt times only the recombiner.  cuOpt normalises by wall clock,
// `4 - ms / 2000` ("normal recombiners take 2000 ms"); here by effort, since
// a clock would break the same-seed replay, with 2 s at FJ's laptop rate
// (`ruin_repair::kRepairEffortPerSecond`) standing in for 2 s of any
// recombiner: FPR's rate, the bound-prop arm's unit, is not measured (a
// deviation).
inline constexpr double kRewardBeatsBest = 8.0;
inline constexpr double kRewardNearBest = 0.2;
inline constexpr double kRewardWorkCeiling = 4.0;
inline constexpr double kRewardWorkFloor = 0.1;
inline constexpr double kRewardNominalEffort = 4.0e8;
inline constexpr double kRewardMarginAbs = 1e-6;
inline constexpr double kRewardMarginRel = 1e-4;

// The reward for an offspring of objective `offspring` (minimisation form;
// `kHighsInf` for none, or an infeasible one), whose better parent has
// `best_of_parents`, against the best known `best`, after `effort` spent.
// A failed recombination is a 0 reward: pass `kHighsInf`.
[[nodiscard]] double recombination_reward(double best_of_parents, double best, double offspring,
                                          size_t effort);

// UCB with exponential recency weighting over the recombiners, as cuOpt's
// `mab_t`: every arm is pulled once first, in index order; then the arm of
// largest `q + sqrt(2 ln t / n)`, where `t` counts `select` calls and `n`
// counts `reward` calls on the arm, ties broken at random.  Each reward
// moves `q` by `kBanditAlpha (r - q)`.
//
// One per worker, no lock: cuOpt's single bandit lives on one host loop,
// while the operator runs on parallel workers here, where a shared bandit's
// arm sequence would depend on thread interleaving and a worker would no
// longer replay from its seed (#234).
class RecombinerBandit {
public:
    explicit RecombinerBandit(uint64_t seed) : rng_(seed) {}

    [[nodiscard]] RecombinerArm select();
    void reward(RecombinerArm arm, double r);

    [[nodiscard]] size_t pulls(RecombinerArm arm) const { return arms_[index(arm)].pulls; }
    [[nodiscard]] double q(RecombinerArm arm) const { return arms_[index(arm)].q; }

private:
    struct Arm {
        size_t pulls = 0;
        double q = kBanditInitialQ;
    };
    static size_t index(RecombinerArm arm) { return static_cast<size_t>(arm); }

    std::array<Arm, kNumRecombinerArms> arms_{};
    size_t steps_ = 0;
    Rng rng_;
};

// What one `RecombineOperator::step` did.  `objective` is the best feasible
// objective the step produced, the offspring's or FJ's from it, or
// `kHighsInf`.
struct RecombineStep {
    RecombinerArm arm = RecombinerArm::kBoundProp;
    bool success = false;
    double objective = kHighsInf;
    double reward = 0.0;
    size_t effort = 0;
};

// cuOpt's recombine-and-local-search step
// (`diversity_manager_t::recombine_and_local_search`) over the sink's pool:
//   1. Two entries, by cuOpt's tournament draw
//      (`population_t::get_two_random`) on the best-first pool:
//      `i = min(U[0, n-1], U[0, n-1])`, `j = min(U[0, n-2], U[0, n-2])`,
//      `j + 1` when `j >= i`, passed in that order, so `i` guides the
//      bound-prop recombiner.  cuOpt draws over a list that starts with a
//      reserved slot for the best feasible point, and its range stops short
//      of the last position, so there the best sits at two positions and the
//      worst is never drawn.  The pool has no such slot, so here the best
//      weighs less and the worst can be drawn: on a pool of ten, the first
//      parent is the best with probability 19% against cuOpt's 36%, and the
//      worst with 1% against none (a deviation).  cuOpt's check that the
//      first two drawn are not twins has no counterpart: the pool holds no
//      twins.
//   2. The bandit's arm recombines them: bound-prop, or the pool's guided
//      crossover (`i`, `j`), or its neighborhood crossover (the better of
//      the two first).  A crossover succeeds when its offspring differs
//      from both parents on the integer columns, bound-prop by its own
//      verdict.  A failure rewards the arm 0 and ends the step, as in
//      cuOpt, which runs no local search then.
//   3. A feasible offspring is offered to the sink as it is (cuOpt adds its
//      offspring to the population after the local search, and its LP
//      offspring beside it); then FJ runs from it for `ls_budget` effort,
//      offering what it finds to the sink, unless the deadline has passed
//      by then, which rewards 0 as cuOpt's local search does.  FJ for cuOpt's
//      local search,
//      since the crossovers' offspring are not repaired.
//      cuOpt's LP over the continuous columns after the local search, and
//      the LP offspring it yields, are not carried over: a feasible
//      offspring's continuous columns are not re-optimised, so its objective
//      and the arm's reward can be worse than cuOpt's on the same integer
//      assignment (a deviation).
//   4. The arm is rewarded with `recombination_reward` on the parents'
//      better objective, the pool's best at the draw, the step's best
//      objective, and the recombiner's effort.  The pool's best at the draw,
//      not at the reward as in cuOpt: FJ offers into the pool during the
//      local search, so by then the best would include the offspring and
//      the reward of 8 could never be paid (a deviation).
//
// One per worker (its bandit and recombiner are per worker, see
// `RecombinerBandit`).  `problem` and `sink` must outlive it.
class RecombineOperator {
public:
    RecombineOperator(const ProblemView& problem, const ExecutionContext& exec, SolutionSink& sink,
                      uint64_t seed, WorkerTrace trace);

    // One step, with `recombine_effort` capping the bound-prop recombiner's
    // search and `ls_budget` FJ's.  None with fewer than two pool entries,
    // and then no arm is pulled.  `ls_budget` is not a cap below FJ's
    // callback period: FJ tests it only at a callback, on a reported point or
    // every 500000 effort, so a step spends up to that much whatever is
    // passed; `RecombineStep::effort` reports what was spent.  `deadline`
    // governs the recombiner and the check before FJ, `exec`'s clock FJ
    // itself, so a `deadline` tighter than `exec`'s does not interrupt FJ.
    [[nodiscard]] std::optional<RecombineStep> step(size_t recombine_effort, size_t ls_budget,
                                                    const Deadline& deadline);

    [[nodiscard]] const RecombinerBandit& bandit() const { return bandit_; }

private:
    // Run `arm` on the parents into `offspring_`; the recombiner's verdict.
    Offspring recombine(RecombinerArm arm, const SolutionPool::EntryRef& first,
                        const SolutionPool::EntryRef& second, size_t recombine_effort,
                        const Deadline& deadline);

    const ProblemView& problem_;
    const ExecutionContext& exec_;
    SolutionSink& sink_;
    BoundPropRecombiner bound_prop_;
    RecombinerBandit bandit_;
    Rng rng_;
    // Its `effort_base` is everything this operator has spent, what its
    // offers and FJ's report.
    WorkerTrace trace_;
    std::vector<bool> integer_mask_;
    std::vector<HighsInt> integer_cols_;
    std::vector<SolutionPool::EntryRef> entries_;
    std::vector<double> offspring_;
};

}  // namespace recombine
