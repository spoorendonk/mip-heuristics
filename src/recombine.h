#pragma once
#include "fpr_core.h"
#include "heuristic_context.h"
#include "prop_engine.h"
#include "rng.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

struct Deadline;

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
// cuOpt normalises by wall clock, `4 - ms / 2000` ("normal recombiners take
// 2000 ms"); here by effort, 2 s at FJ's laptop rate
// (`ruin_repair::kRepairEffortPerSecond`), since FJ's repair dominates the
// operator's work and a clock would break the same-seed replay (a
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

}  // namespace recombine
