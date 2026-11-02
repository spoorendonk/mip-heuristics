#pragma once
#include "fpr_core.h"
#include "heuristic_context.h"
#include "prop_engine.h"
#include "rng.h"

#include <cstddef>
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

// What one recombination produced.  `success` is cuOpt's verdict: the
// offspring is feasible and strictly better than both parents, or it
// differs from both on the integer columns.  `objective` is set only when
// `feasible`.
struct Offspring {
    bool success = false;
    bool feasible = false;
    double objective = kHighsInf;
    size_t effort = 0;
};

// cuOpt's bound-prop recombiner (`bound_prop_recombiner_t::recombine`,
// feasible-guiding branch), with FPR as the completion:
//   1. The guiding parent is the one with the lower objective (the first
//      on a tie); the other is the other.  Both are taken to be feasible,
//      as every pool entry is: cuOpt's branch for an infeasible guiding
//      parent is not carried over.
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
    size_t fix_and_propagate(const std::vector<double>& guiding, size_t num_free,
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

}  // namespace recombine
