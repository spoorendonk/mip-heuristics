#pragma once

#include "mip/HighsCliqueTable.h"
#include "util/HighsInt.h"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Clique covers for the paper's binary-variable rankings
// (Salvagnin, Roberti, Fischetti, "A fix-propagate-repair heuristic for mixed
// integer programming", MPC 17:111-139, 2025, Sect. 4.1 and Figs. 2-3).
//
// Two constructions live here, and they are deliberately *different* covers,
// because the paper defines them that way:
//
//   * `build_clique_cover` is the five-step greedy of Sect. 4.1 — the one
//     `typecl` uses, and the one the paper says `cliques` shares ("The same
//     clique cover is also used in strategy cliques, where however we exploit
//     the analytic center ... to sort variables within each clique").  It is
//     LP-free.
//   * `cliques2_order` is Fig. 3 — a cover built *dynamically* from the clique
//     table and a reference LP solution, one pass over every clique, no
//     partition step.
//
// Both are pure functions over a clique-table snapshot so they can be tested
// against synthetic tables.  They read no solver state themselves; their
// caller (`compute_var_order`) does, and is dispatching-thread-only (#99).
// ---------------------------------------------------------------------------

namespace clique_cover {

using Clique = HighsCliqueTable::Clique;
using CliqueVar = HighsCliqueTable::CliqueVar;

// Tolerance on Fig. 3 line 24's `sum = 1` tightness test.  The figure tests
// exact equality; a reference LP solution is a floating-point vector, so the
// literal sum of a tight clique lands within rounding of 1 rather than on it.
// The test is **two-sided**: a one-sided `sum >= 1 - tol` is only equivalent
// on cliques that are rows of the model, where LP feasibility caps the literal
// sum at 1.  This pass iterates the whole clique table, which also holds
// merged and lifted cliques that no single row bounds, and those can exceed 1
// at a given LP point — a one-sided test would reorder exactly the cliques the
// paper's `sum = 1` skips.
inline constexpr double kCliqueTightnessTol = 1e-6;

// A clique must contribute at least this many *rankable* binaries — columns in
// the binary bucket — before it can become a cover group.  The paper is silent
// because the question does not arise in its setting; here the root domain can
// have fixed some of a clique's columns since the table was built, and a group
// of one variable expresses no ordering the uncovered tail does not already
// express.
inline constexpr HighsInt kMinCliqueGroupSize = 2;

// The output of the Sect. 4.1 greedy: binaries grouped by the clique that
// covers them, plus the ones no clique does.
struct Cover {
    // Covered binaries, concatenated group by group.  Within a group the order
    // is "the order in which they appear in the clique itself" (Sect. 4.1),
    // i.e. the clique table's own entry order.
    std::vector<HighsInt> members;
    // `member_pos[i]` is 1 iff `members[i]` entered its group as a positive
    // literal.  Fig. 2's weights are literal-valued, so `cliques` needs this.
    std::vector<uint8_t> member_pos;
    // Half-open group bounds into `members`; size is `num_groups + 1`.
    std::vector<HighsInt> group_start;
    // Binaries no group covers, in formulation order.  Sect. 4.1: "Binary
    // variables not covered by any clique are moved to the end of the binary
    // bucket, again in formulation order."
    std::vector<HighsInt> uncovered;
    // The first `num_equality_groups` groups are the equality cliques of step
    // 1, in clique-table order; the rest are the step-2 selection, sorted by
    // size descending.
    HighsInt num_equality_groups = 0;

    [[nodiscard]] HighsInt num_groups() const {
        return group_start.empty() ? 0 : static_cast<HighsInt>(group_start.size()) - 1;
    }
};

// Sect. 4.1's five-step greedy.  `cliques` / `entries` are the raw clique
// table (dead slots, marked by `start == -1`, are skipped); `bin` is the
// binary bucket in formulation order; `ncol` sizes the column-indexed scratch.
Cover build_clique_cover(const std::vector<Clique>& cliques, const std::vector<CliqueVar>& entries,
                         const std::vector<HighsInt>& bin, HighsInt ncol);

// Fig. 3.  Returns the whole binary bucket as one ordered list: the cliques it
// selected, in clique-table order, followed by the binaries none of them
// covered, in formulation order.  `dom_lower` / `dom_upper` are the *root
// domain* bounds — the figure's `l` and `u` — which is what makes its
// already-fixed-literal test (lines 9-10 and 17-18) mean anything: a column
// the root propagation fixed is no longer in `bin` at all.
std::vector<HighsInt> cliques2_order(const std::vector<Clique>& cliques,
                                     const std::vector<CliqueVar>& entries,
                                     const std::vector<HighsInt>& bin, HighsInt ncol,
                                     const double* lp_ref, const double* dom_lower,
                                     const double* dom_upper);

}  // namespace clique_cover
