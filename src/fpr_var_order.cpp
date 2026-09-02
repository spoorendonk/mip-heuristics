#include "fpr_var_order.h"

#include "clique_cover.h"
#include "heuristic_common.h"
#include "mip/HighsCliqueTable.h"
#include "mip/HighsDomain.h"
#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

// ===================================================================
// Variable ranking
// ===================================================================

namespace {

// Bucket binary, general-integer, and continuous variables.
struct TypeBuckets {
    std::vector<HighsInt> bin;
    std::vector<HighsInt> gen_int;
    std::vector<HighsInt> cont;
};

// Reads the live root domain rather than a dispatch snapshot
// (`ProblemView::binary`, issue #99).  Legal because every production
// caller reaches this from the dispatching thread, before any parallel
// region opens: `fpr::precompute_var_orders`, `fpr_lp`'s `build_setup`,
// and — since #99 — `scylla::precompute_config_var_orders`.  That last one
// exists precisely because `ScyllaWorker`'s constructor used to compute its
// own order, and `scylla::run` rebuilds a retired worker from inside the
// parallel loop.  It is the same property that makes the clique-table reads
// below safe — `HighsCliqueTable::getCliques()` / `getCliqueEntries()` hand
// out references into state `addIncumbent`'s `extractObjCliques` reallocates —
// and it is load-bearing for both: keep any new caller on the dispatching
// thread.
TypeBuckets bucket_by_type(const HighsMipSolver& mipsolver) {
    const auto* model = mipsolver.model_;
    auto* mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    TypeBuckets b;
    for (HighsInt j = 0; j < ncol; ++j) {
        if (!is_integer(model->integrality_, j)) {
            b.cont.push_back(j);
        } else if (mipdata->getDomain().isBinary(j)) {
            b.bin.push_back(j);
        } else {
            b.gen_int.push_back(j);
        }
    }
    return b;
}

std::vector<HighsInt> concat_buckets(TypeBuckets& b) {
    std::vector<HighsInt> order;
    order.reserve(b.bin.size() + b.gen_int.size() + b.cont.size());
    order.insert(order.end(), b.bin.begin(), b.bin.end());
    order.insert(order.end(), b.gen_int.begin(), b.gen_int.end());
    order.insert(order.end(), b.cont.begin(), b.cont.end());
    return order;
}

// --- LR: formulation order ---
std::vector<HighsInt> rank_lr(const HighsMipSolver& mipsolver) {
    const HighsInt ncol = mipsolver.model_->num_col_;
    std::vector<HighsInt> order(ncol);
    std::ranges::iota(order, 0);
    return order;
}

// --- type: grouped by type, formulation order within ---
std::vector<HighsInt> rank_type(const HighsMipSolver& mipsolver) {
    auto b = bucket_by_type(mipsolver);
    return concat_buckets(b);
}

// --- random: type buckets, random within ---
std::vector<HighsInt> rank_random(const HighsMipSolver& mipsolver, Rng& rng) {
    auto b = bucket_by_type(mipsolver);
    std::shuffle(b.bin.begin(), b.bin.end(), rng);
    std::shuffle(b.gen_int.begin(), b.gen_int.end(), rng);
    std::shuffle(b.cont.begin(), b.cont.end(), rng);
    return concat_buckets(b);
}

// --- locks: sorted by max(uplocks, downlocks) descending within type ---
std::vector<HighsInt> rank_locks(const HighsMipSolver& mipsolver) {
    auto* mipdata = mipsolver.mipdata_.get();
    const auto& up = mipdata->uplocks;
    const auto& dn = mipdata->downlocks;

    auto b = bucket_by_type(mipsolver);
    auto cmp = [&](HighsInt a, HighsInt b_idx) {
        return std::max(up[a], dn[a]) > std::max(up[b_idx], dn[b_idx]);
    };
    // Stable sort preserves formulation order as tiebreak
    std::ranges::stable_sort(b.bin, cmp);
    std::ranges::stable_sort(b.gen_int, cmp);
    // continuous locks are irrelevant for fixing order, but sort for consistency
    std::ranges::stable_sort(b.cont, cmp);
    return concat_buckets(b);
}

// --- typecl: clique cover for binaries, then type ---
// Paper Sect. 4.1's five-step greedy, in `clique_cover::build_clique_cover`:
// equality cliques first (disjoint ones only, in clique-table order), then a
// largest-covering-clique assignment over what is left, sorted by size; within
// a group, the order the variables appear in the clique; binaries no clique
// covers, at the end of the binary bucket in formulation order.
//
// This used to delegate to HiGHS's `cliquePartition`, which is a different
// greedy (#141): no equality pass, no disjointness test, no size sort,
// singletons interleaved rather than appended, and the within-clique order
// replaced by a sort of the column indices.  The paper's construction needs
// the clique table itself — the equality flag and the stored literal order —
// which `cliquePartition` does not expose; the patch adds the two const
// accessors that do (`third_party/highs_patch/apply_patch.cmake`).
std::vector<HighsInt> rank_typecl(const HighsMipSolver& mipsolver) {
    auto* mipdata = mipsolver.mipdata_.get();
    auto b = bucket_by_type(mipsolver);
    if (b.bin.empty()) {
        return concat_buckets(b);
    }

    const HighsCliqueTable& clq = mipdata->cliquetable;
    const clique_cover::Cover cover = clique_cover::build_clique_cover(
        clq.getCliques(), clq.getCliqueEntries(), b.bin, mipsolver.model_->num_col_);

    b.bin = cover.members;
    b.bin.insert(b.bin.end(), cover.uncovered.begin(), cover.uncovered.end());
    return concat_buckets(b);
}

// --- cliques: the same clique cover, analytic-center-weighted within ---
// Paper Sect. 4.1: "The same clique cover is also used in strategy cliques,
// where however we exploit the analytic center x^ac in order to sort variables
// within each clique in the cover."  So the cover is typecl's, verbatim, and
// only the within-group order differs — Fig. 2.
//
// Fig. 2 line 14 reads `w_j <- log(Rand(0,1) / w_j)` with an ascending
// `Sort(vars, w)`.  That is *not* what this code computes and the disagreement
// is deliberate: taken literally the figure's key is `log(u) - log(w)`, whose
// ascending order does favour large weights, but it is a different
// distribution from weighted sampling without replacement, and one closing
// parenthesis moved — `log(Rand(0,1)) / w_j` — turns it into the standard
// Efraimidis-Spirakis key, which is exactly weighted sampling without
// replacement and is what a "weighted discrete distribution" over the analytic
// center is meant to be.  We use the Efraimidis-Spirakis form, sorted
// *descending* (the figure's ascending sort pairs with its own expression, not
// with this one).  Recorded as an ambiguity in the reference, not as a claim
// about what the figure says (#141).
std::vector<HighsInt> rank_cliques(const HighsMipSolver& mipsolver, Rng& rng,
                                   const double* lp_ref) {
    if (lp_ref == nullptr) {
        return rank_typecl(mipsolver);
    }
    auto* mipdata = mipsolver.mipdata_.get();
    auto b = bucket_by_type(mipsolver);
    if (b.bin.empty()) {
        return concat_buckets(b);
    }

    const HighsCliqueTable& clq = mipdata->cliquetable;
    const clique_cover::Cover cover = clique_cover::build_clique_cover(
        clq.getCliques(), clq.getCliqueEntries(), b.bin, mipsolver.model_->num_col_);

    // Fig. 2 lines 5-6 skip a variable with `l_j = u_j`.  That test cannot
    // fire here: `bucket_by_type` puts a column in the binary bucket only when
    // the *root domain* has it at [0, 1], and the cover carries no other
    // column, so the filter is already applied upstream.  It is not dropped,
    // it is hoisted.
    b.bin.clear();
    std::vector<std::pair<HighsInt, double>> keyed;
    for (HighsInt g = 0; g < cover.num_groups(); ++g) {
        keyed.clear();
        for (HighsInt i = cover.group_start[g]; i < cover.group_start[g + 1]; ++i) {
            const HighsInt col = cover.members[i];
            double w = cover.member_pos[i] != 0 ? lp_ref[col] : 1.0 - lp_ref[col];
            w = std::max(w, 1e-10);  // avoid zero weights
            const double u = std::uniform_real_distribution<double>(1e-15, 1.0)(rng);
            keyed.emplace_back(col, std::log(u) / w);
        }
        std::ranges::sort(keyed, [](const auto& a, const auto& c) { return a.second > c.second; });
        for (const auto& [col, key] : keyed) {
            b.bin.push_back(col);
        }
    }
    b.bin.insert(b.bin.end(), cover.uncovered.begin(), cover.uncovered.end());
    return concat_buckets(b);
}

// --- cliques2: dynamic clique cover using LP solution (paper Fig. 3) ---
// A *different* cover from typecl's, by the paper's own description: "we
// construct a clique cover dynamically using both the clique table and a
// reference LP solution, in this case, a zero-objective vertex.  The method is
// quite simple: we just loop over the cliques in the problem, skipping fixed
// variables and non-tight cliques (w.r.t. to the reference LP solution), pick
// the most positive literal in the clique and then the remaining uncovered
// binary variables (again, in the order they appear in the clique)."  One pass
// over the whole table, no partition step — see `clique_cover::cliques2_order`
// for the figure line by line, including the skip of an entire clique whose
// literal is already fixed true.
//
// The reference vector is the zero-objective vertex, and `fpr_lp`'s arm table
// is what binds it there (#128); nothing in this function chooses it.
std::vector<HighsInt> rank_cliques2(const HighsMipSolver& mipsolver, const double* lp_ref) {
    if (lp_ref == nullptr) {
        return rank_typecl(mipsolver);
    }
    auto* mipdata = mipsolver.mipdata_.get();
    auto b = bucket_by_type(mipsolver);
    if (b.bin.empty()) {
        return concat_buckets(b);
    }

    const HighsCliqueTable& clq = mipdata->cliquetable;
    const HighsDomain& dom = mipdata->getDomain();
    b.bin = clique_cover::cliques2_order(clq.getCliques(), clq.getCliqueEntries(), b.bin,
                                         mipsolver.model_->num_col_, lp_ref, dom.col_lower_.data(),
                                         dom.col_upper_.data());
    return concat_buckets(b);
}

}  // namespace

std::vector<HighsInt> compute_var_order(const HighsMipSolver& mipsolver, VarStrategy strategy,
                                        Rng& rng, const double* lp_ref) {
    switch (strategy) {
        case VarStrategy::kLR:
            return rank_lr(mipsolver);
        case VarStrategy::kType:
            return rank_type(mipsolver);
        case VarStrategy::kRandom:
            return rank_random(mipsolver, rng);
        case VarStrategy::kLocks:
            return rank_locks(mipsolver);
        case VarStrategy::kTypecl:
            return rank_typecl(mipsolver);
        case VarStrategy::kCliques:
            return rank_cliques(mipsolver, rng, lp_ref);
        case VarStrategy::kCliques2:
            return rank_cliques2(mipsolver, lp_ref);
        case VarStrategy::kDomainSize:
            // Dynamic strategy: initial order is type-based; actual selection
            // happens at each DFS node in fpr_core via find_smallest_domain.
            return rank_type(mipsolver);
    }
    return rank_type(mipsolver);  // unreachable
}
