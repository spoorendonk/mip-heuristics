#include "fpr_var_order.h"

#include "heuristic_common.h"
#include "mip/HighsCliqueTable.h"
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

TypeBuckets bucket_by_type(const HighsMipSolver& mipsolver) {
    const auto* model = mipsolver.model_;
    auto* mipdata = mipsolver.mipdata_.get();
    const HighsInt ncol = model->num_col_;
    TypeBuckets b;
    for (HighsInt j = 0; j < ncol; ++j) {
        if (!is_integer(model->integrality_, j)) {
            b.cont.push_back(j);
        } else if (mipdata->domain.isBinary(j)) {
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
    std::iota(order.begin(), order.end(), 0);
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
    std::stable_sort(b.bin.begin(), b.bin.end(), cmp);
    std::stable_sort(b.gen_int.begin(), b.gen_int.end(), cmp);
    // continuous locks are irrelevant for fixing order, but sort for consistency
    std::stable_sort(b.cont.begin(), b.cont.end(), cmp);
    return concat_buckets(b);
}

// --- typecl: clique cover for binaries, then type ---
// Paper's greedy algorithm: process equality cliques first, add disjoint ones,
// then remaining. Within each clique, formulation order.
std::vector<HighsInt> rank_typecl(const HighsMipSolver& mipsolver) {
    auto* mipdata = mipsolver.mipdata_.get();
    auto b = bucket_by_type(mipsolver);

    // Build CliqueVar vector for all binary variables (positive literal)
    using CV = HighsCliqueTable::CliqueVar;
    std::vector<CV> clq_vars;
    clq_vars.reserve(b.bin.size());
    for (HighsInt j : b.bin) {
        clq_vars.push_back(CV(j, 1));
    }

    if (clq_vars.empty()) {
        return concat_buckets(b);
    }

    std::vector<HighsInt> partition_start;
    mipdata->cliquetable.cliquePartition(clq_vars, partition_start);

    // Rebuild binary bucket: clique order, formulation order within each clique
    b.bin.clear();
    for (size_t c = 0; c + 1 < partition_start.size(); ++c) {
        HighsInt start = partition_start[c];
        HighsInt end = partition_start[c + 1];
        // Extract columns in this clique
        std::vector<HighsInt> clique_cols;
        for (HighsInt k = start; k < end; ++k) {
            clique_cols.push_back(static_cast<HighsInt>(clq_vars[k].col));
        }
        // Sort by formulation order within clique
        std::sort(clique_cols.begin(), clique_cols.end());
        b.bin.insert(b.bin.end(), clique_cols.begin(), clique_cols.end());
    }

    return concat_buckets(b);
}

// --- cliques: clique partition + analytic-center-weighted random sort ---
// Paper's Fig. 2: within each clique, sort using weighted discrete distribution
// based on analytic center values.
std::vector<HighsInt> rank_cliques(const HighsMipSolver& mipsolver, Rng& rng,
                                   const double* lp_ref) {
    auto* mipdata = mipsolver.mipdata_.get();
    const auto& col_lb = mipsolver.model_->col_lower_;
    const auto& col_ub = mipsolver.model_->col_upper_;
    auto b = bucket_by_type(mipsolver);

    using CV = HighsCliqueTable::CliqueVar;
    std::vector<CV> clq_vars;
    clq_vars.reserve(b.bin.size());
    for (HighsInt j : b.bin) {
        clq_vars.push_back(CV(j, 1));
    }

    if (clq_vars.empty() || !lp_ref) {
        return rank_typecl(mipsolver);
    }

    std::vector<HighsInt> partition_start;
    mipdata->cliquetable.cliquePartition(clq_vars, partition_start);

    // Rebuild binary bucket with weighted random sort within each clique
    b.bin.clear();
    for (size_t c = 0; c + 1 < partition_start.size(); ++c) {
        HighsInt start = partition_start[c];
        HighsInt end = partition_start[c + 1];
        // Collect vars and weights (paper Fig. 2)
        std::vector<std::pair<HighsInt, double>> vars_weights;
        for (HighsInt k = start; k < end; ++k) {
            HighsInt col = static_cast<HighsInt>(clq_vars[k].col);
            // Skip fixed variables
            if (col_ub[col] - col_lb[col] < 1e-6) {
                continue;
            }
            // Weight = x_ac[col] for positive literal (val=1)
            double w = clq_vars[k].val ? lp_ref[col] : 1.0 - lp_ref[col];
            w = std::max(w, 1e-10);  // avoid zero weights
            vars_weights.push_back({col, w});
        }

        if (vars_weights.empty()) {
            continue;
        }

        // Weighted random sort: paper uses log(Rand(0,1))/w as sort key
        // This is the Gumbel-max trick for weighted sampling without replacement
        for (auto& [col, w] : vars_weights) {
            double u = std::uniform_real_distribution<double>(1e-15, 1.0)(rng);
            w = std::log(u) / w;
        }
        std::sort(vars_weights.begin(), vars_weights.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        for (const auto& [col, _] : vars_weights) {
            b.bin.push_back(col);
        }
    }

    return concat_buckets(b);
}

// --- cliques2: dynamic clique cover using LP solution (paper Fig. 3) ---
// For each clique, pick the most positive literal w.r.t. LP solution,
// then remaining uncovered binaries.
std::vector<HighsInt> rank_cliques2(const HighsMipSolver& mipsolver, const double* lp_ref) {
    auto* mipdata = mipsolver.mipdata_.get();
    auto b = bucket_by_type(mipsolver);

    if (b.bin.empty() || !lp_ref) {
        return rank_typecl(mipsolver);
    }

    using CV = HighsCliqueTable::CliqueVar;
    std::vector<CV> clq_vars;
    clq_vars.reserve(b.bin.size());
    for (HighsInt j : b.bin) {
        clq_vars.push_back(CV(j, 1));
    }

    // Use the objective-weighted clique partition variant
    // This sorts within each clique by LP-weighted preference
    std::vector<HighsInt> partition_start;
    std::vector<double> lp_vec(lp_ref, lp_ref + mipsolver.model_->num_col_);
    mipdata->cliquetable.cliquePartition(lp_vec, clq_vars, partition_start);

    // Paper Fig. 3: for each clique, pick best variable, then rest
    b.bin.clear();
    const auto& col_lb = mipsolver.model_->col_lower_;
    const auto& col_ub = mipsolver.model_->col_upper_;

    for (size_t c = 0; c + 1 < partition_start.size(); ++c) {
        HighsInt start = partition_start[c];
        HighsInt end = partition_start[c + 1];

        // Paper Fig. 3: compute sum of LP literal values for tightness check
        double sum = 0.0;
        HighsInt best_col = -1;
        double best_val = -1.0;

        for (HighsInt k = start; k < end; ++k) {
            HighsInt col = static_cast<HighsInt>(clq_vars[k].col);
            double v = clq_vars[k].val ? lp_ref[col] : 1.0 - lp_ref[col];
            sum += v;
            if (col_ub[col] - col_lb[col] < 1e-6) {
                continue;
            }
            if (v > best_val) {
                best_val = v;
                best_col = col;
            }
        }

        // Paper Fig. 3, line 24: only reorder if clique is LP-tight (sum ≈ 1)
        if (best_col >= 0 && sum >= 1.0 - 1e-6) {
            b.bin.push_back(best_col);
            for (HighsInt k = start; k < end; ++k) {
                HighsInt col = static_cast<HighsInt>(clq_vars[k].col);
                if (col != best_col) {
                    b.bin.push_back(col);
                }
            }
        } else {
            // Not tight — keep original clique order
            for (HighsInt k = start; k < end; ++k) {
                b.bin.push_back(static_cast<HighsInt>(clq_vars[k].col));
            }
        }
    }

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
