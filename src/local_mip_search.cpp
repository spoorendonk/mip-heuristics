// Candidate scoring and the Algorithm-2 `infeasible_step` driver.
//
// Declared in local_mip_core.h alongside WorkerCtx / LiftCache.  Split
// from local_mip_core.cpp to keep each translation unit under ~500 LoC
// (issue #66 acceptance criterion).

#include "local_mip_caches.h"
#include "local_mip_core.h"
#include "lp_data/HConst.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace local_mip_detail {

namespace {

void append_candidate(WorkerCtx &ctx, std::vector<BatchCand> &batch, HighsInt j, double delta) {
    double new_val = ctx.clamp_and_round(j, ctx.solution[j] + delta);
    if (std::abs(new_val - ctx.solution[j]) < kEpsZero) {
        return;
    }
    batch.push_back({j, new_val});
}

}  // namespace

// Paper Definitions 5-10: two-level scoring function.
// Progress score (level 1): discrete constraint-transition scores + objective.
// Bonus score (level 2): breakthrough bonus + robustness bonus.
std::pair<double, double> compute_candidate_scores(WorkerCtx &ctx, HighsInt j, double new_val,
                                                   bool best_feasible, double best_obj) {
    double old_val = ctx.solution[j];
    double delta = new_val - old_val;
    if (std::abs(delta) < kEpsZero) {
        return {-std::numeric_limits<double>::infinity(), 0.0};
    }

    ctx.effort += ctx.csc.col_start[j + 1] - ctx.csc.col_start[j];

    // Def 5: progress score for objective
    double obj_delta = ctx.col_cost[j] * delta;
    double new_obj = ctx.current_obj + obj_delta;
    double eps = ctx.epsilon;
    double progress = 0.0;
    if ((!ctx.minimize && new_obj > ctx.current_obj + eps) ||
        (ctx.minimize && new_obj < ctx.current_obj - eps)) {
        progress += static_cast<double>(ctx.obj_weight);  // objective improved
    } else if ((!ctx.minimize && new_obj < ctx.current_obj - eps) ||
               (ctx.minimize && new_obj > ctx.current_obj + eps)) {
        progress -= static_cast<double>(ctx.obj_weight);  // objective worsened
    }

    // Def 8: breakthrough bonus (beats best-found solution)
    double bonus = 0.0;
    if (best_feasible) {
        bool beats_best = ctx.minimize ? (new_obj < best_obj - eps) : (new_obj > best_obj + eps);
        if (beats_best) {
            bonus += static_cast<double>(ctx.obj_weight);
        }
    }

    // Defs 6-7, 9-10: constraint progress + robustness
    for (HighsInt p = ctx.csc.col_start[j]; p < ctx.csc.col_start[j + 1]; ++p) {
        HighsInt i = ctx.csc.col_row[p];
        double coeff = ctx.csc.col_val[p];
        double old_lhs = ctx.lhs[i];
        double new_lhs = old_lhs + coeff * delta;
        double old_viol = ctx.viol_cache.get_or_compute(i, old_lhs, ctx.row_lo[i], ctx.row_hi[i]);
        double new_viol = ctx.compute_violation(i, new_lhs);
        double w = static_cast<double>(ctx.weight[i]);

        // Def 6: constraint progress score
        bool was_viol = (old_viol > kViolTol);
        bool now_viol = (new_viol > kViolTol);
        if (was_viol && !now_viol) {
            progress += w;  // violated → satisfied
        } else if (!was_viol && now_viol) {
            progress -= w;  // satisfied → violated
        } else if (was_viol && now_viol) {
            if (new_viol < old_viol - kViolTol) {
                progress += w;  // still violated, improved
            } else if (new_viol > old_viol + kViolTol) {
                progress -= w;  // still violated, worsened
            }
        }

        // Def 9: robustness bonus — only for transitions into strictly
        // satisfied (was violated or tight, now strictly interior).
        if (!now_viol) {
            bool old_strict =
                !was_viol &&
                (ctx.row_hi[i] >= kHighsInf || old_lhs < ctx.row_hi[i] - ctx.feastol) &&
                (ctx.row_lo[i] <= -kHighsInf || old_lhs > ctx.row_lo[i] + ctx.feastol);
            if (!old_strict) {
                bool new_strict =
                    (ctx.row_hi[i] >= kHighsInf || new_lhs < ctx.row_hi[i] - ctx.feastol) &&
                    (ctx.row_lo[i] <= -kHighsInf || new_lhs > ctx.row_lo[i] + ctx.feastol);
                if (new_strict) {
                    bonus += w;
                }
            }
        }
    }

    return {progress, bonus};
}

bool is_aspiration(const WorkerCtx &ctx, HighsInt j, double new_val, double best_obj,
                   bool best_feasible) {
    if (!best_feasible) {
        return false;
    }
    double delta = new_val - ctx.solution[j];
    double obj_delta = ctx.col_cost[j] * delta;
    double new_obj = ctx.current_obj + obj_delta;
    return ctx.minimize ? (new_obj < best_obj - ctx.epsilon) : (new_obj > best_obj + ctx.epsilon);
}

double compute_breakthrough_delta(const WorkerCtx &ctx, HighsInt j, double cur_obj,
                                  double best_obj) {
    double obj_coeff = ctx.col_cost[j];
    if (std::abs(obj_coeff) < kEpsZero) {
        return 0.0;
    }

    double obj_gap = cur_obj - best_obj;
    if (!ctx.minimize) {
        obj_gap = -obj_gap;
    }

    double delta = -obj_gap / obj_coeff;

    if (ctx.is_int(j)) {
        delta = (obj_coeff > 0) ? std::floor(delta) : std::ceil(delta);
    }
    double new_val = ctx.solution[j] + delta;
    if (new_val < ctx.col_lb[j] || new_val > ctx.col_ub[j]) {
        delta =
            (obj_coeff > 0) ? (ctx.col_lb[j] - ctx.solution[j]) : (ctx.col_ub[j] - ctx.solution[j]);
    }
    return delta;
}

Candidate select_best_from_batch(WorkerCtx &ctx, std::vector<BatchCand> &batch, HighsInt step,
                                 bool aspiration, double best_obj, bool best_feasible) {
    Candidate best;
    for (const auto &c : batch) {
        double delta = c.new_val - ctx.solution[c.var_idx];
        if (std::abs(delta) < kEpsZero) {
            continue;
        }

        if (ctx.is_tabu(c.var_idx, delta, step)) {
            if (!(aspiration &&
                  is_aspiration(ctx, c.var_idx, c.new_val, best_obj, best_feasible))) {
                continue;
            }
        }

        auto [prog, bon] =
            compute_candidate_scores(ctx, c.var_idx, c.new_val, best_feasible, best_obj);

        if (prog > best.score + kViolTol) {
            best = {c.var_idx, c.new_val, prog, bon};
        } else if (prog > best.score - kViolTol) {
            if (bon > best.bonus) {
                best = {c.var_idx, c.new_val, prog, bon};
            }
        }
    }
    ctx.viol_cache.reset();
    return best;
}

Candidate infeasible_step(WorkerCtx &ctx, std::mt19937 &rng, HighsInt step, bool best_feasible,
                          double best_objective, const std::vector<HighsInt> &costed_vars,
                          const std::vector<HighsInt> &binary_vars) {
    ctx.was_infeasible = true;

    auto &batch = ctx.batch;
    auto &sampled = ctx.sampled;

    // --- Phase 1: BMS tight moves from violated constraints ---
    HighsInt num_to_sample = std::min(kBmsConstraints * 3, ctx.violated.size());
    HighsInt num_to_keep = std::min(kBmsConstraints, ctx.violated.size());

    sampled.clear();
    if (num_to_sample == ctx.violated.size()) {
        for (auto ci : ctx.violated) {
            sampled.push_back({ci, ctx.weight[ci]});
        }
    } else {
        for (HighsInt s = 0; s < num_to_sample; ++s) {
            HighsInt idx = static_cast<HighsInt>(rng() % ctx.violated.size());
            sampled.push_back({ctx.violated[idx], ctx.weight[ctx.violated[idx]]});
        }
    }

    if (static_cast<HighsInt>(sampled.size()) > num_to_keep) {
        std::partial_sort(sampled.begin(), sampled.begin() + num_to_keep, sampled.end(),
                          [](const WeightedCon &a, const WeightedCon &b) { return a.w > b.w; });
        sampled.resize(num_to_keep);
    }

    batch.clear();
    HighsInt budget_remaining = kBmsBudget;

    for (auto &[ci, w] : sampled) {
        (void)w;
        if (budget_remaining <= 0) {
            break;
        }
        for (HighsInt k = ctx.ARstart[ci]; k < ctx.ARstart[ci + 1] && budget_remaining > 0; ++k) {
            HighsInt j = ctx.ARindex[k];
            --budget_remaining;
            double delta = ctx.compute_tight_delta(ci, j, ctx.ARvalue[k]);
            append_candidate(ctx, batch, j, delta);
        }
    }

    // --- Phase 1b: Breakthrough moves (only post-feasible, Alg 2 line 5-6) ---
    if (best_feasible) {
        for (HighsInt j : costed_vars) {
            double delta = compute_breakthrough_delta(ctx, j, ctx.current_obj, best_objective);
            append_candidate(ctx, batch, j, delta);
        }
    }

    Candidate cand = select_best_from_batch(ctx, batch, step, true, best_objective, best_feasible);

    // If positive candidate found, done (Alg 2 lines 1-6)
    if (cand.var_idx != -1 && cand.score > kViolTol) {
        return cand;
    }

    // --- Phase 2: MTM in satisfied constraints (Alg 2 lines 7-8) ---
    if (!ctx.satisfied.empty()) {
        batch.clear();
        HighsInt num_sat_sample = std::min(kBmsSatCon, ctx.satisfied.size());
        HighsInt sat_budget = kBmsSatBudget;
        for (HighsInt s = 0; s < num_sat_sample && sat_budget > 0; ++s) {
            HighsInt ci = ctx.satisfied[rng() % ctx.satisfied.size()];
            for (HighsInt k = ctx.ARstart[ci]; k < ctx.ARstart[ci + 1] && sat_budget > 0; ++k) {
                HighsInt j = ctx.ARindex[k];
                --sat_budget;
                double delta = ctx.compute_tight_delta(ci, j, ctx.ARvalue[k]);
                append_candidate(ctx, batch, j, delta);
            }
        }
        auto sat_cand =
            select_best_from_batch(ctx, batch, step, false, best_objective, best_feasible);
        if (sat_cand.var_idx != -1 && sat_cand.score > cand.score + kViolTol) {
            cand = sat_cand;
        }
    }

    if (cand.var_idx != -1 && cand.score > kViolTol) {
        return cand;
    }

    // --- Phase 3: Boolean flip (Alg 2 lines 9-11) ---
    if (!binary_vars.empty()) {
        batch.clear();
        HighsInt nbinary = static_cast<HighsInt>(binary_vars.size());
        HighsInt offset = static_cast<HighsInt>(rng() % nbinary);
        for (HighsInt idx = 0; idx < nbinary && idx < kBoolFlipBudget; ++idx) {
            HighsInt j = binary_vars[(offset + idx) % nbinary];
            double new_val = (ctx.solution[j] < 0.5) ? 1.0 : 0.0;
            if (std::abs(new_val - ctx.solution[j]) < kEpsZero) {
                continue;
            }
            batch.push_back({j, new_val});
        }
        if (!batch.empty()) {
            auto flip_cand =
                select_best_from_batch(ctx, batch, step, true, best_objective, best_feasible);
            if (flip_cand.var_idx != -1 && flip_cand.score > cand.score + kViolTol) {
                cand = flip_cand;
            }
        }
    }

    if (cand.var_idx != -1 && cand.score > kViolTol) {
        return cand;
    }

    // --- Phase 4: Weight update + random constraint fallback (Alg 2 lines 12-14) ---
    ctx.update_weights(rng, /*is_feasible=*/false, best_feasible, best_objective);

    if (!ctx.violated.empty()) {
        batch.clear();
        HighsInt ci = ctx.violated[rng() % ctx.violated.size()];
        for (HighsInt k = ctx.ARstart[ci]; k < ctx.ARstart[ci + 1]; ++k) {
            HighsInt j = ctx.ARindex[k];
            double delta = ctx.compute_tight_delta(ci, j, ctx.ARvalue[k]);
            append_candidate(ctx, batch, j, delta);
        }
        // Breakthrough candidates already scored in Phase 1; skip re-scoring.
        auto fallback =
            select_best_from_batch(ctx, batch, step, false, best_objective, best_feasible);
        if (fallback.var_idx != -1 &&
            (cand.var_idx == -1 || fallback.score > cand.score + kViolTol ||
             (fallback.score > cand.score - kViolTol && fallback.bonus > cand.bonus))) {
            cand = fallback;
        }
    }

    if (cand.var_idx != -1) {
        return cand;
    }

    // --- Phase 5: Perturbation (our addition, last resort) ---
    if (!ctx.violated.empty()) {
        HighsInt ci = ctx.violated[rng() % ctx.violated.size()];
        HighsInt row_len = ctx.ARstart[ci + 1] - ctx.ARstart[ci];
        if (row_len > 0) {
            HighsInt k = ctx.ARstart[ci] + static_cast<HighsInt>(rng() % row_len);
            HighsInt j = ctx.ARindex[k];
            double new_val;
            if (ctx.mipdata->domain.isBinary(j)) {
                new_val = (ctx.solution[j] < 0.5) ? 1.0 : 0.0;
            } else if (ctx.is_int(j)) {
                HighsInt dir = (rng() % 2 == 0) ? 1 : -1;
                new_val = ctx.clamp_and_round(j, ctx.solution[j] + dir);
            } else {
                double range = std::min(ctx.col_ub[j], ctx.col_lb[j] + 1e6) - ctx.col_lb[j];
                double perturbation =
                    std::uniform_real_distribution<double>(-0.1 * range, 0.1 * range)(rng);
                new_val = ctx.clamp_and_round(j, ctx.solution[j] + perturbation);
            }
            if (std::abs(new_val - ctx.solution[j]) > kEpsZero) {
                auto [prog, bon] =
                    compute_candidate_scores(ctx, j, new_val, best_feasible, best_objective);
                ctx.viol_cache.reset();
                cand = {j, new_val, prog, bon};
            }
        }
    }

    if (cand.var_idx != -1) {
        return cand;
    }

    // --- Phase 6: Easy moves (our addition) ---
    {
        batch.clear();
        HighsInt num_easy = std::min(kEasyBudget, ctx.ncol);
        for (HighsInt s = 0; s < num_easy; ++s) {
            HighsInt j = static_cast<HighsInt>(rng() % ctx.ncol);
            double target;
            if (ctx.col_lb[j] > 0) {
                target = ctx.col_lb[j];
            } else if (ctx.col_ub[j] < 0) {
                target = ctx.col_ub[j];
            } else {
                target = 0.0;
            }
            append_candidate(ctx, batch, j, target - ctx.solution[j]);
            // Try: toward lower bound
            if (ctx.col_lb[j] > -1e15 && ctx.col_lb[j] < 0) {
                append_candidate(ctx, batch, j, ctx.col_lb[j] - ctx.solution[j]);
            }
            // Try: toward upper bound
            if (ctx.col_ub[j] < 1e15 && ctx.col_ub[j] > 0) {
                append_candidate(ctx, batch, j, ctx.col_ub[j] - ctx.solution[j]);
            }
            // Try: midpoint for continuous
            if (!ctx.is_int(j) && ctx.col_lb[j] > -1e15 && ctx.col_ub[j] < 1e15) {
                append_candidate(ctx, batch, j,
                                 (ctx.col_lb[j] + ctx.col_ub[j]) * 0.5 - ctx.solution[j]);
            }
        }
        auto easy_cand =
            select_best_from_batch(ctx, batch, step, false, best_objective, best_feasible);
        if (easy_cand.var_idx != -1) {
            cand = easy_cand;
        }
    }

    return cand;
}

}  // namespace local_mip_detail
