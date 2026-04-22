#include "walksat.h"

#include "heuristic_common.h"
#include "prop_engine.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

WalkSatMove walksat_select_move(HighsInt row, const double* solution, const double* lhs_cache,
                                const double* col_lb, const double* col_ub, const PropEngine& data,
                                double noise, Rng& rng, size_t& effort, WalkSatScratch& scratch) {
    // Pull PropEngine pointer accessors into local __restrict copies. The
    // `__restrict` assertions let the compiler keep these in registers and
    // skip aliasing reloads across the hot inner loops below — writes via
    // `scratch.cand` / `scratch.best_indices` would otherwise conservatively
    // force reloads of these const pointers through the accessor.
    const HighsInt* __restrict ar_start = data.ar_start();
    const HighsInt* __restrict ar_index = data.ar_index();
    const double* __restrict ar_value = data.ar_value();
    const HighsInt* __restrict csc_start = data.csc_start();
    const HighsInt* __restrict csc_row = data.csc_row();
    const double* __restrict csc_val = data.csc_val();
    const double* __restrict row_lo = data.row_lo();
    const double* __restrict row_hi = data.row_hi();
    const HighsVarType* __restrict integrality = data.integrality();
    const double feastol = data.feastol();

    const HighsInt row_begin = ar_start[row];
    const HighsInt row_end = ar_start[row + 1];
    const HighsInt row_len = row_end - row_begin;
    if (row_len == 0) [[unlikely]] {
        return {};
    }

    // Loop-invariant row metadata.
    const double row_lo_r = row_lo[row];
    const double row_hi_r = row_hi[row];
    const double ci_lhs = lhs_cache[row];
    const double ci_viol = row_violation(ci_lhs, row_lo_r, row_hi_r);
    const bool overshoot_above = ci_lhs > row_hi_r + feastol;
    const double target_rhs = overshoot_above ? row_hi_r : row_lo_r;
    const double ci_viol_threshold = ci_viol - feastol;  // early-reject bound

    auto& cand = scratch.cand;
    cand.clear();
    cand.reserve(static_cast<size_t>(row_len));
    double best_damage = std::numeric_limits<double>::infinity();

    effort += static_cast<size_t>(row_len);
    for (HighsInt k = row_begin; k < row_end; ++k) {
        const HighsInt j = ar_index[k];
        const double a = ar_value[k];
        if (std::abs(a) < 1e-15) [[unlikely]] {
            continue;
        }

        const double old_val = solution[j];
        double new_val = old_val + (target_rhs - ci_lhs) / a;

        if (integrality[j] != HighsVarType::kContinuous) {
            // Rounding direction depends on (overshoot_above) XOR (a < 0):
            //   overshoot_above && a > 0: floor to decrease lhs
            //   overshoot_above && a < 0: ceil  to decrease lhs
            //   undershoot    && a > 0: ceil  to increase lhs
            //   undershoot    && a < 0: floor to increase lhs
            const bool floor_it = overshoot_above == (a > 0);
            new_val = floor_it ? std::floor(new_val + feastol) : std::ceil(new_val - feastol);
        }
        const double lb = col_lb[j];
        const double ub = col_ub[j];
        if (new_val < lb) {
            new_val = lb;
        } else if (new_val > ub) {
            new_val = ub;
        }

        const double delta_change = new_val - old_val;
        if (std::abs(delta_change) < 1e-15) {
            continue;
        }

        const double new_ci_lhs = ci_lhs + a * delta_change;
        const double new_ci_viol = row_violation(new_ci_lhs, row_lo_r, row_hi_r);
        if (new_ci_viol >= ci_viol_threshold) {
            continue;
        }

        const HighsInt col_begin = csc_start[j];
        const HighsInt col_end = csc_start[j + 1];
        effort += static_cast<size_t>(col_end - col_begin);

        double damage = 0.0;
        const HighsInt* __restrict csc_row_p = csc_row + col_begin;
        const double* __restrict csc_val_p = csc_val + col_begin;
        const HighsInt col_deg = col_end - col_begin;
        for (HighsInt p = 0; p < col_deg; ++p) {
            const HighsInt i2 = csc_row_p[p];
            if (i2 == row) [[unlikely]] {
                continue;
            }
            const double coeff = csc_val_p[p];
            const double old_lhs = lhs_cache[i2];
            const double new_lhs = old_lhs + coeff * delta_change;
            const double lo_i = row_lo[i2];
            const double hi_i = row_hi[i2];
            const double dv =
                row_violation(new_lhs, lo_i, hi_i) - row_violation(old_lhs, lo_i, hi_i);
            if (dv > 0.0) {
                damage += dv;
            }
        }

        if (damage < best_damage) {
            best_damage = damage;
        }
        cand.push_back({j, new_val, damage});
    }

    const size_t ncand = cand.size();
    if (ncand == 0) {
        return {};
    }

    const WalkSatScratch::Candidate* __restrict cand_p = cand.data();

    // WalkSAT selection (paper Fig. 4, lines 17-21)
    if (best_damage > feastol && std::uniform_real_distribution<double>(0.0, 1.0)(rng) < noise) {
        const HighsInt idx =
            std::uniform_int_distribution<HighsInt>(0, static_cast<HighsInt>(ncand) - 1)(rng);
        return {cand_p[idx].var, cand_p[idx].new_val};
    }
    auto& best_indices = scratch.best_indices;
    best_indices.clear();
    best_indices.reserve(ncand);
    const double damage_threshold = best_damage + feastol;
    for (size_t ci = 0; ci < ncand; ++ci) {
        if (cand_p[ci].damage <= damage_threshold) {
            best_indices.push_back(static_cast<HighsInt>(ci));
        }
    }
    const HighsInt idx = best_indices[std::uniform_int_distribution<HighsInt>(
        0, static_cast<HighsInt>(best_indices.size()) - 1)(rng)];
    return {cand_p[idx].var, cand_p[idx].new_val};
}

// ---------------------------------------------------------------------------
// WalkSAT repair walk (paper Fig. 4)
// ---------------------------------------------------------------------------

bool walksat_repair(const PropEngine& data, std::vector<double>& solution,
                    std::vector<double>& lhs_cache, const double* col_lb, const double* col_ub,
                    HighsInt max_iterations, double noise, bool track_best, size_t max_effort,
                    Rng& rng, size_t& effort, WalkSatScratch& scratch) {
    const HighsInt nrow = data.nrow();
    const double feastol = data.feastol();
    const double* row_lo = data.row_lo();
    const double* row_hi = data.row_hi();
    const HighsInt* csc_start = data.csc_start();
    const HighsInt* csc_row = data.csc_row();
    const double* csc_val = data.csc_val();

    // Reuse scratch's violated / violated_pos / sol_undo / lhs_undo to avoid
    // per-call heap allocations.  clear() retains capacity; violated_pos must
    // be sized to nrow and reset to -1 before use.
    auto& violated = scratch.violated;
    auto& violated_pos = scratch.violated_pos;
    auto& sol_undo = scratch.sol_undo;
    auto& lhs_undo = scratch.lhs_undo;
    violated.clear();
    sol_undo.clear();
    lhs_undo.clear();
    if (static_cast<HighsInt>(violated_pos.size()) < nrow) {
        violated_pos.assign(nrow, -1);
    } else {
        std::fill(violated_pos.begin(), violated_pos.begin() + nrow, -1);
    }
    if (violated.capacity() < static_cast<size_t>(nrow)) {
        violated.reserve(nrow);
    }

    auto add_violated = [&](HighsInt i) {
        if (violated_pos[i] != -1) {
            return;
        }
        violated_pos[i] = static_cast<HighsInt>(violated.size());
        violated.push_back(i);
    };
    auto remove_violated = [&](HighsInt i) {
        HighsInt pos = violated_pos[i];
        if (pos == -1) {
            return;
        }
        HighsInt last = violated.back();
        violated[pos] = last;
        violated_pos[last] = pos;
        violated.pop_back();
        violated_pos[i] = -1;
    };

    // Initialize violated set and total violation
    double total_viol = 0.0;
    for (HighsInt i = 0; i < nrow; ++i) {
        double v = row_violation(lhs_cache[i], row_lo[i], row_hi[i]);
        total_viol += v;
        if (is_row_violated(lhs_cache[i], row_lo[i], row_hi[i], feastol)) {
            add_violated(i);
        }
    }

    if (violated.empty()) {
        return true;
    }

    // Delta-based best-state tracking: instead of copying full solution/lhs_cache
    // vectors on each improvement (O(ncol + nrow)), we keep an undo log of
    // changed entries and record a mark at each improvement. Restore replays
    // only the entries after the best mark — O(changes_since_best).
    double best_viol = total_viol;
    HighsInt best_sol_mark = 0;
    HighsInt best_lhs_mark = 0;

    for (HighsInt step = 0; step < max_iterations && !violated.empty(); ++step) {
        if (effort >= max_effort) {
            break;
        }

        HighsInt pick = std::uniform_int_distribution<HighsInt>(
            0, static_cast<HighsInt>(violated.size()) - 1)(rng);
        HighsInt row = violated[pick];

        auto move = walksat_select_move(row, solution.data(), lhs_cache.data(), col_lb, col_ub,
                                        data, noise, rng, effort, scratch);
        if (move.var < 0) {
            continue;
        }

        double old_val = solution[move.var];
        double delta = move.val - old_val;
        if (track_best) {
            sol_undo.push_back({move.var, old_val});
        }
        solution[move.var] = move.val;

        // Apply move: update lhs_cache, violated set, and total_viol incrementally
        effort += csc_start[move.var + 1] - csc_start[move.var];
        for (HighsInt p = csc_start[move.var]; p < csc_start[move.var + 1]; ++p) {
            HighsInt i2 = csc_row[p];
            double old_v = row_violation(lhs_cache[i2], row_lo[i2], row_hi[i2]);
            if (track_best) {
                lhs_undo.push_back({i2, lhs_cache[i2]});
            }
            lhs_cache[i2] += csc_val[p] * delta;
            double new_v = row_violation(lhs_cache[i2], row_lo[i2], row_hi[i2]);
            total_viol += new_v - old_v;

            bool was = violated_pos[i2] != -1;
            bool now = is_row_violated(lhs_cache[i2], row_lo[i2], row_hi[i2], feastol);
            if (was && !now) {
                remove_violated(i2);
            } else if (!was && now) {
                add_violated(i2);
            }
        }

        // Update best-state tracking (paper Fig. 4, lines 23-26)
        if (track_best && total_viol < best_viol) {
            best_viol = total_viol;
            best_sol_mark = static_cast<HighsInt>(sol_undo.size());
            best_lhs_mark = static_cast<HighsInt>(lhs_undo.size());
        }
    }

    // Restore best state if tracking enabled (paper Fig. 4, line 27)
    if (track_best && !violated.empty()) {
        // Undo changes back to the best mark
        for (HighsInt u = static_cast<HighsInt>(sol_undo.size()) - 1; u >= best_sol_mark; --u) {
            solution[sol_undo[u].idx] = sol_undo[u].old_val;
        }
        for (HighsInt u = static_cast<HighsInt>(lhs_undo.size()) - 1; u >= best_lhs_mark; --u) {
            lhs_cache[lhs_undo[u].idx] = lhs_undo[u].old_val;
        }
        // Rebuild violated set from restored state
        for (auto vi : violated) {
            violated_pos[vi] = -1;
        }
        violated.clear();
        for (HighsInt i = 0; i < nrow; ++i) {
            if (is_row_violated(lhs_cache[i], row_lo[i], row_hi[i], feastol)) {
                add_violated(i);
            }
        }
    }

    return violated.empty();
}

// ---------------------------------------------------------------------------
// Greedy 1-opt improvement (paper Section 6)
// ---------------------------------------------------------------------------

void greedy_1opt(const PropEngine& data, std::vector<double>& solution,
                 std::vector<double>& lhs_cache, const double* col_cost, bool minimize,
                 size_t& effort) {
    const HighsInt ncol = data.ncol();
    const HighsInt* __restrict csc_start = data.csc_start();
    const HighsInt* __restrict csc_row = data.csc_row();
    const double* __restrict csc_val = data.csc_val();
    const double* __restrict col_lb = data.col_lb();
    const double* __restrict col_ub = data.col_ub();
    const double* __restrict row_lo = data.row_lo();
    const double* __restrict row_hi = data.row_hi();
    const HighsVarType* __restrict integrality = data.integrality();
    const double feastol = data.feastol();

    double* __restrict sol_p = solution.data();
    double* __restrict lhs_p = lhs_cache.data();

    for (HighsInt j = 0; j < ncol; ++j) {
        if (integrality[j] == HighsVarType::kContinuous) {
            continue;
        }
        const double cj = col_cost[j];
        if (std::abs(cj) < 1e-15) {
            continue;
        }

        // direction = -sign(cj) when minimizing, +sign(cj) when maximizing.
        const double direction = (cj > 0) == minimize ? -1.0 : 1.0;
        const double old_val = sol_p[j];
        double new_val = old_val + direction;
        const double lb = col_lb[j];
        const double ub = col_ub[j];
        if (new_val < lb) {
            new_val = lb;
        } else if (new_val > ub) {
            new_val = ub;
        }
        const double delta = new_val - old_val;
        if (std::abs(delta) < 1e-15) {
            continue;
        }

        const HighsInt col_begin = csc_start[j];
        const HighsInt col_end = csc_start[j + 1];
        const HighsInt col_deg = col_end - col_begin;
        effort += static_cast<size_t>(col_deg);

        const HighsInt* __restrict csc_row_p = csc_row + col_begin;
        const double* __restrict csc_val_p = csc_val + col_begin;

        bool shift_feasible = true;
        for (HighsInt p = 0; p < col_deg; ++p) {
            const HighsInt row = csc_row_p[p];
            const double new_lhs = lhs_p[row] + csc_val_p[p] * delta;
            if (is_row_violated(new_lhs, row_lo[row], row_hi[row], feastol)) {
                shift_feasible = false;
                break;
            }
        }

        if (shift_feasible) {
            for (HighsInt p = 0; p < col_deg; ++p) {
                lhs_p[csc_row_p[p]] += csc_val_p[p] * delta;
            }
            sol_p[j] = new_val;
        }
    }
}
