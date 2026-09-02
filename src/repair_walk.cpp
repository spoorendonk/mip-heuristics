#include "repair_walk.h"

#include "lp_data/HConst.h"
#include "prop_engine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace {

// Paper Sect. 5: "we maintain a short tabu list of the last 3 shifts in
// order to avoid short cycles (whose probability is quite large when there
// are very few violated constraints)".
constexpr HighsInt kTabuLength = 3;

// Paper Sect. 5: "we soft restart from the best state every 10 shifts
// (this is to avoid the repair process to diverge to states from which it
// is very time consuming to recover)".
constexpr HighsInt kSoftRestartPeriod = 10;

// A row's violation on a *partial* assignment (paper Sect. 5): the
// distance between the activity range `[m, mx]` the current domain admits
// and the row's bounds.  Zero unless the row is unsatisfiable by *every*
// completion of the domain, which is what makes it the partial-assignment
// generalization of `row_violation` on a complete assignment (there
// `m == mx == lhs`).
//
// Always finite despite the infinities in play: `mx` can only be
// `+kHighsInf` and `m` only `-kHighsInf` (a `-inf` max activity would need
// a column with `ub == -inf`, a `+inf` min activity one with `lb == +inf`),
// and both of those fail the comparison that would have used them.
double range_violation(double min_act, double max_act, double lo, double hi) {
    if (max_act < lo) {
        return lo - max_act;
    }
    if (min_act > hi) {
        return min_act - hi;
    }
    return 0.0;
}

bool range_violated(double min_act, double max_act, double lo, double hi, double feastol) {
    return max_act < lo - feastol || min_act > hi + feastol;
}

}  // namespace

// Cognitive complexity 63 (threshold 25).  Kept whole: RepairWalk from Sect. 5 in full — the
// activity-range violation bookkeeping, the shift/clip/damage candidate rule and the WalkSAT
// selection are one algorithm and share the per-step state. Decomposing it would move work across a
// worker's inner loop (this runs at every infeasible DFS node), and the closeout takes no
// unmeasured performance risk; the standards also rank fidelity to the reference algorithm above
// mechanical extraction. NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool repair_walk(PropEngine& E, HighsInt max_steps, double noise, size_t max_effort, Rng& rng,
                 size_t& effort_out, RepairWalkScratch& scratch, const Deadline& deadline) {
    // Sect. 5 leans on this explicitly -- "the very same quantities are
    // needed for constraint propagation as well, a fact that is exploited
    // by our implementation" -- so an engine without activities cannot
    // measure a partial assignment's violation at all.  `fpr_attempt_begin`
    // arms every repairing mode; a debug assert rather than a silent
    // degradation, matching `binary_mask`'s treatment next door.
    assert(E.activities_initialized() &&
           "repair_walk requires PropEngine::init_activities() (fpr_attempt_begin arms it)");
    if (!E.activities_initialized()) {
        // Release-build backstop for the assert above: without activities
        // there is nothing to read violations from, and the alternative to
        // returning here is indexing an empty vector.  "Still infeasible"
        // is the honest answer, and it is what the caller does with a
        // repair that changed nothing.
        effort_out = 0;
        return false;
    }

    const HighsInt nrow = E.nrow();
    const double feastol = E.feastol();
    const double* row_lo = E.row_lo();
    const double* row_hi = E.row_hi();
    const HighsInt* ar_start = E.ar_start();
    const HighsInt* ar_index = E.ar_index();
    const double* ar_value = E.ar_value();
    const HighsInt* csc_start = E.csc_start();
    const HighsInt* csc_row = E.csc_row();
    const double* csc_val = E.csc_val();
    const double* col_lb = E.col_lb();
    const double* col_ub = E.col_ub();
    const double* min_act = E.min_activity_data();
    const double* max_act = E.max_activity_data();

    size_t effort = 0;

    auto& violated = scratch.violated;
    auto& violated_pos = scratch.violated_pos;
    if (std::cmp_less(violated_pos.size(), nrow)) {
        violated_pos.assign(nrow, -1);
    } else {
        std::fill(violated_pos.begin(), violated_pos.begin() + nrow, -1);
    }
    violated.clear();
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
        const HighsInt pos = violated_pos[i];
        if (pos == -1) {
            return;
        }
        const HighsInt last = violated.back();
        violated[pos] = last;
        violated_pos[last] = pos;
        violated.pop_back();
        violated_pos[i] = -1;
    };

    double total_viol = 0.0;
    auto rescan = [&]() {
        for (auto vi : violated) {
            violated_pos[vi] = -1;
        }
        violated.clear();
        total_viol = 0.0;
        effort += static_cast<size_t>(nrow);
        for (HighsInt i = 0; i < nrow; ++i) {
            total_viol += range_violation(min_act[i], max_act[i], row_lo[i], row_hi[i]);
            if (range_violated(min_act[i], max_act[i], row_lo[i], row_hi[i], feastol)) {
                add_violated(i);
            }
        }
    };
    rescan();
    if (violated.empty()) {
        effort_out = effort;
        return true;
    }

    // Every repair move goes through `E`, so the engine's own undo stacks
    // are the best-state snapshot: a mark is four integers, and restoring
    // it is the same `backtrack_to` the DFS uses.  No parallel copy of the
    // domain has to exist, and a caller that backtracks past this node
    // unwinds the repair with it.
    struct Mark {
        HighsInt vs;
        HighsInt sol;
        HighsInt act;
        HighsInt pq;
    };
    auto mark_now = [&]() -> Mark {
        return Mark{E.vs_mark(), E.sol_mark(), E.act_mark(),
                    E.pq_initialized() ? E.pq_mark() : HighsInt{-1}};
    };
    auto restore = [&](const Mark& m) { E.backtrack_to(m.vs, m.sol, m.act, m.pq); };

    Mark best_mark = mark_now();
    double best_viol = total_viol;

    auto& cand = scratch.cand;
    auto& best_indices = scratch.best_indices;
    auto& tabu = scratch.tabu;
    tabu.clear();

    HighsInt since_restart = 0;
    for (HighsInt step = 0; step < max_steps && !violated.empty(); ++step) {
        if (effort >= max_effort) {
            break;
        }
        // Polled per step, not on a cadence: a step is O(row degree x
        // column degree) and this runs inside the DFS's own node, so the
        // residual after an expiry is one step rather than the rest of the
        // walk.  The expiry stops the walk and nothing more -- see the
        // header on why it is not signalled outward.
        if (deadline.expired()) {
            break;
        }

        // Pick a violated row uniformly at random (paper Sect. 5).
        const HighsInt row = violated[std::uniform_int_distribution<HighsInt>(
            0, static_cast<HighsInt>(violated.size()) - 1)(rng)];

        const double lo_r = row_lo[row];
        const double hi_r = row_hi[row];
        const double m_r = min_act[row];
        const double mx_r = max_act[row];
        const double cur_viol = range_violation(m_r, mx_r, lo_r, hi_r);
        // Which side the activity range sits on decides both the target
        // bound and which endpoint of the range has to reach it.
        const bool overshoot_above = m_r > hi_r + feastol;
        const double target = overshoot_above ? hi_r : lo_r;
        const double anchor = overshoot_above ? m_r : mx_r;

        cand.clear();
        double best_damage = std::numeric_limits<double>::infinity();
        const HighsInt kbeg = ar_start[row];
        const HighsInt kend = ar_start[row + 1];
        effort += static_cast<size_t>(kend - kbeg);
        for (HighsInt k = kbeg; k < kend; ++k) {
            const HighsInt j = ar_index[k];
            const double a = ar_value[k];
            if (std::abs(a) < 1e-15) {
                continue;
            }
            if (std::ranges::find(tabu, j) != tabu.end()) {
                continue;
            }

            const VarState& vj = E.var(j);
            const double cur_lb = vj.fixed ? vj.val : vj.lb;
            const double cur_ub = vj.fixed ? vj.val : vj.ub;
            // An unbounded interval cannot be translated, and its activity
            // contribution would not move if it were.
            if (cur_lb <= -kHighsInf || cur_ub >= kHighsInf) {
                continue;
            }

            // Shifting the interval by `s` translates *both* activity
            // endpoints of every row containing `j` by `a * s`, whichever
            // sign `a` has -- that identity is what lets a shift be scored
            // without touching the engine.  The minimal shift that would
            // satisfy this row, rounded (paper Sect. 5: "compute the
            // minimal shift that would make the constraint satisfied,
            // round it if the variable is general integer, and then clip
            // it").
            double s = (target - anchor) / a;
            const bool integer = E.is_int(j);
            if (integer) {
                const bool floor_it = overshoot_above == (a > 0);
                s = floor_it ? std::floor(s + feastol) : std::ceil(s - feastol);
            }
            // Clip so the *translated interval* still lies inside the
            // column's structural bounds.  This is where "no domain
            // enlargement" is enforced: the interval keeps its width, so
            // the only freedom is how far it may slide, and both ends of
            // the clip window bracket zero.
            s = std::max(s, col_lb[j] - cur_lb);
            s = std::min(s, col_ub[j] - cur_ub);
            if (integer) {
                // A fractional structural bound could have made the clip
                // fractional; round back toward zero, which can only
                // shrink the shift and so cannot re-break the clip.
                s = (s > 0.0) ? std::floor(s) : std::ceil(s);
            }
            if (std::abs(s) < 1e-15) {
                continue;
            }

            // Skip the variables that do not reduce this row's violation
            // (paper Sect. 5).
            const double shifted = a * s;
            const double new_viol = range_violation(m_r + shifted, mx_r + shifted, lo_r, hi_r);
            if (new_viol >= cur_viol - feastol) {
                continue;
            }

            // Damage: the *increases* in violation over the other rows `j`
            // appears in.  As in the original WalkSAT, improvements are
            // ignored.
            const HighsInt cbeg = csc_start[j];
            const HighsInt cend = csc_start[j + 1];
            effort += static_cast<size_t>(cend - cbeg);
            double damage = 0.0;
            for (HighsInt p = cbeg; p < cend; ++p) {
                const HighsInt i2 = csc_row[p];
                if (i2 == row) {
                    continue;
                }
                const double d = csc_val[p] * s;
                const double dv =
                    range_violation(min_act[i2] + d, max_act[i2] + d, row_lo[i2], row_hi[i2]) -
                    range_violation(min_act[i2], max_act[i2], row_lo[i2], row_hi[i2]);
                if (dv > 0.0) {
                    damage += dv;
                }
            }
            best_damage = std::min(damage, best_damage);
            cand.push_back({j, s, damage});
        }

        if (cand.empty()) {
            continue;
        }

        // WalkSAT selection (paper Sect. 5): a zero-damage shift is taken
        // greedily; otherwise a random candidate with probability `noise`
        // and a minimum-damage one with probability 1 - `noise`.
        HighsInt pick = 0;
        if (best_damage > feastol &&
            std::uniform_real_distribution<double>(0.0, 1.0)(rng) < noise) {
            pick = std::uniform_int_distribution<HighsInt>(
                0, static_cast<HighsInt>(cand.size()) - 1)(rng);
        } else {
            best_indices.clear();
            const double threshold = best_damage + feastol;
            for (size_t ci = 0; ci < cand.size(); ++ci) {
                if (cand[ci].damage <= threshold) {
                    best_indices.push_back(static_cast<HighsInt>(ci));
                }
            }
            pick = best_indices[std::uniform_int_distribution<HighsInt>(
                0, static_cast<HighsInt>(best_indices.size()) - 1)(rng)];
        }
        const HighsInt var = cand[pick].var;
        const double shift = cand[pick].shift;

        // Apply, then re-read the engine's activities rather than trusting
        // the predicted translation: `shift_domain` may clamp, and a
        // fixed-but-wide column collapses to a singleton, so the applied
        // move is not always the predicted one.
        const HighsInt cbeg = csc_start[var];
        const HighsInt cend = csc_start[var + 1];
        effort += 2 * static_cast<size_t>(cend - cbeg);
        double removed = 0.0;
        for (HighsInt p = cbeg; p < cend; ++p) {
            const HighsInt i2 = csc_row[p];
            removed += range_violation(min_act[i2], max_act[i2], row_lo[i2], row_hi[i2]);
        }
        if (!E.shift_domain(var, shift)) {
            // Unreachable given the clip above; a guard, not a path.
            continue;
        }
        double added = 0.0;
        for (HighsInt p = cbeg; p < cend; ++p) {
            const HighsInt i2 = csc_row[p];
            added += range_violation(min_act[i2], max_act[i2], row_lo[i2], row_hi[i2]);
            const bool was = violated_pos[i2] != -1;
            const bool now =
                range_violated(min_act[i2], max_act[i2], row_lo[i2], row_hi[i2], feastol);
            if (was && !now) {
                remove_violated(i2);
            } else if (!was && now) {
                add_violated(i2);
            }
        }
        total_viol += added - removed;

        tabu.push_back(var);
        if (std::cmp_greater(tabu.size(), kTabuLength)) {
            tabu.erase(tabu.begin());
        }

        if (total_viol < best_viol - feastol) {
            best_viol = total_viol;
            best_mark = mark_now();
        }

        if (++since_restart >= kSoftRestartPeriod) {
            since_restart = 0;
            if (total_viol > best_viol + feastol) {
                restore(best_mark);
                rescan();
            }
        }
        // The tabu list deliberately survives a soft restart.  The paper
        // introduces the two tricks side by side and says nothing about
        // their interaction, so this is our reading, not its text: the
        // list is a rolling window over the last three shifts, full stop.
    }

    // Paper Sect. 5's end-of-walk restore, and the reason a failed repair
    // is not simply a wasted node: the node is left in the least-violated
    // state the walk saw, which is the state the DFS below carries on from
    // in a non-backtracking mode.
    if (!violated.empty() && total_viol > best_viol + feastol) {
        restore(best_mark);
        rescan();
    }

    effort_out = effort;
    return violated.empty();
}
