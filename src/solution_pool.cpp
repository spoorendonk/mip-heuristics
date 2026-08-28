#include "solution_pool.h"

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>
#include <utility>

SolutionPool::SolutionPool(int capacity, bool minimize)
    : capacity_(capacity), minimize_(minimize) {}

void SolutionPool::set_on_accept(std::function<void(const std::vector<double>&, int)> callback) {
    on_accept_ = std::move(callback);
}

void SolutionPool::set_integer_mask(std::vector<bool> mask) {
    std::scoped_lock lock(mtx_);
    integer_mask_ = std::move(mask);
    num_integers_ = 0;
    for (bool b : integer_mask_) {
        if (b) {
            ++num_integers_;
        }
    }
}

int SolutionPool::hamming_distance(const std::vector<double>& a,
                                   const std::vector<double>& b) const {
    int dist = 0;
    int n = static_cast<int>(std::min(a.size(), b.size()));
    if (integer_mask_.empty()) {
        return 0;
    }
    for (int j = 0; j < n; ++j) {
        if (integer_mask_[j] && std::round(a[j]) != std::round(b[j])) {
            ++dist;
        }
    }
    return dist;
}

int SolutionPool::num_integers() const {
    return num_integers_;
}

// Cognitive complexity 28 (threshold 25), marginally over.  Kept whole because capacity,
// dominance and diversity replacement are one decision taken over one hold of the pool
// spin-lock: splitting it would either widen the critical section or thread the lock
// through helpers.  Unlike the algorithm cores this is not a hot path — it runs once per
// accepted solution — so the argument here is lock scope, not throughput.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
SolutionPool::AddResult SolutionPool::try_add(double obj, const std::vector<double>& sol,
                                              int source) {
    AddResult result;
    {
        std::scoped_lock lock(mtx_);

        // The best objective offered so far, decided here rather than by
        // the caller because this is the one point where "the best before
        // this offer" is race-free: two workers offering concurrently would
        // otherwise both read the same "before" outside the lock and both
        // call themselves an improvement (#116).  No watermark yet means
        // the solve has no feasible solution at all — the sink seeds the
        // pool from the incumbent at construction — so any offer improves
        // on it.
        //
        // A *watermark* and not `entries_.front()`, which is not the same
        // thing: the diversity path below replaces the entry most similar
        // to the offer, and that can be the front one, so the pool's best
        // objective goes backwards while the solve's does not (HiGHS keeps
        // whatever `addIncumbent` was given).  Reading the front entry
        // would then call the next offer to clear the degraded value an
        // improvement, handing a patience gate a free reset for a solution
        // the incumbent already dominates — which is the very thing #116
        // exists to stop.  A monotone watermark is also exactly what
        // `improving_offers` in `bench/analyze_presolve_probe.py` tracks,
        // and that is where the shipped patience defaults were measured.
        //
        // The margin mirrors that same function; see `kImprovementObjMargin`.
        if (!has_best_seen_) {
            result.improved_best = true;
        } else {
            const double margin = kImprovementObjMargin * std::max(1.0, std::abs(best_seen_));
            result.improved_best =
                minimize_ ? obj < best_seen_ - margin : obj > best_seen_ + margin;
        }

        // Find insertion point (entries_ kept sorted, best first)
        // Heterogeneous (const Entry&, double) on purpose.  That is fine for
        // std::lower_bound but does not model indirect_strict_weak_order, so
        // std::ranges::lower_bound does not accept it — hence the
        // NOLINTs on the three call sites below.
        auto cmp = [this](const Entry& entry, double val) {
            return minimize_ ? entry.objective < val : entry.objective > val;
        };
        // NOLINTNEXTLINE(modernize-use-ranges)
        auto pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);

        if (std::cmp_greater_equal(entries_.size(), capacity_)) {
            auto& worst = entries_.back();
            bool dominated = minimize_ ? obj >= worst.objective : obj <= worst.objective;

            if (!dominated) {
                // Standard path: improves on worst — replace worst.
                entries_.pop_back();
                // NOLINTNEXTLINE(modernize-use-ranges)
                pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);
                entries_.insert(pos, {obj, sol, source});
                result.accepted = true;
            } else if (!integer_mask_.empty() && num_integers_ > 0 && !entries_.empty()) {
                // Diversity-aware path: pool is full and obj doesn't beat worst.
                // Accept if (a) integer mask is set, (b) obj is within tolerance of
                // best, and (c) solution is sufficiently diverse from all entries.
                double best_obj = entries_.front().objective;
                double gap = std::abs(obj - best_obj);
                // Continuous fallback: fraction of |best_obj|, floored to avoid
                // a discontinuous jump near zero.
                double threshold = std::max(kDiversityObjTolerance * std::abs(best_obj),
                                            kDiversityObjTolerance * 1e-6);

                if (gap <= threshold) {
                    // Two different questions, so two different scans.
                    //
                    // *Diversity* is measured against every entry, the best
                    // included: an offer that duplicates the best solution adds
                    // nothing to a crossover pool however different it is from
                    // the rest.
                    //
                    // *Eviction* skips index 0.  This path only runs on an offer
                    // that does not even beat the worst entry, so letting it
                    // replace the best — which it could, whenever the best was
                    // also its nearest neighbour — discards the pool's most
                    // valuable solution to make room for a dominated one.  HiGHS
                    // keeps its own incumbent, so no solution left the solve, but
                    // `copy_best` and the crossover's better parent then work
                    // from something worse until the pool refills.  A diversity
                    // rule has no reason to discard the best solution it holds.
                    int min_dist = std::numeric_limits<int>::max();
                    int evict_dist = std::numeric_limits<int>::max();
                    int most_similar_idx = -1;
                    for (int idx = 0; std::cmp_less(idx, entries_.size()); ++idx) {
                        int dist = hamming_distance(sol, entries_[idx].solution);
                        min_dist = std::min(min_dist, dist);
                        if (idx > 0 && dist < evict_dist) {
                            evict_dist = dist;
                            most_similar_idx = idx;
                        }
                    }

                    double min_frac =
                        static_cast<double>(min_dist) / static_cast<double>(num_integers_);
                    // `most_similar_idx < 0` means the best entry is the only
                    // one there is, so nothing may be evicted.
                    if (min_frac >= kDiversityMinHammingFrac && most_similar_idx > 0) {
                        // Replace the most similar entry other than the best.
                        entries_.erase(entries_.begin() + most_similar_idx);
                        // NOLINTNEXTLINE(modernize-use-ranges)
                        pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);
                        entries_.insert(pos, {obj, sol, source});
                        result.accepted = true;
                    }
                }
            }
        } else {
            entries_.insert(pos, {obj, sol, source});
            result.accepted = true;
        }

        // Advance the watermark only on an offer the pool kept, so a
        // refused solution cannot raise the bar the next one is judged
        // against.  It never moves backwards, which is what makes it
        // track the solve's incumbent rather than the pool's contents.
        if (result.accepted && result.improved_best) {
            best_seen_ = obj;
            has_best_seen_ = true;
        }
    }
    // Invoke callback outside the pool lock to avoid lock inversion: the
    // callback holds its own mutex to serialize concurrent trySolution calls.
    // `improved_best` implies `accepted` in every branch above — an offer
    // that beats the best beats the worst too, so it takes the standard
    // replacement path — but tie them together rather than leaving that to
    // a reader of the admission policy: a staleness gate must never reset
    // on a solution the pool did not keep.
    result.improved_best = result.improved_best && result.accepted;
    if (result.accepted && on_accept_) {
        on_accept_(sol, source);
    }
    return result;
}

SolutionPool::Snapshot SolutionPool::snapshot() {
    std::scoped_lock lock(mtx_);
    if (entries_.empty()) {
        return {false, minimize_ ? std::numeric_limits<double>::infinity()
                                 : -std::numeric_limits<double>::infinity()};
    }
    return {true, entries_[0].objective};
}

bool SolutionPool::is_integer_col(int j) const {
    return !integer_mask_.empty() && std::cmp_less(j, integer_mask_.size()) && integer_mask_[j];
}

// Guided crossover: keep integer values the parents agree on, coin-flip
// everything else.  Caller holds mtx_.
void SolutionPool::guided_crossover(const std::vector<double>& sol_a,
                                    const std::vector<double>& sol_b, Rng& rng,
                                    std::vector<double>& out) const {
    int ncol = static_cast<int>(sol_a.size());
    out.resize(ncol);
    for (int j = 0; j < ncol; ++j) {
        if (is_integer_col(j) && std::round(sol_a[j]) == std::round(sol_b[j])) {
            out[j] = sol_a[j];
        } else {
            // Disagree or continuous — coin flip.
            out[j] = std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? sol_a[j] : sol_b[j];
        }
    }
}

// Neighborhood crossover: the better parent provides the base and only
// disagreeing integer variables are coin-flipped.  Caller holds mtx_.
void SolutionPool::neighborhood_crossover(const std::vector<double>& sol_better,
                                          const std::vector<double>& sol_other, Rng& rng,
                                          std::vector<double>& out) const {
    int ncol = static_cast<int>(sol_better.size());
    out.resize(ncol);
    for (int j = 0; j < ncol; ++j) {
        if (is_integer_col(j) && std::round(sol_better[j]) != std::round(sol_other[j])) {
            out[j] =
                std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? sol_better[j] : sol_other[j];
        } else {
            // Agree or continuous — keep the better parent's value.
            out[j] = sol_better[j];
        }
    }
}

// Biased copy: half the time draw from the better half of the pool, half the
// time from anywhere.  Caller holds mtx_.
void SolutionPool::biased_copy(Rng& rng, std::vector<double>& out) const {
    int pool_size = static_cast<int>(entries_.size());
    int idx = 0;
    if (pool_size > 1 && std::uniform_int_distribution<int>(0, 1)(rng) == 0) {
        idx = std::uniform_int_distribution<int>(0, ((pool_size + 1) / 2) - 1)(rng);
    } else {
        idx = std::uniform_int_distribution<int>(0, pool_size - 1)(rng);
    }
    out = entries_[idx].solution;
}

bool SolutionPool::get_restart(Rng& rng, std::vector<double>& out) {
    std::scoped_lock lock(mtx_);
    if (entries_.empty()) {
        return false;
    }

    int pool_size = static_cast<int>(entries_.size());
    double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

    // Two distinct parents, drawn uniformly.
    auto pick_two_parents = [&](int& a, int& b) {
        a = std::uniform_int_distribution<int>(0, pool_size - 1)(rng);
        b = std::uniform_int_distribution<int>(0, pool_size - 2)(rng);
        if (b >= a) {
            ++b;
        }
    };

    if (roll < 0.4 && pool_size >= 2) {
        int a = 0;
        int b = 0;
        pick_two_parents(a, b);
        guided_crossover(entries_[a].solution, entries_[b].solution, rng, out);
    } else if (roll < 0.7 && pool_size >= 2) {
        int a = 0;
        int b = 0;
        pick_two_parents(a, b);
        // Better parent = lower index (entries_ sorted best-first).
        neighborhood_crossover(entries_[std::min(a, b)].solution, entries_[std::max(a, b)].solution,
                               rng, out);
    } else {
        biased_copy(rng, out);
    }
    return true;
}

std::vector<SolutionPool::Entry> SolutionPool::sorted_entries() {
    std::scoped_lock lock(mtx_);
    return entries_;  // already kept sorted
}

bool SolutionPool::copy_best(std::vector<double>& out) {
    std::scoped_lock lock(mtx_);
    if (entries_.empty()) {
        return false;
    }
    out = entries_[0].solution;  // best — entries_ kept sorted by try_add.
    return true;
}

int SolutionPool::size() {
    std::scoped_lock lock(mtx_);
    return static_cast<int>(entries_.size());
}

// Reads the live `mipdata->incumbent`, which is legal only because every
// caller constructs its `IncumbentSink` on the dispatching thread before
// any worker starts — nothing can be submitting concurrently.  Workers read
// the dispatch snapshot instead (`ProblemView::incumbent`, issue #98).
void seed_pool(SolutionPool& pool, const HighsMipSolver& mipsolver) {
    const auto* model = mipsolver.model_;
    auto* mipdata = mipsolver.mipdata_.get();

    // Build integer mask from model integrality and set on pool.
    const HighsInt ncol = model->num_col_;
    std::vector<bool> int_mask(ncol);
    for (HighsInt j = 0; j < ncol; ++j) {
        int_mask[j] = (model->integrality_[j] != HighsVarType::kContinuous);
    }
    pool.set_integer_mask(std::move(int_mask));

    if (mipdata->incumbent.empty()) {
        return;
    }
    double obj = model->offset_;
    for (HighsInt j = 0; j < ncol; ++j) {
        obj += model->col_cost_[j] * mipdata->incumbent[j];
    }
    // The incumbent came from HiGHS itself (or a prior heuristic that has
    // already been attributed), so on flush HiGHS will recognize it as a
    // duplicate and drop it before logging.  Tag it with the generic
    // kSolutionSourceHeuristic so nothing downstream misattributes it.
    // Discarded on purpose: this is the pool's initial state, not an
    // offer from a heuristic.  There is no dispatch yet and so no
    // staleness counter for the verdict to feed, and the seeded solution
    // must not be counted as production by anything.
    static_cast<void>(pool.try_add(obj, mipdata->incumbent, kSolutionSourceHeuristic));
}
