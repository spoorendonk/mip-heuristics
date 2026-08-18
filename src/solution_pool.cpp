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
bool SolutionPool::try_add(double obj, const std::vector<double>& sol, int source) {
    bool accepted = false;
    {
        std::scoped_lock lock(mtx_);

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
                accepted = true;
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
                    // Compute minimum Hamming distance to any pool entry and track
                    // the index of the most similar entry.
                    int min_dist = std::numeric_limits<int>::max();
                    int most_similar_idx = -1;
                    for (int idx = 0; std::cmp_less(idx, entries_.size()); ++idx) {
                        int dist = hamming_distance(sol, entries_[idx].solution);
                        if (dist < min_dist) {
                            min_dist = dist;
                            most_similar_idx = idx;
                        }
                    }

                    double min_frac =
                        static_cast<double>(min_dist) / static_cast<double>(num_integers_);
                    if (min_frac >= kDiversityMinHammingFrac) {
                        // Replace the most similar entry.
                        entries_.erase(entries_.begin() + most_similar_idx);
                        // NOLINTNEXTLINE(modernize-use-ranges)
                        pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);
                        entries_.insert(pos, {obj, sol, source});
                        accepted = true;
                    }
                }
            }
        } else {
            entries_.insert(pos, {obj, sol, source});
            accepted = true;
        }
    }
    // Invoke callback outside the pool lock to avoid lock inversion: the
    // callback holds its own mutex to serialize concurrent trySolution calls.
    if (accepted && on_accept_) {
        on_accept_(sol, source);
    }
    return accepted;
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
    pool.try_add(obj, mipdata->incumbent, kSolutionSourceHeuristic);
}
