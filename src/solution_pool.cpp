#include "solution_pool.h"

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

SolutionPool::SolutionPool(int capacity, bool minimize)
    : capacity_(capacity), minimize_(minimize) {}

void SolutionPool::set_integer_mask(std::vector<bool> mask) {
    std::lock_guard<HighsSpinMutex> lock(mtx_);
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

bool SolutionPool::try_add(double obj, const std::vector<double>& sol, int source) {
    std::lock_guard<HighsSpinMutex> lock(mtx_);

    // Find insertion point (entries_ kept sorted, best first)
    auto cmp = [this](const Entry& e, double val) {
        return minimize_ ? e.objective < val : e.objective > val;
    };
    auto pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);

    if (static_cast<int>(entries_.size()) >= capacity_) {
        auto& worst = entries_.back();
        bool dominated = minimize_ ? obj >= worst.objective : obj <= worst.objective;

        if (!dominated) {
            // Standard path: improves on worst — replace worst.
            entries_.pop_back();
            pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);
            entries_.insert(pos, {obj, sol, source});
            return true;
        }

        // Diversity-aware path: pool is full and obj doesn't beat worst.
        // Accept if (a) integer mask is set, (b) obj is within tolerance of
        // best, and (c) solution is sufficiently diverse from all entries.
        if (integer_mask_.empty() || num_integers_ == 0 || entries_.empty()) {
            return false;
        }

        double best_obj = entries_.front().objective;
        double gap = std::abs(obj - best_obj);
        // Continuous fallback: fraction of |best_obj|, floored to avoid
        // a discontinuous jump near zero.
        double threshold =
            std::max(kDiversityObjTolerance * std::abs(best_obj), kDiversityObjTolerance * 1e-6);

        if (gap > threshold) {
            return false;
        }

        // Compute minimum Hamming distance to any pool entry and track
        // the index of the most similar entry.
        int min_dist = std::numeric_limits<int>::max();
        int most_similar_idx = -1;
        for (int i = 0; i < static_cast<int>(entries_.size()); ++i) {
            int d = hamming_distance(sol, entries_[i].solution);
            if (d < min_dist) {
                min_dist = d;
                most_similar_idx = i;
            }
        }

        double min_frac = static_cast<double>(min_dist) / static_cast<double>(num_integers_);
        if (min_frac < kDiversityMinHammingFrac) {
            return false;
        }

        // Replace the most similar entry.
        entries_.erase(entries_.begin() + most_similar_idx);
        pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);
        entries_.insert(pos, {obj, sol, source});
        return true;
    }

    entries_.insert(pos, {obj, sol, source});
    return true;
}

SolutionPool::Snapshot SolutionPool::snapshot() {
    std::lock_guard<HighsSpinMutex> lock(mtx_);
    if (entries_.empty()) {
        return {false, minimize_ ? std::numeric_limits<double>::infinity()
                                 : -std::numeric_limits<double>::infinity()};
    }
    return {true, entries_[0].objective};
}

bool SolutionPool::get_restart(Rng& rng, std::vector<double>& out) {
    std::lock_guard<HighsSpinMutex> lock(mtx_);
    if (entries_.empty()) {
        return false;
    }

    int pool_size = static_cast<int>(entries_.size());
    double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

    auto pick_two_parents = [&](int& a, int& b) {
        a = std::uniform_int_distribution<int>(0, pool_size - 1)(rng);
        b = std::uniform_int_distribution<int>(0, pool_size - 2)(rng);
        if (b >= a) {
            ++b;
        }
    };

    if (roll < 0.4 && pool_size >= 2) {
        // Guided crossover: keep agreed integer values, coin-flip
        // disagreements.
        int a, b;
        pick_two_parents(a, b);
        const auto& sol_a = entries_[a].solution;
        const auto& sol_b = entries_[b].solution;
        int ncol = static_cast<int>(sol_a.size());
        out.resize(ncol);
        for (int j = 0; j < ncol; ++j) {
            bool is_int = !integer_mask_.empty() && j < static_cast<int>(integer_mask_.size())
                              ? integer_mask_[j]
                              : false;
            if (is_int && std::round(sol_a[j]) == std::round(sol_b[j])) {
                // Parents agree on this integer value — keep it.
                out[j] = sol_a[j];
            } else {
                // Disagree or continuous — coin flip.
                out[j] = std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? sol_a[j] : sol_b[j];
            }
        }
    } else if (roll < 0.7 && pool_size >= 2) {
        // Neighborhood crossover: better parent provides base, coin-flip
        // only on disagreeing integer variables.
        int a, b;
        pick_two_parents(a, b);
        // Better parent = lower index (entries_ sorted best-first).
        int better = std::min(a, b);
        int other = std::max(a, b);
        const auto& sol_better = entries_[better].solution;
        const auto& sol_other = entries_[other].solution;
        int ncol = static_cast<int>(sol_better.size());
        out.resize(ncol);
        for (int j = 0; j < ncol; ++j) {
            bool is_int = !integer_mask_.empty() && j < static_cast<int>(integer_mask_.size())
                              ? integer_mask_[j]
                              : false;
            if (is_int && std::round(sol_better[j]) != std::round(sol_other[j])) {
                // Integer var where parents disagree — coin flip.
                out[j] = std::uniform_int_distribution<int>(0, 1)(rng) == 0 ? sol_better[j]
                                                                            : sol_other[j];
            } else {
                // Agree or continuous — keep the better parent's value.
                out[j] = sol_better[j];
            }
        }
    } else {
        // Biased copy toward better entries.
        int idx;
        if (pool_size > 1 && std::uniform_int_distribution<int>(0, 1)(rng) == 0) {
            idx = std::uniform_int_distribution<int>(0, (pool_size + 1) / 2 - 1)(rng);
        } else {
            idx = std::uniform_int_distribution<int>(0, pool_size - 1)(rng);
        }
        out = entries_[idx].solution;
    }
    return true;
}

std::vector<SolutionPool::Entry> SolutionPool::sorted_entries() {
    std::lock_guard<HighsSpinMutex> lock(mtx_);
    return entries_;  // already kept sorted
}

int SolutionPool::size() {
    std::lock_guard<HighsSpinMutex> lock(mtx_);
    return static_cast<int>(entries_.size());
}

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
