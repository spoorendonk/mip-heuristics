#include "adaptive/solution_pool.h"

#include <algorithm>

SolutionPool::SolutionPool(int capacity, bool minimize)
    : capacity_(capacity), minimize_(minimize) {}

bool SolutionPool::try_add(double obj, const std::vector<double>& sol) {
  std::lock_guard<HighsSpinMutex> lock(mtx_);
  if (static_cast<int>(entries_.size()) >= capacity_) {
    auto& worst = entries_.back();
    bool dominated = minimize_ ? obj >= worst.objective : obj <= worst.objective;
    if (dominated) return false;
    worst.objective = obj;
    worst.solution = sol;
  } else {
    entries_.push_back({obj, sol});
  }
  std::sort(entries_.begin(), entries_.end(),
            [this](const Entry& a, const Entry& b) {
              return minimize_ ? a.objective < b.objective
                               : a.objective > b.objective;
            });
  return true;
}

SolutionPool::Snapshot SolutionPool::snapshot() {
  std::lock_guard<HighsSpinMutex> lock(mtx_);
  if (entries_.empty()) {
    return {false, minimize_ ? 1e30 : -1e30};
  }
  return {true, entries_[0].objective};
}

bool SolutionPool::get_restart(std::mt19937& rng, std::vector<double>& out) {
  std::lock_guard<HighsSpinMutex> lock(mtx_);
  if (entries_.empty()) return false;

  int pool_size = static_cast<int>(entries_.size());
  double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

  if (roll < 0.5 && pool_size >= 2) {
    // Uniform crossover between 2 random entries
    int a = std::uniform_int_distribution<int>(0, pool_size - 1)(rng);
    int b = std::uniform_int_distribution<int>(0, pool_size - 2)(rng);
    if (b >= a) ++b;
    const auto& sol_a = entries_[a].solution;
    const auto& sol_b = entries_[b].solution;
    int ncol = static_cast<int>(sol_a.size());
    out.resize(ncol);
    for (int j = 0; j < ncol; ++j) {
      out[j] = std::uniform_int_distribution<int>(0, 1)(rng) == 0
                   ? sol_a[j]
                   : sol_b[j];
    }
  } else {
    // Direct copy biased toward better entries
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

bool SolutionPool::get_restart(std::mt19937& rng, int ncol, const double* col_lb,
                               const double* col_ub,
                               bool (*is_binary_fn)(int),
                               bool (*is_int_fn)(int),
                               std::vector<double>& out) {
  std::lock_guard<HighsSpinMutex> lock(mtx_);
  if (entries_.empty()) return false;

  int pool_size = static_cast<int>(entries_.size());
  double roll = std::uniform_real_distribution<double>(0.0, 1.0)(rng);

  if (roll < 0.33 && pool_size >= 2) {
    // Uniform crossover
    int a = std::uniform_int_distribution<int>(0, pool_size - 1)(rng);
    int b = std::uniform_int_distribution<int>(0, pool_size - 2)(rng);
    if (b >= a) ++b;
    const auto& sol_a = entries_[a].solution;
    const auto& sol_b = entries_[b].solution;
    out.resize(ncol);
    for (int j = 0; j < ncol; ++j) {
      out[j] = std::uniform_int_distribution<int>(0, 1)(rng) == 0
                   ? sol_a[j]
                   : sol_b[j];
    }
  } else if (roll < 0.66) {
    // Perturbation of random entry (~15% of vars)
    int idx = std::uniform_int_distribution<int>(0, pool_size - 1)(rng);
    out = entries_[idx].solution;
    for (int j = 0; j < ncol; ++j) {
      if (std::uniform_real_distribution<double>(0.0, 1.0)(rng) < 0.15) {
        if (is_binary_fn(j)) {
          out[j] = (out[j] < 0.5) ? 1.0 : 0.0;
        } else if (is_int_fn(j)) {
          double lo = col_lb[j];
          double hi = std::min(col_ub[j], lo + 100.0);
          out[j] = std::round(
              std::uniform_real_distribution<double>(lo, hi)(rng));
          out[j] = std::max(col_lb[j], std::min(col_ub[j], out[j]));
        } else {
          double lo = col_lb[j];
          double hi = std::min(col_ub[j], lo + 1e6);
          out[j] = std::uniform_real_distribution<double>(lo, hi)(rng);
        }
      }
    }
  } else {
    // Direct copy biased toward better entries
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
