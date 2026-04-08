#include "solution_pool.h"

#include <algorithm>
#include <limits>

SolutionPool::SolutionPool(int capacity, bool minimize)
    : capacity_(capacity), minimize_(minimize) {}

bool SolutionPool::try_add(double obj, const std::vector<double>& sol) {
  std::lock_guard<HighsSpinMutex> lock(mtx_);

  // Find insertion point (entries_ kept sorted, best first)
  auto cmp = [this](const Entry& e, double val) {
    return minimize_ ? e.objective < val : e.objective > val;
  };
  auto pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);

  if (static_cast<int>(entries_.size()) >= capacity_) {
    // Full: only add if better than worst
    auto& worst = entries_.back();
    bool dominated = minimize_ ? obj >= worst.objective : obj <= worst.objective;
    if (dominated) return false;
    // Remove worst, insert at correct position
    entries_.pop_back();
    pos = std::lower_bound(entries_.begin(), entries_.end(), obj, cmp);
    entries_.insert(pos, {obj, sol});
  } else {
    entries_.insert(pos, {obj, sol});
  }
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

std::vector<SolutionPool::Entry> SolutionPool::sorted_entries() {
  std::lock_guard<HighsSpinMutex> lock(mtx_);
  return entries_;  // already kept sorted
}

int SolutionPool::size() {
  std::lock_guard<HighsSpinMutex> lock(mtx_);
  return static_cast<int>(entries_.size());
}

#include "mip/HighsMipSolver.h"
#include "mip/HighsMipSolverData.h"

void seed_pool(SolutionPool &pool, const HighsMipSolver &mipsolver) {
  const auto *model = mipsolver.model_;
  auto *mipdata = mipsolver.mipdata_.get();
  if (mipdata->incumbent.empty()) return;
  const HighsInt ncol = model->num_col_;
  double obj = model->offset_;
  for (HighsInt j = 0; j < ncol; ++j) {
    obj += model->col_cost_[j] * mipdata->incumbent[j];
  }
  pool.try_add(obj, mipdata->incumbent);
}
