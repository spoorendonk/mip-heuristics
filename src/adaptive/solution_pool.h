#pragma once

#include <random>
#include <vector>

#include "parallel/HighsSpinMutex.h"

// Thread-safe solution pool. Keeps top-K solutions sorted by objective.
// Supports restart strategies: crossover, perturbation, and copy.
class SolutionPool {
 public:
  struct Snapshot {
    bool has_solution;
    double best_objective;
  };

  struct Entry {
    double objective;
    std::vector<double> solution;
  };

  SolutionPool(int capacity, bool minimize);

  // Try to add a solution. Returns true if added (improves on worst entry
  // or pool not full).
  bool try_add(double obj, const std::vector<double>& sol);

  // Atomically snapshot feasibility and current best objective.
  Snapshot snapshot();

  // Get a restart solution using one of 3 strategies:
  // - Uniform crossover between 2 random pool entries
  // - Perturbation of random entry (~15% of variables flipped)
  // - Direct copy of random entry (biased toward better)
  // Returns false if pool is empty.
  // col_lb/col_ub/is_binary/is_int are needed for perturbation bounds.
  bool get_restart(std::mt19937& rng, int ncol, const double* col_lb,
                   const double* col_ub,
                   bool (*is_binary_fn)(int),
                   bool (*is_int_fn)(int),
                   std::vector<double>& out);

  // Simpler get_restart that just does crossover or copy (no bounds needed).
  bool get_restart(std::mt19937& rng, std::vector<double>& out);

  // Return sorted entries (best first). Caller should hold no lock.
  std::vector<Entry> sorted_entries();

  int size();

 private:
  mutable HighsSpinMutex mtx_;
  std::vector<Entry> entries_;
  int capacity_;
  bool minimize_;
};
