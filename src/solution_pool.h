#pragma once

#include <random>
#include <vector>

#include "parallel/HighsSpinMutex.h"

class HighsMipSolver;

inline constexpr int kPoolCapacity = 10;

// Thread-safe solution pool. Keeps top-K solutions sorted by objective.
// Supports restart strategies: crossover and copy.
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

  // Get a restart solution using crossover or copy (no bounds needed).
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

// Seed a pool with the current incumbent (if any). Defined inline to
// avoid pulling HighsMipSolver includes into the header — callers
// already include both solution_pool.h and HighsMipSolver.h.
void seed_pool(SolutionPool &pool, const HighsMipSolver &mipsolver);
