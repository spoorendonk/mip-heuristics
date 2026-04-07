#pragma once

#include <random>
#include <vector>

#include "parallel/HighsSpinMutex.h"

// Beta-Bernoulli Thompson Sampling bandit for adaptive heuristic selection.
// Thread-safe when constructed with use_mutex=true (opportunistic mode).
//
// Supports effort-aware selection: when effort is recorded via
// record_effort(), select_effort_aware() weights Thompson samples by
// inverse running-average effort so cheap arms are preferred at equal reward.
class ThompsonSampler {
 public:
  struct ArmStats {
    double alpha;
    double beta;
    int pulls;
    double avg_effort;  // running average wall-clock effort per pull (seconds)
    double mean() const { return alpha / (alpha + beta); }
  };

  // num_arms: number of arms.
  // prior_alpha: array of length num_arms with per-arm alpha priors (beta=1).
  // use_mutex: if true, all operations are spin-mutex protected.
  ThompsonSampler(int num_arms, const double* prior_alpha, bool use_mutex);

  // Sample from each arm's Beta distribution, return arm with highest sample.
  int select(std::mt19937& rng);

  // Effort-aware selection: sample from Beta, then weight by 1/avg_effort.
  // Falls back to plain select() until every arm has at least one effort
  // observation.
  int select_effort_aware(std::mt19937& rng);

  // Update arm with reward in {0, 1, 2, 3}.
  // 0 (infeasible)       → beta  += 1.0
  // 1 (feasible, stale)  → beta  += 0.25
  // 2 (first feasible)   → alpha += 1.0
  // 3 (improved obj)     → alpha += 1.5
  void update(int arm, int reward);

  // Record wall-clock effort (seconds) for an arm pull. Updates the
  // exponential moving average with smoothing factor alpha=0.3.
  void record_effort(int arm, double seconds);

  ArmStats stats(int arm) const;
  int num_arms() const { return static_cast<int>(arms_.size()); }

 private:
  struct ArmState {
    double alpha;
    double beta;
    int pulls;
    double avg_effort;      // EMA of wall-clock seconds per pull
    bool has_effort;        // true once at least one effort observation exists
  };

  // Core sampling logic (no locking — caller must hold mtx_ if needed).
  int select_unlocked(std::mt19937& rng);

  std::vector<ArmState> arms_;
  mutable HighsSpinMutex mtx_;
  bool use_mutex_;
};
