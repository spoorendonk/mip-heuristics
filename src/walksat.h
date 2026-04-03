#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "util/HighsInt.h"

class PropEngine;

struct WalkSatMove {
  HighsInt var = -1;
  double val = 0.0;
};

// Pre-allocated scratch buffers to avoid per-call heap allocations.
struct WalkSatScratch {
  struct Candidate {
    HighsInt var;
    double new_val;
    double damage;
  };
  std::vector<Candidate> cand;
  std::vector<HighsInt> best_indices;
};

// Select a WalkSAT repair move for the given violated row.
// Uses PropEngine for read-only matrix data, bounds, integrality, feastol.
// solution/lhs_cache are the current (possibly infeasible) assignment.
// col_lb/col_ub are clamping bounds for candidate shifts.
// Returns {-1, 0.0} if no valid candidate found.
// Increments effort by coefficient accesses consumed.
WalkSatMove walksat_select_move(HighsInt row, const double* solution,
                                const double* lhs_cache, const double* col_lb,
                                const double* col_ub, const PropEngine& data,
                                double noise, std::mt19937& rng,
                                size_t& effort, WalkSatScratch& scratch);

// Run flat WalkSAT repair walk (paper Fig. 4) on solution/lhs_cache.
// Returns true if feasible. Modifies solution/lhs_cache in place.
// total_viol is the current total violation (updated incrementally).
bool walksat_repair(const PropEngine& data, std::vector<double>& solution,
                    std::vector<double>& lhs_cache, const double* col_lb,
                    const double* col_ub, HighsInt max_iterations, double noise,
                    bool track_best, size_t max_effort, std::mt19937& rng,
                    size_t& effort);

// Greedy 1-opt: shift each integer variable by ±1 toward better objective
// if the shift maintains feasibility. Modifies solution/lhs_cache in place.
void greedy_1opt(const PropEngine& data, std::vector<double>& solution,
                 std::vector<double>& lhs_cache, const double* col_cost,
                 bool minimize, size_t& effort);
