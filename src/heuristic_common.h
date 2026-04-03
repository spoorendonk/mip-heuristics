#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "lp_data/HConst.h"
#include "util/HighsInt.h"

// Variable state used during fix-and-propagate.
struct VarState {
  double lb, ub, val;
  bool fixed;
};

struct HeuristicResult {
  bool found_feasible = false;
  std::vector<double> solution;
  double objective = std::numeric_limits<double>::infinity();
  size_t effort = 0;

  static HeuristicResult failed(size_t e = 0) {
    HeuristicResult r;
    r.effort = e;
    return r;
  }
};

struct CscMatrix {
  std::vector<HighsInt> col_start;
  std::vector<HighsInt> col_row;
  std::vector<double> col_val;
};

inline CscMatrix build_csc(HighsInt ncol, HighsInt nrow,
                           const std::vector<HighsInt> &ARstart,
                           const std::vector<HighsInt> &ARindex,
                           const std::vector<double> &ARvalue) {
  const HighsInt nnz = static_cast<HighsInt>(ARindex.size());
  CscMatrix csc;
  csc.col_start.assign(ncol + 1, 0);
  for (HighsInt k = 0; k < nnz; ++k) {
    csc.col_start[ARindex[k] + 1]++;
  }
  for (HighsInt j = 0; j < ncol; ++j) {
    csc.col_start[j + 1] += csc.col_start[j];
  }
  csc.col_row.resize(nnz);
  csc.col_val.resize(nnz);
  {
    std::vector<HighsInt> pos(csc.col_start);
    for (HighsInt i = 0; i < nrow; ++i) {
      for (HighsInt k = ARstart[i]; k < ARstart[i + 1]; ++k) {
        HighsInt j = ARindex[k];
        csc.col_row[pos[j]] = i;
        csc.col_val[pos[j]] = ARvalue[k];
        pos[j]++;
      }
    }
  }
  return csc;
}

inline bool is_integer(const std::vector<HighsVarType> &integrality,
                       HighsInt j) {
  return integrality[j] != HighsVarType::kContinuous;
}

// Tolerance hierarchy:
//   feastol  (~1e-6)  — from solver, used for feasibility checks
//   kViolTol (5e-7)   — local_mip local-search violation threshold
//   1e-15             — numerical zero (avoids division/move on zero-delta)

// Row violation: how much lhs exceeds [lo, hi] bounds.
inline double row_violation(double lhs, double lo, double hi) {
  return std::max(0.0, lhs - hi) + std::max(0.0, lo - lhs);
}

// Whether a row is violated beyond the given feasibility tolerance.
inline bool is_row_violated(double lhs, double lo, double hi, double feastol) {
  return lhs > hi + feastol || lhs < lo - feastol;
}

// Clamp value to [lb, ub], rounding if integer.
inline double clamp_round(double val, double lb, double ub, bool integer) {
  if (integer) {
    val = std::round(val);
  }
  return std::max(lb, std::min(ub, val));
}

// Effort budget for presolve heuristics, scaled by mip_heuristic_effort.
// Base budget nnz << 12 at default effort 0.05; scales linearly.
inline size_t heuristic_effort_budget(size_t nnz, double mip_heuristic_effort) {
  if (mip_heuristic_effort <= 0.0) {
    return 0;
  }
  constexpr int kBaseShift = 12;
  double scale = mip_heuristic_effort / 0.05;
  return static_cast<size_t>(static_cast<double>(nnz << kBaseShift) * scale);
}
