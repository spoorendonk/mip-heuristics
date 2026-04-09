#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

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

inline CscMatrix build_csc(HighsInt ncol, HighsInt nrow, const std::vector<HighsInt> &ARstart,
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

inline bool is_integer(const std::vector<HighsVarType> &integrality, HighsInt j) {
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

// Seed constants for deterministic per-worker RNG seeding.
constexpr uint32_t kBaseSeedOffset = 42;
constexpr uint32_t kSeedStride = 997;

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

// ---------------------------------------------------------------------------
// Memory-aware worker count caps
// ---------------------------------------------------------------------------

// Total physical RAM in bytes (0 on failure).
inline size_t total_system_memory() {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullTotalPhys);
    }
    return 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<size_t>(pages) * static_cast<size_t>(page_size);
    }
    return 0;
#endif
}

// Fraction of system RAM available as budget for parallel workers.
constexpr double kMemoryBudgetFraction = 0.5;

// Per-worker memory estimates (bytes).
// FJ: FeasibilityJumpSolver duplicates cols, rows, and constraint matrix.
inline size_t estimate_worker_memory_fj(HighsInt ncol, HighsInt nrow, size_t nnz) {
    // col_value, bounds, costs, score arrays ~ 6*ncol doubles
    // row state ~ 2*nrow doubles
    // constraint matrix indices + values ~ nnz*(sizeof(int)+sizeof(double))
    return static_cast<size_t>(ncol) * 6 * sizeof(double) +
           static_cast<size_t>(nrow) * 2 * sizeof(double) +
           nnz * (sizeof(HighsInt) + sizeof(double));
}

// LocalMIP: WorkerCtx owns solution, lhs, weight, tabu, lift cache, index sets.
inline size_t estimate_worker_memory_local_mip(HighsInt ncol, HighsInt nrow) {
    // solution(ncol) + lhs(nrow) + weight(nrow*8) = ncol*8 + nrow*16
    // tabu_inc_until(ncol) + tabu_dec_until(ncol) = 2*ncol*sizeof(HighsInt)
    // LiftCache: lo,hi,score(ncol doubles) + dirty,in_positive(ncol bools) = 3*ncol*8 + 2*ncol
    // ViolCache: cache(nrow doubles) + used(nrow ints) = nrow*8 + nrow*sizeof(HighsInt)
    // IndexedSet violated + satisfied: 2*(elements(nrow) + pos(nrow)) = 4*nrow*sizeof(HighsInt)
    // best_solution(ncol) + costed_vars + binary_vars ~ 2*ncol*sizeof(HighsInt)
    return static_cast<size_t>(ncol) * (4 * sizeof(double) + 2 * sizeof(HighsInt)) +
           static_cast<size_t>(nrow) *
               (2 * sizeof(double) + sizeof(uint64_t) + 5 * sizeof(HighsInt));
}

// Scylla parallel: M FPR result vectors, each holding ncol doubles.
inline size_t estimate_worker_memory_scylla(HighsInt ncol, int num_results) {
    return static_cast<size_t>(num_results) * static_cast<size_t>(ncol) * sizeof(double);
}

// Max workers that fit within the memory budget for a given per-worker cost.
// Returns at least 1 (always allow one worker).
inline int max_workers_for_memory(size_t per_worker_bytes) {
    if (per_worker_bytes == 0) {
        return std::numeric_limits<int>::max();
    }
    size_t total = total_system_memory();
    if (total == 0) {
        // Cannot determine RAM — don't cap.
        return std::numeric_limits<int>::max();
    }
    auto budget = static_cast<size_t>(static_cast<double>(total) * kMemoryBudgetFraction);
    int max_w = static_cast<int>(budget / per_worker_bytes);
    return std::max(max_w, 1);
}
