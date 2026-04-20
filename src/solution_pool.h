#pragma once

#include "parallel/HighsSpinMutex.h"

#include <random>
#include <vector>

class HighsMipSolver;

inline constexpr int kPoolCapacity = 10;

// Diversity-aware insertion constants.
// Accept a solution within this fraction of best objective if it adds diversity.
inline constexpr double kDiversityObjTolerance = 0.10;
// Minimum Hamming distance (as fraction of integer vars) to qualify as diverse.
inline constexpr double kDiversityMinHammingFrac = 0.05;

// Thread-safe solution pool. Keeps top-K solutions sorted by objective.
// Supports restart strategies: guided crossover, neighborhood crossover,
// and biased copy.  When an integer mask is set, insertion is
// diversity-aware: solutions that don't improve the worst objective can
// still enter the pool if they are structurally different from existing
// entries (measured by Hamming distance on integer variables).
class SolutionPool {
public:
    struct Snapshot {
        bool has_solution;
        double best_objective;
    };

    struct Entry {
        double objective;
        std::vector<double> solution;
        // Per-entry provenance tag (one of the kSolutionSource* constants
        // from HiGHS's HighsMipSolverData.h).  Carried so portfolio flushes
        // can attribute each solution to the bandit arm / heuristic that
        // produced it, rather than falling back on the generic
        // kSolutionSourceHeuristic tag.
        int source;
    };

    SolutionPool(int capacity, bool minimize);

    // Set the integer variable mask.  Must be called before diversity-aware
    // insertion can take effect.  is_integer[j] == true iff variable j is
    // integer.  Thread-safe (acquires lock).
    void set_integer_mask(std::vector<bool> mask);

    // Try to add a solution. Returns true if added.
    // `source` is one of the kSolutionSource* constants and is stored on
    // the inserted entry for later provenance-aware flushing.
    // Insertion policy (when pool is full):
    //   1. If obj improves on worst: replace worst (standard).
    //   2. Else if obj is within kDiversityObjTolerance of best and Hamming
    //      diversity exceeds kDiversityMinHammingFrac: replace most similar.
    bool try_add(double obj, const std::vector<double>& sol, int source);

    // Atomically snapshot feasibility and current best objective.
    Snapshot snapshot();

    // Get a restart solution via one of three strategies (roll order):
    //   [0.0, 0.4)  — guided crossover: keep agreed integer values, coin-flip
    //                  disagreements.
    //   [0.4, 0.7)  — neighborhood crossover: better parent provides base,
    //                  coin-flip only on disagreeing integers.
    //   [0.7, 1.0)  — biased copy toward better entries.
    //
    // Post-crossover repair is handled naturally by the calling heuristic
    // (FPR, LocalMIP, etc.) which treats the restart as an initial solution
    // and runs its own feasibility restoration.
    bool get_restart(std::mt19937& rng, std::vector<double>& out);

    // Return sorted entries (best first). Caller should hold no lock.
    std::vector<Entry> sorted_entries();

    int size();

private:
    // Hamming distance on integer variables between two solutions.
    // Caller must hold mtx_.
    int hamming_distance(const std::vector<double>& a, const std::vector<double>& b) const;

    // Number of integer variables (cached from integer_mask_).
    int num_integers() const;

    mutable HighsSpinMutex mtx_;
    std::vector<Entry> entries_;
    int capacity_;
    bool minimize_;
    std::vector<bool> integer_mask_;  // true for integer variables
    int num_integers_ = 0;            // cached count of integer vars
};

// Seed a pool with the current incumbent (if any). Defined inline to
// avoid pulling HighsMipSolver includes into the header — callers
// already include both solution_pool.h and HighsMipSolver.h.
void seed_pool(SolutionPool& pool, const HighsMipSolver& mipsolver);
