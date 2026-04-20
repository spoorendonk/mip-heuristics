#pragma once

#include "heuristic_common.h"
#include "util/HighsInt.h"

#include <limits>
#include <vector>

namespace local_mip_detail {

// --- File-scope constants (paper + engineering) ---
inline constexpr double kViolTol = 5e-7;
inline constexpr HighsInt kRestartInterval = 200000;
inline constexpr HighsInt kTermCheckInterval = 1000;
inline constexpr HighsInt kActivityPeriod = 100000;
inline constexpr double kSmoothProb = 3e-4;
inline constexpr HighsInt kBmsConstraints = 12;
inline constexpr HighsInt kBmsBudget = 2250;
inline constexpr HighsInt kBmsSatCon = 1;
inline constexpr HighsInt kBmsSatBudget = 80;
inline constexpr HighsInt kBoolFlipBudget = 5000;
inline constexpr HighsInt kEasyBudget = 5;
inline constexpr HighsInt kTabuBase = 3;
inline constexpr HighsInt kTabuVar = 10;
inline constexpr HighsInt kFeasibleRecheckPeriod = 100;
inline constexpr HighsInt kFeasiblePlateau = 5000;
inline constexpr double kEpsZero = 1e-15;

// Forward declaration: LiftCache::recompute_* take a WorkerCtx& (defined
// in local_mip_core.h).  Defined here to keep LiftCache colocated with
// the other caches and the shared constants, matching the issue's
// suggested split.
struct WorkerCtx;

// --- IndexedSet: O(1) add/remove with iteration ---
struct IndexedSet {
    std::vector<HighsInt> elements;
    std::vector<HighsInt> pos;  // -1 = absent

    explicit IndexedSet(HighsInt n) : pos(n, -1) { elements.reserve(n); }

    void add(HighsInt i) {
        if (pos[i] != -1) {
            return;
        }
        pos[i] = static_cast<HighsInt>(elements.size());
        elements.push_back(i);
    }

    void remove(HighsInt i) {
        HighsInt p = pos[i];
        if (p == -1) {
            return;
        }
        HighsInt last = elements.back();
        elements[p] = last;
        pos[last] = p;
        elements.pop_back();
        pos[i] = -1;
    }

    bool contains(HighsInt i) const { return pos[i] != -1; }
    bool empty() const { return elements.empty(); }
    HighsInt size() const { return static_cast<HighsInt>(elements.size()); }
    HighsInt operator[](HighsInt idx) const { return elements[idx]; }

    void clear() {
        for (HighsInt e : elements) {
            pos[e] = -1;
        }
        elements.clear();
    }

    auto begin() const { return elements.begin(); }
    auto end() const { return elements.end(); }
};

// --- ViolCache: memoize row violations within a candidate batch ---
struct ViolCache {
    std::vector<double> cache;
    std::vector<HighsInt> used;
    static constexpr double kSentinel = -1.0;

    explicit ViolCache(HighsInt n) : cache(n, kSentinel) { used.reserve(n); }

    double get_or_compute(HighsInt i, double lhs_i, double row_lo_i, double row_hi_i) {
        if (cache[i] >= 0.0) {
            return cache[i];
        }
        double v = row_violation(lhs_i, row_lo_i, row_hi_i);
        cache[i] = v;
        used.push_back(i);
        return v;
    }

    void reset() {
        for (HighsInt i : used) {
            cache[i] = kSentinel;
        }
        used.clear();
    }
};

// --- Candidate structs ---
struct Candidate {
    HighsInt var_idx = -1;
    double new_val = 0.0;
    double score = -std::numeric_limits<double>::infinity();
    double bonus = 0.0;
};

struct BatchCand {
    HighsInt var_idx;
    double new_val;
};

struct WeightedCon {
    HighsInt ci;
    uint64_t w;
};

// --- LiftCache: cached lift bounds / scores per variable ---
//
// Methods are defined in local_mip_core.cpp because they depend on the
// full WorkerCtx definition.
struct LiftCache {
    std::vector<double> lo, hi, score;
    std::vector<bool> dirty;
    std::vector<HighsInt> dirty_list;
    bool all_dirty = true;
    std::vector<HighsInt> positive_list;
    std::vector<bool> in_positive;
    const std::vector<HighsInt> *costed_vars = nullptr;

    explicit LiftCache(HighsInt ncol)
        : lo(ncol), hi(ncol), score(ncol), dirty(ncol, true), in_positive(ncol, false) {
        dirty_list.reserve(ncol);
        positive_list.reserve(ncol);
    }

    void mark_dirty(HighsInt j) {
        if (!dirty[j]) {
            dirty[j] = true;
            dirty_list.push_back(j);
        }
    }

    void mark_all_dirty() {
        all_dirty = true;
        dirty_list.clear();
        std::fill(dirty.begin(), dirty.end(), true);
        positive_list.clear();
        std::fill(in_positive.begin(), in_positive.end(), false);
    }

    void recompute_one(HighsInt j, WorkerCtx &ctx);
    void recompute_all(WorkerCtx &ctx);
};

}  // namespace local_mip_detail
