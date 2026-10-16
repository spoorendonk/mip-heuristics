#pragma once

#include <atomic>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

// The least-violated infeasible point a local search has retained (#188),
// for a caller that wants a start before any solution exists — ReXi's LNS
// (Mexi & Rehfeldt, Sect. 2.3) starts from "the latest retained point of
// smallest constraint violation from the local search" while its pool is
// empty.
//
// Caller-owned and opt-in: point `ExecutionContext::least_violated` at one
// and LocalMIP's workers offer it the least-violated state each of them
// passes through; leave it null (the default, and what the HiGHS adapter
// does) and nothing is measured or copied.  FJ does not feed it.
//
// The violation is the sum, over the rows the search counts as violated
// (outside their bounds by more than `feastol`), of each row's distance to
// its bounds — in the view's column space, which for a view built with the
// core `make_problem` is the caller's model's own: columns are not
// transformed, and every point the search holds is inside the column
// bounds, so a row is the only thing a point can violate.  Only infeasible
// states are offered; a feasible one goes to the `SolutionSink`.  Ties go
// to the latest offer.
//
// Thread-safe: any number of workers may offer while any thread reads.
// The point is published as an immutable shared vector, so a reader copies
// a pointer under a short lock and reads the vector outside it, as
// `SolutionPool`'s entries are (#180).
class LeastViolatedPoint {
public:
    struct Point {
        std::shared_ptr<const std::vector<double>> x;
        double violation;
    };

    // The retained point, or nothing before the first offer.
    [[nodiscard]] std::optional<Point> best() const {
        std::scoped_lock lock(mu_);
        if (!x_) {
            return std::nullopt;
        }
        return Point{x_, violation_};
    }

    // Whether an offer at `violation` would be taken: lock-free, so a
    // worker can skip building one that would not.
    [[nodiscard]] bool would_take(double violation) const {
        return violation <= best_violation_.load(std::memory_order_relaxed);
    }

    // Retain `x` when `violation` is at most the retained one's; returns
    // whether it did.
    bool offer(double violation, const std::vector<double>& x) {
        if (!would_take(violation)) {
            return false;
        }
        auto point = std::make_shared<const std::vector<double>>(x);
        std::scoped_lock lock(mu_);
        if (violation > violation_) {
            return false;
        }
        x_ = std::move(point);
        violation_ = violation;
        best_violation_.store(violation, std::memory_order_relaxed);
        return true;
    }

private:
    mutable std::mutex mu_;
    std::shared_ptr<const std::vector<double>> x_;
    double violation_ = std::numeric_limits<double>::infinity();
    // `violation_`, readable without the lock; a stale read only lets an
    // offer through to the locked re-check.
    std::atomic<double> best_violation_{std::numeric_limits<double>::infinity()};
};
