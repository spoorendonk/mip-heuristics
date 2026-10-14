// SolutionPool's pointer entries (#180): the overload equivalence, the
// lifetime of a handed-out pointer, and concurrent use.  Needs nothing of
// HiGHS but the spin-lock header, so `mip_heuristics_pool_tsan_tests` builds
// this file alone with `src/solution_pool.cpp` under ThreadSanitizer.

#include "rng.h"
#include "solution_pool.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vector>

// Source tags.  Any ints do: the pool stores them and compares nothing.
constexpr int kSourceA = 1;
constexpr int kSourceB = 2;
constexpr int kSourceC = 3;

namespace {

// A random offer stream built to exercise every admission path: a small
// pool, objectives in a narrow band so the diversity tolerance admits
// dominated offers, and binary columns so their Hamming distances vary.
struct Offer {
    double obj;
    std::vector<double> sol;
};

std::vector<Offer> make_offers(int count, int ncol, std::uint64_t seed) {
    Rng rng(seed);
    std::vector<Offer> offers;
    offers.reserve(count);
    for (int i = 0; i < count; ++i) {
        Offer offer{std::uniform_real_distribution<double>(100.0, 108.0)(rng),
                    std::vector<double>(ncol)};
        for (auto& v : offer.sol) {
            v = static_cast<double>(std::uniform_int_distribution<int>(0, 1)(rng));
        }
        offers.push_back(std::move(offer));
    }
    return offers;
}

}  // namespace

TEST_CASE("SolutionPool: the pointer try_add decides every offer as the vector one does",
          "[pool]") {
    constexpr int kNcol = 40;
    const auto offers = make_offers(/*count=*/400, kNcol, /*seed=*/7);

    SolutionPool by_ref(/*capacity=*/4);
    SolutionPool by_ptr(/*capacity=*/4);
    by_ref.set_integer_mask(std::vector<bool>(kNcol, true));
    by_ptr.set_integer_mask(std::vector<bool>(kNcol, true));

    int diversity_admissions = 0;
    for (const auto& offer : offers) {
        const double worst_before = by_ref.size() == 4 ? by_ref.sorted_entries().back().objective
                                                       : std::numeric_limits<double>::infinity();
        const auto a = by_ref.try_add(offer.obj, offer.sol, kSourceA);
        const auto b = by_ptr.try_add(
            offer.obj, std::make_shared<const std::vector<double>>(offer.sol), kSourceA);
        REQUIRE(a.accepted == b.accepted);
        REQUIRE(a.improved_best == b.improved_best);
        if (a.accepted && offer.obj >= worst_before) {
            ++diversity_admissions;
        }
    }
    // The stream must actually reach the path the scan outside the lock
    // changed, or the comparison above proves nothing about it.
    REQUIRE(diversity_admissions > 0);

    const auto ea = by_ref.sorted_entries();
    const auto eb = by_ptr.sorted_entries();
    REQUIRE(ea.size() == eb.size());
    for (size_t i = 0; i < ea.size(); ++i) {
        REQUIRE(ea[i].objective == eb[i].objective);
        REQUIRE(ea[i].source == eb[i].source);
        REQUIRE(ea[i].solution == eb[i].solution);
    }

    // Same entries and the same RNG state give the same restart draws.
    Rng ra(11);
    Rng rb(11);
    std::vector<double> out_a;
    std::vector<double> out_b;
    for (int i = 0; i < 200; ++i) {
        REQUIRE(by_ref.get_restart(ra, out_a));
        REQUIRE(by_ptr.get_restart(rb, out_b));
        REQUIRE(out_a == out_b);
    }
}

TEST_CASE("SolutionPool: an entry_refs pointer outlives the entry's eviction", "[pool]") {
    SolutionPool pool(/*capacity=*/2);
    REQUIRE(pool.try_add(10.0, {1.0, 2.0}, kSourceA).accepted);
    REQUIRE(pool.try_add(20.0, {3.0, 4.0}, kSourceB).accepted);

    std::vector<SolutionPool::EntryRef> refs;
    pool.entry_refs(refs);
    REQUIRE(refs.size() == 2);
    REQUIRE(refs[0].objective == 10.0);
    REQUIRE(refs[1].objective == 20.0);
    REQUIRE(refs[1].source == kSourceB);
    const SolutionPool::SolutionPtr held = refs[1].solution;

    // Both better than 20: the obj-20 entry is evicted on the first.
    REQUIRE(pool.try_add(5.0, {5.0, 6.0}, kSourceC).accepted);
    REQUIRE(pool.try_add(1.0, {7.0, 8.0}, kSourceC).accepted);
    for (const auto& entry : pool.sorted_entries()) {
        REQUIRE(entry.objective != 20.0);
    }
    REQUIRE(*held == std::vector<double>{3.0, 4.0});

    // A refill reuses `refs`' capacity and replaces its contents.
    const auto* storage = refs.data();
    pool.entry_refs(refs);
    REQUIRE(refs.data() == storage);
    REQUIRE(refs.size() == 2);
    REQUIRE(refs[0].objective == 1.0);
    REQUIRE(*refs[0].solution == std::vector<double>{7.0, 8.0});
}

// Also run under ThreadSanitizer by `mip_heuristics_pool_tsan_tests`.
TEST_CASE("SolutionPool: concurrent offers, reads and restarts on pointer entries",
          "[pool][thread-safety]") {
    constexpr int kNcol = 32;
    SolutionPool pool(/*capacity=*/6);
    pool.set_integer_mask(std::vector<bool>(kNcol, true));

    constexpr int kNumThreads = 4;
    constexpr int kOpsPerThread = 300;
    // Catch2 assertions are not thread-safe: workers count what they see,
    // and the test asserts on the count after the join.
    std::atomic<int> violations{0};
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&pool, &violations, t]() {
            auto check = [&](bool ok) {
                if (!ok) {
                    violations.fetch_add(1, std::memory_order_relaxed);
                }
            };
            const auto offers = make_offers(kOpsPerThread, kNcol, 100 + t);
            Rng rng(42 + t);
            std::vector<SolutionPool::EntryRef> refs;
            std::vector<double> buf;
            for (const auto& offer : offers) {
                if (t % 2 == 0) {
                    static_cast<void>(pool.try_add(offer.obj, offer.sol, kSourceA));
                } else {
                    static_cast<void>(pool.try_add(
                        offer.obj, std::make_shared<const std::vector<double>>(offer.sol),
                        kSourceB));
                }
                pool.entry_refs(refs);
                for (size_t i = 1; i < refs.size(); ++i) {
                    check(refs[i - 1].objective <= refs[i].objective);
                }
                for (const auto& ref : refs) {
                    check(ref.solution->size() == static_cast<size_t>(kNcol));
                }
                if (pool.copy_best(buf)) {
                    check(buf.size() == static_cast<size_t>(kNcol));
                }
                if (pool.get_restart(rng, buf)) {
                    check(buf.size() == static_cast<size_t>(kNcol));
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    REQUIRE(violations.load() == 0);

    const auto entries = pool.sorted_entries();
    REQUIRE(entries.size() == 6);
    for (size_t i = 1; i < entries.size(); ++i) {
        REQUIRE(entries[i - 1].objective <= entries[i].objective);
    }
}
