#pragma once

// xoshiro256++ — fast, high-quality PRNG used throughout the MIP heuristics.
//
// Replaces std::mt19937 (previously ~23% of instruction references in
// callgrind's Portfolio bell5 trace). The C++ standard library's Mersenne
// Twister has a large 2.5 KB state, slow refill (`_M_gen_rand`), and
// expensive double generation via `std::generate_canonical`. xoshiro256++
// has 32 bytes of state, produces 64 bits per call with just a few
// arithmetic ops, and passes BigCrush/PractRand.
//
// Reference implementation: https://prng.di.unimi.it/xoshiro256plusplus.c
// (public domain, translated faithfully to C++23). SplitMix64 is used
// internally to expand a single uint64_t seed to the required 256 bits of
// state — this is the standard seeding recipe recommended by the same
// authors (https://prng.di.unimi.it/splitmix64.c).
//
// The class satisfies the C++ UniformRandomBitGenerator named requirement,
// so it plugs into `std::uniform_int_distribution`, `std::generate_canonical`,
// `std::gamma_distribution`, `std::shuffle`, etc. unchanged.

#include <cstdint>
#include <limits>

class Xoshiro256PlusPlus {
public:
    using result_type = std::uint64_t;

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    // Construct from a single 64-bit seed.  Uses SplitMix64 to expand into
    // the 256-bit state.  SplitMix64 is a full-period generator, so distinct
    // seeds give distinct state sequences even for tiny seeds like 0 or 1.
    explicit Xoshiro256PlusPlus(std::uint64_t seed = 0) noexcept { reseed(seed); }

    void reseed(std::uint64_t seed) noexcept {
        std::uint64_t z = seed;
        s_[0] = splitmix64(z);
        s_[1] = splitmix64(z);
        s_[2] = splitmix64(z);
        s_[3] = splitmix64(z);
        // SplitMix64 can emit an all-zero word exceedingly rarely; but the
        // xoshiro state must not be all-zero overall.  A single non-zero
        // word is sufficient — the four draws above already guarantee this
        // for every representable seed, but assert-by-construction is cheap.
        if ((s_[0] | s_[1] | s_[2] | s_[3]) == 0) {
            s_[0] = 0x9E3779B97F4A7C15ULL;  // golden-ratio fallback
        }
    }

    result_type operator()() noexcept {
        const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

private:
    static constexpr std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }

    // SplitMix64: reference https://prng.di.unimi.it/splitmix64.c.
    // `z` is advanced in place; each call returns the next output.
    static std::uint64_t splitmix64(std::uint64_t &z) noexcept {
        z += 0x9E3779B97F4A7C15ULL;
        std::uint64_t x = z;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    std::uint64_t s_[4];
};

// Project-wide PRNG alias.  Every heuristic uses `Rng` in preference to
// `std::mt19937`; defined here so including `rng.h` is sufficient.
using Rng = Xoshiro256PlusPlus;
