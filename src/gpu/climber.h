#pragma once

// A batch of GPU feasibility-jump climbers (#210, #262, #272), each with its
// own state: the point, every constraint's LHS and weight, the violated set,
// the objective weight, and every variable's jump value and score with the
// good set built from them.  `init` builds a climber's state as upstream FJ's
// `FeasibilityJumpSolver::init` does (`resetIncumbent`, then `resetMoves` for
// each variable in turn), with both #139 fixes; `run` takes upstream's
// `solve` steps (#263) for every climber in one launch, one persistent thread
// block each, up to the next point upstream calls its callback.  The blocks
// share only the read-only model, so a climber's steps and bytes are the ones
// it would take alone.  Like `device_model.h`, it names no CUDA type, and
// every call reports a CUDA failure through `error` and returns false (or
// null).

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace gpu_fj {

class DeviceModel;

// Upstream's `CALLBACK_EFFORT`, `kFjCallbackEffort` on the CPU side: the
// effort between two of `solve`'s callbacks, and so a `Climber::run` chunk.
inline constexpr std::uint64_t kCallbackEffort = 500000;

// Climbers per streaming multiprocessor in `default_climbers`' batch.
inline constexpr std::int32_t kClimbersPerSm = 1;

// The device's multiprocessor count times `kClimbersPerSm`, or 0 with
// `error` set.
[[nodiscard]] std::int32_t default_climbers(std::string& error);

class Climber {
public:
    // `climbers` climbers on `model`, which must outlive them.  The device
    // memory they need is checked against the free memory first, and a
    // batch that does not fit is refused: null, with `error` set.  One batch
    // serves one thread at a time.
    static std::unique_ptr<Climber> create(const DeviceModel& model, std::int32_t climbers,
                                           std::string& error);
    ~Climber();
    Climber(const Climber&) = delete;
    Climber& operator=(const Climber&) = delete;

    [[nodiscard]] std::int32_t count() const;

    // Upstream's `init` of climber `k` from the point `start` (`ncol` values
    // within the model's bounds): every weight 1, the objective weight 0,
    // LHS and the violated set, then each variable's jump value, score and
    // good-set membership.  Every counter restarts, the effort at `init`'s
    // `nNonzeros` plus each `resetMoves`'s cells, and `seed` seeds the
    // climber's random stream.  The other climbers are left as they are.
    [[nodiscard]] bool init(std::int32_t k, std::span<const double> start, std::uint64_t seed,
                            std::string& error);

    // Where `run` stopped a climber: upstream's callback points.
    struct Chunk {
        std::uint64_t total_effort = 0;
        // `FJStatus::effortSinceLastImprovement`.
        std::uint64_t effort_since_improvement = 0;
        // The climber stopped at a new best feasible point, `objective`;
        // `read_best` copies it.
        bool improved = false;
        double objective = 0.0;
        // No step is left to take: the model has no column, or a selection
        // found no variable (v1.16 `solve`'s `UINT_MAX` break); every later
        // `run` stops it at once.
        bool finished = false;
    };

    // Upstream's `solve` loop for every climber from where the last call
    // stopped it: each step selects a variable, at a local minimum bumping
    // the weights, and jumps it, until the effort since the last callback
    // passes `kFjCallbackEffort` or the point is a new best feasible one by
    // upstream's relative-improvement test, or `max_steps` steps are taken.
    // `chunks` gets one `Chunk` per climber.  Every climber must have had an
    // `init`.
    [[nodiscard]] bool run(std::uint64_t max_steps, std::span<Chunk> chunks, std::string& error);

    // Climber `k`'s best feasible point found since its `init`, `ncol`
    // values; empty while there is none.
    [[nodiscard]] bool read_best(std::int32_t k, std::vector<double>& x, std::string& error) const;

    // Sets climber `k`'s objective weight and recomputes every variable's
    // move under its current weights and point, as `init`'s `resetMoves`
    // loop does.  Upstream changes the objective weight only at a local
    // minimum and never recomputes every move after; the tests use this to
    // exercise the objective term, which `init`'s zero weight leaves out.
    [[nodiscard]] bool reset_moves(std::int32_t k, double objective_weight, std::string& error);

    // A host copy of a climber's state, in upstream's layout: `violated` and
    // `good` in the order upstream's lists hold them.
    struct State {
        std::vector<double> x;
        std::vector<double> lhs;
        std::vector<double> weight;
        std::vector<std::int32_t> violated;
        std::vector<std::int32_t> violated_pos;  // -1 when satisfied
        std::vector<double> jump;
        std::vector<double> score;
        std::vector<std::int32_t> good;
        std::vector<std::int32_t> good_pos;  // -1 when not good
        double objective_weight = 0.0;
        double weight_increment = 0.0;
        double objective = 0.0;  // `incumbentObjective`
        std::uint64_t total_effort = 0;
        std::uint64_t steps = 0;
        std::uint64_t bumps = 0;
        std::int32_t last_var = -1;  // the variable the last step moved
    };
    [[nodiscard]] bool read(std::int32_t k, State& state, std::string& error) const;

private:
    struct Impl;
    explicit Climber(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace gpu_fj
