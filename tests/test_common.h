#pragma once

// Shared helpers for the Catch2 test suite.
//
// This header gathers the small number of helpers that more than one
// per-topic test translation unit needs.  It intentionally keeps the
// surface area small — anything used by a single file stays in that
// file as a file-local helper.

#include "Highs.h"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// Path to the HiGHS-provided `check/instances/` directory, injected by
// CMake via `INSTANCES_DIR`.  Defined inline so every translation unit
// that includes this header gets its own const reference.
inline const std::string kInstancesDir = INSTANCES_DIR;

// `setOptionValue` that fails the test when the option does not exist.
// HiGHS returns `kError` for an unknown name and otherwise does nothing,
// so a test that sets a renamed option silently keeps the solve at its
// defaults instead of the configuration it asked for.  That matters most
// where an option is *disabling* something: the test then measures a
// completely different solve and can still pass.
template <typename T>
inline void require_option(Highs& h, const std::string& name, T value) {
    REQUIRE(h.setOptionValue(name, value) == HighsStatus::kOk);
}

// Solve `inst` with the requested (portfolio × opportunistic) cell of
// the execution matrix and return the final objective.  Used by the
// mode-matrix cross-heuristic parity tests.
inline double solve_mode(const char* inst, bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}

// Solve `inst` and return every log message HiGHS emitted, in order.
// `configure` receives the `Highs` object after logging is wired up and
// before the model is read, so it can set any option the caller needs.
//
// A callback is the only way to observe HiGHS's MIP display and its
// `log_dev_level=3` traces — neither exists in any info field — so this is
// the capture primitive the whole suite shares.  Declaration order below is
// load-bearing: `capture` outlives `h`, so the callback can never fire
// against a destroyed buffer.
template <typename Configure>
inline std::vector<std::string> solve_capturing_log(const char* inst, Configure&& configure) {
    struct LogCapture {
        std::mutex mtx;
        std::vector<std::string> lines;
    };
    LogCapture capture;

    Highs h;
    h.setOptionValue("output_flag", true);
    h.setOptionValue("log_to_console", false);

    auto log_cb = [](int callback_type, const std::string& message,
                     const HighsCallbackOutput* /*out*/, HighsCallbackInput* /*in*/,
                     void* user_data) {
        if (callback_type != kCallbackLogging) {
            return;
        }
        auto* cap = static_cast<LogCapture*>(user_data);
        std::lock_guard<std::mutex> lock(cap->mtx);
        cap->lines.emplace_back(message);
    };

    REQUIRE(h.setCallback(HighsCallbackFunctionType(log_cb), &capture) == HighsStatus::kOk);
    REQUIRE(h.startCallback(kCallbackLogging) == HighsStatus::kOk);
    std::forward<Configure>(configure)(h);
    REQUIRE(h.readModel(kInstancesDir + "/" + inst) == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);

    std::lock_guard<std::mutex> lock(capture.mtx);
    return capture.lines;
}

// Every solution-source code HiGHS printed on a MIP display line of
// `lines`, in emission order.
//
// The source code is the single character HiGHS's `printSolutionSourceKey`
// maps a `kSolutionSource*` value to: `A` FPR, `D` fpr_lp, `M` LocalMIP,
// `G` Scylla, `J` FeasibilityJump, plus upstream's own codes.  Asserting on
// these is the only way a test can tell *which* heuristic found a solution.
//
// Display line format is `" %s %7s ..."` with the one-character code at
// offset 1 (`HighsMipSolverData::printDisplayLine`).  `kSolutionSourceNone`
// and `kSolutionSourceCleanup` both render as a space and are skipped —
// they carry no attribution.
inline std::string source_codes(const std::vector<std::string>& lines) {
    std::string codes;
    for (const auto& line : lines) {
        if (line.size() >= 3 && line[0] == ' ' && line[2] == ' ' && line[1] != ' ') {
            codes.push_back(line[1]);
        }
    }
    return codes;
}

// Convenience composition of the two helpers above.
template <typename Configure>
inline std::string solve_capturing_source_codes(const char* inst, Configure&& configure) {
    return source_codes(solve_capturing_log(inst, std::forward<Configure>(configure)));
}

// Whether `lines` carries a `[Sequential] heur=<heur> effort=<N>` trace
// with a non-zero effort, i.e. whether that heuristic actually ran and
// consumed budget.  Requires `log_dev_level=3` on the solve.
//
// This is a weaker signal than a solution-source code — it proves the
// heuristic ran, not that it found anything — so prefer `source_codes`
// where the code is reliably emitted.
inline bool heuristic_reported_effort(const std::vector<std::string>& lines,
                                      const std::string& heur) {
    const std::string tag = "[Sequential] heur=" + heur + " effort=";
    for (const auto& line : lines) {
        const auto pos = line.find(tag);
        if (pos != std::string::npos &&
            std::strtoull(line.c_str() + pos + tag.size(), nullptr, 10) > 0) {
            return true;
        }
    }
    return false;
}

// Solve flugpl with every custom heuristic disabled in the requested
// (portfolio × opportunistic) cell — verifies none of the mode paths
// blocks HiGHS's built-in B&B fallback.
inline double solve_mode_no_heuristics(bool portfolio, bool opp) {
    Highs h;
    h.setOptionValue("output_flag", false);
    h.setOptionValue("mip_heuristic_portfolio", portfolio);
    h.setOptionValue("mip_heuristic_opportunistic", opp);
    h.setOptionValue("mip_heuristic_run_fpr", false);
    h.setOptionValue("mip_heuristic_run_local_mip", false);
    h.setOptionValue("mip_heuristic_run_feasibility_jump", false);
    h.setOptionValue("mip_heuristic_run_scylla", false);
    REQUIRE(h.readModel(std::string(INSTANCES_DIR) + "/flugpl.mps") == HighsStatus::kOk);
    REQUIRE(h.run() == HighsStatus::kOk);
    double obj;
    h.getInfoValue("objective_function_value", obj);
    return obj;
}
