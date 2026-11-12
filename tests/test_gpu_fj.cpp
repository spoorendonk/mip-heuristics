// The GPU feasibility jump's model (#261).  The host half runs in every
// tree: `HostModel` must hand the device the constraints and bounds
// `FjWorker` hands upstream FJ.  The `[gpu]` cases, compiled only in a
// `MIP_HEURISTICS_CUDA` tree, compare the device's row activities with the
// host's; they fail, never skip, on a machine with no usable device.

#include "core_test_support.h"
#include "gpu/host_model.h"
#include "Highs.h"
#include "test_paths.h"

#ifdef MIP_HEURISTICS_HAS_GPU_FJ
#include "gpu/device_model.h"

#include <string>
#endif

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using namespace core_test;

namespace {

constexpr std::array<const char*, 4> kInstances = {"egout.mps", "flugpl.mps", "bell5.mps",
                                                   "p0548.mps"};

void read(Highs& highs, const char* instance) {
    highs.setOptionValue("output_flag", false);
    REQUIRE(highs.readModel(instances_dir() + "/" + instance) == HighsStatus::kOk);
}

}  // namespace

TEST_CASE("gpu_fj: the host model expands rows as FjWorker does", "[gpu_fj]") {
    for (const char* instance : kInstances) {
        INFO(instance);
        Highs highs;
        read(highs, instance);
        const Setup s(highs.getLp(), highs.getOptions().log_options);
        const ProblemView& p = s.problem;
        const gpu_fj::HostModel host(p);
        const gpu_fj::GpuFjModel m = host.pod();
        REQUIRE(host.legal_bounds());
        REQUIRE(m.ncol == p.ncol);
        REQUIRE(m.nrow == p.nrow);
        REQUIRE(std::cmp_equal(m.nnz, p.nnz));

        // Every finite row bound is one constraint, `>=` before `<=`, in row
        // order.
        std::int32_t con = 0;
        for (HighsInt row = 0; row < p.nrow; ++row) {
            const double lower = (*p.row_lower)[row];
            const double upper = (*p.row_upper)[row];
            if (std::isfinite(lower)) {
                REQUIRE(con < m.ncon);
                REQUIRE(m.con_row[con] == row);
                REQUIRE(m.con_sense[con] == gpu_fj::ConSense::kGte);
                REQUIRE(m.con_rhs[con] == lower);
                ++con;
            }
            if (std::isfinite(upper)) {
                REQUIRE(con < m.ncon);
                REQUIRE(m.con_row[con] == row);
                REQUIRE(m.con_sense[con] == gpu_fj::ConSense::kLte);
                REQUIRE(m.con_rhs[con] == upper);
                ++con;
            }
        }
        REQUIRE(con == m.ncon);

        for (HighsInt col = 0; col < p.ncol; ++col) {
            const bool is_int = (*p.integrality)[col] != HighsVarType::kContinuous;
            REQUIRE(m.col_is_int[col] == (is_int ? 1 : 0));
            if (is_int) {
                REQUIRE(m.col_lower[col] == std::ceil((*p.col_lower)[col] - p.feastol));
                REQUIRE(m.col_upper[col] == std::floor((*p.col_upper)[col] + p.feastol));
            } else {
                REQUIRE(m.col_lower[col] == (*p.col_lower)[col]);
                REQUIRE(m.col_upper[col] == (*p.col_upper)[col]);
            }
        }
    }
}

TEST_CASE("gpu_fj: an empty integer domain is not legal for FJ", "[gpu_fj]") {
    Highs highs;
    read(highs, "egout.mps");
    const Setup s(highs.getLp(), highs.getOptions().log_options);
    // `make_problem` refuses such a column itself, so the view is edited
    // after it: a caller building its own view could hand one over.
    ProblemView p = s.problem;
    HighsInt col = 0;
    while ((*p.integrality)[col] == HighsVarType::kContinuous) {
        ++col;
    }
    std::vector<double> lower = *p.col_lower;
    std::vector<double> upper = *p.col_upper;
    // Rounded inward, [0.2, 0.8] holds no integer.
    lower[col] = 0.2;
    upper[col] = 0.8;
    p.col_lower = &lower;
    p.col_upper = &upper;
    const gpu_fj::HostModel host(p);
    REQUIRE_FALSE(host.legal_bounds());
}

#ifdef MIP_HEURISTICS_HAS_GPU_FJ
TEST_CASE("gpu_fj: device row activities equal the host's", "[gpu]") {
    std::string error;
    const double init = gpu_fj::init_context_seconds(error);
    INFO(error);
    REQUIRE(init >= 0.0);
    WARN("CUDA context init: " << init << " s");

    for (const char* instance : kInstances) {
        INFO(instance);
        Highs highs;
        read(highs, instance);
        const Setup s(highs.getLp(), highs.getOptions().log_options);
        const ProblemView& p = s.problem;
        const gpu_fj::HostModel host(p);
        const gpu_fj::GpuFjModel m = host.pod();
        auto device = gpu_fj::DeviceModel::upload(m, error);
        INFO(error);
        REQUIRE(device != nullptr);

        // A point with no special structure: inside each column's bounds,
        // off their midpoint, and nonzero wherever a bound allows.
        std::vector<double> x(static_cast<size_t>(m.ncol));
        for (std::int32_t j = 0; j < m.ncol; ++j) {
            const double up = m.col_upper[j];
            double lo = -10.0;
            if (std::isfinite(m.col_lower[j])) {
                lo = m.col_lower[j];
            } else if (std::isfinite(up)) {
                lo = up - 10.0;
            }
            const double hi = std::isfinite(up) ? up : lo + 10.0;
            const double frac = 0.3 + (0.4 * ((j * 7919) % 13) / 13.0);
            x[j] = lo + ((hi - lo) * frac);
        }
        std::vector<double> want(static_cast<size_t>(m.nrow));
        for (std::int32_t i = 0; i < m.nrow; ++i) {
            double sum = 0.0;
            for (std::int32_t k = m.row_start[i]; k < m.row_start[i + 1]; ++k) {
                sum += m.row_value[k] * x[m.row_index[k]];
            }
            want[i] = sum;
        }
        std::vector<double> got(static_cast<size_t>(m.nrow), std::nan(""));
        REQUIRE(device->row_activity(x, got, error));
        // Bitwise: the same additions in the same order, FMA off both sides.
        REQUIRE(got == want);
    }
}
#endif
