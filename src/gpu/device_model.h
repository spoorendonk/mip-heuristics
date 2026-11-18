#pragma once

// A device copy of a `GpuFjModel` (#261), owned for the life of a GPU FJ
// run and read by its climbers (`climber.h`).  Included from `.cu` and host
// files alike, so it names no CUDA type: the device buffers sit behind
// `Impl`.  Every call reports a CUDA failure through `error` and returns
// false (or null) rather than throwing, so a caller can fall back to the
// CPU workers.

#include "gpu_fj_model.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace gpu_fj {

// The most cells the related-variable table may hold (#277): for each
// column, its cells' constraints' other columns, with each one's cells in
// those constraints.  It grows with the sum of squared row lengths, so a
// model past the cap, or one whose table does not fit on the device, runs
// without it: a step then scans each neighbour's whole column.
inline constexpr std::int64_t kRelatedMaxCells = std::int64_t{1} << 24U;

// How many cells the table's build gathers between looks at `upload`'s stop
// predicate (#304).  It looks between a column's rows, so one dense column
// cannot hold a stop back; only that column's sort runs without a look.
inline constexpr std::size_t kRelatedPollCells = 4096;

class DeviceModel {
public:
    // Copies `host` to the device, with its related-variable table when that
    // holds at most `max_related_cells` cells and fits, or returns null with
    // `error` set.  Building that table is the one long host step, so it
    // polls `stopped` as it goes: once that returns true the upload gives
    // up and returns null with `error` empty (#304).
    static std::unique_ptr<DeviceModel> upload(const GpuFjModel& host, std::string& error,
                                               std::int64_t max_related_cells = kRelatedMaxCells,
                                               const std::function<bool()>& stopped = {});
    ~DeviceModel();
    DeviceModel(const DeviceModel&) = delete;
    DeviceModel& operator=(const DeviceModel&) = delete;

    [[nodiscard]] std::int32_t ncol() const;
    [[nodiscard]] std::int32_t ncon() const;
    // Whether the related-variable table was uploaded.
    [[nodiscard]] bool has_related_table() const;

private:
    // Reads the device arrays.
    friend class Climber;
    struct Impl;
    explicit DeviceModel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Seconds to create this process's CUDA context on the current device, or a
// negative value with `error` set.  Measured once, the first call in a
// process; a later call finds the context made and returns its own (small)
// cost.
double init_context_seconds(std::string& error);

}  // namespace gpu_fj
