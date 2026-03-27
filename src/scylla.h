#pragma once

#include <cstddef>

class HighsMipSolver;

namespace scylla {

// Run N parallel ScyllaFPR restarts with LP-guided scoring (B&B dive).
void run(HighsMipSolver &mipsolver, size_t max_effort);

} // namespace scylla
