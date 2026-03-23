#pragma once

class HighsMipSolver;

namespace scylla {

// Run N parallel ScyllaFPR restarts with LP-guided scoring (B&B dive).
void run(HighsMipSolver& mipsolver);

}  // namespace scylla
