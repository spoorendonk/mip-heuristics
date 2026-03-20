#pragma once

class HighsMipSolver;

namespace portfolio {

// Run presolve-based portfolio (pre-root): FPR, LocalMIP, FJ arms.
void run_presolve(HighsMipSolver& mipsolver);

// Run N parallel ScyllaFPR restarts with score perturbation (B&B dive).
void run_scylla_parallel(HighsMipSolver& mipsolver);

}  // namespace portfolio
