#pragma once

class HighsMipSolver;

namespace portfolio {

// Run presolve-based portfolio (pre-root): FPR, LocalMIP arms.
void run_presolve(HighsMipSolver& mipsolver);

// Run LP-based portfolio (B&B dive): ScyllaFPR arm.
void run_lp_based(HighsMipSolver& mipsolver);

}  // namespace portfolio
