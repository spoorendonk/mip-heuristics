#pragma once

class HighsMipSolver;

namespace portfolio {

// Run presolve-based portfolio (pre-root): FPR, LocalMIP, FJ arms.
void run_presolve(HighsMipSolver& mipsolver);

}  // namespace portfolio
