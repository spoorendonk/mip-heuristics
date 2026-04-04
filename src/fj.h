#pragma once
#include <cstddef>
class HighsMipSolver;
namespace fj {
// Returns true if FJ detected proven infeasibility (empty integer domain).
bool run(HighsMipSolver &mipsolver, size_t max_effort);
}
