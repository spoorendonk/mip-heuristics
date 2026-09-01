#pragma once

// Deliberately its own header, not folded into fpr_lp.h (issue #128).
//
// The patch inserts `#include "fpr_lp.h"` directly into HiGHS's own
// `mip/HighsMipSolver.cpp`, which is compiled as part of the fetched `highs`
// CMake target at `CMAKE_CXX_STANDARD 11` (HiGHS's own CMakeLists.txt) — not
// this project's C++23.  `fpr_lp.h` is kept to a bare forward declaration of
// `HighsMipSolver` for exactly that reason: anything it pulls in transitively
// reaches a C++11 translation unit.  `fpr_strategies.h` (needed for
// `NamedConfig`) drags in `heuristic_common.h`, which uses `inline
// constexpr` variables (C++17) and multi-statement `constexpr` functions
// (C++14) — both hard errors under `-std=c++11` (confirmed empirically:
// `HighsMipSolver.cpp` failed with "body of constexpr function ... not a
// return-statement" and "inline variables are only available with
// -std=c++17" the one time this content briefly lived in fpr_lp.h). So the
// test-facing arm/reference-class types live here instead, included only by
// `fpr_lp.cpp` (already a C++23 TU, already including fpr_strategies.h) and
// by `tests/test_fpr_lp.cpp` — never by fpr_lp.h.

#include "fpr_strategies.h"

#include <vector>

namespace fpr_lp {

// Which LP reference vector an LP-dependent FPR arm's strategy is defined
// against (Salvagnin, Roberti, Fischetti, MPC 17:111-139, 2025, Sect. 4.1,
// Sect. 6.3): the zero-obj analytic center (paper's Class 2), the zero-obj
// simplex vertex (Class 3a), or the full-obj LP solution (Class 3b).
enum class LpRefClass {
    kAnalyticCenter,
    kZeroObjVertex,
    kFullObjLp,
};

// Test hook: one entry of the LP-arm portfolio, as `lp_arm_table()` reports
// it (issue #128).
struct LpArmInfo {
    const char* name = nullptr;
    NamedConfig config{};
    LpRefClass ref_class = LpRefClass::kAnalyticCenter;
};

// Test hook: the full LP-arm portfolio, read from the single source of
// truth (`kLpArmTable` in fpr_lp.cpp) that binds each arm's strategy to a
// `ref_class`.  Exists so a test can assert every arm's `ref_class` against
// what its own strategy needs — e.g. `cliques2`'s ranking
// (`fpr_var_order.cpp`'s `rank_cliques2`) reads its LP reference for both
// the clique-tightness test and the per-clique ranking, so an arm using it
// must report `kZeroObjVertex` — without duplicating the mapping
// `build_setup` itself uses: both read `kLpArmTable`, so the arm-to-class
// assignment cannot drift apart again the way #128 found it.
//
// This covers only *half* of #128's mechanism, though: which reference
// vector a `ref_class` names — the other half — is a second mapping, owned
// by `select_ref` below, and a cold review of the first version of this
// fix found it untested: reprogramming `select_ref`'s `kZeroObjVertex` case
// to return the full-obj LP pointer reintroduced #128's bug and the whole
// suite still passed, because nothing exercised that function in
// isolation. `select_ref` is what closes that gap.
std::vector<LpArmInfo> lp_arm_table();

// Selects the LP reference vector `ref_class` names, out of the three
// `build_setup` computes per dispatch (`ac`: zero-obj analytic center,
// `zv`: zero-obj simplex vertex, `lp`: full-obj LP solution). A standalone,
// pure function — not a switch inlined at the call site — precisely so
// `tests/test_fpr_lp.cpp` can call it directly for every `LpRefClass`
// enumerator and check the returned pointer against the vector its name
// promises, closing the gap `lp_arm_table()`'s comment above describes.
//
// No default case, and no pointer pre-assigned before the switch: an
// enumerator this function doesn't handle must not silently resolve to a
// pointer that looks like a legitimate answer (`lp_ptr` used to be exactly
// that plausible-looking wrong default). `-Werror=switch` on the
// `mip_heuristics` target (CMakeLists.txt) turns a case an edit forgets to
// add into a compile error, for exactly this switch and every other one in
// our own code — the same protection the `select_ref`-level test gives the
// enumerators that already exist, extended to one added later.
const double* select_ref(LpRefClass ref_class, const double* ac, const double* zv,
                         const double* lp);

}  // namespace fpr_lp
