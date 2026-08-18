# Patch script for HiGHS: insert heuristic call sites and options
# Called by FetchContent PATCH_COMMAND
# Idempotent: safe to run multiple times.

# ── The clean-rebuild incantation, stated once ───────────────────────────────
# Nearly every failure below needs the same remedy for the same reason: this
# script decides "already applied?" by searching for text it previously
# inserted, so a tree patched by an older version of the script cannot be
# rewritten in place.  The rule and its rationale live in CONTRIBUTING.md
# under "The clean-rebuild rule"; this string keeps the command itself in
# front of whoever hit the error, and keeps nineteen copies of it from
# drifting apart.
string(CONCAT CLEAN_REBUILD
    "Clean the HiGHS source tree and rebuild "
    "(see CONTRIBUTING.md, \"The clean-rebuild rule\"): "
    "rm -rf build/_deps/highs-src build/_deps/highs-subbuild build/CMakeCache.txt "
    "&& cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build")

# Detect source layout: v1.13+ uses highs/ subdirectory
if(EXISTS "${SOURCE_DIR}/highs/mip")
    set(MIP_DIR "${SOURCE_DIR}/highs/mip")
    set(LP_DATA_DIR "${SOURCE_DIR}/highs/lp_data")
    set(IO_DIR "${SOURCE_DIR}/highs/io")
else()
    set(MIP_DIR "${SOURCE_DIR}/src/mip")
    set(LP_DATA_DIR "${SOURCE_DIR}/src/lp_data")
    set(IO_DIR "${SOURCE_DIR}/src/io")
endif()

# ── Patch HighsIO.cpp: self-identify patched binaries in the log header ──
# A patched build prints the same "Running HiGHS <version> (git hash: ...)"
# banner as an unpatched build of the same tag, so logs from the two are
# otherwise indistinguishable when no custom option is set — a mislabeled
# benchmark results directory would be undetectable.  Emit one marker line
# right after the banner, unconditionally (highsLogHeader is the single
# banner site used by both the CLI and the library API).  Keep the leading
# token "mip-heuristics patch" stable: bench scripts and humans grep for it.
file(READ "${IO_DIR}/HighsIO.cpp" IO_CONTENT)
string(FIND "${IO_CONTENT}" "mip-heuristics patch" _marker_found)
if(_marker_found EQUAL -1)
    string(REPLACE
      "  highsLogUser(log_options, HighsLogType::kInfo, \"%s\\n\",\n               HighsExternalApi::thirdPartyNoticeHeader().c_str());"
      "  highsLogUser(log_options, HighsLogType::kInfo, \"%s\\n\",\n               HighsExternalApi::thirdPartyNoticeHeader().c_str());\n\n  highsLogUser(log_options, HighsLogType::kInfo,\n               \"mip-heuristics patch active (custom MIP presolve heuristics; \"\n               \"spoorendonk/mip-heuristics)\\n\");"
      IO_CONTENT "${IO_CONTENT}")

    string(FIND "${IO_CONTENT}" "mip-heuristics patch" _marker_check)
    if(_marker_check EQUAL -1)
        message(FATAL_ERROR
            "HighsIO.cpp patch failed: the thirdPartyNoticeHeader anchor in "
            "highsLogHeader no longer matches (upstream reformat?). "
            "Update the marker patch in third_party/highs_patch/apply_patch.cmake. "
            "${CLEAN_REBUILD}")
    endif()

    file(WRITE "${IO_DIR}/HighsIO.cpp" "${IO_CONTENT}")
    message(STATUS "Applied patched-binary log marker to HighsIO.cpp")
else()
    message(STATUS "Log marker already applied to HighsIO.cpp, skipping")
endif()

# ── Patch HighsOptions.h: register custom heuristic options ──
file(READ "${LP_DATA_DIR}/HighsOptions.h" OPTIONS_CONTENT)

# Defensive check: reject a tree patched by an older version of this script.
#
# Every per-file idempotency check below keys on text this script itself
# inserted, so none of them can tell a tree patched by an *earlier* version
# from a current one: the sentinel is present in both, the block is skipped,
# and the stale layout survives into the build as a silently wrong option set.
#
# Two probes, because the option set has been renamed out from under the
# marker once already:
#   * the marker itself, for any tree patched since it was introduced;
#   * the identifiers this script *used* to insert and no longer does, for
#     trees older than the marker.  A retired name in the tree means the old
#     block was applied and its text is still there — the new block would be
#     appended alongside it and the failure would surface as an unrelated
#     compile error in mode_dispatch.cpp.  None of these is an upstream
#     identifier, and none is a substring of anything this script inserts
#     (`mip_heuristic_preset` does not occur inside
#     `mip_heuristic_presolve_effort` — they diverge at `prese`/`preso`).
#
# Bump PATCH_VERSION whenever any inserted text changes, and add any
# identifier this script stops inserting to the retired list.
set(PATCH_VERSION "6")
string(FIND "${OPTIONS_CONTENT}" "mip-heuristics patch version ${PATCH_VERSION}" _patch_version_found)
if(_patch_version_found EQUAL -1)
    # No `;` in an entry: set() builds a cmake list and would split on it,
    # truncating the explanation and adding a phantom probe for the tail.
    set(_retired_options
        "mip_heuristic_preset:renamed to mip_heuristic_suite"
        "mip_heuristic_run_fpr:folded into mip_heuristic_suite"
        "mip_heuristic_run_local_mip:folded into mip_heuristic_suite"
        "mip_heuristic_run_scylla:folded into mip_heuristic_suite"
        "mip_heuristic_portfolio:removed with the Thompson portfolio"
        "mip_heuristic_opportunistic:removed, opportunistic is the only parallel mode")
    foreach(_entry IN LISTS _retired_options)
        string(REGEX REPLACE ":.*$" "" _retired_ident "${_entry}")
        string(REGEX REPLACE "^[^:]*:" "" _retired_why "${_entry}")
        string(FIND "${OPTIONS_CONTENT}" "${_retired_ident}" _retired_idx)
        if(NOT _retired_idx EQUAL -1)
            message(FATAL_ERROR
                "HighsOptions.h still carries '${_retired_ident}' (${_retired_why}). "
                "The tree was patched by an older version of apply_patch.cmake; "
                "an in-place rewrite is not safe. "
                "${CLEAN_REBUILD}")
        endif()
    endforeach()
    string(FIND "${OPTIONS_CONTENT}" "mip-heuristics patch version" _patch_marker_found)
    if(NOT _patch_marker_found EQUAL -1)
        message(FATAL_ERROR
            "HighsOptions.h was patched by an older version of apply_patch.cmake "
            "(expected 'mip-heuristics patch version ${PATCH_VERSION}'). "
            "The inserted text has changed since; an in-place rewrite is not safe. "
            "${CLEAN_REBUILD}")
    endif()
endif()

# ── Add the mip_heuristic_suite string option ──
# One single-valued option selects which custom heuristics run:
# off | fj | fpr | local_mip | scylla | all (default "all").  It replaced the
# three mip_heuristic_run_* bools and mip_heuristic_preset in #93.
#
# All three insertions anchor on *upstream's* mip_heuristic_run_shifting
# text.  Anchoring on our own inserted text is what made the previous option
# blocks a chain: deleting one silently dropped the next from the build.
string(FIND "${OPTIONS_CONTENT}" "mip_heuristic_suite" _suite_found)
if(_suite_found EQUAL -1)
    # Member variable: insert after mip_heuristic_run_shifting.  The version
    # marker rides along on this line — it is the stale-tree probe above.
    string(REPLACE
      "bool mip_heuristic_run_shifting;\n"
      "bool mip_heuristic_run_shifting;\n  std::string mip_heuristic_suite;  // mip-heuristics patch version ${PATCH_VERSION}\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Constructor initializer list: insert after mip_heuristic_run_shifting(false),
    string(REPLACE
      "mip_heuristic_run_shifting(false),\n"
      "mip_heuristic_run_shifting(false),\n        mip_heuristic_suite(\"all\"),\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Record registration: insert after the mip_heuristic_run_shifting record block
    string(REPLACE
      "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n                                       \"Use the Shifting heuristic\", advanced,\n                                       &mip_heuristic_run_shifting, false);\n    records.push_back(record_bool);"
      "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n                                       \"Use the Shifting heuristic\", advanced,\n                                       &mip_heuristic_run_shifting, false);\n    records.push_back(record_bool);\n\n    record_string = new OptionRecordString(\"mip_heuristic_suite\",\n                                          \"Custom MIP heuristic suite: \\\"off\\\", \\\"fj\\\", \\\"fpr\\\", \\\"local_mip\\\", \\\"scylla\\\" or \\\"all\\\"\", advanced,\n                                          &mip_heuristic_suite, \"all\");\n    records.push_back(record_string);"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Sanity checks: all three insertions must land.  The failure mode is
    # silent rather than loud — if only the *record registration* REPLACE
    # misses (upstream reformats the Shifting record block it anchors on),
    # the member and ctor init are still there, so HighsOptions.h compiles;
    # the option is simply never registered, keeps its ctor default, and
    # every attempt to set it fails with no diagnostic anywhere.
    #
    # Match the member declaration *without* its trailing semicolon: cmake
    # splits a matched string containing `;` into list elements, which would
    # make list(LENGTH) report 2 for a single hit.  The `std::string ` prefix
    # is what keeps this from also matching the ctor init or the record.
    string(REGEX MATCHALL "std::string mip_heuristic_suite" _suite_member_hits "${OPTIONS_CONTENT}")
    list(LENGTH _suite_member_hits _suite_member_count)
    string(REGEX MATCHALL "mip_heuristic_suite\\(\"all\"\\)" _suite_ctor_hits "${OPTIONS_CONTENT}")
    list(LENGTH _suite_ctor_hits _suite_ctor_count)
    # string(FIND), not REGEX MATCHALL: the record text contains semicolons.
    string(FIND "${OPTIONS_CONTENT}" "OptionRecordString(\"mip_heuristic_suite\"" _suite_record_idx)
    if(NOT _suite_member_count EQUAL 1 OR NOT _suite_ctor_count EQUAL 1 OR _suite_record_idx EQUAL -1)
        message(FATAL_ERROR
            "HighsOptions.h post-patch sanity check failed for "
            "mip_heuristic_suite (member=${_suite_member_count}, "
            "ctor=${_suite_ctor_count}, record_idx=${_suite_record_idx}). "
            "Upstream HiGHS likely reformatted HighsOptions.h so one of the "
            "three mip_heuristic_run_shifting anchors no longer matches. "
            "${CLEAN_REBUILD}")
    endif()
    file(WRITE "${LP_DATA_DIR}/HighsOptions.h" "${OPTIONS_CONTENT}")
    message(STATUS "Applied mip_heuristic_suite option to HighsOptions.h")
else()
    message(STATUS "mip_heuristic_suite option already applied to HighsOptions.h, skipping")
endif()

# ── Keep mip_heuristic_effort at the vanilla default 0.05 ──
# Historical: the patch used to raise the default to 0.30 because the
# presolve heuristics were starved at 0.05.  That overloaded one option
# with two meanings (B&B LP-iteration fraction for RENS/RINS vs nnz-based
# presolve budget).  The presolve budget now has its own knob,
# mip_heuristic_presolve_effort (default 0.30, added below), and
# mip_heuristic_effort keeps upstream's exact semantics and default so a
# patched binary at default options matches vanilla's B&B heuristic
# budget.  The rewrite below downgrades in-place source trees that still
# carry the old raised default; a FATAL_ERROR fires if upstream ever
# reformats the OptionRecordDouble line and neither substring matches —
# we'd silently ship the wrong default otherwise.
file(READ "${LP_DATA_DIR}/HighsOptions.h" OPTIONS_CONTENT)
string(FIND "${OPTIONS_CONTENT}" "&mip_heuristic_effort, 0.0, 0.05, 1.0" _effort_default_found)
string(FIND "${OPTIONS_CONTENT}" "&mip_heuristic_effort, 0.0, 0.30, 1.0" _effort_patched_found)
if(NOT _effort_patched_found EQUAL -1)
    string(REPLACE
      "&mip_heuristic_effort, 0.0, 0.30, 1.0"
      "&mip_heuristic_effort, 0.0, 0.05, 1.0"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")
    file(WRITE "${LP_DATA_DIR}/HighsOptions.h" "${OPTIONS_CONTENT}")
    message(STATUS "Reverted mip_heuristic_effort default to vanilla 0.05")
elseif(NOT _effort_default_found EQUAL -1)
    message(STATUS "mip_heuristic_effort default already at vanilla 0.05, skipping")
else()
    message(FATAL_ERROR
        "HighsOptions.h post-patch sanity check failed: neither the pristine "
        "'&mip_heuristic_effort, 0.0, 0.05, 1.0' nor the legacy-patched "
        "'&mip_heuristic_effort, 0.0, 0.30, 1.0' substring was found. "
        "Upstream HiGHS likely reformatted the option-record block so the "
        "exact-string REPLACE pattern no longer matches. "
        "${CLEAN_REBUILD}")
endif()

# ── Add mip_heuristic_presolve_effort double option ──
# Effort budget multiplier for the custom presolve heuristics (FPR,
# LocalMIP, Scylla).  Split off from mip_heuristic_effort so the latter
# keeps vanilla B&B semantics; see the block above.  Default 0.30 keeps
# the presolve budget identical to the previous overloaded default.
#
# Anchored on upstream's mip_heuristic_run_shifting text, like the suite
# block above.  It used to anchor on the *preset* block's inserted text —
# including the whole multi-line record registration — so deleting that
# block in #93 would have silently dropped this option from the build.
# Both blocks insert directly after the same upstream anchor, so this one
# (running second) lands between the anchor and the suite option in the
# member list *and* the ctor initializer list, keeping the two in the same
# relative order and out of -Wreorder's way.
file(READ "${LP_DATA_DIR}/HighsOptions.h" OPTIONS_CONTENT)
string(FIND "${OPTIONS_CONTENT}" "mip_heuristic_presolve_effort" _presolve_effort_found)
if(_presolve_effort_found EQUAL -1)
    # Member variable: insert after mip_heuristic_run_shifting
    string(REPLACE
      "bool mip_heuristic_run_shifting;\n"
      "bool mip_heuristic_run_shifting;\n  double mip_heuristic_presolve_effort;\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Constructor initializer: insert after mip_heuristic_run_shifting(false),
    string(REPLACE
      "mip_heuristic_run_shifting(false),\n"
      "mip_heuristic_run_shifting(false),\n        mip_heuristic_presolve_effort(0.0),\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Record registration: insert after the mip_heuristic_run_shifting record block
    string(REPLACE
      "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n                                       \"Use the Shifting heuristic\", advanced,\n                                       &mip_heuristic_run_shifting, false);\n    records.push_back(record_bool);"
      "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n                                       \"Use the Shifting heuristic\", advanced,\n                                       &mip_heuristic_run_shifting, false);\n    records.push_back(record_bool);\n\n    record_double = new OptionRecordDouble(\n        \"mip_heuristic_presolve_effort\",\n        \"Effort budget multiplier for custom presolve heuristics\", advanced,\n        &mip_heuristic_presolve_effort, 0.0, 0.30, 1.0);\n    records.push_back(record_double);"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Sanity checks: all three insertions must land.
    string(REGEX MATCHALL "double mip_heuristic_presolve_effort" _pe_member_hits "${OPTIONS_CONTENT}")
    list(LENGTH _pe_member_hits _pe_member_count)
    string(REGEX MATCHALL "mip_heuristic_presolve_effort\\(0\\.0\\)" _pe_ctor_hits "${OPTIONS_CONTENT}")
    list(LENGTH _pe_ctor_hits _pe_ctor_count)
    string(FIND "${OPTIONS_CONTENT}" "\"mip_heuristic_presolve_effort\"" _pe_record_idx)
    if(NOT _pe_member_count EQUAL 1 OR NOT _pe_ctor_count EQUAL 1 OR _pe_record_idx EQUAL -1)
        message(FATAL_ERROR
            "HighsOptions.h post-patch sanity check failed for "
            "mip_heuristic_presolve_effort (member=${_pe_member_count}, "
            "ctor=${_pe_ctor_count}, record_idx=${_pe_record_idx}). "
            "An anchor REPLACE pattern no longer matches. "
            "${CLEAN_REBUILD}")
    endif()
    file(WRITE "${LP_DATA_DIR}/HighsOptions.h" "${OPTIONS_CONTENT}")
    message(STATUS "Applied mip_heuristic_presolve_effort option to HighsOptions.h")
else()
    message(STATUS "mip_heuristic_presolve_effort option already applied, skipping")
endif()

# ── Patch HighsMipSolverData.h: add capture overload + custom solution source enums ──
file(READ "${MIP_DIR}/HighsMipSolverData.h" MIPDATA_H)

# Add heuristic_effort_used field
string(FIND "${MIPDATA_H}" "heuristic_effort_used" _effort_field_found)
if(_effort_field_found EQUAL -1)
    string(REPLACE
      "double heuristic_effort;"
      "double heuristic_effort;\n  size_t heuristic_effort_used = 0;"
      MIPDATA_H "${MIPDATA_H}")
    file(WRITE "${MIP_DIR}/HighsMipSolverData.h" "${MIPDATA_H}")
    message(STATUS "Applied heuristic_effort_used field to HighsMipSolverData.h")
else()
    message(STATUS "heuristic_effort_used field already applied, skipping")
endif()

# ── Cannibalization instrumentation counters (issue #95) ──
# Four patch-added fields with no upstream reader, plus one for our own
# presolve wall time.  They exist to answer the closeout question "do the
# custom presolve heuristics starve HiGHS's own RENS/RINS, or delay the
# root LP?", which needs measurements on both sides of the patch boundary.
#
# Anchored on *upstream's* `double heuristic_effort;`, the same text the
# heuristic_effort_used block above anchors on, rather than on that
# block's inserted line — chaining onto our own text is what epic #88
# coupling C warns about.  Both blocks insert directly after the anchor
# and this one runs second, so the layout is
#   double heuristic_effort;  <these fields>  size_t heuristic_effort_used;
# and either block can be removed without touching the other.
#
# `std::atomic` because the B&B-dive RENS/RINS call site runs inside
# `runTask` and executes concurrently across workers when `parallel=on`;
# a plain `++` there would be a data race.  The root-node counters and
# both doubles are written single-threaded, but share the type for
# uniformity at the read sites.
string(FIND "${MIPDATA_H}" "rens_calls" _instr_fields_found)
if(_instr_fields_found EQUAL -1)
    string(REPLACE
      "#include <vector>"
      "#include <atomic>\n#include <vector>"
      MIPDATA_H "${MIPDATA_H}")
    string(REPLACE
      "  double heuristic_effort;"
      "  double heuristic_effort;\n  // mip-heuristics: cannibalization instrumentation (issue #95).\n  // Incremented at existing call sites; no upstream reader, no control\n  // flow touched, so these read identically on a patched suite=off run\n  // and on an unpatched binary.\n  std::atomic<size_t> rens_calls{0};\n  std::atomic<size_t> rins_calls{0};\n  std::atomic<size_t> rcfix_calls{0};\n  // RENS invocations from the *root* call site only, a subset of\n  // rens_calls.  The root gate (upper_limit finite plus\n  // moreHeuristicsAllowed()) is where a presolve-found incumbent\n  // suppresses RENS, so the root count is the cannibalization signal and\n  // the whole-solve total alone would hide it.\n  std::atomic<size_t> rens_root_calls{0};\n  // LP iterations our own dive heuristic charged to\n  // heuristic_lp_iterations / total_lp_iterations.  Those two counters\n  // are shared, so without this an analyst reading them as \"HiGHS's own\n  // heuristic work\" is really reading ours as well.  Written by\n  // src/effort_ledger.cpp.\n  int64_t fpr_lp_lp_iterations = 0;\n  // Elapsed solve seconds when the first root LP solve started; negative\n  // until it does.  A pure read of mipsolver.timer_.\n  double root_lp_time = -1.0;\n  // Wall seconds the custom presolve heuristic chain spent before the\n  // root node, setup included.  Written by src/effort_ledger.cpp on the\n  // dispatching thread; read by heuristics::log_solve_summary.\n  double presolve_heuristic_time = 0.0;"
      MIPDATA_H "${MIPDATA_H}")

    # Sanity checks: a missed anchor here is silent — the fields simply
    # never appear and every call site below fails to compile with an
    # error that points at HiGHS rather than at this script.
    string(REGEX MATCHALL "std::atomic<size_t> rens_calls" _instr_rens_hits "${MIPDATA_H}")
    list(LENGTH _instr_rens_hits _instr_rens_count)
    string(REGEX MATCHALL "std::atomic<size_t> rens_root_calls" _instr_rroot_hits "${MIPDATA_H}")
    list(LENGTH _instr_rroot_hits _instr_rroot_count)
    string(REGEX MATCHALL "int64_t fpr_lp_lp_iterations" _instr_fprlp_hits "${MIPDATA_H}")
    list(LENGTH _instr_fprlp_hits _instr_fprlp_count)
    string(REGEX MATCHALL "double root_lp_time = -1.0" _instr_root_hits "${MIPDATA_H}")
    list(LENGTH _instr_root_hits _instr_root_count)
    string(REGEX MATCHALL "#include <atomic>" _instr_atomic_hits "${MIPDATA_H}")
    list(LENGTH _instr_atomic_hits _instr_atomic_count)
    if(NOT _instr_rens_count EQUAL 1 OR NOT _instr_rroot_count EQUAL 1
       OR NOT _instr_fprlp_count EQUAL 1 OR NOT _instr_root_count EQUAL 1
       OR NOT _instr_atomic_count EQUAL 1)
        message(FATAL_ERROR
            "HighsMipSolverData.h post-patch sanity check failed for the "
            "instrumentation counters (rens=${_instr_rens_count}, "
            "rens_root=${_instr_rroot_count}, fpr_lp_iters=${_instr_fprlp_count}, "
            "root_lp_time=${_instr_root_count}, atomic_include=${_instr_atomic_count}). "
            "Upstream HiGHS likely reformatted the 'double heuristic_effort;' "
            "member or the include block. "
            "${CLEAN_REBUILD}")
    endif()
    file(WRITE "${MIP_DIR}/HighsMipSolverData.h" "${MIPDATA_H}")
    message(STATUS "Applied instrumentation counters to HighsMipSolverData.h")
else()
    message(STATUS "Instrumentation counters already applied, skipping")
endif()

string(FIND "${MIPDATA_H}" "feasibilityJumpCapture" _fj_h_found)
if(_fj_h_found EQUAL -1)
    string(REPLACE
      "HighsModelStatus feasibilityJump();"
      "HighsModelStatus feasibilityJump();\n  HighsModelStatus feasibilityJumpCapture(std::vector<double>& captured_solution, double& captured_obj, size_t& captured_effort, size_t max_effort = 0, const std::vector<double>* hint_incumbent = nullptr, int seed_override = -1);"
      MIPDATA_H "${MIPDATA_H}")

    file(WRITE "${MIP_DIR}/HighsMipSolverData.h" "${MIPDATA_H}")
    message(STATUS "Applied feasibilityJumpCapture declaration to HighsMipSolverData.h")
else()
    message(STATUS "feasibilityJumpCapture patch already applied, skipping")
endif()

# Add per-heuristic solution source enum entries.
# Idempotency key includes kSolutionSourceFprLp (added after the initial
# four so the patch can detect a stale checkout that has the first four
# but is missing FprLp and still re-apply only the missing entry).  The
# older sentinel kSolutionSourceFPR alone is not sufficient.
string(FIND "${MIPDATA_H}" "kSolutionSourceFprLp" _src_enum_found)
if(_src_enum_found EQUAL -1)
    # If the stale first-four patch is present, strip it so the full
    # replacement below inserts the complete set in one go.
    string(REPLACE
      "  kSolutionSourceTrivialZ,            // z\n  kSolutionSourceFPR,                 // A (fix-propagate-repair)\n  kSolutionSourceLocalMIP,            // M (local MIP search)\n  kSolutionSourceScylla,              // G (Scylla)\n  kSolutionSourceFJ,                  // J (feasibility jump)\n  kSolutionSourceCleanup,"
      "  kSolutionSourceTrivialZ,            // z\n  kSolutionSourceCleanup,"
      MIPDATA_H "${MIPDATA_H}")
    string(REPLACE
      "  kSolutionSourceTrivialZ,            // z\n  kSolutionSourceCleanup,"
      "  kSolutionSourceTrivialZ,            // z\n  kSolutionSourceFPR,                 // A (fix-propagate-repair)\n  kSolutionSourceFprLp,               // D (LP-dependent FPR, B&B dive)\n  kSolutionSourceLocalMIP,            // M (local MIP search)\n  kSolutionSourceScylla,              // G (Scylla)\n  kSolutionSourceFJ,                  // J (feasibility jump)\n  kSolutionSourceCleanup,"
      MIPDATA_H "${MIPDATA_H}")

    # Sanity check: the source-enum insert must produce exactly one occurrence
    # of kSolutionSourceFprLp. If it does not, an upstream reformat likely
    # broke the REPLACE pattern above, leaving the file malformed.
    string(REGEX MATCHALL "kSolutionSourceFprLp" _h_fprlp_hits "${MIPDATA_H}")
    list(LENGTH _h_fprlp_hits _h_fprlp_count)
    if(NOT _h_fprlp_count EQUAL 1)
        message(FATAL_ERROR
            "HighsMipSolverData.h post-patch sanity check failed: "
            "expected exactly 1 occurrence of 'kSolutionSourceFprLp', got ${_h_fprlp_count}. "
            "Upstream HiGHS likely reformatted the source-enum block so the exact-string "
            "REPLACE patterns no longer match. "
            "${CLEAN_REBUILD}")
    endif()
    file(WRITE "${MIP_DIR}/HighsMipSolverData.h" "${MIPDATA_H}")
    message(STATUS "Applied custom solution source enums to HighsMipSolverData.h")
else()
    message(STATUS "Custom solution source enums already applied, skipping")
endif()

# ── Patch HighsMipSolverData.cpp: add source strings + fix key display ──
file(READ "${MIP_DIR}/HighsMipSolverData.cpp" MIPDATA_CPP)

# Idempotency key on kSolutionSourceFprLp so the patch re-applies on
# top of the stale first-four-sources layout when upgrading in place.
string(FIND "${MIPDATA_CPP}" "kSolutionSourceFprLp" _src_cpp_found)
if(_src_cpp_found EQUAL -1)
    # If the stale first-four-sources patch is present, strip the
    # source-to-string chain and the 4-group limits back to pristine so
    # the replacements below insert the complete set in one go.
    string(REPLACE
      "} else if (solution_source == kSolutionSourceFPR) {\n    if (code) return \"A\";\n    return \"FPR\";\n  } else if (solution_source == kSolutionSourceLocalMIP) {\n    if (code) return \"M\";\n    return \"Local MIP\";\n  } else if (solution_source == kSolutionSourceScylla) {\n    if (code) return \"G\";\n    return \"Scylla\";\n  } else if (solution_source == kSolutionSourceFJ) {\n    if (code) return \"J\";\n    return \"FJ\";\n  } else if (solution_source == kSolutionSourceCleanup) {\n    if (code) return \" \";\n    return \"\";"
      "} else if (solution_source == kSolutionSourceCleanup) {\n    if (code) return \" \";\n    return \"\";"
      MIPDATA_CPP "${MIPDATA_CPP}")
    string(REPLACE
      "std::vector<int> limits = {4, 9, 14, 18, last_enum};"
      "std::vector<int> limits = {4, 9, 14, last_enum};"
      MIPDATA_CPP "${MIPDATA_CPP}")

    # Add source-to-string entries before kSolutionSourceCleanup
    string(REPLACE
      "} else if (solution_source == kSolutionSourceCleanup) {\n    if (code) return \" \";\n    return \"\";"
      "} else if (solution_source == kSolutionSourceFPR) {\n    if (code) return \"A\";\n    return \"FPR\";\n  } else if (solution_source == kSolutionSourceFprLp) {\n    if (code) return \"D\";\n    return \"FPR LP\";\n  } else if (solution_source == kSolutionSourceLocalMIP) {\n    if (code) return \"M\";\n    return \"Local MIP\";\n  } else if (solution_source == kSolutionSourceScylla) {\n    if (code) return \"G\";\n    return \"Scylla\";\n  } else if (solution_source == kSolutionSourceFJ) {\n    if (code) return \"J\";\n    return \"FJ\";\n  } else if (solution_source == kSolutionSourceCleanup) {\n    if (code) return \" \";\n    return \"\";"
      MIPDATA_CPP "${MIPDATA_CPP}")

    # Update printSolutionSourceKey limits for the 5 new entries (one extra
    # group), and drop that group again at mip_heuristic_suite=off.
    #
    # `off` is the patch-overhead row of the benchmark matrix: it must be
    # indistinguishable from an unpatched binary, and a legend advertising
    # FPR / FPR LP / Local MIP / Scylla / FJ when none of them can run is a
    # visible difference.  The literal {4, 9, 14, 19} is deliberate — reusing
    # `last_enum` here would print [14, 24) and list the five custom sources
    # in the *third* group instead.  With the literal, the printed key is
    # byte-identical to vanilla's: same four groups over indices 0..18, same
    # trailing-semicolon logic (limits.size() is 4 in both).  The enum values
    # themselves stay registered — printSolutionSourceKey's group limits are
    # positional index literals and renumbering them corrupts the legend.
    string(REPLACE
      "std::vector<int> limits = {4, 9, 14, last_enum};"
      "std::vector<int> limits = {4, 9, 14, 19, last_enum};\n  if (mipsolver.options_mip_->mip_heuristic_suite == \"off\")\n    limits = {4, 9, 14, 19};  // mip-heuristics: vanilla-equivalent key"
      MIPDATA_CPP "${MIPDATA_CPP}")

    # Sanity checks: the source-to-string insert must produce exactly one
    # kSolutionSourceFprLp branch, one "FPR LP" display string, and one
    # updated printSolutionSourceKey limits vector. If any count is wrong,
    # an upstream reformat likely broke the strip-and-restore REPLACEs,
    # leaving the file with a missing or duplicated insert.
    string(REGEX MATCHALL "kSolutionSourceFprLp" _cpp_fprlp_hits "${MIPDATA_CPP}")
    list(LENGTH _cpp_fprlp_hits _cpp_fprlp_count)
    if(NOT _cpp_fprlp_count EQUAL 1)
        message(FATAL_ERROR
            "HighsMipSolverData.cpp post-patch sanity check failed: "
            "expected exactly 1 occurrence of 'kSolutionSourceFprLp', got ${_cpp_fprlp_count}. "
            "Upstream HiGHS likely reformatted the source-to-string chain so the exact-string "
            "REPLACE patterns no longer match. "
            "${CLEAN_REBUILD}")
    endif()
    string(REGEX MATCHALL "\"FPR LP\"" _cpp_fprlp_str_hits "${MIPDATA_CPP}")
    list(LENGTH _cpp_fprlp_str_hits _cpp_fprlp_str_count)
    if(NOT _cpp_fprlp_str_count EQUAL 1)
        message(FATAL_ERROR
            "HighsMipSolverData.cpp post-patch sanity check failed: "
            "expected exactly 1 occurrence of '\"FPR LP\"', got ${_cpp_fprlp_str_count}. "
            "Upstream HiGHS likely reformatted the source-to-string chain so the exact-string "
            "REPLACE patterns no longer match. "
            "${CLEAN_REBUILD}")
    endif()
    string(REGEX MATCHALL "\\{4, 9, 14, 19, last_enum\\}" _cpp_limits_hits "${MIPDATA_CPP}")
    list(LENGTH _cpp_limits_hits _cpp_limits_count)
    if(NOT _cpp_limits_count EQUAL 1)
        message(FATAL_ERROR
            "HighsMipSolverData.cpp post-patch sanity check failed: "
            "expected exactly 1 occurrence of '{4, 9, 14, 19, last_enum}', got ${_cpp_limits_count}. "
            "Upstream HiGHS likely reformatted printSolutionSourceKey so the limits-vector "
            "REPLACE pattern no longer matches. "
            "${CLEAN_REBUILD}")
    endif()
    file(WRITE "${MIP_DIR}/HighsMipSolverData.cpp" "${MIPDATA_CPP}")
    message(STATUS "Applied solution source strings to HighsMipSolverData.cpp")
else()
    message(STATUS "Solution source strings already applied, skipping")
endif()

# ── Root-node instrumentation: rcfix/RENS counts + root LP timestamp (#95) ──
# Independent of the source-string block above: its own sentinel, its own
# anchors, its own check.  Three insertions, all of them additive
# statements placed immediately before an existing one — no upstream
# statement is moved, reordered or made conditional, which is the
# equivalence invariant these counters exist to be measured against.
#
# `rootReducedCost` is called from nowhere else in HiGHS, so the root
# block is the only place `rcfix_calls` can be counted; RENS is called
# here *and* in the B&B dive lambda, and both sites increment so the
# reported count is the whole solve's.  RINS has no root call site.
#
# `root_lp_time` is guarded on its own sentinel value rather than on
# `numRestarts`: `evaluateRootNode` runs again after a restart, and the
# question the number answers ("when did the root LP first get to start,
# given what ran before it") is about the first one.
file(READ "${MIP_DIR}/HighsMipSolverData.cpp" MIPDATA_CPP2)
string(FIND "${MIPDATA_CPP2}" "rcfix_calls" _root_instr_found)
if(_root_instr_found EQUAL -1)
    string(REPLACE
      "      heuristics.rootReducedCost(worker);"
      "      rcfix_calls.fetch_add(1, std::memory_order_relaxed);\n      heuristics.rootReducedCost(worker);"
      MIPDATA_CPP2 "${MIPDATA_CPP2}")
    # Both counters at the root site: `rens_calls` is the whole-solve
    # total (this site plus the dive site, Patch D below), `rens_root_calls`
    # isolates the root gate, which is where a presolve-found incumbent
    # actually suppresses RENS.
    string(REPLACE
      "      heuristics.RENS(worker, rootlpsol);"
      "      rens_calls.fetch_add(1, std::memory_order_relaxed);\n      rens_root_calls.fetch_add(1, std::memory_order_relaxed);\n      heuristics.RENS(worker, rootlpsol);"
      MIPDATA_CPP2 "${MIPDATA_CPP2}")
    # The declaration (`HighsLpRelaxation::Status status = ...`) is what
    # makes this the *first* of the eleven evaluateRootLp calls in the
    # file; every later one assigns to the already-declared `status`.
    string(REPLACE
      "  profiling->start(kMipClockEvaluateRootLp);\n  HighsLpRelaxation::Status status = evaluateRootLp(worker);"
      "  if (root_lp_time < 0.0)\n    root_lp_time = mipsolver.timer_.read();  // mip-heuristics (#95)\n  profiling->start(kMipClockEvaluateRootLp);\n  HighsLpRelaxation::Status status = evaluateRootLp(worker);"
      MIPDATA_CPP2 "${MIPDATA_CPP2}")

    # Sanity checks: every miss here is silent.  A missed counter reports
    # zero RENS/rcfix calls forever, which reads as "the custom heuristics
    # starved them completely" — the exact conclusion this instrumentation
    # exists to test, arrived at from a broken anchor.  A missed
    # root_lp_time leaves the field at -1 and the [Root] line useless.
    string(REGEX MATCHALL "rcfix_calls.fetch_add" _root_rcfix_hits "${MIPDATA_CPP2}")
    list(LENGTH _root_rcfix_hits _root_rcfix_count)
    string(REGEX MATCHALL "rens_calls.fetch_add" _root_rens_hits "${MIPDATA_CPP2}")
    list(LENGTH _root_rens_hits _root_rens_count)
    string(REGEX MATCHALL "rens_root_calls.fetch_add" _root_rroot_hits "${MIPDATA_CPP2}")
    list(LENGTH _root_rroot_hits _root_rroot_count)
    string(REGEX MATCHALL "root_lp_time = mipsolver.timer_.read\\(\\)" _root_time_hits "${MIPDATA_CPP2}")
    list(LENGTH _root_time_hits _root_time_count)
    if(NOT _root_rcfix_count EQUAL 1 OR NOT _root_rens_count EQUAL 1
       OR NOT _root_rroot_count EQUAL 1 OR NOT _root_time_count EQUAL 1)
        message(FATAL_ERROR
            "HighsMipSolverData.cpp post-patch sanity check failed for the "
            "root-node instrumentation (rcfix=${_root_rcfix_count}, "
            "rens=${_root_rens_count}, rens_root=${_root_rroot_count}, "
            "root_lp_time=${_root_time_count}). "
            "Upstream HiGHS likely reformatted evaluateRootNode so one of the "
            "call-site anchors no longer matches. "
            "${CLEAN_REBUILD}")
    endif()
    file(WRITE "${MIP_DIR}/HighsMipSolverData.cpp" "${MIPDATA_CPP2}")
    message(STATUS "Applied root-node instrumentation to HighsMipSolverData.cpp")
else()
    message(STATUS "Root-node instrumentation already applied, skipping")
endif()

# ── Patch HighsFeasibilityJump.cpp: add capture implementation ──
file(READ "${MIP_DIR}/HighsFeasibilityJump.cpp" FJ_CONTENT)

string(FIND "${FJ_CONTENT}" "feasibilityJumpCapture" _fj_found)
if(_fj_found EQUAL -1)
    # Append the capture variant after the original function
    string(APPEND FJ_CONTENT "\n\
HighsModelStatus HighsMipSolverData::feasibilityJumpCapture(\n\
    std::vector<double>& captured_solution, double& captured_obj,\n\
    size_t& captured_effort, size_t max_effort,\n\
    const std::vector<double>* hint_incumbent, int seed_override) {\n\
  const HighsLp* model = this->mipsolver.model_;\n\
  const HighsLogOptions& log_options = mipsolver.options_mip_->log_options;\n\
  double sense_multiplier = static_cast<double>(model->sense_);\n\
\n\
#ifdef HIGHSINT64\n\
  highsLogUser(log_options, HighsLogType::kInfo,\n\
               \"Feasibility Jump code isn't currently compatible \"\n\
               \"with a 64-bit HighsInt: skipping Feasibility Jump\\n\");\n\
  return HighsModelStatus::kNotset;\n\
#else\n\
\n\
  bool found_integer_feasible_solution = false;\n\
  std::vector<double> col_value(model->num_col_, 0.0);\n\
  double objective_function_value = 0.0;\n\
\n\
  const auto& inc = hint_incumbent ? *hint_incumbent : incumbent;\n\
  const bool use_incumbent = !inc.empty();\n\
\n\
  const int fj_seed = (seed_override >= 0) ? seed_override : mipsolver.options_mip_->random_seed;\n\
  auto solver = external_feasibilityjump::FeasibilityJumpSolver(\n\
      log_options,\n\
      fj_seed,\n\
      epsilon,\n\
      feastol);\n\
\n\
  for (HighsInt col = 0; col < model->num_col_; ++col) {\n\
    double lower = model->col_lower_[col];\n\
    double upper = model->col_upper_[col];\n\
\n\
    external_feasibilityjump::VarType fjVarType;\n\
    if (model->integrality_[col] == HighsVarType::kContinuous) {\n\
      fjVarType = external_feasibilityjump::VarType::Continuous;\n\
    } else {\n\
      fjVarType = external_feasibilityjump::VarType::Integer;\n\
      lower = std::ceil(lower - feastol);\n\
      upper = std::floor(upper + feastol);\n\
    }\n\
\n\
    const bool legal_bounds = lower <= upper && lower < kHighsInf &&\n\
                              upper > -kHighsInf && !std::isnan(lower) &&\n\
                              !std::isnan(upper);\n\
    if (!legal_bounds) {\n\
      return HighsModelStatus::kInfeasible;\n\
    }\n\
    solver.addVar(fjVarType, lower, upper,\n\
                  sense_multiplier * model->col_cost_[col]);\n\
\n\
    double initial_assignment = 0.0;\n\
    if (use_incumbent && std::isfinite(inc[col])) {\n\
      initial_assignment = std::max(lower, std::min(upper, inc[col]));\n\
    } else {\n\
      if (std::isfinite(lower)) {\n\
        initial_assignment = lower;\n\
      } else if (std::isfinite(upper)) {\n\
        initial_assignment = upper;\n\
      }\n\
    }\n\
    col_value[col] = initial_assignment;\n\
  }\n\
\n\
  HighsSparseMatrix a_matrix;\n\
  a_matrix.createRowwise(model->a_matrix_);\n\
\n\
  for (HighsInt row = 0; row < model->num_row_; ++row) {\n\
    bool hasFiniteLower = std::isfinite(model->row_lower_[row]);\n\
    bool hasFiniteUpper = std::isfinite(model->row_upper_[row]);\n\
    if (hasFiniteLower || hasFiniteUpper) {\n\
      HighsInt row_num_nz = a_matrix.start_[row + 1] - a_matrix.start_[row];\n\
      auto row_index = a_matrix.index_.data() + a_matrix.start_[row];\n\
      auto row_value = a_matrix.value_.data() + a_matrix.start_[row];\n\
      if (hasFiniteLower) {\n\
        solver.addConstraint(external_feasibilityjump::RowType::Gte,\n\
                             model->row_lower_[row], row_num_nz, row_index,\n\
                             row_value, 0);\n\
      }\n\
      if (hasFiniteUpper) {\n\
        solver.addConstraint(external_feasibilityjump::RowType::Lte,\n\
                             model->row_upper_[row], row_num_nz, row_index,\n\
                             row_value, 0);\n\
      }\n\
    }\n\
  }\n\
\n\
  const HighsInt nnz = a_matrix.numNz();\n\
  const size_t kMaxTotalEffort = (max_effort > 0) ? max_effort : ((size_t)nnz << 10);\n\
  const size_t kMaxEffortSinceLastImprovement = std::min((size_t)nnz << 8, (max_effort > 0) ? max_effort : ((size_t)nnz << 8));\n\
\n\
  size_t last_total_effort = 0;\n\
  auto fjControlCallback =\n\
      [=, &col_value, &found_integer_feasible_solution,\n\
       &objective_function_value, &last_total_effort](external_feasibilityjump::FJStatus status)\n\
      -> external_feasibilityjump::CallbackControlFlow {\n\
    last_total_effort = status.totalEffort;\n\
    if (status.solution != nullptr) {\n\
      found_integer_feasible_solution = true;\n\
      col_value = std::vector<double>(status.solution,\n\
                                      status.solution + status.numVars);\n\
      objective_function_value =\n\
          model->offset_ + sense_multiplier * status.solutionObjectiveValue;\n\
    }\n\
    if (status.effortSinceLastImprovement > kMaxEffortSinceLastImprovement ||\n\
        status.totalEffort > kMaxTotalEffort) {\n\
      return external_feasibilityjump::CallbackControlFlow::Terminate;\n\
    } else {\n\
      return external_feasibilityjump::CallbackControlFlow::Continue;\n\
    }\n\
  };\n\
\n\
  solver.solve(col_value.data(), fjControlCallback);\n\
  captured_effort = last_total_effort;\n\
\n\
  if (found_integer_feasible_solution) {\n\
    captured_solution = std::move(col_value);\n\
    captured_obj = objective_function_value;\n\
  }\n\
  return HighsModelStatus::kNotset;\n\
#endif\n\
}\n")

    file(WRITE "${MIP_DIR}/HighsFeasibilityJump.cpp" "${FJ_CONTENT}")
    message(STATUS "Applied feasibilityJumpCapture to HighsFeasibilityJump.cpp")
else()
    message(STATUS "feasibilityJumpCapture already applied, skipping")
endif()

# ── Patch feasibilityjump.hh: add resume parameter to solve() ──
file(READ "${MIP_DIR}/feasibilityjump.hh" FJ_HH)

string(FIND "${FJ_HH}" "bool resume = false" _fj_resume_found)
if(_fj_resume_found EQUAL -1)
    string(REPLACE
      "  int solve(double* initialValues,\n            std::function<CallbackControlFlow(FJStatus)> callback) {\n    assert(callback);\n    highsLogDev(logOptions, HighsLogType::kInfo,\n                FJ_LOG_PREFIX\n                \"starting solve. weightUpdateDecay=%g, relaxContinuous=%d  \\n\",\n                weightUpdateDecay, problem.usedRelaxContinuous);\n\n    init(initialValues);\n\n    effortAtLastLogging = -kMinEffortToLogging;  // Enabling step=0 logging\n    int num_logging_lines_since_header = 0;"
      "  int solve(double* initialValues,\n            std::function<CallbackControlFlow(FJStatus)> callback,\n            bool resume = false) {\n    assert(callback);\n    if (!resume) {\n      highsLogDev(logOptions, HighsLogType::kInfo,\n                  FJ_LOG_PREFIX\n                  \"starting solve. weightUpdateDecay=%g, relaxContinuous=%d  \\n\",\n                  weightUpdateDecay, problem.usedRelaxContinuous);\n      init(initialValues);\n      effortAtLastLogging = -kMinEffortToLogging;  // Enabling step=0 logging\n    }\n    int num_logging_lines_since_header = 0;"
      FJ_HH "${FJ_HH}")

    file(WRITE "${MIP_DIR}/feasibilityjump.hh" "${FJ_HH}")
    message(STATUS "Applied resume parameter patch to feasibilityjump.hh")
else()
    message(STATUS "Resume parameter patch already applied to feasibilityjump.hh, skipping")
endif()

# ── Patch standalone feasibilityJump() to store effort ──
file(READ "${MIP_DIR}/HighsFeasibilityJump.cpp" FJ_CONTENT2)
string(FIND "${FJ_CONTENT2}" "heuristic_effort_used" _fj_effort_found)
if(_fj_effort_found EQUAL -1)
    # Add effort capture variable to original FJ callback
    string(REPLACE
      "  auto fjControlCallback =\n      [=, &col_value, &found_integer_feasible_solution,\n       &objective_function_value](external_feasibilityjump::FJStatus status)\n      -> external_feasibilityjump::CallbackControlFlow {"
      "  size_t fj_last_effort = 0;\n  auto fjControlCallback =\n      [=, &col_value, &found_integer_feasible_solution,\n       &objective_function_value, &fj_last_effort](external_feasibilityjump::FJStatus status)\n      -> external_feasibilityjump::CallbackControlFlow {\n    fj_last_effort = status.totalEffort;"
      FJ_CONTENT2 "${FJ_CONTENT2}")

    # Store effort after solve
    string(REPLACE
      "  solver.solve(col_value.data(), fjControlCallback);\n\n  if (found_integer_feasible_solution) {\n    // Initial assignments"
      "  solver.solve(col_value.data(), fjControlCallback);\n  heuristic_effort_used += fj_last_effort;\n\n  if (found_integer_feasible_solution) {\n    // Initial assignments"
      FJ_CONTENT2 "${FJ_CONTENT2}")

    # Silent if it misses: the `+=` simply never appears and vanilla FJ's
    # effort goes unaccounted forever.  That is the standalone call site
    # Patch A now runs at `suite=off`, so a miss makes the patch-overhead
    # row of the benchmark matrix under-report its effort.
    string(FIND "${FJ_CONTENT2}" "heuristic_effort_used += fj_last_effort;" _fj_effort_check)
    if(_fj_effort_check EQUAL -1)
        message(FATAL_ERROR
            "HighsFeasibilityJump.cpp post-patch sanity check failed: "
            "'heuristic_effort_used += fj_last_effort;' not found after patching. "
            "Upstream HiGHS likely reformatted the standalone feasibilityJump() "
            "callback so an exact-string anchor no longer matches. "
            "${CLEAN_REBUILD}")
    endif()

    file(WRITE "${MIP_DIR}/HighsFeasibilityJump.cpp" "${FJ_CONTENT2}")
    message(STATUS "Applied effort tracking to standalone feasibilityJump()")
else()
    message(STATUS "Standalone FJ effort tracking already applied, skipping")
endif()

# ── Patch HighsMipSolver.cpp: insert heuristic call sites ──
file(READ "${MIP_DIR}/HighsMipSolver.cpp" CONTENT)

# Defensive check: the presolve budget used to be derived from
# mip_heuristic_effort before the option split introduced
# mip_heuristic_presolve_effort.  The idempotency sentinel below
# ('heuristics::run_presolve') is present in both layouts, so an in-place
# upgrade would silently keep the old call site and starve the presolve
# heuristics at the reverted 0.05 default.  Force a clean rebuild.
string(FIND "${CONTENT}" "heuristic_effort_budget(nnz, options_mip_->mip_heuristic_effort)" _stale_presolve_budget)
if(NOT _stale_presolve_budget EQUAL -1)
    message(FATAL_ERROR
        "HighsMipSolver.cpp derives the presolve heuristic budget from "
        "'mip_heuristic_effort'; this was split into "
        "'mip_heuristic_presolve_effort'. "
        "${CLEAN_REBUILD}")
endif()

string(FIND "${CONTENT}" "heuristics::run_presolve" _found)
if(_found EQUAL -1)
    # Add includes at top (after existing includes)
    string(REPLACE
      "#include \"mip/HighsMipSolver.h\""
      "#include \"mip/HighsMipSolver.h\"\n#include \"fpr_lp.h\"\n#include \"heuristic_common.h\"\n#include \"mode_dispatch.h\""
      CONTENT "${CONTENT}")

    # Patch A: hand the standalone FJ call site over to the custom presolve
    # block, except at mip_heuristic_suite=off.
    #
    # `off` is the patch-overhead / vanilla-equivalence row of the benchmark
    # matrix, so it must run exactly what an unpatched binary runs — which
    # includes upstream's single-threaded FeasibilityJump.  At every other
    # suite value FJ is either off or run by our parallel infrastructure, and
    # letting the native call site fire too would double-run it.
    # `mip_heuristic_run_feasibility_jump` is upstream's own option and keeps
    # its meaning: false disables FJ here and in the custom chain alike.
    string(REPLACE
      "if (options_mip_->mip_heuristic_run_feasibility_jump) {"
      "if (options_mip_->mip_heuristic_suite == \"off\" &&\n        options_mip_->mip_heuristic_run_feasibility_jump) { // native FJ only at suite=off"
      CONTENT "${CONTENT}")

    # Patch A2: insert custom heuristics block via mode_dispatch
    string(REPLACE
      "    }\n    // End of pre-root-node heuristics"
      "    }\n    {\n      const size_t nnz = mipdata_->ARindex_.size();\n      const size_t budget = heuristic_effort_budget(nnz, options_mip_->mip_heuristic_presolve_effort);\n      if (heuristics::run_presolve(*this, budget)) {\n        modelstatus_ = HighsModelStatus::kInfeasible;\n        cleanupSolve();\n        return;\n      }\n    }\n\n    // End of pre-root-node heuristics"
      CONTENT "${CONTENT}")

    # This block had no check at all, and it is the one whose miss is
    # worst: if A2's anchor stops matching while A's still does, the tree
    # compiles and links, but `heuristics::run_presolve` is never called
    # *and* vanilla's standalone FJ only runs at suite=off — a binary that
    # runs no primal heuristics whatsoever, in every configuration a
    # benchmark would use, while still printing the "mip-heuristics patch
    # active" banner.  The mirror case (A misses, A2 lands) double-runs FJ
    # and quietly invalidates the vanilla-equivalence row of the benchmark
    # matrix (epic #88 coupling I).  Neither shows up as a build failure,
    # so nothing else would catch it.
    string(FIND "${CONTENT}" "heuristics::run_presolve" _presolve_check)
    string(FIND "${CONTENT}" "native FJ only at suite=off" _fj_off_check)
    if(_presolve_check EQUAL -1 OR _fj_off_check EQUAL -1)
        message(FATAL_ERROR
            "HighsMipSolver.cpp presolve patch failed "
            "(run_presolve=${_presolve_check}, fj_disable=${_fj_off_check}). "
            "Upstream HiGHS likely restructured the pre-root-node heuristics "
            "block so an exact-string anchor no longer matches. "
            "Please update Patch A/A2 in third_party/highs_patch/apply_patch.cmake. "
            "${CLEAN_REBUILD}")
    endif()

    file(WRITE "${MIP_DIR}/HighsMipSolver.cpp" "${CONTENT}")
    message(STATUS "Applied presolve heuristic patches to HighsMipSolver.cpp")
else()
    message(STATUS "Presolve heuristic patches already applied to HighsMipSolver.cpp, skipping")
endif()

# ── Patch C: insert fpr_lp::run after RENS/RINS in the B&B dive heuristics ──
# Separate idempotency block so it can be applied independently of A/A2.
# HiGHS 1.15 moved the RENS/RINS block into a runHeuristics() lambda; the
# anchor is the profiling stop + infeasible() return that ends the lambda.
# The call is deliberately bare: gating (mip_heuristic_suite) and budget
# derivation (shared RENS/RINS LP-iteration headroom) live inside
# fpr_lp::run so the patch string stays minimal.
file(READ "${MIP_DIR}/HighsMipSolver.cpp" CONTENT)

# Defensive check: the pre-split insertion passed an nnz-based budget and
# gated on mip_heuristic_run_fpr at the call site.  The sentinel below
# ('fpr_lp::run') is present in both layouts; the old two-argument call no
# longer matches the fpr_lp::run signature, so force a clean rebuild with
# an actionable message instead of a confusing compile error.
string(FIND "${CONTENT}" "fpr_lp::run(*this, heuristic_effort_budget" _stale_fprlp_call)
if(NOT _stale_fprlp_call EQUAL -1)
    message(FATAL_ERROR
        "HighsMipSolver.cpp contains the pre-split two-argument fpr_lp::run "
        "call site.  fpr_lp now derives its budget from the shared RENS/RINS "
        "LP-iteration headroom internally. "
        "${CLEAN_REBUILD}")
endif()

string(FIND "${CONTENT}" "fpr_lp::run" _fprlp_found)
if(_fprlp_found EQUAL -1)
    string(REPLACE
      "    if (!mipdata_->parallelLockActive())\n      profiling_->stop(kMipClockDivePrimalHeuristics);\n\n    return worker.getGlobalDomain().infeasible();"
      "    fpr_lp::run(*this);\n    if (!mipdata_->parallelLockActive())\n      profiling_->stop(kMipClockDivePrimalHeuristics);\n\n    return worker.getGlobalDomain().infeasible();"
      CONTENT "${CONTENT}")

    string(FIND "${CONTENT}" "fpr_lp::run" _fprlp_check)
    if(_fprlp_check EQUAL -1)
        message(FATAL_ERROR
            "HighsMipSolver.cpp post-patch sanity check failed: "
            "'fpr_lp::run' not found after patching. "
            "Upstream HiGHS likely restructured the runHeuristics block so the "
            "exact-string anchor no longer matches. "
            "Please update Patch C in third_party/highs_patch/apply_patch.cmake. "
            "${CLEAN_REBUILD}")
    endif()

    file(WRITE "${MIP_DIR}/HighsMipSolver.cpp" "${CONTENT}")
    message(STATUS "Applied fpr_lp B&B dive patch to HighsMipSolver.cpp")
else()
    message(STATUS "fpr_lp B&B dive patch already applied, skipping")
endif()

# ── Patch D: count HiGHS's own RENS/RINS calls in the B&B dive (#95) ──
# Deliberately anchored on the `mipdata_->heuristics.RENS(` /
# `.RINS(` call statements rather than on the profiling stop that Patch C
# above uses, even though both live in the same `runHeuristics` lambda:
# the two blocks must stay independently editable, and re-using Patch C's
# anchor would make changing either one silently shift the other.
#
# Both increments are additive statements placed immediately before an
# existing call.  Nothing is reordered and nothing becomes conditional,
# so the dive behaves exactly as an unpatched binary does.  The counters
# are atomic because `runHeuristics` is invoked from `runTask` and runs
# concurrently across B&B workers when `parallel=on`.
file(READ "${MIP_DIR}/HighsMipSolver.cpp" CONTENT)
string(FIND "${CONTENT}" "mipdata_->rens_calls" _dive_instr_found)
if(_dive_instr_found EQUAL -1)
    string(REPLACE
      "        mipdata_->heuristics.RENS("
      "        mipdata_->rens_calls.fetch_add(1, std::memory_order_relaxed);\n        mipdata_->heuristics.RENS("
      CONTENT "${CONTENT}")
    string(REPLACE
      "        mipdata_->heuristics.RINS("
      "        mipdata_->rins_calls.fetch_add(1, std::memory_order_relaxed);\n        mipdata_->heuristics.RINS("
      CONTENT "${CONTENT}")

    # Silent on a miss: the dive calls stop being counted and the
    # `[Native]` line under-reports, which reads as starvation that did
    # not happen.
    string(REGEX MATCHALL "mipdata_->rens_calls.fetch_add" _dive_rens_hits "${CONTENT}")
    list(LENGTH _dive_rens_hits _dive_rens_count)
    string(REGEX MATCHALL "mipdata_->rins_calls.fetch_add" _dive_rins_hits "${CONTENT}")
    list(LENGTH _dive_rins_hits _dive_rins_count)
    if(NOT _dive_rens_count EQUAL 1 OR NOT _dive_rins_count EQUAL 1)
        message(FATAL_ERROR
            "HighsMipSolver.cpp post-patch sanity check failed for the dive "
            "RENS/RINS counters (rens=${_dive_rens_count}, rins=${_dive_rins_count}). "
            "Upstream HiGHS likely reformatted the runHeuristics lambda so an "
            "exact-string anchor no longer matches. "
            "Please update Patch D in third_party/highs_patch/apply_patch.cmake. "
            "${CLEAN_REBUILD}")
    endif()

    file(WRITE "${MIP_DIR}/HighsMipSolver.cpp" "${CONTENT}")
    message(STATUS "Applied dive RENS/RINS counters to HighsMipSolver.cpp")
else()
    message(STATUS "Dive RENS/RINS counters already applied, skipping")
endif()

# ── Patch E: emit the once-per-solve [Native] / [Root] lines (#95) ──
# `cleanupSolve` is the single funnel every exit path of
# `HighsMipSolver::run` goes through, including the early returns that
# never reach the root node, so one call here reports exactly once per
# solve without adding a return path of its own.  The emission is placed
# before the final display line so the instrumentation precedes HiGHS's
# solving report rather than interleaving with it.
#
# The declaration comes from the `mode_dispatch.h` include added by
# Patch A/A2 above.  That is the one cross-block dependency here, and it
# fails loudly (a compile error naming `heuristics::log_solve_summary`)
# rather than silently, which is why it is not duplicated.
file(READ "${MIP_DIR}/HighsMipSolver.cpp" CONTENT)
string(FIND "${CONTENT}" "log_solve_summary" _summary_found)
if(_summary_found EQUAL -1)
    string(REPLACE
      "  // Force a final logging line\n  mipdata_->printDisplayLine(kSolutionSourceCleanup);"
      "  heuristics::log_solve_summary(*this);\n  // Force a final logging line\n  mipdata_->printDisplayLine(kSolutionSourceCleanup);"
      CONTENT "${CONTENT}")

    string(REGEX MATCHALL "heuristics::log_solve_summary" _summary_hits "${CONTENT}")
    list(LENGTH _summary_hits _summary_count)
    if(NOT _summary_count EQUAL 1)
        message(FATAL_ERROR
            "HighsMipSolver.cpp post-patch sanity check failed: expected exactly "
            "1 occurrence of 'heuristics::log_solve_summary', got ${_summary_count}. "
            "Upstream HiGHS likely reformatted cleanupSolve so the final-display-line "
            "anchor no longer matches. "
            "Please update Patch E in third_party/highs_patch/apply_patch.cmake. "
            "${CLEAN_REBUILD}")
    endif()

    file(WRITE "${MIP_DIR}/HighsMipSolver.cpp" "${CONTENT}")
    message(STATUS "Applied solve-summary instrumentation to HighsMipSolver.cpp")
else()
    message(STATUS "Solve-summary instrumentation already applied, skipping")
endif()
