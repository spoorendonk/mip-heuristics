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
# The version marker is the one probe: a tree carrying any marker other than
# the current one is rejected outright rather than rewritten in place.
#
# Bump PATCH_VERSION whenever any inserted text changes.  It is per-tree
# state, not a per-change counter: version 8 covered two independent
# changes to inserted text that landed together — the four per-heuristic
# effort option records plus the now-argument-less run_presolve call site
# (#110), and the suite option's registered description, which enumerated
# six values as if exhaustive (#112).  Two such changes need one bump, not
# two; what the marker has to distinguish is trees, not commits.
#
# Version 9 is #106's calibration surface: the four
# `mip_heuristic_<name>_stall` option records, the
# `mip_heuristic_presolve_only` record, and the presolve-only early exit
# inserted into HighsMipSolver.cpp beside the run_presolve call site.
#
# Version 10 widens the four effort records' upper bound from 1.0 to
# `kEffortMax` (#113).  The bound is what a *record* carries, so it is
# inserted text and needs the bump even though nothing else moved.
set(PATCH_VERSION "10")
string(FIND "${OPTIONS_CONTENT}" "mip-heuristics patch version ${PATCH_VERSION}" _patch_version_found)
if(_patch_version_found EQUAL -1)
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
# One option selects which custom heuristics run: the alias "off" (none) or
# "all" (every one), or a comma-separated list of fj, fpr, local_mip, scylla
# (default "all").  It replaced the three mip_heuristic_run_* bools and
# mip_heuristic_preset in #93; the list form arrived in #112.
#
# The value is interpreted in `heuristics::effective_flags`, not here — HiGHS
# does not validate string option *values*, so this registration only has to
# name the option and describe it.  The description is the one part of #112
# that is inserted text, and it is worth the PATCH_VERSION bump: it is the
# only documentation of the legal values that ships *inside* the binary, and
# the enumeration it carried listed six values as if they were exhaustive.
#
# The path that echoes it is `Highs::writeOptions(<filename>)` — an API call,
# not a CLI flag.  Do not go looking for it in `highs --options_file`: the
# CLI dump is `highs.writeOptions("", true)` in `app/RunHighs.cpp`, whose
# `report_only_deviations` argument emits `Set option ... to "<value>"` lines
# and no descriptions at all.  What makes the description reachable from the
# full dump is an accident of *where* this record lands: `reportOptions`
# skips every record whose `advanced` flag is set, and the insertion anchor
# below puts ours ahead of the point where `setOptionRecords` flips its local
# `advanced` to true, so the option registers non-advanced and survives that
# filter.  Moving the anchor past that point would silence the description
# without changing a character of it.
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
      "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n                                       \"Use the Shifting heuristic\", advanced,\n                                       &mip_heuristic_run_shifting, false);\n    records.push_back(record_bool);\n\n    record_string = new OptionRecordString(\"mip_heuristic_suite\",\n                                          \"Custom MIP heuristic suite: comma-separated list of \\\"fj\\\", \\\"fpr\\\", \\\"local_mip\\\", \\\"scylla\\\", or the alias \\\"off\\\" (none) or \\\"all\\\" (every one)\", advanced,\n                                          &mip_heuristic_suite, \"all\");\n    records.push_back(record_string);"
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

# ── Assert mip_heuristic_effort keeps the vanilla default 0.05 ──
# The patch leaves upstream's own B&B heuristic knob exactly as it is, so a
# patched binary at default options matches vanilla's heuristic budget.  This
# is a sanity check, not an edit: a FATAL_ERROR fires if upstream ever changes
# the default or reformats the OptionRecordDouble line, because we would
# otherwise silently ship against a different budget than the docs claim.
file(READ "${LP_DATA_DIR}/HighsOptions.h" OPTIONS_CONTENT)
string(FIND "${OPTIONS_CONTENT}" "&mip_heuristic_effort, 0.0, 0.05, 1.0" _effort_default_found)
if(_effort_default_found EQUAL -1)
    message(FATAL_ERROR
        "HighsOptions.h sanity check failed: '&mip_heuristic_effort, 0.0, "
        "0.05, 1.0' not found. Upstream HiGHS changed the default or "
        "reformatted the option-record block. "
        "${CLEAN_REBUILD}")
endif()

# ── Add the per-heuristic calibration options ──
# One effort-budget multiplier per presolve heuristic (#110), replacing the
# single shared mip_heuristic_presolve_effort and the kWeight* constants
# that split it.  `src/mode_dispatch.cpp` reads each one and sizes that
# heuristic's dispatch with `heuristic_effort_budget(nnz, value)`:
# `nnz << 12` effort units at the anchor 0.05, linear in the value.  No
# shared envelope means raising one heuristic's budget no longer lowers
# the others', which is what makes a per-heuristic calibration possible.
#
# The defaults are derived from what the shared envelope handed each
# heuristic before the split:
#   * fj 0.0125 -> `nnz << 10` per worker, exactly vanilla HiGHS's
#     hardcoded single-thread FJ limit, which is what our N workers each
#     ran on before.  FJ's option is a *per-worker* allowance; the other
#     three size a whole dispatch (see mode_dispatch.cpp).
#   * fpr / local_mip / scylla: 0.30 * w/sum(w) for the retired weights
#     2.99 / 6.16 / 1.00, i.e. the 29.5% / 60.7% / 9.9% split of the 0.30
#     envelope they had at suite=all.
#
# How close that is, stated once and referenced from CLAUDE.md and
# docs/PARAMETERS.md:
#
#   These four defaults are the closest scalar approximation to the
#   retired scheme, not an exact reproduction of it, and no scalar can be
#   exact.  The old envelope handed a heuristic
#   `budget × max(1 − N/(80e), ¼) × w/Σw_enabled`, which depends on two
#   runtime facts a constant here cannot see: the worker count `N`, and
#   which *other* heuristics the suite enabled.  These reproduce the
#   `suite=all` share with the worker-count term dropped, so they are
#   exact at neither end.  At `suite=all` they run 1.04x the old budget
#   at `N=1`, 1.33x at `N=6`, 2x at `N=12` and 4x from `N=18`, where the
#   old quarter-floor capped the FJ deduction; at a single-heuristic
#   suite — which used to hand the sole enabled heuristic the entire
#   envelope — they run 0.29x / 0.61x / 0.10x for fpr / local_mip /
#   scylla.  FJ is the one exact case, at every `N` and every suite.  The
#   anchor is a choice, not a constraint: the deviation is smallest at
#   the low worker counts the test suite actually runs at (1–2 on CI,
#   `(hardware_concurrency()+1)/2` locally), and erring high is the cheap
#   direction once stall thresholds are absolute (#111) — an over-large
#   budget is truncated by a gate that fires, an under-large one is a
#   hard cap nothing recovers.  Treat all four as the starting point of
#   #106's calibration, not the result of one.
#
# All three insertions of every option anchor on *upstream's*
# mip_heuristic_run_shifting text, like the suite block above.  Anchoring
# on our own inserted text is what made the older option blocks a chain,
# where deleting one silently dropped the next from the build.
#
# Ordering invariant, and the reason this is one loop rather than four
# hand-written blocks: every insertion lands directly after the same
# anchor, so the resulting declaration order is the *reverse* of the
# application order.  That is harmless only while the member list and the
# constructor initializer list come out in the same relative sequence —
# otherwise GCC's -Wreorder fires on HighsOptions' constructor, which is a
# warning about our patch in a file nobody reads.  One loop applying the
# member and ctor insertions together, in one iteration order, makes that
# true by construction: both lists come out in the reverse of the list
# below, with the suite option last (its block ran first).  Add an option
# by appending to the list below — of any kind; the loop dispatches on
# the entry's `kind` field — never by writing a separate block with an
# ordering of its own.  A second loop would keep the invariant only by
# accident.
#
# Record registrations are appended after the same anchor too, so they
# come out in that same reversed order; nothing depends on it, but each
# one still needs its own sanity check.  A missed *record* insertion is
# the silent failure: the header still compiles, the option keeps its
# constructor default, and every setOptionValue for it fails with no
# diagnostic anywhere.
#
# ── The stall-threshold options, and the presolve-only switch (#106) ──
#
# The four `mip_heuristic_<name>_stall` options were `constexpr` values in
# each heuristic's own header until #106.  They are the parameter that
# actually limits a presolve dispatch — a 64x sweep of the LocalMIP effort
# option moved median presolve wall time by under 4%, because the stall
# gate, not the budget, is what stops the search — and a constant cannot
# be swept without a rebuild per point, so the calibration could not reach
# it at all.
#
# Units are effort units per constraint-matrix nonzero, and they are **not
# comparable across heuristics**: FJ counts step units, FPR and LocalMIP
# coefficient accesses, Scylla PDLP iterations x nnz.  Scope follows each
# heuristic's effort option — FJ's is per worker, the other three size a
# whole dispatch (see `kChain` in src/mode_dispatch.cpp).  The defaults
# reproduce the pre-#111 gate at each heuristic's shipped default effort,
# rounded to the neighbouring power of two, and are provisional.
#
# **0 means no staleness gate at all**, not "give up immediately" — see
# `stall_threshold` in src/heuristic_common.h.  That is why the range is
# `[0, kHighsIInf]` and why the zero end is load-bearing: a search of the
# stall axis needs a point where the gate provably never fires.
#
# `mip_heuristic_presolve_only` exits the solve after the presolve
# heuristic chain, before the root LP, keeping whatever incumbent the
# chain produced.  It is what makes a presolve-heuristic measurement
# possible at all: in a full solve a heuristic runs for ~2 s of a 60 s
# limit and B&B owns the rest, so the campaign's primal-integral metric
# dilutes the thing being tuned into seed noise.  The two alternatives
# that look like they should work do not — `mip_max_nodes = 0` is checked
# inside the B&B loop, so the root LP and the dive heuristics all run
# first, and `mip_root_presolve_only` controls where presolve is applied,
# not termination.  The early exit itself is inserted into
# HighsMipSolver.cpp further down.

# The upper bound of an effort option.  1.0 was the natural ceiling while
# the option only ever expressed a fraction of a fixed envelope, and it is
# still the top of the range anything ships or tunes at.  #113's
# calibration probe needs one configuration the bound cannot express: a
# budget so large that it never binds, so the only thing that stops a
# heuristic is the wall clock and the trace measures the heuristic rather
# than the setting being derived from it.  `1e4` is `2e5` times the
# vanilla anchor — unreachable inside any per-run cap the campaign uses,
# and two orders of magnitude below where `nnz << 12` times it would
# trouble a `size_t` on the largest MIPLIB model.
set(kEffortMax "1e4")

# No `;` and no `:` in a list entry: set() builds a cmake list and would
# split on the first, and the field split below uses the second.
# Fields are <identifier>:<kind>:<record default>:<record bound>:<record
# description>, where <kind> is one of double / int / bool.  <record
# bound> is the record's upper bound, or `-` for a kind that has none
# (bool); it sits *before* the description because the description is the
# one field allowed to contain anything, so it has to be last.
set(_patch_options
    "mip_heuristic_fj_effort:double:0.0125:${kEffortMax}:Per-worker effort budget multiplier for the FeasibilityJump presolve heuristic"
    "mip_heuristic_fpr_effort:double:0.0884:${kEffortMax}:Effort budget multiplier for the FPR presolve heuristic"
    "mip_heuristic_local_mip_effort:double:0.1821:${kEffortMax}:Effort budget multiplier for the LocalMIP presolve heuristic"
    "mip_heuristic_scylla_effort:double:0.0296:${kEffortMax}:Effort budget multiplier for the Scylla presolve heuristic"
    "mip_heuristic_fj_stall:int:256:kHighsIInf:Per-worker staleness threshold for the FeasibilityJump presolve heuristic, in effort units per matrix nonzero (0 disables the gate)"
    "mip_heuristic_fpr_stall:int:2048:kHighsIInf:Staleness threshold for the FPR presolve heuristic, in effort units per matrix nonzero (0 disables the gate)"
    "mip_heuristic_local_mip_stall:int:4096:kHighsIInf:Staleness threshold for the LocalMIP presolve heuristic, in effort units per matrix nonzero (0 disables the gate)"
    "mip_heuristic_scylla_stall:int:512:kHighsIInf:Staleness threshold for the Scylla presolve heuristic, in effort units per matrix nonzero (0 disables the gate)"
    "mip_heuristic_presolve_only:bool:false:-:Exit the solve after the presolve heuristic chain, before the root LP, keeping the incumbent it found")

# The upstream record block all four record insertions anchor on, spelled
# once: four copies of a four-line exact-match string is four chances for
# one of them to drift.
string(CONCAT _shifting_record
    "record_bool = new OptionRecordBool(\"mip_heuristic_run_shifting\",\n"
    "                                       \"Use the Shifting heuristic\", advanced,\n"
    "                                       &mip_heuristic_run_shifting, false);\n"
    "    records.push_back(record_bool);")

file(READ "${LP_DATA_DIR}/HighsOptions.h" OPTIONS_CONTENT)
set(_opt_applied FALSE)
foreach(_entry IN LISTS _patch_options)
    string(REGEX REPLACE "^([^:]+):.*$" "\\1" _opt_ident "${_entry}")
    string(REGEX REPLACE "^[^:]+:([^:]+):.*$" "\\1" _opt_kind "${_entry}")
    string(REGEX REPLACE "^[^:]+:[^:]+:([^:]+):.*$" "\\1" _opt_default "${_entry}")
    string(REGEX REPLACE "^[^:]+:[^:]+:[^:]+:([^:]+):.*$" "\\1" _opt_bound "${_entry}")
    string(REGEX REPLACE "^[^:]+:[^:]+:[^:]+:[^:]+:(.*)$" "\\1" _opt_desc "${_entry}")

    # Everything the kind decides, in one place.  `_opt_ctor_value` is
    # deliberately *not* the record default: OptionRecord* writes the record
    # default over the member at registration, so the constructor initializer
    # only decides what a *missed* record insertion would leave behind — a
    # stable zero rather than an uninitialised read.  `_opt_ctor_regex` is
    # that same value escaped for the sanity check below.
    #
    # The lower bound is a property of the kind — every one of these is
    # non-negative, and `0` means the same thing for all of them (an inert
    # heuristic, a disabled gate).  The *upper* bound is the entry's own,
    # because it stopped being uniform when #113 needed an effort budget
    # that cannot bind; a bool carries `-` and uses neither.
    if(_opt_kind STREQUAL "double")
        set(_opt_cxx_type "double")
        set(_opt_ctor_value "0.0")
        set(_opt_ctor_regex "\\(0\\.0\\)")
        set(_opt_record_var "record_double")
        set(_opt_record_ctor "OptionRecordDouble")
        set(_opt_record_args "0.0, ${_opt_default}, ${_opt_bound}")
    elseif(_opt_kind STREQUAL "int")
        set(_opt_cxx_type "HighsInt")
        set(_opt_ctor_value "0")
        set(_opt_ctor_regex "\\(0\\)")
        set(_opt_record_var "record_int")
        set(_opt_record_ctor "OptionRecordInt")
        set(_opt_record_args "0, ${_opt_default}, ${_opt_bound}")
    elseif(_opt_kind STREQUAL "bool")
        set(_opt_cxx_type "bool")
        set(_opt_ctor_value "false")
        set(_opt_ctor_regex "\\(false\\)")
        set(_opt_record_var "record_bool")
        set(_opt_record_ctor "OptionRecordBool")
        set(_opt_record_args "${_opt_default}")
    else()
        message(FATAL_ERROR
            "apply_patch.cmake: unknown option kind '${_opt_kind}' for "
            "${_opt_ident}. Add it to the kind dispatch in this file; the "
            "entry format is "
            "<identifier>:<kind>:<default>:<bound>:<description>.")
    endif()

    # A missing field does not fail the regexes above — it leaves the
    # variable holding the whole entry — so an entry written in the old
    # four-field format would reach the record as a bound spelled like a
    # description, and the first sign of it would be a HiGHS compile error
    # a full clean rebuild away.
    if(NOT _opt_kind STREQUAL "bool" AND NOT _opt_bound MATCHES "^[A-Za-z0-9_.+-]+$")
        message(FATAL_ERROR
            "apply_patch.cmake: option '${_opt_ident}' has no usable upper "
            "bound (got '${_opt_bound}'). The entry format is "
            "<identifier>:<kind>:<default>:<bound>:<description>.")
    endif()

    string(FIND "${OPTIONS_CONTENT}" "${_opt_ident}" _opt_found)
    if(NOT _opt_found EQUAL -1)
        message(STATUS "${_opt_ident} option already applied, skipping")
        continue()
    endif()

    # Member variable: insert after mip_heuristic_run_shifting
    string(REPLACE
      "bool mip_heuristic_run_shifting;\n"
      "bool mip_heuristic_run_shifting;\n  ${_opt_cxx_type} ${_opt_ident};\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Constructor initializer: insert after mip_heuristic_run_shifting(false),
    string(REPLACE
      "mip_heuristic_run_shifting(false),\n"
      "mip_heuristic_run_shifting(false),\n        ${_opt_ident}(${_opt_ctor_value}),\n"
      OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Record registration: insert after the mip_heuristic_run_shifting record
    string(CONCAT _opt_record
      "${_shifting_record}\n"
      "\n"
      "    ${_opt_record_var} = new ${_opt_record_ctor}(\n"
      "        \"${_opt_ident}\",\n"
      "        \"${_opt_desc}\", advanced,\n"
      "        &${_opt_ident}, ${_opt_record_args});\n"
      "    records.push_back(${_opt_record_var});")
    string(REPLACE "${_shifting_record}" "${_opt_record}" OPTIONS_CONTENT "${OPTIONS_CONTENT}")

    # Sanity checks: all three insertions must land, per option.
    #
    # Match the member declaration *without* its trailing semicolon: cmake
    # splits a matched string containing `;` into list elements, which would
    # make list(LENGTH) report 2 for a single hit.  The type prefix is what
    # keeps this from also matching the ctor init or the record.
    string(REGEX MATCHALL "${_opt_cxx_type} ${_opt_ident}" _opt_member_hits "${OPTIONS_CONTENT}")
    list(LENGTH _opt_member_hits _opt_member_count)
    string(REGEX MATCHALL "${_opt_ident}${_opt_ctor_regex}" _opt_ctor_hits "${OPTIONS_CONTENT}")
    list(LENGTH _opt_ctor_hits _opt_ctor_count)
    # string(FIND), not REGEX MATCHALL: the record text contains semicolons.
    # The quoted identifier occurs only in the record — member and ctor
    # spell it bare.
    string(FIND "${OPTIONS_CONTENT}" "\"${_opt_ident}\"" _opt_record_idx)
    if(NOT _opt_member_count EQUAL 1 OR NOT _opt_ctor_count EQUAL 1 OR _opt_record_idx EQUAL -1)
        message(FATAL_ERROR
            "HighsOptions.h post-patch sanity check failed for "
            "${_opt_ident} (member=${_opt_member_count}, "
            "ctor=${_opt_ctor_count}, record_idx=${_opt_record_idx}). "
            "Upstream HiGHS likely reformatted HighsOptions.h so one of the "
            "three mip_heuristic_run_shifting anchors no longer matches. "
            "${CLEAN_REBUILD}")
    endif()
    set(_opt_applied TRUE)
    message(STATUS "Applied ${_opt_ident} option to HighsOptions.h")
endforeach()
# Only on a change: an unconditional write would restamp the header's mtime
# on every configure and rebuild all of HiGHS behind it.
if(_opt_applied)
    file(WRITE "${LP_DATA_DIR}/HighsOptions.h" "${OPTIONS_CONTENT}")
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
string(FIND "${MIPDATA_H}" "kSolutionSourceFprLp" _src_enum_found)
if(_src_enum_found EQUAL -1)
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

string(FIND "${MIPDATA_CPP}" "kSolutionSourceFprLp" _src_cpp_found)
if(_src_cpp_found EQUAL -1)
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

# ── Patch feasibilityjump.hh: drop upstream FJ's per-bump log line ──────
# `updateWeights()` logs "Reached a local minimum." at kVerbose, once per
# weight bump, from every parallel FJ worker, each with an fflush.  Our own
# `[Heur]` / `[HeurSol]` instrumentation is kVerbose too, so anything that
# reads a trace pays for this line as well — and it is not a rounding cost.
# Measured on one 30 s presolve-only probe run of `50v-10` (2745 nonzeros,
# 16 workers): **453 MB**, of which 99.8 % is this line, which puts the
# 233-instance probe of #113 past 400 GB.
#
# It is also a measurement error and not only a disk problem.  Such a run is
# bounded by the wall clock, so an fflush per bump is time FJ does not spend
# searching, and it lands asymmetrically — FJ logs this per bump while the
# other three heuristics barely log at all, so a traced FJ dispatch is a
# different regime from an untraced one.  That is where nearly all of the
# documented 1.1-4.4x `log_dev_level=3` cost lives.
#
# The line is dropped rather than moved to a quieter level: kVerbose is the
# quietest level there is, and adding an option to gate it is a larger
# change to upstream code than removing one diagnostic.
#
# Consequence to keep in mind: at `log_dev_level=3` a patched binary's FJ
# output is no longer identical to vanilla's, and `suite=off` is faster than
# vanilla there.  The vanilla-equivalence claim is at the default log level,
# which is where `bench/check_vanilla_equivalence.py` makes it.
file(READ "${MIP_DIR}/feasibilityjump.hh" FJ_CONTENT)
string(FIND "${FJ_CONTENT}" "Reached a local minimum" _fj_log_found)
if(NOT _fj_log_found EQUAL -1)
    string(REPLACE
      "    highsLogDev(logOptions, HighsLogType::kVerbose,\n                FJ_LOG_PREFIX \"Reached a local minimum.\\n\");"
      "    // mip-heuristics: per-bump local-minimum log removed.  It is\n    // kVerbose, the same level our [Heur]/[HeurSol] trace needs, and one\n    // fflushed line per weight bump per worker is 99.8% of a traced run's\n    // log volume and most of level 3's wall-time cost."
      FJ_CONTENT "${FJ_CONTENT}")

    string(FIND "${FJ_CONTENT}" "Reached a local minimum" _fj_log_check)
    if(NOT _fj_log_check EQUAL -1)
        message(FATAL_ERROR
            "feasibilityjump.hh patch failed: the updateWeights() log anchor "
            "no longer matches (upstream reformat?). Update the FJ log patch "
            "in third_party/highs_patch/apply_patch.cmake. ${CLEAN_REBUILD}")
    endif()

    file(WRITE "${MIP_DIR}/feasibilityjump.hh" "${FJ_CONTENT}")
    message(STATUS "Removed upstream FJ per-bump log line from feasibilityjump.hh")
else()
    message(STATUS "FJ per-bump log line already removed, skipping")
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

# Defensive check, and the one place a retired option name still earns its
# keep: this file carries no version marker, so the HighsOptions.h gate
# above cannot speak for it, and the idempotency sentinel below
# ('heuristics::run_presolve') is present in every layout the call site has
# ever had.  An in-place upgrade would therefore keep the old call site,
# which computed one shared budget from mip_heuristic_presolve_effort and
# passed it in — an option that no longer exists, and an arity run_presolve
# no longer has.  Force a clean rebuild with an actionable message rather
# than a confusing compile error.
string(FIND "${CONTENT}" "mip_heuristic_presolve_effort" _stale_presolve_budget)
if(NOT _stale_presolve_budget EQUAL -1)
    message(FATAL_ERROR
        "HighsMipSolver.cpp derives one shared presolve heuristic budget from "
        "'mip_heuristic_presolve_effort'; that option was replaced by the four "
        "per-heuristic 'mip_heuristic_<name>_effort' options, which "
        "heuristics::run_presolve now reads itself. "
        "${CLEAN_REBUILD}")
endif()

string(FIND "${CONTENT}" "heuristics::run_presolve" _found)
if(_found EQUAL -1)
    # Add includes at top (after existing includes)
    string(REPLACE
      "#include \"mip/HighsMipSolver.h\""
      "#include \"mip/HighsMipSolver.h\"\n#include \"fpr_lp.h\"\n#include \"mode_dispatch.h\""
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

    # Patch A2: insert custom heuristics block via mode_dispatch.  The call
    # is bare: each heuristic's budget comes from its own
    # mip_heuristic_<name>_effort option, derived inside run_presolve, so
    # the patch string carries no budget arithmetic to drift out of sync
    # with the source tree.
    #
    # Patch A3 rides along directly behind it: the mip_heuristic_presolve_only
    # early exit (#106).  It takes the same path the infeasibility exit above
    # takes — this is the last point in `run()` before `evaluateRootNode`, so
    # returning here is what "before the root LP" means — with a status that
    # keeps the incumbent instead of discarding it.
    #
    # The status is `kSolutionLimit`, and the choice is not arbitrary:
    #
    #  * It is what HiGHS itself assigns when a user-configured *search-size*
    #    limit stops the solve — `mip_max_nodes`, `mip_max_leaves` and
    #    `mip_max_improving_sols` all set it in `HighsMipSolverData::
    #    checkLimits`.  Presolve-only is that same kind of limit; it is
    #    morally `mip_max_nodes = 0`, which is the option this feature exists
    #    because HiGHS checks *inside* the B&B loop, after the root LP.
    #  * `cleanupSolve` overwrites only `kNotset` and `kInfeasible`, so this
    #    survives to the caller.  `kNotset` would be rewritten to `kOptimal`
    #    whenever the chain found anything — claiming optimality for a solve
    #    that never computed a dual bound — and `kInfeasible` would do the
    #    same, having lied in the log line on the way.
    #  * It maps to `HighsStatus::kWarning`, so a caller sees that the solve
    #    did not run to completion, and the solution is still extracted:
    #    `Highs::callSolveMip` keys that off `solution_objective_`, never off
    #    the model status.
    #  * It stays honest when the chain found nothing: status kSolutionLimit,
    #    infinite primal bound, solution status "-".
    #
    # No `!submip` guard: presolve-only never reaches B&B, so it never
    # reaches the dive heuristics that build sub-MIPs, and no sub-MIP can be
    # constructed under this option in the first place.
    string(REPLACE
      "    }\n    // End of pre-root-node heuristics"
      "    }\n    if (heuristics::run_presolve(*this)) {\n      modelstatus_ = HighsModelStatus::kInfeasible;\n      cleanupSolve();\n      return;\n    }\n    if (options_mip_->mip_heuristic_presolve_only) {\n      // mip-heuristics: stop before the root LP, keeping the incumbent.\n      // kSolutionLimit is what HiGHS assigns for its own search-size\n      // limits (mip_max_nodes/leaves/improving_sols), and cleanupSolve\n      // leaves it alone, so the presolve incumbent is reported as found.\n      modelstatus_ = HighsModelStatus::kSolutionLimit;\n      cleanupSolve();\n      return;\n    }\n\n    // End of pre-root-node heuristics"
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
    #
    # A3's miss is quieter still: `mip_heuristic_presolve_only` would be a
    # registered option that silently does nothing, so a presolve-only
    # screen would run full solves and score them as if they had stopped
    # after presolve.  It shares A2's anchor, so in practice it lands or
    # misses with it — check it anyway, because it is the half whose
    # failure produces plausible numbers rather than none.
    string(FIND "${CONTENT}" "heuristics::run_presolve" _presolve_check)
    string(FIND "${CONTENT}" "native FJ only at suite=off" _fj_off_check)
    string(FIND "${CONTENT}" "mip_heuristic_presolve_only" _presolve_only_check)
    if(_presolve_check EQUAL -1 OR _fj_off_check EQUAL -1 OR _presolve_only_check EQUAL -1)
        message(FATAL_ERROR
            "HighsMipSolver.cpp presolve patch failed "
            "(run_presolve=${_presolve_check}, fj_disable=${_fj_off_check}, "
            "presolve_only=${_presolve_only_check}). "
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

