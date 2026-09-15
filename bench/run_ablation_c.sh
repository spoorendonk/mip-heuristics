#!/usr/bin/env bash
# Ablation C — what `fpr_lp` contributes, and whether it contributes anything
# at all (issue #165).
#
# Usage:
#   bench/run_ablation_c.sh capability [hours]    stage C0 — the gate
#   bench/run_ablation_c.sh contribution [hours]  stage C1 — the paired arm
#   bench/run_ablation_c.sh status
#
# Like every other campaign stage this is `run_mipfeas.sh` with a different
# environment rather than a different runner (#109), so the tree, the `.opts`
# record and the log shapes match and `analyze_results.py` reads them without
# special cases.  Read the tree with `bench/analyze_ablation_c.py`.
#
# ── the two stages, and why in this order ───────────────────────────────────
#
# `fpr_lp` ships **off** (`mip_heuristic_fpr_lp_effort = 0`) and has never been
# measured.  Its budget is zero-sum against upstream's RENS/RINS LP-iteration
# envelope, so shipping it unmeasured is not neutral in either direction.
#
# **C0 is a capability test, and that is the whole point of running it first.**
# Its output is a *count* — how many dive dispatches happened, and how many
# produced an accepted incumbent — so it needs no power calculation and a
# result of zero is decisive at any n.  Ablation B's expensive lesson was
# sizing a mean-difference experiment to what was affordable and then reading
# its null as an answer; a count sidesteps that entirely.
#
# C0 gives `fpr_lp` every advantage it could ask for: it runs **alone**, owns
# the whole envelope with RENS/RINS disabled, and gets a per-call budget that
# cannot bind.  That is deliberate — the shipped chain squeezes it from three
# sides (RENS/RINS drain the envelope first, `headroom_iters <= 0`, and
# `max_effort < nnz << 8`), so a null measured there invites "you never gave
# it a chance".  C0 removes the objection rather than arguing with it.
#
# It is also self-diagnosing, which is what makes a short limit safe: the run
# reports its own adequacy.  Plentiful dispatches with zero yields is a
# decisive null; near-zero dispatches means the limit was too short and the
# right response is to extend it, not to conclude.
#
# **C1 only runs if C0 yields.**  It completes the `E-shipped-plus-fprlp` arm
# — the shipped defaults with `mip_heuristic_fpr_lp_effort=1.0` and nothing
# else changed — against `D-shipped`, whose runs are already on disk.  That
# pairing is much better powered than anything in Ablation B, because the two
# arms differ in exactly one option, so configuration cancels along with
# instance difficulty.
#
# ── what is deliberately NOT set ────────────────────────────────────────────
#
# **No `threads=1`.**  It was in the issue's original design to stop
# `parallelLockActive()` skipping the heuristic, and that premise is wrong:
# `src/fpr_lp.cpp` says the lock is held only under multi-worker B&B, which is
# not HiGHS's default, and both existing arms confirm it — every log reads
# `Thread count 16 (of 32 threads). Using 1 max workers. Parallel search off`.
# Pinning it would have measured a regime nothing ships in.
#
# `MIPFEAS_DEV_LOG=1` on C0 and **not** on C1, which is not an oversight in
# either direction.  C0 needs the `[Heur] name=fpr_lp phase=dive` lines: they
# are the dispatch count, and they are emitted at `log_dev_level=3` only.  C1
# must not have them, because its 20 existing runs are untraced and its
# `D-shipped` pair is untraced, and tracing is not free — mixing traced and
# untraced runs inside one paired arm would confound the comparison with the
# tracing.  C1 reads its yields from the ordinary log instead: the `D`
# solution-source character is in the incumbent lines at any log level.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
RESULTS="${ABLATION_C_RESULTS:-$REPO/bench/results/ablation_c}"
FINALISTS="$REPO/bench/results/finalists"

# A binary outside build/, because `pre-push` runs `rm -rf build` and would
# delete it out from under a running campaign (CLAUDE.md).  This is also the
# exact binary both existing finalist arms ran on, which C1 requires: its 20
# runs and their `D-shipped` pair must not straddle a rebuild.
BINARY="${MIP_HEURISTICS_BINARY:-$REPO/bench/results/irace/bin/highs}"

# The stratified 49 the confirmation stage uses.  C0 shares it with C1 so a
# fire label and a paired outcome land on the same instance and can be read
# against each other.
INSTANCES="${ABLATION_C_INSTANCES:-$REPO/bench/instances_confirm48.txt}"

# 120 s: long enough for B&B to reach dive nodes, short enough that the gate
# costs ~1.3 h.  The stage reports its own dispatch count, so an inadequate
# limit shows up as "too few dispatches to conclude" rather than as a null.
C0_TIME_LIMIT="${ABLATION_C0_TIME_LIMIT:-120}"

cmd_capability() {
	local hours=${1:-3}
	# Every option here is verified present on the binary by run_benchmark.py's
	# own pre-flight (it probes each key and refuses before the first solve).
	local opts=(
		# The per-call budget cannot bind.  The share multiplies the whole
		# `min(headroom, cap)` since #164, so a value above 1.0 buys a deeper
		# dive rather than being absorbed by the cap.
		"mip_heuristic_fpr_lp_effort=100"
		# fpr_lp owns the envelope: the two heuristics it is zero-sum against
		# do not draw from it first.
		"mip_heuristic_run_rens=false"
		"mip_heuristic_run_rins=false"
		# ...and the envelope itself is at its ceiling.
		"mip_heuristic_effort=1.0"
	)
	echo "=== C0 capability: fpr_lp alone, unbounded, ${C0_TIME_LIMIT}s, $(grep -c '^[^#]' "$INSTANCES") instances"
	MIPFEAS_CONFIGS="fpr_lp" \
	MIPFEAS_OUTPUT="$RESULTS/capability" \
	MIPFEAS_INSTANCES="$INSTANCES" \
	MIPFEAS_TIME_LIMIT="$C0_TIME_LIMIT" \
	MIPFEAS_BINARY="$BINARY" \
	MIPFEAS_DEV_LOG=1 \
	MIPFEAS_ANALYZE=0 \
	MIPFEAS_EXTRA_OPTIONS="${opts[*]}" \
		"$REPO/bench/run_mipfeas.sh" next "$hours"
}

cmd_contribution() {
	local hours=${1:?usage: contribution <hours>}
	# Not a second definition of the arm: it is a `finalists.json` entry, run
	# by the launcher that owns it.  `run_mipfeas.sh` resumes per
	# (config, instance, seed), so a repeated call adds what is missing and
	# repeats nothing.
	#
	# The arm is `F-selected-plus-fprlp` -- the configuration #107 selected,
	# plus `fpr_lp`.  Its control is that configuration's own 49 confirmation
	# runs, already on disk, so this stage costs one arm.
	#
	# `E-shipped-plus-fprlp` is C1's first run, complete at n=49 and kept as
	# the record of what `fpr_lp` did against the *previous* four-heuristic
	# vector.  That background was retired by #107, so the contrast had to be
	# remeasured rather than carried over -- the arm is superseded, not wrong.
	#
	# **Deliberately on the same binary as the control**
	# (`run_finalists.sh`'s own default, `bench/results/irace/bin/highs`),
	# not on the freshly built one.  Both arms pass all nine options
	# explicitly, so the binary's *defaults* are not read by either, and
	# using the control's binary removes the build as a variable.  The
	# PATCH_VERSION bumps between that binary and today's changed defaults,
	# help strings and comments only.
	local arm="${ABLATION_C_ARM:-F-selected-plus-fprlp}"
	echo "=== C1 contribution: $arm against its fpr_lp-free control"
	CONFIRM_INSTANCES="$INSTANCES" \
	FINALISTS_RESULTS="$FINALISTS" \
	FINALISTS_ONLY="$arm" \
		"$REPO/bench/run_finalists.sh" confirm "$hours"
}

# `set -o pipefail` is on, and a glob that matches nothing makes `ls` exit
# non-zero, which would abort the whole status report on the first arm not yet
# run.  Counting through a subshell that swallows that is the point.
count_logs() {
	{ ls "$@" 2>/dev/null || true; } | wc -l
}

cmd_status() {
	local total
	total=$(grep -c '^[^#]' "$INSTANCES")
	printf 'C0 capability    %s / %s runs\n' \
		"$(count_logs "$RESULTS/capability/fpr_lp/seed0"/*.log)" "$total"
	printf 'C1 contribution  %s / %s runs (F-selected-plus-fprlp)\n' \
		"$(count_logs "$FINALISTS/confirm/F-selected-plus-fprlp"/*/seed0/*.log)" "$total"
	printf "C1 control       %s / %s runs (B'-mix-cheapest, already on disk)\n" \
		"$(count_logs "$FINALISTS/confirm/B'-mix-cheapest"/*/seed0/*.log)" "$total"
	printf 'C1 superseded    %s / %s runs (E-shipped-plus-fprlp, previous background)\n' \
		"$(count_logs "$FINALISTS/confirm/E-shipped-plus-fprlp"/*/seed0/*.log)" "$total"
}

case "${1:-status}" in
capability) shift; cmd_capability "$@" ;;
contribution) shift; cmd_contribution "$@" ;;
status) cmd_status ;;
*) echo "usage: $0 {capability [hours]|contribution <hours>|status}" >&2; exit 1 ;;
esac
