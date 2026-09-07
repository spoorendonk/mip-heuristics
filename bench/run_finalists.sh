#!/usr/bin/env bash
# #107 stage 2: validate the finalists on held-out instances, then confirm
# them with full timed solves.
#
# Usage:
#   bench/run_finalists.sh heldout [count]   presolve-only, instances the
#                                            search never saw
#   bench/run_finalists.sh confirm <hours>   full timed solves at the
#                                            campaign limit, chunked
#   bench/run_finalists.sh status
#
# The finalists are `bench/finalists.json`, and the selection rule that
# produced them is fixed in `bench/irace/PREREGISTRATION.md`. This script
# executes; it chooses nothing.
#
# ── why two stages, and why the second is not optional ──────────────────────
#
# `heldout` answers "does the selection survive instances it was not tuned
# on" -- the search optimises over the 90-instance tuning set, and the winner
# of a search over hundreds of configurations is biased upward by roughly
# `sigma * sqrt(2 ln n)`. It runs the *same* presolve-only objective, so it
# is directly comparable with the search and cheap: 143 instances at seconds
# each.
#
# `confirm` runs on `bench/instances_confirm.txt` -- a stratified 20-instance
# subset of the tuning set, not the whole of it.  That is a deviation from the
# pre-registration and is recorded there as a dated amendment, signed before
# any confirmation run: the full stage costs ~9.7 h per finalist (measured),
# the design is paired so instance difficulty cancels, and reduced power
# resolves into the "indistinguishable -> simpler" clause the selection rule
# already carries.  n=20 is a starter; extend with CONFIRM_INSTANCES and
# record the extension in the amendment.
#
# `confirm` answers a different question from `heldout`, and no amount of
# screening can answer it. The screen scores presolve-exit gap; the campaign scores primal
# integral over 600 s. A configuration that spends more presolve time to exit
# with a better incumbent can still lose, because that time comes out of the
# B&B search which is where most of the integral is earned. Measured cost:
# ~9.7 h per finalist over the tuning set, so this is chunked and resumable
# rather than run in one window.
#
# Runs go through run_benchmark.py so the tree, the .opts record and the log
# shapes match every other campaign stage and `analyze_results.py` reads them
# without special cases.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
FINALISTS="$HERE/finalists.json"
RESULTS="${FINALISTS_RESULTS:-$REPO/bench/results/finalists}"

# A binary outside build/, because `pre-push` runs `rm -rf build` and would
# delete it out from under a running campaign (CLAUDE.md).
BINARY="${MIP_HEURISTICS_BINARY:-$REPO/bench/results/irace/bin/highs}"

names() { python3 -c "import json;print(' '.join(json.load(open('$FINALISTS'))['finalists']))"; }

# The `mip_heuristic_*` options one finalist is, as --extra-options arguments.
opts_for() {
	python3 - "$1" <<'PY'
import json, sys
name = sys.argv[1]
f = json.load(open("bench/finalists.json"))["finalists"][name]
out = []
for h, v in f["efforts"].items():
    out.append(f"mip_heuristic_{h}_effort={v}")
for h, v in f["patiences"].items():
    out.append(f"mip_heuristic_{h}_patience={v}")
print(" ".join(out))
PY
}

# The harness config name for a finalist: the `+`-spelled suite of exactly the
# heuristics whose effort is positive.  Effort 0 is off, so the suite is
# *derived* from the eight numbers rather than stored beside them -- a stored
# suite could disagree with them, and `run_benchmark.py` raises on a name it
# does not know, which is what keeps this honest.  The finalist name is the
# results *tree*; the config name is the suite, because that is the harness's
# own vocabulary.
config_for() {
	python3 - "$1" <<'CFG'
import json, sys
f = json.load(open("bench/finalists.json"))["finalists"][sys.argv[1]]
on = [h for h, v in f["efforts"].items() if v > 0]
print("+".join(on) if len(on) < 4 else "all")
CFG
}

cmd_heldout() {
	local count=${1:-0}
	local list="$RESULTS/heldout-instances.txt"
	mkdir -p "$RESULTS"
	# PLATO minus the tuning set, derived rather than stored: a third tracked
	# list would drift from the two it is defined by.
	comm -23 \
		<(awk 'NF && $1 !~ /^#/ {print $1}' "$REPO/bench/instances_plato.txt" | sort) \
		<(awk 'NF && $1 !~ /^#/ {print $1}' "$REPO/bench/instances_tuning.txt" | sort) \
		> "$list"
	echo "held-out instances: $(wc -l < "$list")"

	local name opts config
	for name in $(names); do
		opts=$(opts_for "$name")
		config=$(config_for "$name")
		echo "=== $name (config $config) : $opts"
		if [ "$count" != "0" ]; then export PLATO_COUNT="$count"; else unset PLATO_COUNT || true; fi
		PLATO_CONFIGS="$config" \
		PLATO_OUTPUT="$RESULTS/heldout/$name" \
		PLATO_INSTANCES="$list" \
		PLATO_TIME_LIMIT=60 \
		PLATO_BINARY="$BINARY" \
		PLATO_DEV_LOG=1 \
		PLATO_ANALYZE=0 \
		PLATO_EXTRA_OPTIONS="$opts mip_heuristic_presolve_only=true" \
			"$REPO/bench/run_plato.sh" next 6
	done
}

cmd_confirm() {
	local hours=${1:?usage: confirm <hours>}
	local name opts config
	for name in $(names); do
		opts=$(opts_for "$name")
		config=$(config_for "$name")
		echo "=== $name (config $config, full limit) : $opts"
		PLATO_CONFIGS="$config" \
		PLATO_OUTPUT="$RESULTS/confirm/$name" \
		PLATO_INSTANCES="${CONFIRM_INSTANCES:-$REPO/bench/instances_confirm.txt}" \
		PLATO_TIME_LIMIT=600 \
		PLATO_BINARY="$BINARY" \
		PLATO_ANALYZE=0 \
		PLATO_EXTRA_OPTIONS="$opts" \
			"$REPO/bench/run_plato.sh" next "$hours"
	done
}

cmd_status() {
	local name stage
	for stage in heldout confirm; do
		echo "$stage  ($RESULTS/$stage)"
		for name in $(names); do
			printf '  %-18s %s runs\n' "$name" \
				"$(ls "$RESULTS/$stage/$name"/*/seed0/*.log 2>/dev/null | wc -l)"
		done
	done
}

case "${1:-status}" in
heldout) shift; cmd_heldout "$@" ;;
confirm) shift; cmd_confirm "$@" ;;
status) cmd_status ;;
*) echo "usage: $0 {heldout [count]|confirm <hours>|status}" >&2; exit 1 ;;
esac
