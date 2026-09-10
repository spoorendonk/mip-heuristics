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

# Arms to run.  `confirm: false` in finalists.json records a selection that
# is kept for the record but not run -- see its `excluded_because`.
#
# `FINALISTS_ONLY` names arms explicitly and overrides that flag.  It exists
# for a stage that owns one arm rather than the finalist set: Ablation C
# (#165) runs `E-shipped-plus-fprlp` against a `D-shipped` already on disk,
# and flipping `confirm` in the tracked JSON to do that would mean editing the
# record of what Ablation B selected in order to run something that is not one
# of its finalists.  Named arms are validated against the JSON, so a typo is
# an error rather than a silently empty run.
names() {
	python3 - "$FINALISTS" "${FINALISTS_ONLY:-}" <<'NAMES'
import json, sys

finalists = json.load(open(sys.argv[1]))["finalists"]
only = sys.argv[2].split()
if only:
    missing = [n for n in only if n not in finalists]
    if missing:
        sys.exit("unknown finalist(s): " + " ".join(missing))
    print(" ".join(only))
else:
    print(" ".join(k for k, v in finalists.items() if v.get("confirm", True)))
NAMES
}

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
# `extra_options` is for a finalist that differs from the eight numbers in some
# other way -- #165's arm, which is the incumbent plus `fpr_lp`. It is spelled
# out rather than derived because `fpr_lp` is not one of the eight: it draws
# from upstream's RENS/RINS LP-iteration envelope, not from `nnz << 10`, so it
# has no effort entry to read.
out.extend(f.get("extra_options", []))
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
# An explicit suite in extra_options wins: it is the only way to name a
# heuristic that is not one of the four efforts, which is what #165's arm needs.
for opt in f.get("extra_options", []):
    if opt.startswith("mip_heuristic_suite="):
        print(opt.split("=", 1)[1].replace(",", "+"))
        raise SystemExit
on = [h for h, v in f["efforts"].items() if v > 0]
print("+".join(on) if len(on) < 4 else "all")
CFG
}

cmd_heldout() {
	local count=${1:-0}
	# A stratified 48 drawn from the 143 held-out instances -- the same size as
	# the confirmation's analysis set, deliberately.  Different n would mean
	# different power, and "detected on tuning but not on held-out" would then
	# be confounded with "less power on held-out", which is precisely the
	# comparison this stage exists to make cleanly.
	local list="${HELDOUT_INSTANCES:-$REPO/bench/instances_heldout48.txt}"
	mkdir -p "$RESULTS"
	echo "held-out instances: $(grep -c '^[^#]' "$list")"

	local name opts config arms
	# Assign before looping: `for x in $(f)` discards f's exit status, so an
	# unknown FINALISTS_ONLY name would run zero arms and report success.
	arms=$(names)
	for name in $arms; do
		opts=$(opts_for "$name")
		config=$(config_for "$name")
		echo "=== $name (config $config) : $opts"
		if [ "$count" != "0" ]; then export PLATO_COUNT="$count"; else unset PLATO_COUNT || true; fi
		PLATO_CONFIGS="$config" \
		PLATO_OUTPUT="$RESULTS/heldout/$name" \
		PLATO_INSTANCES="$list" \
		PLATO_TIME_LIMIT=600 \
		PLATO_BINARY="$BINARY" \
		PLATO_DEV_LOG=1 \
		PLATO_ANALYZE=0 \
		PLATO_EXTRA_OPTIONS="$opts" \
			"$REPO/bench/run_plato.sh" next 6
	done
}

cmd_confirm() {
	local hours=${1:?usage: confirm <hours>}
	local name opts config arms
	# Assign before looping: `for x in $(f)` discards f's exit status, so an
	# unknown FINALISTS_ONLY name would run zero arms and report success.
	arms=$(names)
	for name in $arms; do
		opts=$(opts_for "$name")
		config=$(config_for "$name")
		echo "=== $name (config $config, full limit) : $opts"
		PLATO_CONFIGS="$config" \
		PLATO_OUTPUT="$RESULTS/confirm/$name" \
		PLATO_INSTANCES="${CONFIRM_INSTANCES:-$REPO/bench/instances_confirm48.txt}" \
		PLATO_TIME_LIMIT=600 \
		PLATO_BINARY="$BINARY" \
		PLATO_ANALYZE=0 \
		PLATO_EXTRA_OPTIONS="$opts" \
			"$REPO/bench/run_plato.sh" next "$hours"
	done
}

cmd_status() {
	local name stage arms
	arms=$(names)
	for stage in heldout confirm; do
		echo "$stage  ($RESULTS/$stage)"
		for name in $arms; do
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
