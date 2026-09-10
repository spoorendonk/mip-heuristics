#!/usr/bin/env bash
# #108 — the headline run: the selected configuration over the full PLATO
# `mipfeas` list at 600 s, against the vanilla baseline from #105.
#
# Usage:
#   bench/run_headline.sh next <hours>   run within a window, resume safely
#   bench/run_headline.sh until <HH:MM>  run until a wall-clock time today/tomorrow
#   bench/run_headline.sh status
#   bench/run_headline.sh report
#
# `run_plato.sh` with this stage's environment, like every other campaign
# stage (#109).  It executes; it chooses nothing.
#
# ── the configuration, and why it carries no options ────────────────────────
#
# #107's selected configuration is **the shipped defaults**.  Ablation B found
# nothing that beat them: A (fj alone) was separated and worst, and B, B' and
# D' were all indistinguishable from D-shipped, which resolves into the
# pre-registered "indistinguishable -> simpler" clause.  So the selected
# configuration is what the binary already does, and this stage passes **no**
# `--extra-options` at all.
#
# That is deliberate rather than convenient.  Ablation B's arms passed their
# eight numbers explicitly because they were *candidates*; passing D-shipped's
# eight here would produce a `.opts` that differs from the shipped defaults in
# the fourth decimal (`addd29c` rounded the patience column, and
# `finalists.json` still records the pre-rounding values), so the headline
# would measure a configuration that ships nowhere.  `fpr_lp` stays off at its
# own default, which Ablation C measured (#165).
#
# ── what is NOT reused ──────────────────────────────────────────────────────
#
# Ablation B's 49 confirmation runs are not a head start on this, for three
# separate reasons, any one of which is enough: they live in a different tree
# (`results/finalists/`), they cover 49 stratified instances rather than the
# full 233, and they were given their eight numbers explicitly, so they are
# not default-options runs.  The vanilla arm *is* reused -- 233/233 at seed 0
# from #105, already in this tree -- and `--skip-existing` protects it.
#
# ── no developer logging ────────────────────────────────────────────────────
#
# `PLATO_DEV_LOG` is deliberately unset.  These are the headline timings and
# `log_dev_level=3` is not the same run; attribution comes from the
# solution-source characters in the ordinary log, which need no tracing.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"

# The selected configuration, as a harness config name.  `all` names every
# suite token; `fpr_lp` among them runs only if its effort is non-zero, and
# its shipped default is 0 (#165).
CONFIG="${HEADLINE_CONFIG:-all}"
# **One seed by default, and three is the deliverable rather than the gate.**
# #108 requires at least three, because the execution mode is opportunistic
# parallel and a point estimate would hide run-to-run spread -- that stands
# for the reported result.  It does not have to be paid up front: seed 0 over
# all 233 is one full pass (~24 h) and already resolves ~9% against vanilla at
# this benchmark's paired sd, which is enough to decide whether there is an
# effect worth spending two more passes on.  It is also *symmetric* with the
# baseline, which #105 ran at one seed.
#
# The order matters as much as the count.  `run_benchmark.py` iterates
# instance -> seed -> config, so a three-seed window buys all three seeds on a
# prefix of instances rather than one seed across many: at a fixed number of
# hours, three seeds covers a third of the instances.  A gate wants breadth.
#
# `HEADLINE_SEEDS="0 1 2"` after the gate passes; resume is per
# (config, instance, seed), so nothing already run is repeated.
SEEDS="${HEADLINE_SEEDS:-0}"

# A binary outside build/, because `pre-push` runs `rm -rf build` and would
# delete it out from under a multi-day campaign (CLAUDE.md).  Staged by hand
# rather than built here: which binary produced the headline is the one fact
# the whole comparison rests on, so it is named, not discovered.
BINARY="${MIP_HEURISTICS_BINARY:-$REPO/bench/results/plato/bin/highs}"

plato() {
	PLATO_CONFIGS="$CONFIG" \
	PLATO_SEEDS="$SEEDS" \
	PLATO_INSTANCES="$REPO/bench/instances_plato.txt" \
	PLATO_OUTPUT="$REPO/bench/results/plato" \
	PLATO_TIME_LIMIT=600 \
	PLATO_BINARY="$BINARY" \
	PLATO_ANALYZE=0 \
		"$REPO/bench/run_plato.sh" "$@"
}

cmd_next() {
	local hours=${1:?usage: next <hours>}
	[ -x "$BINARY" ] || { echo "ERROR: no binary at $BINARY" >&2; exit 1; }
	# Refuse a stock HiGHS outright: it has none of the options this stage's
	# config sets, so every run would exit 255 and the tree would fill with
	# .log.err.  run_benchmark.py probes this too; failing here is faster and
	# names the binary.
	#
	# Invoked with no arguments, the way `check_vanilla_binary` does it: HiGHS
	# prints its banner and then complains about the missing model, so the
	# marker is on stdout and the exit status is non-zero.  Captured rather
	# than piped into grep, because `set -o pipefail` would then report the
	# binary's own exit status and every patched build would look unpatched.
	local banner
	banner=$("$BINARY" 2>&1 || true)
	case "$banner" in
	*"mip-heuristics patch active"*) ;;
	*)
		echo "ERROR: $BINARY is not a patched build" >&2
		exit 1
		;;
	esac
	plato next "$hours"
}

cmd_until() {
	# The window a human actually wants: "run until 08:00 and stop".  Whole
	# hours only, because run_plato.sh does its budget arithmetic in bash
	# integers -- rounded *down*, so the last instance launched still has its
	# full 600 s inside the window rather than past it.
	local target=${1:?usage: until <HH:MM>}
	local hours
	hours=$(python3 - "$target" <<'PY'
import sys
from datetime import datetime, timedelta

hh, mm = (int(x) for x in sys.argv[1].split(":"))
now = datetime.now()
end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
if end <= now:
    end += timedelta(days=1)
hours = int((end - now).total_seconds() // 3600)
if hours < 1:
    sys.exit(f"{sys.argv[1]} is less than an hour away")
print(hours)
PY
	)
	echo "== running for ${hours}h, to finish before $target"
	cmd_next "$hours"
}

cmd_report() {
	local out="$REPO/bench/results/plato"
	echo "=== headline: all 233, paired against vanilla"
	python3 "$REPO/bench/analyze_results.py" "$out" --configs "$CONFIG" vanilla \
		--time-limit 600 --baseline --summary
	echo
	echo "=== secondary: the held-out complement of the tuning set"
	python3 "$REPO/bench/analyze_results.py" "$out" --configs "$CONFIG" vanilla \
		--time-limit 600 --baseline --summary \
		--instances "$REPO/bench/instances_plato.txt" \
		--exclude-instances "$REPO/bench/instances_tuning.txt"
	echo
	echo "=== per-heuristic attribution of accepted incumbents"
	python3 "$REPO/bench/analyze_results.py" "$out" --configs "$CONFIG" vanilla \
		--time-limit 600 --attribution
}

case "${1:-status}" in
next) shift; cmd_next "$@" ;;
until) shift; cmd_until "$@" ;;
report) cmd_report ;;
status) plato status ;;
*) echo "usage: $0 {next <hours>|until <HH:MM>|status|report}" >&2; exit 1 ;;
esac
