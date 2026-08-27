#!/usr/bin/env bash
# Everything the #113 probe produces, from the logs, in one command.
#
# Usage:
#   bench/derive_from_probe.sh [probe-tree] [vanilla-tree]
#
# Defaults: bench/results/probe/preprobe and bench/results/plato/vanilla.
#
# The probe is a results tree of presolve-only runs (bench/run_presolve_probe.sh).
# Everything below is derived *from those logs* — nothing is carried by hand
# between the steps, and every artifact records the command that regenerates
# it plus a digest of its inputs.  Same trees plus same seed reproduce every
# file byte for byte; none of them carries a timestamp.
#
#   informative.txt  instances where the presolve chain produced the reported
#                    incumbent, as a union over the four single-heuristic arms
#   hard_tier.txt    its complement, each with the reason it is out
#   report.txt       counts, the budget-headroom check, and the per-heuristic
#                    effort trajectories
#   defaults.json    the derived parameter vector: effort and patience per
#                    heuristic, with the worker count they are valid at.
#
#                    effort   = p50 of the yield knee over the dispatches that
#                               *finished* improving.  The knee is the budget
#                               that reached a dispatch's last incumbent
#                               improvement: spend beyond it bought nothing on
#                               that instance.  Barren dispatches have no knee
#                               and are excluded -- they are a cost question,
#                               not a budget question -- and a dispatch still
#                               improving when the cap fired is right-censored,
#                               since its true knee is larger than what it was
#                               seen to spend.
#
#                    patience = min(p95 of the inter-improvement gaps, a
#                               fraction of the ceiling).  The p95 is the
#                               retention statement: at most 5 % of the
#                               improvements that would ever arrive are cut
#                               off.  The clamp is the cost statement: a barren
#                               dispatch spends exactly the patience, on 30-46 %
#                               of dispatches depending on the heuristic, so it
#                               must stay well under the ceiling.
#
#                    Both are only valid at the worker count they were measured
#                    at, which is why the JSON records it beside them: FJ's
#                    budget is per worker and the other three are per dispatch,
#                    so changing the count *reallocates* budget between
#                    heuristics rather than rescaling it.
#   instances_tuning.txt  (in bench/) the stratified tuning subset, sampled
#                    from the informative set and stratified on *vanilla*
#                    time-to-first-feasible, which is why the second tree is
#                    needed: the probe cannot stratify on itself without
#                    conditioning the instance set on the outcome it measures
#
# The two derivations are separate scripts on purpose and are both callable
# directly — this is the order and the arguments, not a third implementation.
set -euo pipefail

PROBE="${1:-bench/results/probe/preprobe}"
VANILLA="${2:-bench/results/plato/vanilla}"
# Tracked, not beside the tree: `bench/results*` is gitignored, so artifacts
# written there vanish from the repository and the numbers behind a shipped
# default would live only on the machine that ran the probe.
OUT="${OUT:-bench/ablation_effort}"
mkdir -p "$OUT"
# 90 of 75-100: the size #113 asks for, which buys back the selection noise a
# wide search over a 25-instance subset would have carried.
SIZE="${TUNING_SIZE:-90}"
SEED="${TUNING_SEED:-0}"

echo "== reading the probe: $PROBE"
python3 bench/analyze_presolve_probe.py "$PROBE" \
	--informative-output "$OUT/informative.txt" \
	--hard-tier-output "$OUT/hard_tier.txt" \
	--defaults-output "$OUT/defaults.json" \
	--report-output "$OUT/report.txt"

echo
echo "== drawing the tuning set: $VANILLA, stratified on time-to-first-feasible"
python3 bench/make_tuning_set.py "$VANILLA" \
	--informative-instances "$OUT/informative.txt" \
	--size "$SIZE" --seed "$SEED" \
	--output bench/instances_tuning.txt

echo
echo "Wrote:"
for f in "$OUT/informative.txt" "$OUT/hard_tier.txt" "$OUT/defaults.json" \
	"$OUT/report.txt" bench/instances_tuning.txt; do
	printf '  %-46s %s\n' "$f" "$(wc -l <"$f") lines"
done
echo
echo "The derived parameter vector:"
python3 - "$OUT/defaults.json" <<'PY'
import json, sys

data = json.load(open(sys.argv[1]))
p = data["provenance"]
print(f"  measured at {p['workers_observed']} workers over "
      f"{p['runs_traced']} traced run(s) on {p['instances_analysed']} instance(s)")
for name, h in sorted(data["heuristics"].items()):
    effort = "-" if h["effort"] is None else f"{h['effort']:.4f}"
    patience = "-" if h["patience"] is None else str(h["patience"])
    stale = "-" if h["stale_fraction"] is None else f"{100 * h['stale_fraction']:.1f}%"
    print(f"  {name:<10} effort {effort:>9} (shipped {h['effort_shipped']:.4f})"
          f"   patience {patience:>10}   stale {stale:>6}")
PY
