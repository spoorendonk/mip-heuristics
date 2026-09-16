# Ablation B — the joint search over mix, effort and patience

What was run, what it found, and what a paper can and cannot claim from it.
Companion to `bench/ablation_effort/` (Ablation A, which produced the search
centres and the instance sets) and `bench/ablation_fprlp/` (Ablation C).

**Headline: the search's cheapest three-heuristic survivor ships, and Scylla is
disabled.** `B'-mix-cheapest` — fj 0.3317, fpr 3.161, local_mip 3.2865,
scylla 0 — is 12.6% better than the four-heuristic vector on the 48 held-out
instances (ratio 0.874, CI [0.782, 0.977]), the only arm separated from it at
the campaign metric. Scylla goes to 0 because it produced **zero accepted
incumbents in ~380 runs** while dispatching on every solve.

## Reproducing

| step | command | artifacts |
|---|---|---|
| free search, 3 cost weights | `bench/run_irace.sh all` | `bench/results/irace/<tag>/` |
| constrained search (all four forced on) | `IRACE_DIR=bench/irace-all4 IRACE_RESULTS=bench/results/irace-all4 bench/run_irace.sh 1_600` | `bench/results/irace-all4/1_600/` |
| apply the selection rule | `bench/analyze_irace.py <results dir>` | `selection*.json` here |
| confirmation, 600 s | `bench/run_finalists.sh confirm <hours>` | `bench/results/finalists/confirm/` |
| held-out, 600 s | `bench/run_finalists.sh heldout` | `bench/results/finalists/heldout/` |
| held-out, one named arm | `FINALISTS_ONLY="B'-mix-cheapest" bench/run_finalists.sh heldout` | same tree |
| attribution, any tree | `bench/analyze_results.py <tree> --configs <arms> --attribution` | the tables below |
| score either | `bench/compare_finalists.sh {confirm,heldout} --time-limit 600` | `tables.txt` here |

Results trees are gitignored (they are ~3 GB); every derived artifact needed to
read the result is in this directory.

## Design

* **Objective, pre-registered** in `bench/irace/PREREGISTRATION.md`, signed at
  commit `01780d0` before the first experiment: `cost = gap + lambda * tau`,
  where `gap` is the primal gap of the presolve-exit incumbent (capped at 1,
  penalty 2.0 when nothing is found) and `tau` is the heuristics' own wall time.
* **Space:** eight real parameters — four efforts, four patiences — plus two
  discrete switches per heuristic so a log-sampled real can reach the exact
  values 0 (does not run) and 0 (no staleness gate). Domains are per-heuristic,
  centred on Ablation A's measured yield knees.
* **Cost-weight sweep:** `lambda` in {1/1200, 1/600, 1/300}; 1/600 is the
  derived value `g(0)/T` at the campaign's 600 s limit, the others bracket it.
* **Budget:** 3000 experiments per lambda, `firstTest = 5`, elitist, seed 1,
  90 tuning instances, presolve-only at 60 s, 16 workers, CPU build.
* **Selection is by outcome *and* by attribution.** The objective scores how
  well a configuration did; it cannot see what each part of it contributed, so
  a heuristic that does nothing is invisible to it however much compute is
  spent. Per-heuristic solution-source counts are read from the same logs and
  are what retires a component.

## What the searches selected

| search | lambda | survivors | selection (pre-registered tie-break) |
|---|---|---|---|
| free | 1/1200 | 2 | `fj,fpr,local_mip` — total effort 12.06 |
| free | **1/600** | 6 | **`fj` alone — 0.16** |
| free | 1/300 | 6 | `fj,fpr,local_mip` — 10.78 |
| constrained (all four on) | 1/600 | 5 | `fj,fpr,local_mip,scylla` — 10.55 |

**The sweep is formally unstable**: the middle cost weight selects the sparse
configuration while both extremes select the mix. It is *not* monotone in the
cost weight, so the instability is not evidence that cost drives the mix —
13 of the 14 free-search survivors are `fj,fpr,local_mip`, and the three
`fj`-only ones all sit at lambda = 1/600, where the "fewer heuristics" clause
of the tie-break then decides.

**Scylla is absent from all 14 free-search survivors at every cost weight.**

## Confirmation and held-out, at the campaign metric

Five arms: **A** = `fj` alone (irace's pick at the derived lambda, and the
pooled-rule pick over all 14 survivors); **B**, **B'** = two well-separated
points on the mix axis (10.78 and 6.78 total effort); **D** = the
four-heuristic vector Ablation A's per-heuristic measurement left in place;
**D'** = the constrained search's winner.

Scored tables are in `tables.txt`. SGM of the primal integral at 600 s:

| arm | total effort | confirm-48 | pilot-free 29 | held-out 48 |
|---|---|---|---|---|
| A-fj-only | 0.16 | 15.815 | 15.262 | — |
| B-mix-lambda300 | 10.78 | 12.091 | 11.465 | 18.479 |
| **B'-mix-cheapest** | **6.78** | 12.991 | 12.894 | **17.12** |
| D-shipped | 29.85 | **11.944** | **10.550** | 19.580 |
| D'-tuned-all4 | 10.55 | 12.834 | 11.646 | — |

**A is separated and worst** on every analysis set (~30% behind) and is by far
the most erratic per instance (paired sd **1.745**, against ~0.46 for the
others). **Nothing among B, B', D, D' is separable on the tuning-set
confirmation, and the ordering there is unstable to the analysis set** — a
single instance flips which of B and D is nominally best between the 48 and the
49.

## Attribution is what retires Scylla

Per-heuristic incumbent improvements, read off ordinary logs:

| tree | runs | FJ | FPR | LocalMIP | **Scylla** |
|---|---|---|---|---|---|
| confirm, `D-shipped` | 49 | 1756 | 19 | 1521 | **0** |
| held-out, `D-shipped` | 48 | 301 | 27 | 195 | **0** |
| confirm, `D'-tuned-all4` (scylla 0.4907) | 49 | 1616 | 14 | 1622 | **0** |
| full mipfeas, four heuristics | 233 | 24763 | 139 | 4651 | **0** |

Scylla is neither disabled nor skipped in those runs: a traced solve emits
`[Heur] name=scylla phase=presolve` every time, and on `50v-10` reports
`found=0` after spending **220.7 ms of a 342 ms chain** at the worst throughput
of the four. The criterion is **incumbent improvements**, this project's own
definition of productive, not pool acceptances — the two differ by five orders
of magnitude.

That is the same thing the searches say in a different form: Scylla is absent
from all 14 free-search survivors at every cost weight, and the constrained
search that forced it on placed it at 0.4907, an order of magnitude below the
3.068 Ablation A derived for it running alone.

## Applying the tie-break

Rule 2 of `bench/irace/PREREGISTRATION.md`: among statistically
indistinguishable configurations the **simpler** one is chosen — fewer
heuristics enabled, then lower total effort, because "fewer heuristics is less
to defend and a tie is a legitimate result".

Confirmation removed `A-fj-only` from the indistinguishable set, being
separated and ~30% worse on every analysis set. That leaves the **eleven
three-heuristic survivors**, all tied on the first clause, so the second
decides: lowest total effort, which is `B'-mix-cheapest` at 6.78.

Selecting the one arm lacking an out-of-sample check would be indefensible, so
`B'` was run on the held-out 48 — the same set and the same binary as the other
two arms:

| arm | heuristics | total effort | held-out SGM |
|---|---|---|---|
| **B'-mix-cheapest** | 3 | **6.78** | **17.12** |
| B-mix-lambda300 | 3 | 10.78 | 18.48 |
| D-shipped | 4 | 29.85 | 19.58 |

Paired, n=48:

| comparison | ratio | 95% CI | t | win / tie / loss |
|---|---|---|---|---|
| **B' / D** | **0.874** | **[0.782, 0.977]** | **-2.37** | 31 / 7 / 10 |
| B' / B | 0.926 | [0.850, 1.009] | -1.76 | 29 / 9 / 10 |
| B / D | 0.944 | [0.837, 1.064] | -0.95 | 28 / 7 / 13 |

So B' is not only the rule's pick: it is **separated from D**, which no arm was
on the tuning-set confirmation. The held-out ordering is monotone in total
effort — 6.78 -> 17.12, 10.78 -> 18.48, 29.85 -> 19.58 — and the paired sd is
0.30-0.42 rather than the 1.22 of patched-vs-vanilla, which is why n=48 can
separate a 12.6% effect at all.

**One honest observation for a write-up:** B' is nominally *worse* than B on
the tuning-set confirmation (12.99 vs 12.09) and nominally *better* out of
sample. That is the direction selection bias predicts, and it is an argument
for trusting the held-out ordering over the tuning-set one.

## The power calculation, which is the methodological finding

For a paired comparison at 80% power, `n = 8 * sd^2 / ln(1+delta)^2`. The
measured per-instance sd of the paired log-ratio is **0.44-0.57**. At
sd = 0.464:

| n | smallest detectable difference |
|---|---|
| 48 (this stage) | 20.6% |
| 90 (whole tuning set) | 14.8% |
| 233 (full mipfeas, one seed) | 9.0% |
| 699 (233 x 3 seeds) | **5.1%** |

The differences among B/B'/D/D' on the tuning-set confirmation are **0-8%**. So
they are not merely unresolved there — **they are unresolvable with this
benchmark**, even by spending a whole headline campaign on the question. That
is a result about the method, and it is the main thing to carry into a
write-up: *within the tested range, the presolve configuration does not
measurably move the campaign metric on the tuning set*, and what separates B'
from D is the out-of-sample comparison plus the attribution above.

## What the searches said about patience

Ablation A set patience to the *cost bound* (`effort/4`), because on three of
four heuristics the measured p95 wait was orders of magnitude above it. B
searched all four patiences jointly, so it is the only evidence about the axis
that comes from tuning rather than from a clamp.

Across all 19 survivors of both searches, counting only those running the
heuristic in question:

| heuristic | gate off (patience 0) | below its clamp | at/above its clamp |
|---|---|---|---|
| fj | 13 of 19 | 6 | 0 |
| fpr | **0 of 16** | 13 | 3 |
| local_mip | 8 of 16 | 1 | 7 |
| scylla | **0 of 5** | 0 | 5 |

* **FPR is the one heuristic that always wants a real gate** — never off, and
  13 of 16 values sit strictly below the clamp, so they are doing work rather
  than being clamped to it.
* **FJ mostly wants no gate at all**, and when it has one the value is tiny
  (0.03-0.16). That agrees with Ablation A, whose probe ran FJ ungated.
* **Scylla's searched patience spans 0.337-0.9695**, mid-range against the
  0.767 Ablation A derived. Worth contrasting with Scylla's *effort*, which is
  the one place the searches disagree sharply with that measurement
  (0.49-0.55 against 3.068).

**A quarter of the patience axis was degenerate, and that qualifies the null.**
`patience_threshold` clamps to `effort/4`, so any sampled value at or above its
own clamp is behaviourally identical to every other such value. 15 of the 56
survivor-parameters land there — **all 5 for Scylla and 7 of 16 for
LocalMIP**. For those two the patience dimension was effectively unsearched.
The domains were set knowing this could happen (`bench/irace/parameters.txt`
records the reasoning: the top of each patience range has to reach `effort/4`
at the *top* of the effort range, or a configuration sampling a large effort
could not express the loosest gate at all), and the trade was accepted as
costing resolution rather than correctness. Clamped values map onto one real
configuration, they are not nonsense — but "patience is not separable" partly
reflects a space where many samples were equivalent by construction, not only
a flat landscape.

Fixing it properly means searching the *ratio* `patience/effort` in (0, 0.25],
which `run_target.py`'s CLI does not express today. Worth doing if the patience
axis is ever revisited.

**The ceiling divisor itself is fixed at 4, and that is a decision rather than
an omission.** It is `kPatienceCeilingDivisor`, a `constexpr`, so no search can
reach it — but it is also upstream's own ratio rather than one this project
chose: HiGHS's FeasibilityJump pairs `kMaxTotalEffort = nnz << 10` with
`kMaxEffortSinceLastImprovement = nnz << 8`, exactly a quarter, and `nnz << 10`
is the same constant our effort options are denominated in. Adopting the
reference implementation's ratio needs less justification than deviating from
it would. B's null also spans the whole reachable range `[0, effort/4]`,
including 21 of 56 survivor-parameters with the gate entirely off, so a
landscape that flat inside the range gives no reason to expect its edge to
matter.

### Verdict on the patience axis: closed

**No further work is warranted on patience, and this is the argument rather
than a shrug.**

What makes the null trustworthy is *where* the search had coverage. For the two
heuristics that genuinely explored below the clamp — **FJ** (13 gates off, 6
below the clamp, 0 clamped) and **FPR** (13 below, 3 clamped) — the axis was
well sampled and still nothing separated. The degeneracy is concentrated in
**Scylla** (all 5 clamped) and **LocalMIP** (7 of 16), so the coverage gap sits
where the evidence is weakest rather than where the conclusion rests.

The one concentrated signal points *toward* the current setting: **FPR never
wants its gate off (0 of 16 survivors)**, and FPR ships with a gate.

**Scope of the claim, stated precisely.** This is a null at 20.6% resolution:
*no evidence that any reachable patience beats `effort/4`* — not *`effort/4` is
optimal*. From the power table above, no affordable experiment upgrades that.

**Method note if the axis is ever reopened**: search the *ratio*
`patience/effort` in (0, 0.25] rather than the absolute value, so samples
cannot clamp onto each other.

## Two answers the searches did give

* **Joint calibration is not better than per-heuristic calibration.** D' tuned
  all four together and came out level with (nominally behind) D, whose values
  came from Ablation A measuring each heuristic alone. The constrained search
  optimised presolve-exit gap; that objective does not transfer.
* **The mix's effort level does not matter between 6.8 and 10.8.** B' uses 37%
  less total effort than B and scores level on the tuning set, consistent with
  the 16x scatter in `fpr` effort across the eleven mix survivors.

## What this says about method, which is the transferable part

* **An objective that scores outcomes cannot retire a component.** Every stage
  here measures how well a configuration did, never what each part of it
  contributed. A heuristic that does nothing is invisible to that, and stays
  invisible however much compute is spent — attribution is a `grep` away and
  belongs in the analysis from the start.
* **"Keep the incumbent on a null" needs an incumbent.** Before a release,
  every configuration is a candidate; a default that merely happens to be in
  the tree is not an incumbent and does not defeat a tie-break.
* **A pre-registered tie-break is only worth having if it is applied when it
  bites.**

## Scope limits a paper must state

* **`fpr_lp` was disabled in every arm.** It ships at effort 0 and draws from
  upstream's RENS/RINS LP-iteration envelope, so the conclusion is "these
  defaults, *with `fpr_lp` off*". Ablation C measures it.
* **One seed.** Deliberate: n and seeds enter the standard error the same way
  and instances also buy coverage. Run-to-run variation sits inside some of
  these margins, so a narrow ordering here is not a result.
* **Internal-pilot design.** A 20-instance pilot estimated the variance and 19
  of its instances are in the 48. The sensitivity analysis on the 29 pilot-free
  instances reproduces every conclusion, so the dependency is demonstrated
  immaterial rather than assumed to be. Note internal-pilot bias inflates
  *false positives*, and the tuning-set finding here is a null.
* **Only the root dispatch is screened**, and ~18% of mipfeas is invisible to a
  presolve-only screen (Ablation A's informative set is 191 of 233).
