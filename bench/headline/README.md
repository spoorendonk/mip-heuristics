# The headline: the shipped configuration against vanilla HiGHS (#108)

`mipfeas`, all 233 instances, 600 s, one seed, 16 workers, CPU build.
Companion to `bench/ablation_effort/` (A), `bench/ablation_search/` (B) and
`bench/ablation_fprlp/` (C), which chose the configuration this measures.
These are the numbers the paper reports
([arXiv:2609.22938](https://arxiv.org/abs/2609.22938), Table 5 and Section 4.6).

**Headline: the primal-integral SGM falls by 23.7% over the full set** — ratio
0.763, 95% CI [0.684, 0.852], p < 0.001 — **and by 12.9% on the 143 instances
never used for tuning** — 0.871, [0.782, 0.970], p = 0.012 — with a first
feasible solution on 3 more instances. **At the limit the patched solver is
slightly ahead**, and the difference separates only over the full set. The
contribution is *sooner*, which is what the benchmark's metric measures and
what a feasibility heuristic should claim.

## Scoring

Every number here is under the benchmark's published formula, the default of
`bench/analyze_results.py` (`--formula mipfeas`), as its own
`create_primalintegral.py` computes it: the penalty is 2 before the first
incumbent, 1 when the incumbent and the reference have opposite signs, and
otherwise `|z - z*| / max(|z|, |z*|, 1)` rounded to six decimals; the primal
integral is the integral of that penalty over [0, 600 s] divided by 600, so it
lies in [0, 2]; the reference z* is the benchmark's own
(`bench/mipfeas_optimal_objective.csv`, copied verbatim from its results
archive), never a virtual best; every SGM is over all instances at shift 0.001,
a run with no solution scoring 2 in gap and integral and 600 s in T1st.

The campaign's stages were run and read under an earlier scoring
(`--formula campaign`: gap-seconds, no incumbent scored 1, MIPLIB `.solu`
references improved by any better observed primal), and the ablation write-ups
under `bench/ablation_*/` still are. The headline was re-read under the
published formula so that the paper and this page report the same numbers; the
runs did not change.

## Reproducing

| step | command |
|---|---|
| run | `bench/run_headline.sh until 08:00` (or `next <hours>`), repeat until `status` shows 233/233 |
| the tables | `bench/run_headline.sh report` -> `tables.txt` |
| paired statistics | `bench/run_headline.sh paired` -> `paired.txt` |

Results trees are gitignored; everything needed to read the result is here.

## Configuration

The arm is the **shipped binary at default options**. Its `.opts` carries the
seed and nothing else — the `all` config sets no option, and no
`--extra-options` is passed — so this measures what a user gets rather than a
hand-assembled vector. The defaults are `B'-mix-cheapest`
(`bench/ablation_search/`): fj 0.3317, fpr 3.161, local_mip 3.2865,
**scylla 0 (disabled)**, patiences 0 / 0.3372 / 3.1943 / 0, with
`mip_heuristic_fpr_lp_effort = 0` (`bench/ablation_fprlp/`).

`mip_heuristic_effort` is at upstream's own default of 0.05 and is asserted at
configure time, so the B&B dive-heuristic budget — what RENS and RINS draw
from — is bit-identical to vanilla's. The only difference between the arms is
what runs during presolve.

**The baseline is a separately built unpatched binary** of the same tag (#105),
not a setting on the patched one.

## Result

| set (n) | arm | #Feasible | #Wins | SGM T1st (s=1) | SGM final gap | **SGM primal integral** |
|---|---|---|---|---|---|---|
| full set (233) | **patched (B')** | **214** | **49** | **5.57** | **0.0054** | **0.0484** |
| | vanilla | 211 | 36 | 7.08 | 0.0066 | 0.0637 |
| tuning set (90) | **patched (B')** | 86 | **19** | **3.06** | **0.0043** | **0.0344** |
| | vanilla | 86 | 15 | 4.32 | 0.0056 | 0.0562 |
| held-out (143) | **patched (B')** | **128** | **30** | **7.89** | **0.0062** | **0.0600** |
| | vanilla | 125 | 21 | 9.52 | 0.0072 | 0.0690 |

Wins are strictly better objectives at the limit. Paired per instance on the
primal integral (full output in `paired.txt`):

| set | n | ratio | 95% CI | t | p | better/tied/worse |
|---|---|---|---|---|---|---|
| all 233 | 233 | 0.763 | [0.684, 0.852] | −4.82 | <0.001 | 114/24/95 |
| **held-out 143** | **143** | **0.871** | **[0.782, 0.970]** | **−2.50** | **0.012** | 63/19/61 |
| tuning 90 | 90 | 0.619 | [0.496, 0.772] | −4.26 | <0.001 | 51/5/34 |
| held-out, unseen 95 | 95 | 0.861 | [0.746, 0.994] | −2.04 | 0.041 | 43/11/41 |

**The held-out number is the fairer estimate.** The tuning set was drawn from
instances some heuristic solves alone (`bench/ablation_effort/`), so it reads
38% better against the held-out 13% — the selection bias the split exists to
quantify. Of the 143 held-out instances, 48 were already seen in Ablation B's
held-out check; the last row is the 95 the campaign never saw before this
stage, and it reads the same.

## Three qualifications that travel with it

**1. The win is in magnitude, not frequency.** 63 better against 61 worse on
held-out; 114/95 over all 233. A sign test finds nothing on held-out
(p = 0.86; 0.19 over all 233). The heuristics do not win more often — the
decrease comes from a minority of instances with large gains. Both measures
belong in any write-up; quoting the SGM alone would misrepresent the shape of
the result.

**2. At the limit it is slightly ahead, not level.** The patched arm finds a
solution on 4 instances where vanilla finds none, each first found by one of
our heuristics, against 1 the other way, where vanilla's solution arrives as
the solve closes. Paired over the 210 instances both arms made feasible, its
final gap is smaller on 42 and larger on 34, a ratio of 0.836, [0.718, 0.974],
p = 0.022; on the held-out complement that is 0.871, [0.720, 1.053] over 125
instances, which does not separate. HiGHS's own machinery holds **188 of 214**
final answers. The claim is that HiGHS reaches a good solution sooner, and is
slightly ahead when the clock stops.

**3. The held-out effect sits at the edge of what 143 instances can resolve.**
At the held-out sd of 0.659 the n=143 resolves a **14.3% decrease** at 80%
power, and the observed effect is a **12.9% decrease**. Quote the interval,
not the point: significance at this power implies the point estimate is likely
overstated. Over the full set the resolution is 14.5% against an observed
23.7%, comfortably separated.

*Both percentages are read on the same side of the ratio, and that is not a
pedantic detail.* A minimum detectable log effect `d = (z_0.975 + z_0.8) *
sd / sqrt(n)` has two percentage readings — `exp(d) - 1`, how much bigger
vanilla is than patched, and `1 - exp(-d)`, how much smaller patched is than
vanilla. `paired.txt` prints the second (`mdd80`), the side an improvement is
quoted on.

## Statistical conventions, stated rather than inherited

Each has a textbook alternative that gives different numbers, so which one
produced a figure is part of the figure.

* **Paired on the log-ratio** of the primal integral, `log((arm + s) / (control
  + s))` with the SGM shift `s = 1e-3`. The integral spans orders of magnitude
  across instances, and pairing cancels instance difficulty — the dominant
  variance component in cross-instance MIP benchmarking.
* **Intervals are normal**, `exp(mean +- 1.96 * se)`, not Student-t.
* **The sign test is the normal approximation**, without continuity
  correction.
* **Minimum detectable effect** is `1 - exp(-d)` for an improvement and
  `exp(d) - 1` for a regression — always read on the same side of the ratio as
  the effect it is compared with.
* **The final gap is paired over the instances both arms made feasible**, at
  the same shift; an instance one arm never solved has no gap to pair.

## Where the gain comes from

Most first solutions in the patched arm come from FeasibilityJump (114 of the
214 instances made feasible), and inside the chain FPR and LocalMIP add to what
FJ finds: on 71 of those 114 a later heuristic found a better solution,
LocalMIP on 62 and FPR on 16, and on those the median gap went from 0.624 at
FJ's best to 0.382. LocalMIP restarts from the incumbent, so part of its gain
is refining FJ's solution.

**That does not make this a result about parallelism.** Our FJ differs from
vanilla's in three confounded ways: 16 opportunistic workers against one call,
a per-worker budget that totals ~5x vanilla's single FJ allowance, and two
corrected upstream defects (the negative-coefficient jump value and the sign of
the objective term in the move score). Vanilla runs its own FJ, so the
comparison already includes FJ-vs-FJ; separating the three is not attempted.

## Attribution

| | patched #First | #Best | vanilla #First | #Best |
|---|---|---|---|---|
| FJ | 114 | 20 | 97 | 10 |
| FPR | 16 | 2 | — | — |
| LocalMIP | 5 | 4 | — | — |
| HiGHS/other | 79 | **188** | 114 | **201** |

Ours find the first feasible solution on 135 of 214 instances against vanilla's
97 of 211, and hold the final best on 26 against 10.

## The second arm: does the simplification cost anything?

`all-prev-vector` is the four-heuristic vector Ablation B replaced, measured
over the same 233 against the same baseline — so this is a 233-instance paired
comparison of the two patched configurations, at ~9% resolution against the
20.6% at which Ablation B's n=49 tuning-set comparison returned a null.

```
B' / prev-4-heuristic   n=233  ratio 0.965  CI [0.906, 1.027]  t=-1.13  p=0.26
                               better 140 / tied 26 / worse 67   sign p=4e-7
```

**The two measures disagree, and both are reported.** B′ is not separated on
magnitude, but wins twice as often; a few large losses offset the wins. The
reading: dropping Scylla and cutting total effort from 29.85 to 6.78 costs
nothing and helps slightly and often, by amounts too small to move an SGM. The
headline does not depend on which of the two was picked: the previous vector
reads 0.791, [0.697, 0.898] against vanilla over the 233.

## Deviations from the specified design, stated

**One seed, not the three this issue asks for.** Seed 0 was run as a gate and
the campaign stopped there.

Seeds shrink only the within-instance variance component. The between-instance
component does not move, because the instances are the same 143 — and the
baseline is *also* a single seed, so averaging `k` patched seeds removes at
most half the seed noise even as `k` grows. Writing `f` for the share of
variance that is seed noise, and assuming it splits evenly between the two
arms, `Var(k) = sd^2 * [1 - f/2 + f/(2k)]`. At the held-out sd of 0.659 and
n = 143, on the decrease side:

| `f` | k=1 | k=2 | **k=3** | k=∞ |
|---|---|---|---|---|
| 25% | 14.3% | 13.9% | **13.7%** | 13.4% |
| 50% | 14.3% | 13.4% | **13.1%** | 12.5% |
| 100% (not attainable) | 14.3% | 12.5% | **11.8%** | 10.3% |

**Three seeds reach the observed 12.9% only once `f` is above roughly 55%** —
so the honest statement is that extra seeds *might* have resolved this, not
that they could not. With one seed `f` is unestimable: there are no replicates,
so the table is bracketing rather than measurement.

What that makes the stopping decision is a judgement rather than an
impossibility argument. It stands on what a tighter interval would have bought:
the held-out effect is already separated at p = 0.012, nothing about what ships
turns on the width, and two further passes (~48 h) would at best have moved a
12.9% point estimate inside a slightly narrower band. Two seeds would have been
the informative spend — that is the smallest `k` from which `f` can be
estimated at all — and it was not made.

**Two binaries.** `all-prev-vector` was produced by the PATCH_VERSION 23
build; `all` by PATCH_VERSION 24. They differ in the default option values,
which is the thing under comparison. The PATCH_VERSION 23 binary was not
retained — its configuration is recoverable from commit `7056a0f`, and its logs
carry the patch marker.

**Both patched arms ran at default options**, so their `.opts` files are
byte-identical and the directory name is the only thing distinguishing them.
That is why the previous arm is `all-prev-vector` rather than `all`.

**Killed runs.** Four, listed at the top of `tables.txt`: three patched
(`germanrr`, `neos-3046615-murg`, `neos-5114902-kasavu`) and one vanilla
(`neos-5114902-kasavu`), kept as truncated logs. They score correctly: T1st and
the primal integral read incumbent lines, not the final report.

## Comparability

Adopting the `mipfeas` instance list, time limit, reference objectives and
scoring formula makes this *definitionally* the same benchmark. It does **not**
make the absolute numbers comparable with the rankings published on
Mittelmann's server at [plato.asu.edu](https://plato.asu.edu/bench.html), which
are measured on different hardware. The defensible claim is patched versus
vanilla on one machine under the `mipfeas` definition.
