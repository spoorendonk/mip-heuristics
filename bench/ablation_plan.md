## Ablation B+ — confirm on a properly sized set, and tune the shipped shape

**Why B is not finished.** Two defects in what was run, both mine:

* the held-out stage ran **presolve-only at 60 s**, so it scores the *screen's*
  question on unseen instances, not the campaign's 600 s primal integral. It
  cannot confirm or refute a ranking taken at the full limit;
* the **stepwise extension never happened**. n=20 was registered as a starter
  with "extend if the top arms are inside noise"; the arms were inside noise
  and I concluded anyway.

**Sizing, now measured rather than assumed.** Paired log-ratio B/D over the 17
comparable instances has **sd = 0.444** per instance, and a paired mean of
**-0.056 (ratio 0.945)** — B is ~5.5% better per instance, not the 15.5% the
SGM headline implies, which is carried by a few instances.

| true difference | n for 80% power |
|---|---|
| 20% | 48 |
| 15% | 81 |
| 10% | 174 |
| 5%  | 664 |

### B+ steps, in dependency order

0. **Effort-tune the shipped shape first, because it feeds step 1.**
   `bench/irace-all4/` forces all four heuristics on and searches only the
   eight effort/patience values; 12 irace parameters, 3000 experiments,
   lambda = 1/600, the 90 tuning instances, presolve-only at 60 s. Running
   2026-09-08. Its winner **replaces** D-shipped as the four-heuristic arm,
   because D's values were measured one heuristic at a time and never tuned
   together — so if the tuned vector differs, the 20 completed D runs are
   superseded and re-run with it.
1. **Extend the confirmation to n=49**, full solves at **600 s**, arms
   **B, D, D'** — the mix from the free search, the shipped incumbent, and
   step 0's tuned four-heuristic vector.

   **A-fj-only is dropped, on evidence rather than for economy.** It was
   worst on the confirmation (15.67 against B 12.52 and D 14.83), worst on
   held-out by 51% (35.99 against D 23.87), and found solutions on **71 of
   143 instances against D's 87** — 16 fewer. Re-confirming a settled loss
   would cost 29 runs.

   New runs: B +29, D +29, D' +49 = 107, about 12 h. `bench/instances_confirm48.txt` is
   the stratified 48-draw **union** the original 20, so the n=20 result nests
   inside the extension and no completed run is discarded; the union is 49
   rather than 48 because largest-remainder allocation at a different size
   shifts which instances a stratum picks (the 48-draw shares 19 of the
   original 20). 29 new runs per arm. Stop rule fixed in advance: if the top
   two are within the detectable margin at n=49, extend to **n=81** rather
   than declare a winner.
2. **Held-out at the campaign metric — still required, but only for the top
   two.** Extending step 1 fixes *power*; it does not fix *selection
   optimism*, because every arm was selected on tuning-set data and the
   extension draws more tuning-set instances. Those are different failure
   modes and the first cannot cure the second. The evidence that it matters
   here is direct: the presolve-only held-out stage already **reversed** the
   confirmation's ranking once. So after step 1 names a winner, run **winner
   vs incumbent only** on a stratified subset of the 143 held-out instances,
   at **600 s full solves** — not presolve-only at 60 s, which is what made
   the first attempt unable to speak to a campaign-metric ranking. Two arms,
   same stepwise sizing.
3. **One seed, deliberately.** Two seeds would be the statistically sound
   choice and would roughly halve the margin that can be resolved, but the
   compute is better spent on instances than on replicates at this stage:
   n enters the standard error the same way and also broadens coverage.
   Seeds are for the final run (#108), which is scored on three. The cost is
   recorded rather than hidden: run-to-run variation sits inside some of the
   margins being compared here, so a narrow ordering at any n in this stage
   is not a result.

### Not worth separate arms

The other 12 survivors are minor variations of `fj,fpr,local_mip` at scattered
efforts (FPR 2.1-32.9, LocalMIP 1.98-13.4). That scatter is itself evidence
the mix is not sharply determined, and averaging over near-duplicates buys
nothing.

## Ablation C — fpr_lp effort share (postponed, method recorded)

**Sweep, not irace.** irace needs >=504 experiments even for one parameter;
at 600 s full solves that is ~85 h. A fixed sweep of 4-5 share values over the
same instance subset is ~11-14 h and answers the same question, because the
space is one-dimensional and monotone-ish — irace's advantage is navigating
high-dimensional spaces, and there is nothing here to navigate.

* **Background:** the winning configuration from B+ (not D by default — pick
  what B+ selects), held fixed.
* **Arms:** `mip_heuristic_fpr_lp_effort` in {0 (off), 0.25, 0.5, 1.0, 2.0},
  paired on identical instances at **600 s**. Effort 0 is the control and is
  the current shipped default.
* **n:** same sizing rule as B+ — start at 48, extend on the same stop rule.
* **Required alongside:** `log_dev_level=3`, or the `[Heur] name=fpr_lp
  phase=dive` lines do not exist and the fire rate is unmeasurable. This is
  the error that made the first attempt worthless.
* **Fire rate is bimodal and must be reported per instance, not averaged.**
  Measured over 10 instances at 120 s: 4 fire and heavily (35, 36, 10, 6
  dispatches), 6 do not fire at all. An arm's mean effect will be diluted by
  the instances where the heuristic never ran, so the analysis must condition
  on the dive having executed.
* **Envelope share is descriptive, not tuned, unless measured directly:**
  fpr_lp charges back to `heuristic_lp_iterations`, so the share it actually
  takes from RENS/RINS is readable from the logs at each sweep point, and that
  is what makes the sweep a *calibration* rather than an on/off test.
