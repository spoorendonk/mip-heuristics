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

### B+ steps

1. **Extend the confirmation to n=48**, full solves at **600 s**, reusing the
   20 already done (+28 instances), arms **A, B, D**. ~9 h/arm-set at the
   measured rate; chunked and resumable. Stop rule fixed in advance: if the
   top two are within the n=48 detectable margin, extend to **n=81** rather
   than declare a winner.
2. **Held-out at the campaign metric.** Repeat on held-out instances at
   **600 s full solves**, not presolve-only — a stratified subset of the 143,
   sized the same way. This is the only stage that can say whether a ranking
   generalises on the metric #108 scores.
3. **Effort-tune the shipped shape (new).** D's eight values come from
   Ablation A, where each heuristic was measured **alone** at an unbindable
   budget; the four have never been tuned *together*. The free search never
   proposed a 4-heuristic configuration because it always dropped Scylla —
   a selection made on the screen metric that we now know overfits. So run a
   **constrained irace**: `fj`, `fpr`, `local_mip`, `scylla` all forced on,
   searching only the eight effort/patience values. Smaller space than the
   free search, and it answers the question nobody has asked: *is D's vector
   any good?* Add the winner as a fourth confirmation arm.
4. **Two seeds minimum** in the confirmation from here on. Everything above is
   one seed, and run-to-run variation is inside the margins being compared.

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
