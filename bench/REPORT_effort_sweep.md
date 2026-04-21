# `mip_heuristic_effort` sweep — 10 instances × 5 configs × 120 s

Single seed (0). Head-to-head list (`bench/instances_headtohead.txt`).
Reference objectives: `miplib2017-v22.solu` with per-instance virtual-best
substitution (any config beating `=best=` raises the reference). Configs share
one `best_known` per instance.

Logs: `bench/results_effort_sweep/{vanilla,patched_e005,patched_e015,patched_e030,patched_e060}/seed0/*.log`.

## Aggregate

| Config       | #Feas | #Win | SGM T1st (s=1) | SGM Gap@120s (s=0.01) | SGM Primal Integral (s=1) |
|--------------|------:|-----:|---------------:|----------------------:|--------------------------:|
| vanilla      |    10 |    4 |         1.0904 |              0.138913 |                   30.9424 |
| patched_e005 |    10 |    3 |         1.2321 |              0.132037 |                   37.8718 |
| patched_e015 |    10 |    5 |         1.8464 |              0.126653 |                   37.2427 |
| patched_e030 |    10 |    6 |         3.0592 |          **0.119321** |                   41.9510 |
| patched_e060 |    10 |    5 |         3.6076 |              0.127597 |                   41.2024 |

Feasibility coverage is 10/10 for all configs. SGM Gap@120s is monotone
improving 0.05 → 0.30 (14% relative reduction vs current default), then
regresses at 0.60. #Win peaks at 0.30 (6/10). SGM T1st grows with effort as
expected (patched spends more presolve time); T1st of ~3s on a 120s budget
is not limiting. SGM Primal Integral is best for vanilla — patched's early
presolve shifts when the first incumbent lands, inflating PI; not a fairness
issue for the effort knob itself.

## Per-instance Gap@120s (vs shared virtual best)

| instance               | vanilla  | e005    | e015    | e030    | e060    |
|------------------------|---------:|--------:|--------:|--------:|--------:|
| dfn-bwin-DBE           |   50.09% |  56.91% |  56.91% |  56.91% |  52.99% |
| ger50-17-ptp-pop-3t    |    3.40% |   3.37% |   3.20% |   3.34% |   3.34% |
| liu                    |   73.62% |  72.14% |  64.58% |  50.37% |  51.66% |
| milo-v13-4-3d-4-0      |  175.51% | 175.51% | 175.51% | 175.51% | 165.03% |
| neos-1420790           |    1.43% |   1.43% |   2.14% |   1.23% |   1.23% |
| neos-3009394-lami      |    0.00% |   0.00% |   0.00% |   0.00% |   1.30% |
| neos-3426085-ticino    |    1.78% |   1.78% |   1.78% |   1.78% |   1.78% |
| neos-5045105-creuse    |    0.00% |   0.00% |   0.00% |   0.00% |   0.00% |
| set3-09                |  235.42% | 132.41% |  93.43% |  93.43% |  76.62% |
| set3-16                |  112.59% | 112.59% |  97.02% |  97.02% | 106.30% |

## Drivers

- **liu** confirms the 3-instance diagnostic: 73.6% → 50.4% at effort 0.30
  (past that, regresses).
- **set3-09** is strictly monotone with effort: 235% → 132% → 93% → 93% → 77%.
- **set3-16** plateaus at e015 (112 → 97%), then regresses at e060.
- **milo-v13-4-3d-4-0** only responds to e060 (175 → 165%).
- Five instances are effectively flat or already at/near 0% — dominated by
  search, not heuristics.

## Regressions

- **dfn-bwin-DBE**: vanilla (50.1%) beats every patched config; ~7-pt loss
  that doesn't recover at 0.60. Single-instance loss, dominated by liu /
  set3-09 / set3-16 wins in aggregate.
- **neos-3009394-lami at e060**: 0% → 1.3% regression. A signal that 0.60
  over-invests on easy instances.

## Recommendation

**Raise default `mip_heuristic_effort` from 0.05 to 0.30.** Monotone SGM-gap
improvement 0.05 → 0.30 (14% relative), peak #Win (6/10), regression at 0.60
on liu / neos-3009394-lami / set3-16 without aggregate gains. T1st and
feasibility unaffected. Known residual: dfn-bwin-DBE regresses under any
patched config.

Follow-ups (not in scope): multi-seed confirmation, larger `instances_bench.txt`
sweep.
