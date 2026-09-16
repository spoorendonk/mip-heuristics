# Reference papers

The PDFs in this directory are the source papers that each custom heuristic is based on. Each heuristic's C++ entry point lives in `src/` and deviates from the paper only where explicitly noted in code comments.

**All four are redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**, which is what the table's attribution is for. Three are the publishers' own open-access versions (Springer's `© The Author(s)` CC BY for the two *Mathematical Programming Computation* papers, Dagstuhl's LIPIcs CC BY for the CP 2024 one); the Scylla file is the CC BY 4.0 arXiv preprint [arXiv:2307.03466v2](https://arxiv.org/abs/2307.03466), not the Springer proceedings version, and the citation beside it names the published venue.

| Heuristic | Paper (file in this directory) | Primary sources | Citation |
|-----------|--------------------------------|-----------------|----------|
| **FPR** (Fix, Propagate, Repair) | [`A fix-propagate-repair heuristic for mixed integer programming.pdf`](./A%20fix-propagate-repair%20heuristic%20for%20mixed%20integer%20programming.pdf) | `src/fpr.cpp`, `src/fpr_core.cpp`, `src/fpr_strategies.h`, `src/fpr_var_order.cpp`, `src/fpr_val_select.cpp` (Table 3 variable/value rules), `src/clique_cover.cpp` (Sect. 4.1 clique covers, Figs. 2-3), `src/prop_engine.cpp`, `src/walksat.cpp`, `src/repair_walk.cpp` (Fig. 1 lines 7-8, in-tree repair), `src/repair_search.cpp` (Fig. 5) | Salvagnin, Roberti, Fischetti, *Mathematical Programming Computation* 17, 111–139, 2025. [doi:10.1007/s12532-024-00269-5](https://doi.org/10.1007/s12532-024-00269-5) |
| **FPR_LP** (LP-dependent FPR, paper Classes 2–3) | Same paper as FPR | `src/fpr_lp.cpp` (LP setup + arm assignment); `src/fpr_lp_arms.h` (the arm table and its reference classes); `src/fpr_lp_refs.cpp` (LP reference points); rounding kernel shared with FPR — `src/fpr_core.cpp`, `src/fpr_strategies.h`, `src/fpr_var_order.cpp`, `src/fpr_val_select.cpp`, `src/prop_engine.cpp`, `src/walksat.cpp`, `src/repair_walk.cpp`, `src/repair_search.cpp` | Same citation as FPR; runs during B&B dive using the root LP solution |
| **LocalMIP** | [`An Efficient Local Search Solver for Mixed Integer Programming.pdf`](./An%20Efficient%20Local%20Search%20Solver%20for%20Mixed%20Integer%20Programming.pdf) | `src/local_mip.cpp` (dispatch), `src/local_mip_core.cpp`, `src/local_mip_search.cpp`, `src/local_mip_worker.cpp`, `src/local_mip_caches.h` (incremental structures), `src/local_mip_construction.cpp` (cold-start sweep) | Peng Lin, Mengchuan Zou, Shaowei Cai, *Proc. CP 2024* (LIPIcs vol. 307), Article 19, pp. 19:1–19:19. [doi:10.4230/LIPIcs.CP.2024.19](https://doi.org/10.4230/LIPIcs.CP.2024.19) |
| **Scylla** (feasibility pump with PDLP) | [`Scylla: a matrix-free fix-propagate-and-project heuristic for mixed-integer optimization.pdf`](./Scylla%3A%20a%20matrix-free%20fix-propagate-and-project%20heuristic%20for%20mixed-integer%20optimization.pdf) (arXiv preprint) | `src/scylla.cpp`, `src/scylla_worker.cpp`, `src/pump_common.h`, `src/contested_pdlp.cpp`; the FPR rounding kernel above supplies Algorithm 1.1 line 12. **Ships disabled** (`mip_heuristic_scylla_effort` defaults to `0`) | Mexi, Besançon, Bolusani, Chmiela, Hoen, Gleixner, *OR Proceedings 2023*, 57–63. [doi:10.1007/978-3-031-58405-3_9](https://doi.org/10.1007/978-3-031-58405-3_9) |
| **FeasibilityJump** | [`Feasibility Jump_ an LP-free Lagrangian MIP heuristic.pdf`](./Feasibility%20Jump_%20an%20LP-free%20Lagrangian%20MIP%20heuristic.pdf) | `src/fj.cpp`, `src/fj_worker.cpp` (dispatch and workers around HiGHS's own FJ); `third_party/highs_patch/apply_patch.cmake` corrects two upstream defects in HiGHS's `feasibilityjump.hh` (#139) — the negative-coefficient jump value of eq. (5)/(6) and Algorithm 1's pre-bound slope, and the sign of Sect. 2.6's objective term — so the FJ that runs is HiGHS's implementation made faithful to this paper, not HiGHS's unmodified | Luteberget, Sartor, *Mathematical Programming Computation* 15, 365–388, 2023. [doi:10.1007/s12532-023-00234-8](https://doi.org/10.1007/s12532-023-00234-8) |

## Benchmark

The end-to-end evaluation uses the **`mipfeas`** benchmark: 233 instances
selected from the MIPLIB 2017 benchmark collection with the known-infeasible
ones excluded, scored by the primal integral.

| | Source |
|---|---|
| **`mipfeas`** (instance set and metric) | Bussieck, Dirkse (GAMS), [*Expanding the Focus: Introducing the mipfeas Benchmark*](https://www.gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark/), 17 March 2026. A collaboration including Mittelmann, ZIB and NVIDIA. |
| Published results | Hans Mittelmann's benchmark server at Arizona State, [plato.asu.edu](https://plato.asu.edu/bench.html). |

`PLATO` is the hostname of that server, not a name for the benchmark and not an
acronym; it names the place results are posted, nothing else. Use `mipfeas` for
the benchmark, its instance list, its metric and its protocol.
