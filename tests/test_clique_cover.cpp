// Tests for the paper's clique-cover variable rankings (issue #141).
//
// Salvagnin, Roberti, Fischetti, "A fix-propagate-repair heuristic for mixed
// integer programming", MPC 17:111-139, 2025, Sect. 4.1 and Figs. 2-3.
//
// These assert the paper's *properties* — equality cliques first, selected
// cliques in descending size order, the within-clique order preserved, an
// uncovered tail in formulation order, and a clique with a literal already
// fixed true skipped in its entirety — rather than pinning an opaque
// permutation, which would fail on any legal change to the tie-breaks and
// tell a reader nothing about which rule broke.
//
// The cover functions are pure over a clique-table snapshot, so the unit
// cases build synthetic tables directly; the last case runs the whole
// `compute_var_order` path on a bundled instance to pin the invariant every
// caller depends on (a permutation of [0, ncol) with the type buckets in
// order).

#include "clique_cover.h"
#include "fpr_var_order.h"
#include "heuristic_common.h"
#include "Highs.h"
#include "mip/HighsCliqueTable.h"
#include "parallel/HighsParallel.h"
#include "test_common.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

namespace {

using clique_cover::Clique;
using clique_cover::CliqueVar;

// A synthetic clique table.  `add` appends one clique's literals to the flat
// entry array and records its slot; `kill` marks the last slot dead the way
// `HighsCliqueTable::removeClique` does, so the "dead slots are skipped" rule
// is exercised rather than assumed.
struct FakeTable {
    std::vector<Clique> cliques;
    std::vector<CliqueVar> entries;

    void add(const std::vector<CliqueVar>& lits, bool equality) {
        Clique cl{};
        cl.start = static_cast<HighsInt>(entries.size());
        cl.end = cl.start + static_cast<HighsInt>(lits.size());
        cl.origin = kHighsIInf;
        cl.numZeroFixed = 0;
        cl.equality = equality;
        cliques.push_back(cl);
        entries.insert(entries.end(), lits.begin(), lits.end());
    }

    void kill_last() { cliques.back().start = -1; }
};

std::vector<CliqueVar> pos(const std::vector<HighsInt>& cols) {
    std::vector<CliqueVar> out;
    out.reserve(cols.size());
    for (HighsInt c : cols) {
        out.emplace_back(c, 1);
    }
    return out;
}

std::vector<HighsInt> iota_cols(HighsInt ncol) {
    std::vector<HighsInt> bin(static_cast<size_t>(ncol));
    for (HighsInt j = 0; j < ncol; ++j) {
        bin[j] = j;
    }
    return bin;
}

// The cover flattened the way `rank_typecl` flattens it.
std::vector<HighsInt> flatten(const clique_cover::Cover& cover) {
    std::vector<HighsInt> out = cover.members;
    out.insert(out.end(), cover.uncovered.begin(), cover.uncovered.end());
    return out;
}

std::vector<HighsInt> group(const clique_cover::Cover& cover, HighsInt g) {
    return {cover.members.begin() + cover.group_start[g],
            cover.members.begin() + cover.group_start[g + 1]};
}

// The literal signs of one group, as `1` / `0` — what Fig. 2's weights read.
std::vector<int> group_signs(const clique_cover::Cover& cover, HighsInt g) {
    std::vector<int> out;
    for (HighsInt i = cover.group_start[g]; i < cover.group_start[g + 1]; ++i) {
        out.push_back(cover.member_pos[i]);
    }
    return out;
}

}  // namespace

TEST_CASE("clique_cover: equality cliques come before inequality cliques", "[clique-cover]") {
    // A *larger* inequality clique is registered first.  Sect. 4.1 processes
    // equality cliques before anything else, so size does not rescue it: the
    // equality group leads even though it is smaller.
    FakeTable t;
    t.add(pos({0, 1, 2, 3}), /*equality=*/false);
    t.add(pos({4, 5}), /*equality=*/true);

    const auto cover = clique_cover::build_clique_cover(t.cliques, t.entries, iota_cols(6), 6);

    REQUIRE(cover.num_equality_groups == 1);
    REQUIRE(cover.num_groups() == 2);
    CHECK(group(cover, 0) == std::vector<HighsInt>{4, 5});
    CHECK(group(cover, 1) == std::vector<HighsInt>{0, 1, 2, 3});
}

TEST_CASE("clique_cover: an equality clique overlapping an accepted one is dropped",
          "[clique-cover]") {
    // "adding a clique to the cover if and only if it is disjoint w.r.t. to
    // all the ones that have already been added".  The second equality clique
    // shares column 1, so the whole clique is rejected — not trimmed to its
    // free members — and column 5 has no other clique, so it lands in the
    // formulation-order tail.
    FakeTable t;
    t.add(pos({0, 1}), /*equality=*/true);
    t.add(pos({1, 5}), /*equality=*/true);

    const auto cover = clique_cover::build_clique_cover(t.cliques, t.entries, iota_cols(6), 6);

    REQUIRE(cover.num_equality_groups == 1);
    CHECK(group(cover, 0) == std::vector<HighsInt>{0, 1});

    // The rejected clique is rejected *as an equality clique*, not struck from
    // the table: step 2 counts "how many binary variables are covered by each
    // clique", and it still covers column 5.  So 5 is grouped, not tailed —
    // and there is no size floor keeping a one-variable group out, because the
    // paper's greedy has none.
    REQUIRE(cover.num_groups() == 2);
    CHECK(group(cover, 1) == std::vector<HighsInt>{5});
    CHECK(cover.uncovered == std::vector<HighsInt>{2, 3, 4});
}

TEST_CASE("clique_cover: selected cliques are sorted by size, descending", "[clique-cover]") {
    // Registered smallest-first, so table order alone would produce the
    // reverse.  Sect. 4.1's last step is "finally sorting the selected
    // cliques by size".
    FakeTable t;
    t.add(pos({0, 1}), false);
    t.add(pos({2, 3, 4, 5}), false);
    t.add(pos({6, 7, 8}), false);

    const auto cover = clique_cover::build_clique_cover(t.cliques, t.entries, iota_cols(9), 9);

    REQUIRE(cover.num_equality_groups == 0);
    REQUIRE(cover.num_groups() == 3);
    CHECK(group(cover, 0) == std::vector<HighsInt>{2, 3, 4, 5});
    CHECK(group(cover, 1) == std::vector<HighsInt>{6, 7, 8});
    CHECK(group(cover, 2) == std::vector<HighsInt>{0, 1});
}

TEST_CASE("clique_cover: within a clique, the table's entry order is kept", "[clique-cover]") {
    // "Within each clique, variable are then sorted according to the order in
    // which they appear in the clique itself."  The retired implementation
    // sorted the column indices instead, which is what this pins against.
    FakeTable t;
    t.add(pos({5, 2, 9}), false);
    t.add(pos({7, 1}), true);

    const auto cover = clique_cover::build_clique_cover(t.cliques, t.entries, iota_cols(10), 10);

    REQUIRE(cover.num_groups() == 2);
    CHECK(group(cover, 0) == std::vector<HighsInt>{7, 1});
    CHECK(group(cover, 1) == std::vector<HighsInt>{5, 2, 9});
}

TEST_CASE("clique_cover: uncovered binaries are appended in formulation order", "[clique-cover]") {
    // "Binary variables not covered by any clique are moved to the end of the
    // binary bucket, again in formulation order."  Columns 0 and 4 are in no
    // clique; column 6's clique is a dead slot, so it is uncovered too.
    FakeTable t;
    t.add(pos({3, 1}), false);
    t.add(pos({6, 5}), false);
    t.kill_last();
    t.add(pos({2, 7}), false);

    const auto cover = clique_cover::build_clique_cover(t.cliques, t.entries, iota_cols(8), 8);

    CHECK(cover.uncovered == std::vector<HighsInt>{0, 4, 5, 6});
    CHECK(flatten(cover).size() == 8);
}

TEST_CASE("clique_cover: every binary appears exactly once", "[clique-cover]") {
    // The permutation invariant `compute_var_order` documents.  Overlapping
    // cliques, a non-binary column inside a clique (8, absent from `bin`), a
    // dead slot and a repeated literal all have to leave it intact.
    FakeTable t;
    t.add(pos({0, 1, 2}), true);
    t.add(pos({1, 3, 4}), false);
    t.add(pos({2, 3, 8}), false);
    t.add(pos({5, 5, 6}), false);
    t.add(pos({4, 7}), false);
    t.kill_last();

    const std::vector<HighsInt> bin{0, 1, 2, 3, 4, 5, 6, 7};
    const auto cover = clique_cover::build_clique_cover(t.cliques, t.entries, bin, 9);

    auto flat = flatten(cover);
    std::ranges::sort(flat);
    CHECK(flat == bin);
}

TEST_CASE("clique_cover: a shared binary goes to the largest covering clique", "[clique-cover]") {
    // Sect. 4.1 step 2, bullet 2.  Column 3 is covered by both cliques; the
    // larger one must take it.  Every other cover case here uses *disjoint*
    // cliques, where the assignment is forced and this rule is unfalsifiable.
    //
    // The sizes are chosen so bullet 4 cannot paper over a wrong pick: send 3
    // to the smaller clique instead and the two end up with three variables
    // each, which is not the strict inequality the local adjustment moves on,
    // so the groups come out {0,1,2} / {3,4,5} and both assertions fail.
    FakeTable t;
    t.add(pos({0, 1, 2, 3}), false);
    t.add(pos({3, 4, 5}), false);

    const auto cover = clique_cover::build_clique_cover(t.cliques, t.entries, iota_cols(6), 6);

    REQUIRE(cover.num_groups() == 2);
    CHECK(group(cover, 0) == std::vector<HighsInt>{0, 1, 2, 3});
    CHECK(group(cover, 1) == std::vector<HighsInt>{4, 5});
}

TEST_CASE("clique_cover: the local adjustment moves a stranded variable", "[clique-cover]") {
    // Sect. 4.1 step 2, bullet 4 — "doing some local adjustments (e.g., switch
    // a variable to a larger selected clique)".  Pass 1 ranks by *potential*
    // size, pass 2 by how many variables each clique actually kept, and the
    // two orders have to disagree for the adjustment to be observable at all.
    //
    // R has the largest potential and takes columns 0-6.  P's potential (6)
    // beats Q's (3), so pass 1 sends column 9 to P — which R has otherwise
    // emptied, leaving P with one variable against Q's two.  The adjustment
    // moves 9 into Q.  Make pass 2 a no-op and P survives as a third,
    // one-variable group holding 9, so both assertions below fail.
    FakeTable t;
    t.add(pos({0, 1, 2, 3, 4, 5, 6}), false);
    t.add(pos({0, 1, 2, 3, 4, 9}), false);
    t.add(pos({7, 8, 9}), false);

    const auto cover = clique_cover::build_clique_cover(t.cliques, t.entries, iota_cols(10), 10);

    REQUIRE(cover.num_groups() == 2);
    CHECK(group(cover, 0) == std::vector<HighsInt>{0, 1, 2, 3, 4, 5, 6});
    CHECK(group(cover, 1) == std::vector<HighsInt>{7, 8, 9});
    CHECK(cover.uncovered.empty());
}

TEST_CASE("clique_cover: a group records its literals' signs", "[clique-cover]") {
    // `member_pos` is the only thing carrying the literal's sign out of the
    // cover, and it is what Fig. 2's weight reads (`x^ac_j` for a positive
    // literal, `1 - x^ac_j` for a negative one).  Every other cover case here
    // builds positive-only cliques, so nothing else would notice the field
    // being wrong.
    FakeTable t;
    t.add({CliqueVar(2, 0), CliqueVar(0, 1), CliqueVar(1, 0)}, false);

    const auto cover = clique_cover::build_clique_cover(t.cliques, t.entries, iota_cols(3), 3);

    REQUIRE(cover.num_groups() == 1);
    CHECK(group(cover, 0) == std::vector<HighsInt>{2, 0, 1});
    CHECK(group_signs(cover, 0) == std::vector<int>{0, 1, 0});
}

TEST_CASE("cliques2: a clique with a literal already fixed true is skipped entirely",
          "[clique-cover]") {
    // Fig. 3 lines 9-10 and 17-18.  The retired implementation skipped such a
    // literal only as a best-variable candidate and still emitted the clique's
    // members in clique order; the figure abandons the clique.  Column 1 is
    // fixed to 1, so columns 0 and 2 reach the order through the tail, in
    // formulation order rather than clique order (2 before 0 would be clique
    // order).
    FakeTable t;
    t.add(pos({2, 1, 0}), false);

    const std::vector<double> lp{0.5, 1.0, 0.5};
    const std::vector<double> lb{0.0, 1.0, 0.0};
    const std::vector<double> ub{1.0, 1.0, 1.0};
    const std::vector<HighsInt> bin{0, 2};

    const auto order =
        clique_cover::cliques2_order(t.cliques, t.entries, bin, 3, lp.data(), lb.data(), ub.data());
    CHECK(order == std::vector<HighsInt>{0, 2});
}

TEST_CASE("cliques2: the most positive literal leads a tight clique", "[clique-cover]") {
    // Fig. 3 lines 24-29: append `bestVar`, then the rest "in the order they
    // appear in the clique".
    FakeTable t;
    t.add(pos({0, 1, 2}), false);

    const std::vector<double> lp{0.2, 0.1, 0.7};
    const std::vector<double> lb{0.0, 0.0, 0.0};
    const std::vector<double> ub{1.0, 1.0, 1.0};

    const auto order = clique_cover::cliques2_order(t.cliques, t.entries, iota_cols(3), 3,
                                                    lp.data(), lb.data(), ub.data());
    CHECK(order == std::vector<HighsInt>{2, 0, 1});
}

TEST_CASE("cliques2: a clique that is not LP-tight contributes nothing", "[clique-cover]") {
    // Fig. 3 line 24 gates on `sum = 1`.  The literal sum here is 0.5, so the
    // clique is skipped and its members reach the order through the
    // formulation-order tail — column 2 no longer leads.
    FakeTable t;
    t.add(pos({0, 1, 2}), false);

    const std::vector<double> lp{0.2, 0.1, 0.2};
    const std::vector<double> lb{0.0, 0.0, 0.0};
    const std::vector<double> ub{1.0, 1.0, 1.0};

    const auto order = clique_cover::cliques2_order(t.cliques, t.entries, iota_cols(3), 3,
                                                    lp.data(), lb.data(), ub.data());
    CHECK(order == std::vector<HighsInt>{0, 1, 2});
}

TEST_CASE("cliques2: the tightness test is two-sided", "[clique-cover]") {
    // A merged or lifted clique is bounded by no single row, so its literal
    // sum at a given LP point can exceed 1.  Fig. 3 tests `sum = 1`; a
    // one-sided `sum >= 1 - tol` would reorder this clique.
    FakeTable t;
    t.add(pos({0, 1, 2}), false);

    const std::vector<double> lp{0.9, 0.8, 0.7};
    const std::vector<double> lb{0.0, 0.0, 0.0};
    const std::vector<double> ub{1.0, 1.0, 1.0};

    const auto order = clique_cover::cliques2_order(t.cliques, t.entries, iota_cols(3), 3,
                                                    lp.data(), lb.data(), ub.data());
    CHECK(order == std::vector<HighsInt>{0, 1, 2});
}

TEST_CASE("cliques2: negative literals weigh 1-x and die on u_j = 0", "[clique-cover]") {
    // Fig. 3's else branch (lines 16-23).  Clique A is {~0, 1}: literal ~0 is
    // worth 1 - 0.1 = 0.9 and literal 1 is worth 0.1, so the sum is tight and
    // column 0 leads.  Clique B carries ~2 with u_2 = 0 — the literal is
    // already true — so B is skipped and column 3 falls to the tail.
    FakeTable t;
    t.add({CliqueVar(0, 0), CliqueVar(1, 1)}, false);
    t.add({CliqueVar(2, 0), CliqueVar(3, 1)}, false);

    const std::vector<double> lp{0.1, 0.1, 0.0, 0.9};
    const std::vector<double> lb{0.0, 0.0, 0.0, 0.0};
    const std::vector<double> ub{1.0, 1.0, 0.0, 1.0};
    const std::vector<HighsInt> bin{0, 1, 3};

    const auto order =
        clique_cover::cliques2_order(t.cliques, t.entries, bin, 4, lp.data(), lb.data(), ub.data());
    CHECK(order == std::vector<HighsInt>{0, 1, 3});
}

TEST_CASE("compute_var_order: clique strategies return a bucketed permutation",
          "[clique-cover][heuristic]") {
    // The end-to-end invariant every caller relies on: a permutation of
    // [0, ncol) with binaries first, then general integers, then continuous.
    // This harness's clique table is *empty* — `build_bare_mipsolver` turns
    // HiGHS presolve off and it is `HPresolve` that extracts cliques — so
    // what this pins is the plumbing and the invariant on the degenerate
    // path, not the cover.  The cover itself is covered by the synthetic and
    // real-`HighsCliqueTable` cases above and, end to end, by the
    // characterization solves, which run presolve.
    highs::parallel::initialize_scheduler();
    Highs highs;
    highs.setOptionValue("output_flag", false);
    HighsCallback cb(&highs);
    auto mipsolver = build_bare_mipsolver(highs, cb, "p0548.mps");
    const HighsInt ncol = mipsolver->model_->num_col_;
    REQUIRE(ncol > 0);

    Rng rng(12345);
    const std::vector<double> lp_ref(static_cast<size_t>(ncol), 0.25);

    for (auto strategy : {VarStrategy::kTypecl, VarStrategy::kCliques, VarStrategy::kCliques2}) {
        auto order = compute_var_order(*mipsolver, strategy, rng, lp_ref.data());
        REQUIRE(order.size() == static_cast<size_t>(ncol));

        HighsInt last_bucket = 0;
        for (HighsInt j : order) {
            HighsInt bucket = 2;
            if (is_integer(mipsolver->model_->integrality_, j)) {
                bucket = mipsolver->mipdata_->getDomain().isBinary(j) ? 0 : 1;
            }
            CHECK(bucket >= last_bucket);
            last_bucket = bucket;
        }

        std::ranges::sort(order);
        for (HighsInt j = 0; j < ncol; ++j) {
            REQUIRE(order[j] == j);
        }
    }
}

TEST_CASE("clique_cover: the patched accessors expose a real HighsCliqueTable", "[clique-cover]") {
    // The synthetic-table cases above would pass against accessors that
    // returned the wrong arrays, and `build_bare_mipsolver` cannot help: it
    // turns HiGHS presolve off, and it is `HPresolve` that extracts cliques,
    // so that harness's table is empty and every clique strategy degenerates
    // to `type`.  Drive a real `HighsCliqueTable` instead — `doAddClique` is
    // public — so the patch's `getCliques()` / `getCliqueEntries()` are held
    // to the slot bounds, the entry order and the equality flag they promise.
    constexpr HighsInt kNcol = 6;
    HighsCliqueTable table(kNcol);
    auto packing = pos({4, 0, 2});
    auto partitioning = pos({5, 1});
    table.doAddClique(packing.data(), static_cast<HighsInt>(packing.size()),
                      /*equality=*/false);
    table.doAddClique(partitioning.data(), static_cast<HighsInt>(partitioning.size()),
                      /*equality=*/true);

    const auto cover = clique_cover::build_clique_cover(
        table.getCliques(), table.getCliqueEntries(), iota_cols(kNcol), kNcol);

    REQUIRE(cover.num_equality_groups == 1);
    REQUIRE(cover.num_groups() == 2);
    CHECK(group(cover, 0) == std::vector<HighsInt>{5, 1});
    CHECK(group(cover, 1) == std::vector<HighsInt>{4, 0, 2});
    CHECK(cover.uncovered == std::vector<HighsInt>{3});
}
