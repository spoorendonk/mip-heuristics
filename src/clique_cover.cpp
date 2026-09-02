#include "clique_cover.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace clique_cover {

namespace {

// A slot is live iff `start` is not -1; `removeClique` recycles slots through
// the table's free list and leaves the dead ones in place.
bool is_live(const Clique& cl) {
    return cl.start != -1;
}

std::vector<uint8_t> binary_mask(const std::vector<HighsInt>& bin, HighsInt ncol) {
    std::vector<uint8_t> is_bin(static_cast<size_t>(ncol), 0);
    for (HighsInt j : bin) {
        is_bin[j] = 1;
    }
    return is_bin;
}

// --- Sect. 4.1 step 1: equality cliques ------------------------------------
//
// "We first process equality cliques (if any): this is done by just iterating
// once over the equality cliques, in the order in which they appear in the
// model, and adding a clique to the cover if and only if it is disjoint w.r.t.
// to all the ones that have already been added, so that we can keep track of
// the equality status."
//
// Clique-table slot order stands in for model order.  `Clique::origin` names
// the originating row, but HiGHS resets it to -1 or kHighsIInf during clique
// merging, extraction from cuts and `rebuild`/`buildFrom`, so it is not a key
// the whole table can be sorted on; slot order is insertion order except where
// a removed clique's slot has been recycled.  That approximation is the one
// fidelity gap in this step.
//
// `stamp` de-duplicates a column that appears twice in one clique.  It is
// caller-owned so the two phases can each get a clean one.
HighsInt add_equality_groups(const std::vector<Clique>& cliques,
                             const std::vector<CliqueVar>& entries,
                             const std::vector<uint8_t>& is_bin, std::vector<uint8_t>& covered,
                             std::vector<HighsInt>& stamp, Cover& cover) {
    HighsInt num_groups = 0;
    std::vector<HighsInt> picked;
    const auto num_cliques = static_cast<HighsInt>(cliques.size());

    for (HighsInt c = 0; c < num_cliques; ++c) {
        const Clique& cl = cliques[c];
        if (!is_live(cl) || !cl.equality) {
            continue;
        }

        picked.clear();
        bool disjoint = true;
        for (HighsInt k = cl.start; k < cl.end; ++k) {
            const auto j = static_cast<HighsInt>(entries[k].col);
            if (is_bin[j] == 0) {
                continue;
            }
            if (covered[j] != 0) {
                disjoint = false;
                break;
            }
            if (stamp[j] == c) {
                continue;
            }
            stamp[j] = c;
            picked.push_back(k);
        }
        // Empty, not "small": a clique that contributes no rankable binary is
        // no group at all.  There is deliberately no size threshold — the
        // paper's greedy has none, and one here would be a deviation, not a
        // gap being filled (a clique overlapping an accepted equality group is
        // left with one rankable binary in the paper's own setting too).
        if (!disjoint || picked.empty()) {
            continue;
        }

        cover.group_start.push_back(static_cast<HighsInt>(cover.members.size()));
        for (HighsInt k : picked) {
            const auto j = static_cast<HighsInt>(entries[k].col);
            covered[j] = 1;
            cover.members.push_back(j);
            cover.member_pos.push_back(static_cast<uint8_t>(entries[k].val));
        }
        ++num_groups;
    }
    return num_groups;
}

// The column -> covering-cliques incidence, restricted to the binaries step 1
// left uncovered, held as CSR (counts, prefix sum, flat array) rather than a
// vector per column: the whole of step 2 is then linear in the number of
// clique entries.
struct ColCliqueIndex {
    std::vector<HighsInt> start;  // size ncol + 1
    std::vector<HighsInt> clq;    // flat, `start[j] .. start[j+1]`
    std::vector<HighsInt> size;   // per clique: uncovered binaries it covers
};

ColCliqueIndex build_incidence(const std::vector<Clique>& cliques,
                               const std::vector<CliqueVar>& entries,
                               const std::vector<uint8_t>& is_bin,
                               const std::vector<uint8_t>& covered, std::vector<HighsInt>& stamp,
                               HighsInt ncol) {
    const auto num_cliques = static_cast<HighsInt>(cliques.size());
    ColCliqueIndex idx;
    idx.start.assign(static_cast<size_t>(ncol) + 1, 0);
    idx.size.assign(static_cast<size_t>(num_cliques), 0);

    // Sect. 4.1 step 2 bullet 1: "counting how many binary variables are
    // covered by each clique".
    auto scan = [&](auto&& on_hit) {
        std::ranges::fill(stamp, HighsInt{-1});
        for (HighsInt c = 0; c < num_cliques; ++c) {
            const Clique& cl = cliques[c];
            if (!is_live(cl)) {
                continue;
            }
            for (HighsInt k = cl.start; k < cl.end; ++k) {
                const auto j = static_cast<HighsInt>(entries[k].col);
                if (is_bin[j] == 0 || covered[j] != 0 || stamp[j] == c) {
                    continue;
                }
                stamp[j] = c;
                on_hit(j, c);
            }
        }
    };

    scan([&](HighsInt j, HighsInt c) {
        ++idx.start[j + 1];
        ++idx.size[c];
    });
    for (HighsInt j = 0; j < ncol; ++j) {
        idx.start[j + 1] += idx.start[j];
    }
    idx.clq.assign(static_cast<size_t>(idx.start[ncol]), 0);

    std::vector<HighsInt> fill(idx.start.begin(), idx.start.end() - 1);
    scan([&](HighsInt j, HighsInt c) { idx.clq[fill[j]++] = c; });
    return idx;
}

// Sect. 4.1 step 2 bullets 2-4: assign each remaining binary to the largest
// clique covering it, recount the selected cliques, then make the local
// adjustment the paper names ("switch a variable to a larger selected
// clique").  Ties keep the lowest-numbered / incumbent clique, so the result
// is deterministic.
void assign_to_cliques(const std::vector<HighsInt>& bin, const std::vector<uint8_t>& covered,
                       const ColCliqueIndex& idx, std::vector<HighsInt>& assign,
                       std::vector<HighsInt>& assigned) {
    for (HighsInt j : bin) {
        if (covered[j] != 0) {
            continue;
        }
        HighsInt best = -1;
        // Any clique covering `j` is a candidate: "assigning each binary
        // variable to the largest clique covering it", with no floor on what
        // counts as a clique.
        HighsInt best_size = 0;
        for (HighsInt p = idx.start[j]; p < idx.start[j + 1]; ++p) {
            const HighsInt c = idx.clq[p];
            if (idx.size[c] > best_size) {
                best_size = idx.size[c];
                best = c;
            }
        }
        assign[j] = best;
        if (best >= 0) {
            ++assigned[best];
        }
    }

    for (HighsInt j : bin) {
        const HighsInt cur = assign[j];
        if (covered[j] != 0 || cur < 0) {
            continue;
        }
        HighsInt best = cur;
        HighsInt best_size = assigned[cur];
        for (HighsInt p = idx.start[j]; p < idx.start[j + 1]; ++p) {
            const HighsInt c = idx.clq[p];
            if (assigned[c] > best_size) {
                best_size = assigned[c];
                best = c;
            }
        }
        if (best != cur) {
            --assigned[cur];
            ++assigned[best];
            assign[j] = best;
        }
    }
}

// Sect. 4.1 step 2 bullet 5, "finally sorting the selected cliques by size",
// then emitting each group's members in the order in which they appear in the
// clique itself.  The sort is stable and descending: the ordered list of
// cliques is meant to put the most constrained groups first.
void emit_selected_groups(const std::vector<Clique>& cliques, const std::vector<CliqueVar>& entries,
                          const std::vector<HighsInt>& assign,
                          const std::vector<HighsInt>& assigned, std::vector<uint8_t>& covered,
                          Cover& cover) {
    std::vector<HighsInt> selected;
    const auto num_cliques = static_cast<HighsInt>(assigned.size());
    for (HighsInt c = 0; c < num_cliques; ++c) {
        if (assigned[c] > 0) {
            selected.push_back(c);
        }
    }
    std::ranges::stable_sort(selected,
                             [&](HighsInt a, HighsInt b) { return assigned[a] > assigned[b]; });

    for (HighsInt c : selected) {
        const Clique& cl = cliques[c];
        cover.group_start.push_back(static_cast<HighsInt>(cover.members.size()));
        for (HighsInt k = cl.start; k < cl.end; ++k) {
            const auto j = static_cast<HighsInt>(entries[k].col);
            if (assign[j] != c || covered[j] != 0) {
                continue;
            }
            covered[j] = 1;
            cover.members.push_back(j);
            cover.member_pos.push_back(static_cast<uint8_t>(entries[k].val));
        }
    }
}

// One clique's Fig. 3 scan (lines 3-23): the literal sum, the most positive
// still-free literal, and whether a literal is already fixed true — in which
// case the whole clique is dead and the caller drops it.
struct CliqueScan {
    double sum = 0.0;
    HighsInt best_var = -1;
    double best_value = 0.0;
    bool skip = false;
};

CliqueScan scan_clique(const std::vector<CliqueVar>& entries, const Clique& cl,
                       const double* lp_ref, const double* dom_lower, const double* dom_upper) {
    CliqueScan out;
    for (HighsInt k = cl.start; k < cl.end; ++k) {
        const CliqueVar lit = entries[k];
        const auto j = static_cast<HighsInt>(lit.col);
        // Fig. 3 lines 9-10 / 17-18: a literal already fixed *true* makes the
        // whole clique dead — every other literal is then implied false, so
        // there is nothing left to rank.  The figure skips the clique, not
        // merely the candidate.
        const bool positive = lit.val != 0;
        if (positive ? dom_lower[j] == 1.0 : dom_upper[j] == 0.0) {
            out.skip = true;
            return out;
        }
        const double v = positive ? lp_ref[j] : 1.0 - lp_ref[j];
        out.sum += v;
        // Lines 13 / 21: the candidate must still be free to take the value
        // the literal asks for.
        const bool free_for_true = positive ? dom_upper[j] == 1.0 : dom_lower[j] == 0.0;
        if (v > out.best_value && free_for_true) {
            out.best_var = j;
            out.best_value = v;
        }
    }
    return out;
}

}  // namespace

Cover build_clique_cover(const std::vector<Clique>& cliques, const std::vector<CliqueVar>& entries,
                         const std::vector<HighsInt>& bin, HighsInt ncol) {
    Cover cover;
    if (bin.empty()) {
        cover.group_start.push_back(0);
        return cover;
    }

    const std::vector<uint8_t> is_bin = binary_mask(bin, ncol);
    std::vector<uint8_t> covered(static_cast<size_t>(ncol), 0);
    std::vector<HighsInt> stamp(static_cast<size_t>(ncol), -1);

    cover.num_equality_groups =
        add_equality_groups(cliques, entries, is_bin, covered, stamp, cover);

    const ColCliqueIndex idx = build_incidence(cliques, entries, is_bin, covered, stamp, ncol);
    std::vector<HighsInt> assign(static_cast<size_t>(ncol), -1);
    std::vector<HighsInt> assigned(cliques.size(), 0);
    assign_to_cliques(bin, covered, idx, assign, assigned);
    emit_selected_groups(cliques, entries, assign, assigned, covered, cover);

    cover.group_start.push_back(static_cast<HighsInt>(cover.members.size()));

    for (HighsInt j : bin) {
        if (covered[j] == 0) {
            cover.uncovered.push_back(j);
        }
    }
    return cover;
}

std::vector<HighsInt> cliques2_order(const std::vector<Clique>& cliques,
                                     const std::vector<CliqueVar>& entries,
                                     const std::vector<HighsInt>& bin, HighsInt ncol,
                                     const double* lp_ref, const double* dom_lower,
                                     const double* dom_upper) {
    std::vector<HighsInt> order;
    if (bin.empty()) {
        return order;
    }
    order.reserve(bin.size());

    const std::vector<uint8_t> is_bin = binary_mask(bin, ncol);
    std::vector<uint8_t> covered(static_cast<size_t>(ncol), 0);

    // Fig. 3 lines 25-29: "Append(S, j)".  A clique table is not a partition,
    // so a column can appear in many cliques; the prose is explicit that only
    // "the remaining uncovered binary variables" follow the best one.
    auto append = [&](HighsInt j) {
        if (is_bin[j] == 0 || covered[j] != 0) {
            return;
        }
        covered[j] = 1;
        order.push_back(j);
    };

    const auto num_cliques = static_cast<HighsInt>(cliques.size());
    for (HighsInt c = 0; c < num_cliques; ++c) {
        const Clique& cl = cliques[c];
        if (!is_live(cl)) {
            continue;
        }

        const CliqueScan scan = scan_clique(entries, cl, lp_ref, dom_lower, dom_upper);

        // Fig. 3 line 24: `bestVar != None and sum = 1`.  A clique that is not
        // LP-tight contributes nothing at all — its members reach the order
        // through a later clique or through the formulation-order tail.
        if (scan.skip || scan.best_var < 0 || std::abs(scan.sum - 1.0) > kCliqueTightnessTol) {
            continue;
        }

        // `best_var` is a clique column, which need not be in the binary
        // bucket: a column the root domain moved off [0, 1] is not rankable
        // and `append` drops it, so the group is emitted headless and the
        // second-best literal leads.  The figure's own `l_j`/`u_j` guards
        // already exclude the fixed cases, which is why no construction of
        // this was found; the code absorbs it rather than special-casing it.
        append(scan.best_var);
        for (HighsInt k = cl.start; k < cl.end; ++k) {
            const auto j = static_cast<HighsInt>(entries[k].col);
            if (j != scan.best_var) {
                append(j);
            }
        }
    }

    for (HighsInt j : bin) {
        if (covered[j] == 0) {
            order.push_back(j);
        }
    }
    return order;
}

}  // namespace clique_cover
