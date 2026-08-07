"""Tests for SpilloverNetwork: hand-constructed small networks, since
this module is pure logic (matrix ops, DFS path search, diffing) that's
easiest to verify against known-by-hand expected results rather than
real market data.

Threshold-sensitive fixture values are derived from
SpilloverNetwork.EDGE_THRESHOLD rather than hardcoded, so these tests
keep passing if that constant is recalibrated again in the future (as
it already has been once, from 1% to 2% — see docs/notes.md).
"""

import pandas as pd
import pytest

from src.network import SpilloverNetwork

NODES = ["A", "B", "C", "D"]

THRESHOLD = SpilloverNetwork.EDGE_THRESHOLD
TINY = 0.0001  # "edge absent": safely below any plausible threshold
WEAK_ABOVE = THRESHOLD + 0.005  # "edge present but weak": just above threshold

# rows = target, columns = source. Each row sums to 1.0, matching a real
# FEVD matrix's convention. Designed so that:
#   - A->B and B->C are strong (0.40, 0.50): the indirect path A->B->C
#     compounds to 0.20, stronger than the weak direct edge A->C
#     (WEAK_ABOVE, just above threshold) -- assumes THRESHOLD stays
#     well under ~0.19, true for any threshold in the 0.5%-5% range
#     this project has considered.
#   - D exports ~0 to everyone (every D-> cell is TINY), so D has no
#     outgoing paths at all.
BASE_MATRIX = pd.DataFrame(
    [
        [1 - 0.100 - 0.049 - TINY, 0.100, 0.049, TINY],  # target A
        [0.400, 1 - 0.400 - 0.040 - TINY, 0.040, TINY],  # target B
        [WEAK_ABOVE, 0.500, 1 - WEAK_ABOVE - 0.500 - TINY, TINY],  # target C
        [TINY, TINY, 0.300, 1 - TINY - TINY - 0.300],  # target D
    ],
    index=NODES,
    columns=NODES,
)


def make_network(matrix: pd.DataFrame = BASE_MATRIX) -> SpilloverNetwork:
    return SpilloverNetwork(matrix, NODES)


def test_get_edge_weight_is_raw_lookup_including_diagonal():
    net = make_network()
    assert net.get_edge_weight("B", "A") == pytest.approx(0.100)
    assert net.get_edge_weight("A", "C") == pytest.approx(WEAK_ABOVE)
    # Diagonal is not special-cased by this method.
    assert net.get_edge_weight("A", "A") == pytest.approx(BASE_MATRIX.loc["A", "A"])


def test_out_degree_centrality_excludes_diagonal():
    net = make_network()
    # C's outgoing edges: C->A=0.049, C->B=0.040, C->D=0.300 (excludes C->C)
    assert net.out_degree_centrality("C") == pytest.approx(0.049 + 0.040 + 0.300)


def test_in_degree_centrality_excludes_diagonal():
    net = make_network()
    # A's incoming edges: B->A=0.100, C->A=0.049, D->A=TINY (excludes A->A)
    assert net.in_degree_centrality("A") == pytest.approx(0.100 + 0.049 + TINY)


def test_find_all_paths_finds_direct_and_indirect_routes():
    net = make_network()
    paths = net.find_all_paths("A", "C")
    assert ["A", "C"] in paths
    assert ["A", "B", "C"] in paths
    assert len(paths) == 2


def test_find_all_paths_excludes_edges_below_threshold():
    net = make_network()
    # D's outgoing edges are all TINY, below EDGE_THRESHOLD.
    assert net.find_all_paths("D", "A") == []


def test_strongest_path_picks_highest_product_not_fewest_hops():
    net = make_network()
    # Direct A->C = WEAK_ABOVE; indirect A->B->C = 0.40 * 0.50 = 0.20 (stronger).
    assert net.strongest_path("A", "C") == ["A", "B", "C"]


def test_strongest_path_returns_none_when_no_path_exists():
    net = make_network()
    assert net.strongest_path("D", "A") is None


def test_diff_detects_appeared_and_disappeared_edges():
    below_threshold = THRESHOLD / 2
    above_threshold = THRESHOLD + 0.03

    other_matrix = BASE_MATRIX.copy()
    other_matrix.loc["C", "A"] = below_threshold  # A->C drops below threshold
    other_matrix.loc["A", "D"] = above_threshold  # D->A rises above threshold

    net = make_network()
    other_net = make_network(other_matrix)

    diff = net.diff(other_net)

    assert ("D", "A") in diff.edges_appeared
    assert ("A", "C") in diff.edges_disappeared
    assert diff.delta.loc["C", "A"] == pytest.approx(below_threshold - WEAK_ABOVE)
    assert diff.delta.loc["A", "D"] == pytest.approx(above_threshold - TINY)


def test_diff_rejects_mismatched_node_order():
    net = make_network()
    other_net = SpilloverNetwork(
        BASE_MATRIX.loc[["B", "A", "C", "D"], ["B", "A", "C", "D"]],
        ["B", "A", "C", "D"],
    )
    with pytest.raises(ValueError):
        net.diff(other_net)
