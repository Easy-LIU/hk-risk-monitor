"""Tests for SpilloverNetwork: hand-constructed small networks, since
this module is pure logic (matrix ops, DFS path search, diffing) that's
easiest to verify against known-by-hand expected results rather than
real market data.
"""

import numpy as np
import pandas as pd
import pytest

from src.network import SpilloverNetwork

NODES = ["A", "B", "C", "D"]

# rows = target, columns = source. Each row sums to 1.0, matching a real
# FEVD matrix's convention. Designed so that:
#   - A->B and B->C are strong (0.40, 0.50): the indirect path A->B->C
#     compounds to 0.20, stronger than the weak direct edge A->C (0.02).
#   - D exports ~0 to everyone (every D-> cell is 0.001, below the 0.01
#     threshold), so D has no outgoing paths at all.
BASE_MATRIX = pd.DataFrame(
    [
        [0.850, 0.100, 0.049, 0.001],  # target A
        [0.400, 0.559, 0.040, 0.001],  # target B
        [0.020, 0.500, 0.479, 0.001],  # target C
        [0.001, 0.001, 0.300, 0.698],  # target D
    ],
    index=NODES,
    columns=NODES,
)


def make_network(matrix: pd.DataFrame = BASE_MATRIX) -> SpilloverNetwork:
    return SpilloverNetwork(matrix, NODES)


def test_get_edge_weight_is_raw_lookup_including_diagonal():
    net = make_network()
    assert net.get_edge_weight("B", "A") == pytest.approx(0.100)
    assert net.get_edge_weight("A", "C") == pytest.approx(0.020)
    # Diagonal is not special-cased by this method.
    assert net.get_edge_weight("A", "A") == pytest.approx(0.850)


def test_out_degree_centrality_excludes_diagonal():
    net = make_network()
    # C's outgoing edges: C->A=0.049, C->B=0.040, C->D=0.300 (excludes C->C=0.479)
    assert net.out_degree_centrality("C") == pytest.approx(0.049 + 0.040 + 0.300)


def test_in_degree_centrality_excludes_diagonal():
    net = make_network()
    # A's incoming edges: B->A=0.100, C->A=0.049, D->A=0.001 (excludes A->A=0.850)
    assert net.in_degree_centrality("A") == pytest.approx(0.100 + 0.049 + 0.001)


def test_find_all_paths_finds_direct_and_indirect_routes():
    net = make_network()
    paths = net.find_all_paths("A", "C")
    assert ["A", "C"] in paths
    assert ["A", "B", "C"] in paths
    assert len(paths) == 2


def test_find_all_paths_excludes_edges_below_threshold():
    net = make_network()
    # D's outgoing edges are all 0.001, below EDGE_THRESHOLD=0.01.
    assert net.find_all_paths("D", "A") == []


def test_strongest_path_picks_highest_product_not_fewest_hops():
    net = make_network()
    # Direct A->C = 0.02; indirect A->B->C = 0.40 * 0.50 = 0.20 (stronger).
    assert net.strongest_path("A", "C") == ["A", "B", "C"]


def test_strongest_path_returns_none_when_no_path_exists():
    net = make_network()
    assert net.strongest_path("D", "A") is None


def test_diff_detects_appeared_and_disappeared_edges():
    other_matrix = BASE_MATRIX.copy()
    other_matrix.loc["C", "A"] = 0.005  # A->C: 0.02 -> 0.005, drops below threshold
    other_matrix.loc["A", "D"] = 0.050  # D->A: 0.001 -> 0.05, rises above threshold

    net = make_network()
    other_net = make_network(other_matrix)

    diff = net.diff(other_net)

    assert ("D", "A") in diff.edges_appeared
    assert ("A", "C") in diff.edges_disappeared
    assert diff.delta.loc["C", "A"] == pytest.approx(0.005 - 0.020)
    assert diff.delta.loc["A", "D"] == pytest.approx(0.050 - 0.001)


def test_diff_rejects_mismatched_node_order():
    net = make_network()
    other_net = SpilloverNetwork(
        BASE_MATRIX.loc[["B", "A", "C", "D"], ["B", "A", "C", "D"]],
        ["B", "A", "C", "D"],
    )
    with pytest.raises(ValueError):
        net.diff(other_net)
