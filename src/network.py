"""Single-timestamp spillover transmission network, backed by a numpy
adjacency matrix. See docs/design.md sections 4, 6, and 7 for the
rationale behind the adjacency-matrix representation, the edge
threshold, and the multiplicative path-weight combination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class NetworkDiff:
    """Element-wise comparison between two SpilloverNetworks."""

    node_names: list[str]
    delta: pd.DataFrame  # other.matrix - self.matrix, same shape as either
    edges_appeared: list[tuple[str, str]]  # below threshold in self, above in other
    edges_disappeared: list[tuple[str, str]]  # above threshold in self, below in other


class SpilloverNetwork:
    """A single time point's transmission network. Edge (source, target)
    weight is source's contribution to target's forecast error variance."""

    # Calibrated on the real 2015-2026 rolling sample; see docs/design.md
    # section 6 for the empirical justification (chosen for signal
    # composition, not event count) and the caveat that this is a
    # sample-specific calibration, not a universal constant.
    EDGE_THRESHOLD = 0.02

    def __init__(self, fevd_matrix: pd.DataFrame, node_names: list[str]):
        self.node_names = list(node_names)
        self._index = {name: i for i, name in enumerate(self.node_names)}
        self.matrix = fevd_matrix.loc[self.node_names, self.node_names].to_numpy()

    def get_edge_weight(self, source: str, target: str) -> float:
        """Raw lookup, no thresholding: source's contribution to target's
        forecast error variance. Callers that care about the edge
        threshold apply it themselves; this method makes no judgment."""
        return self.matrix[self._index[target], self._index[source]]

    def out_degree_centrality(self, node: str) -> float:
        """Sum of node's outgoing edge weights to all other nodes — how
        much risk this node exports. Excludes the diagonal: a node's own
        idiosyncratic share is not risk transmitted to another node."""
        i = self._index[node]
        row_mask = np.arange(len(self.node_names)) != i
        # matrix[target, source]; node's outgoing weight to target t is matrix[t, i]
        return self.matrix[row_mask, i].sum()

    def in_degree_centrality(self, node: str) -> float:
        """Sum of node's incoming edge weights from all other nodes — how
        vulnerable this node is to external shocks. Excludes the
        diagonal: a node's own idiosyncratic share is not an external
        vulnerability."""
        i = self._index[node]
        col_mask = np.arange(len(self.node_names)) != i
        return self.matrix[i, col_mask].sum()

    def find_all_paths(self, source: str, target: str) -> list[list[str]]:
        """DFS over edges with weight > EDGE_THRESHOLD, visiting each node
        at most once. Self-loops are never part of a path."""
        paths: list[list[str]] = []

        def dfs(current: str, visited: set[str], path: list[str]):
            if current == target and len(path) > 1:
                paths.append(list(path))
                return
            for candidate in self.node_names:
                if candidate == current or candidate in visited:
                    continue
                if self.get_edge_weight(current, candidate) <= self.EDGE_THRESHOLD:
                    continue
                visited.add(candidate)
                path.append(candidate)
                dfs(candidate, visited, path)
                path.pop()
                visited.remove(candidate)

        dfs(source, {source}, [source])
        return paths

    def strongest_path(self, source: str, target: str) -> list[str] | None:
        """The path (among find_all_paths' results) that maximizes the
        product of edge weights along it. None if no path exists."""
        paths = self.find_all_paths(source, target)
        if not paths:
            return None

        def path_weight(path: list[str]) -> float:
            weight = 1.0
            for a, b in zip(path[:-1], path[1:]):
                weight *= self.get_edge_weight(a, b)
            return weight

        return max(paths, key=path_weight)

    def diff(self, other: "SpilloverNetwork") -> NetworkDiff:
        """Element-wise comparison against another network over the same
        nodes. Edge appearance/disappearance is judged by EDGE_THRESHOLD
        crossing, not by the raw delta's sign or magnitude."""
        if self.node_names != other.node_names:
            raise ValueError("diff() requires both networks to share the same node order")

        delta = pd.DataFrame(
            other.matrix - self.matrix, index=self.node_names, columns=self.node_names
        )

        appeared = []
        disappeared = []
        for source in self.node_names:
            for target in self.node_names:
                if source == target:
                    continue
                before = self.get_edge_weight(source, target)
                after = other.get_edge_weight(source, target)
                if before <= self.EDGE_THRESHOLD < after:
                    appeared.append((source, target))
                elif after <= self.EDGE_THRESHOLD < before:
                    disappeared.append((source, target))

        return NetworkDiff(
            node_names=self.node_names,
            delta=delta,
            edges_appeared=appeared,
            edges_disappeared=disappeared,
        )
