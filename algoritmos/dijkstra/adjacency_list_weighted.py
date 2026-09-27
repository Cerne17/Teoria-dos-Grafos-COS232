from collections.abc import Iterable, Iterator
from pathlib import Path

Edge = tuple[int, int, float]


class AdjacencyList:
    """Weighted directed graph over vertices 0..n-1.

    Each vertex keeps a dict {target: weight} of its outgoing arcs, so
    lookups, insertions and removals of a single arc are O(1) on average,
    while iterating a vertex's neighbours stays O(out_degree). Parallel arcs
    are not allowed: inserting an existing arc overwrites its weight.
    """

    def __init__(self, graph_size: int, edges: Iterable[Edge] = ()) -> None:
        if graph_size < 0:
            raise ValueError(f"graph_size can not be < 0. graph_size = {graph_size}")
        self._grph_size = graph_size
        self._adj_list: list[dict[int, float]] = [{} for _ in range(graph_size)]
        self._in_degree = [0] * graph_size
        self._edge_count = 0
        for source, target, weight in edges:
            self.add_edge(source, target, weight)

    @classmethod
    def from_file(cls, path: str | Path, one_indexed: bool = True) -> "AdjacencyList":
        """Reads the course format: first line n, then one 'u v w' arc per line.

        With one_indexed, file vertices 1..n are mapped to 0..n-1. O(n + m)
        """
        with open(path) as f:
            graph = cls(int(f.readline()))
            offset = 1 if one_indexed else 0
            for line in f:
                if not line.strip():
                    continue
                u, v, w = line.split()
                graph.add_edge(int(u) - offset, int(v) - offset, float(w))
        return graph

    # ---- size / membership -------------------------------------------------

    def __len__(self) -> int:
        """|V|. O(1)"""
        return self._grph_size

    @property
    def edge_count(self) -> int:
        """|E|. O(1)"""
        return self._edge_count

    def __contains__(self, v: int) -> bool:
        """O(1)"""
        return 0 <= v < self._grph_size

    def __iter__(self) -> Iterator[int]:
        """Iterates the vertices. O(V)"""
        return iter(range(self._grph_size))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(|V| = {len(self)}, |E| = {self.edge_count})"

    # ---- arcs --------------------------------------------------------------

    def add_edge(self, source: int, target: int, weight: float) -> bool:
        """Inserts source -> target with the given weight. O(1) average

        Returns True if the arc is new, False if it already existed (its
        weight is overwritten).
        """
        self._check(source)
        self._check(target)
        out = self._adj_list[source]
        is_new = target not in out
        if is_new:
            self._in_degree[target] += 1
            self._edge_count += 1
        out[target] = weight
        return is_new

    def remove_edge(self, source: int, target: int) -> bool:
        """Removes source -> target. Returns False if it did not exist. O(1) average"""
        self._check(source)
        self._check(target)
        if self._adj_list[source].pop(target, None) is None:
            return False
        self._in_degree[target] -= 1
        self._edge_count -= 1
        return True

    def has_edge(self, source: int, target: int) -> bool:
        """O(1) average"""
        self._check(source)
        return target in self._adj_list[source]

    def weight(self, source: int, target: int) -> float:
        """Weight of source -> target. Raises KeyError if the arc does not exist. O(1) average"""
        self._check(source)
        try:
            return self._adj_list[source][target]
        except KeyError:
            raise KeyError(f"no arc {source} -> {target}") from None

    def neighbors(self, v: int) -> Iterator[tuple[int, float]]:
        """Outgoing (target, weight) pairs of v. O(out_degree(v))"""
        self._check(v)
        return iter(self._adj_list[v].items())

    def __getitem__(self, v: int) -> dict[int, float]:
        """Read-only view of v's outgoing arcs as {target: weight}. O(1)"""
        self._check(v)
        return self._adj_list[v]

    def edges(self) -> Iterator[Edge]:
        """Every arc as (source, target, weight). O(V + E)"""
        for source, out in enumerate(self._adj_list):
            for target, weight in out.items():
                yield source, target, weight

    # ---- degrees -----------------------------------------------------------

    def out_degree(self, v: int) -> int:
        """O(1)"""
        self._check(v)
        return len(self._adj_list[v])

    def in_degree(self, v: int) -> int:
        """O(1), kept up to date on every insertion/removal"""
        self._check(v)
        return self._in_degree[v]

    # ---- derived graphs ----------------------------------------------------

    def reverse(self) -> "AdjacencyList":
        """Transpose graph (every arc flipped). Useful for Kosaraju. O(V + E)"""
        return AdjacencyList(self._grph_size, ((v, u, w) for u, v, w in self.edges()))

    def has_negative_weight(self) -> bool:
        """True if any arc is negative, i.e. Dijkstra is not safe to use. O(V + E)"""
        return any(w < 0 for _, _, w in self.edges())

    def _check(self, v: int) -> None:
        if not 0 <= v < self._grph_size:
            raise IndexError(f"vertex {v} out of range [0, {self._grph_size})")


if __name__ == "__main__":
    g = AdjacencyList(4, [(0, 1, 4.0), (0, 2, 1.0), (2, 1, 2.0), (1, 3, 5.0)])
    print(g)
    for u in g:
        print(u, "->", list(g.neighbors(u)))
    print("reverse:", list(g.reverse().edges()))
