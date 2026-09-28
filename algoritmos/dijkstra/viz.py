"""Draws an AdjacencyList with networkx + matplotlib.

Vertices are shown 1-indexed, matching the graph files. Usage:

    python viz.py graphs/graph_02.txt             # opens a window
    python viz.py graphs/graph_02.txt -o g02.png  # saves to file
    python viz.py graphs/graph_03.txt -s 1 -t 5 --gif g03.gif   # Dijkstra run as a GIF
"""

import argparse
from collections.abc import Collection, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

from adjacency_list_weighted import AdjacencyList

# Curve arcs so that u -> v and v -> u are drawn as two distinct arrows
CONNECTION_STYLE = "arc3,rad=0.15"


def to_networkx(graph: AdjacencyList) -> nx.DiGraph:
    """Converts to a networkx DiGraph with 1-indexed nodes and 'weight' on each arc. O(V + E)"""
    g = nx.DiGraph()
    g.add_nodes_from(v + 1 for v in graph)
    g.add_weighted_edges_from((u + 1, v + 1, w) for u, v, w in graph.edges())
    return g


def _fmt(x: float) -> str:
    return "∞" if x == float("inf") else f"{x:g}"


def _node_color(
    v: int, source: int | None, explored: Collection[int], frontier: Collection[int]
) -> str:
    if v == source:
        return "tab:orange"
    if v in explored:
        return "tab:green"
    if v in frontier:
        return "goldenrod"
    return "tab:blue"


def draw(
    graph: AdjacencyList,
    title: str = "",
    source: int | None = None,
    distances: Sequence[float] | None = None,
    tree_edges: Sequence[tuple[int, int]] = (),
    explored: Collection[int] = (),
    frontier: Collection[int] = (),
    path: Sequence[int] = (),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draws the graph on ax (a new figure if None).

    All vertex arguments are 0-indexed, like AdjacencyList:
      source      highlighted in a different colour
      distances   written outside each vertex (e.g. Dijkstra's output)
      tree_edges  arcs (u, v) highlighted, e.g. the shortest-path tree
      explored    vertices whose distance is final (green)
      frontier    discovered but not yet final (yellow)
      path        a route as a vertex sequence, drawn in purple over the tree
    """
    g = to_networkx(graph)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    # Circle keeps small graphs readable: no arc crosses through a vertex
    pos = nx.circular_layout(sorted(g.nodes))
    tree = {(u + 1, v + 1) for u, v in tree_edges}
    node_colors = [_node_color(v - 1, source, explored, frontier) for v in g.nodes]
    route = {(u + 1, v + 1) for u, v in zip(path, path[1:])}
    edge_colors = ["tab:purple" if e in route else "tab:red" if e in tree else "gray"
                   for e in g.edges]  # fmt: skip
    edge_widths = [4 if e in route else 2.5 if e in tree else 1.2 for e in g.edges]

    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=600)
    nx.draw_networkx_labels(g, pos, ax=ax, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        arrowsize=18,
        node_size=600,
        connectionstyle=CONNECTION_STYLE,
    )
    nx.draw_networkx_edge_labels(
        g,
        pos,
        ax=ax,
        edge_labels={(u, v): _fmt(w) for u, v, w in g.edges(data="weight")},
        connectionstyle=CONNECTION_STYLE,
        font_size=9,
    )

    if distances is not None:
        for v, (x, y) in pos.items():
            ax.annotate(
                f"d={_fmt(distances[v - 1])}",
                (x, y),
                # Push the label radially outward, away from the arcs
                xytext=(30 * x, 30 * y),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="tab:green",
            )

    # Room around the circle for the distance labels
    ax.margins(0.2)
    ax.set_title(title or repr(graph))
    ax.set_axis_off()
    return ax


def show(graph: AdjacencyList, output: str | Path | None = None, **kwargs) -> None:
    """Draws and either saves to output or opens a window."""
    ax = draw(graph, **kwargs)
    if output:
        ax.figure.savefig(output, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(ax.figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="graph file: n, then 'u v w' lines")
    parser.add_argument("-o", "--output", type=Path, help="save image instead of showing")
    parser.add_argument("-s", "--source", type=int, default=1, help="1-indexed start (for --gif)")
    parser.add_argument("-t", "--target", type=int, help="1-indexed destination (for --gif)")
    parser.add_argument("--gif", type=Path, help="save a Dijkstra run from -s as an animated GIF")
    parser.add_argument("--fps", type=float, default=1.0, help="GIF steps per second")
    args = parser.parse_args()

    graph = AdjacencyList.from_file(args.path)
    if args.gif:
        import steps

        steps.export_gif(args.gif, graph, args.source - 1,
                         None if args.target is None else args.target - 1,
                         fps=args.fps, title=f"Dijkstra on {args.path.stem} from {args.source}")  # fmt: skip
        print(f"saved {args.gif}")
    else:
        show(graph, args.output, title=args.path.stem)
