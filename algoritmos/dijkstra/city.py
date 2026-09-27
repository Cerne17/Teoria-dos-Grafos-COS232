"""The fictional city from the paper sketch, drawn as a map.

Intersections are vertices (1-indexed in graphs/city.txt), lanes are arcs: a
two-way street is two arcs, drawn as two lanes (right-hand traffic). Weights are
street lengths in metres, centre to centre of the intersections on the 10 m grid.

    python city.py              # opens a window
    python city.py -o city.png  # saves to file
    python city.py -s 19 -t 4   # shortest path from 19 to 4
    python city.py -s 19        # shortest paths from 19 to everywhere
"""

import argparse
import math
from collections.abc import Collection, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

import viz
from adjacency_list_weighted import AdjacencyList
from dijkstra import Dijkstra

GRAPH_FILE = Path(__file__).parent / "graphs" / "city.txt"

# City limits in metres, origin at the sketch's top-left corner, y grows downward
WIDTH, HEIGHT = 130, 120

# Centre of each intersection, index = vertex (0-indexed)
POSITIONS: list[tuple[float, float]] = [
    (5, 5), (45, 5), (85, 5), (125, 5),                  # 1  2  3  4
    (5, 25), (25, 25), (45, 25),                         # 5  6  7
    (25, 45), (45, 45), (85, 45),                        # 8  9  10
    (5, 75), (25, 75), (85, 75), (105, 75), (125, 75),   # 11 12 13 14 15
    (5, 95), (45, 95), (85, 95),                         # 16 17 18
    (5, 115), (45, 115), (105, 115), (125, 115),         # 19 20 21 22
]  # fmt: skip

# City blocks as polygons, in metres; everything else is street
BLOCKS: list[list[tuple[float, float]]] = [
    [(10, 10), (40, 10), (40, 20), (10, 20)],
    [(50, 10), (80, 10), (80, 40), (50, 40)],
    [(90, 10), (120, 10), (120, 70), (90, 70)],
    [(10, 30), (20, 30), (20, 70), (10, 70)],
    [(30, 50), (80, 50), (80, 70), (30, 70)],
    [(10, 80), (80, 80), (80, 90), (10, 90)],
    [(90, 80), (100, 80), (100, 110), (50, 110), (50, 100), (90, 100)],  # the L
    [(110, 80), (120, 80), (120, 110), (110, 110)],
    [(10, 100), (40, 100), (40, 110), (10, 110)],
]

# The block between 6, 7, 8 and 9 is a roundabout: its four lanes are drawn as
# arcs of the circle through those intersections. Weights stay the grid's 20 m.
ROUNDABOUT_CENTER = (35.0, 35.0)
ROUNDABOUT_NODES = {5, 6, 7, 8}  # 0-indexed 6, 7, 8, 9
ROUNDABOUT_RADIUS = math.dist(ROUNDABOUT_CENTER, (25, 25))
ISLAND_RADIUS = 8.0

NODE_RADIUS = 3.2
LANE_OFFSET = 2.2  # distance from the street's centre line to each lane
COLORS = {"street": "#d4d4d4", "block": "#cfe3c3", "block_edge": "#7a9a6a", "path": "tab:purple"}


def load() -> AdjacencyList:
    return AdjacencyList.from_file(GRAPH_FILE)


def path_from_parents(parent: Sequence[int | None], target: int) -> list[int]:
    """Walks parent pointers back from target; [target] alone if it has no parent."""
    path = [target]
    while (p := parent[path[-1]]) is not None:
        path.append(p)
    return path[::-1]


def shortest_paths(graph: AdjacencyList, source: int) -> tuple[list[float], list[int | None]]:
    """Runs Dijkstra and returns (distances, parent)."""
    parent: list[int | None] = []

    def keep_parent(u, distances, discovered, current_parent):
        parent[:] = current_parent

    distances = Dijkstra(graph, source, keep_parent)
    return distances, parent


def shortest_path(graph: AdjacencyList, source: int, target: int) -> tuple[float, list[int]]:
    """(length, vertices from source to target); (inf, []) if target is unreachable."""
    distances, parent = shortest_paths(graph, source)
    if distances[target] == float("inf"):
        return distances[target], []
    return distances[target], path_from_parents(parent, target)


Point = tuple[float, float]
TRIM = NODE_RADIUS + 0.8  # arrows stop this far from the intersection centre


def _lane(u: int, v: int, two_way: bool) -> tuple[list[Point], Point]:
    """Points of the arrow u -> v, trimmed to the node circles, and where to label it."""
    if u in ROUNDABOUT_NODES and v in ROUNDABOUT_NODES:
        return _roundabout_lane(u, v)
    (x0, y0), (x1, y1) = POSITIONS[u], POSITIONS[v]
    length = math.hypot(x1 - x0, y1 - y0)
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    # With y pointing down, (-dy, dx) is the right-hand side of the direction
    off = LANE_OFFSET if two_way else 0.0
    ox, oy = -dy * off, dx * off
    return (
        [(x0 + dx * TRIM + ox, y0 + dy * TRIM + oy), (x1 - dx * TRIM + ox, y1 - dy * TRIM + oy)],
        ((x0 + x1) / 2, (y0 + y1) / 2),
    )


def _roundabout_lane(u: int, v: int) -> tuple[list[Point], Point]:
    """Quarter arc of the roundabout circle from u to v, the short way round."""
    cx, cy = ROUNDABOUT_CENTER
    r = ROUNDABOUT_RADIUS
    a0 = math.atan2(POSITIONS[u][1] - cy, POSITIONS[u][0] - cx)
    a1 = math.atan2(POSITIONS[v][1] - cy, POSITIONS[v][0] - cx)
    sweep = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi  # in (-pi, pi]
    trim = math.copysign(TRIM / r, sweep)
    start, end = a0 + trim, a0 + sweep - trim
    angles = [start + (end - start) * k / 24 for k in range(25)]
    mid = a0 + sweep / 2
    return (
        [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in angles],
        (cx + r * math.cos(mid), cy + r * math.sin(mid)),
    )


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
    """Draws the city map; arguments mean the same as in viz.draw (0-indexed).

    path is a route as a vertex sequence, drawn in purple over everything else.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    tree = set(tree_edges)
    path_arcs = set(zip(path, path[1:]))

    ax.add_patch(Rectangle((0, 0), WIDTH, HEIGHT, color=COLORS["street"], zorder=0))
    # Pave the ring before the blocks, so they cover whatever spills over
    ax.add_patch(
        plt.Circle(ROUNDABOUT_CENTER, ROUNDABOUT_RADIUS + 3, color=COLORS["street"], zorder=0.5)
    )
    for corners in BLOCKS:
        ax.add_patch(
            Polygon(corners, facecolor=COLORS["block"], edgecolor=COLORS["block_edge"], zorder=1)
        )
    ax.add_patch(
        plt.Circle(
            ROUNDABOUT_CENTER, ISLAND_RADIUS,
            facecolor=COLORS["block"], edgecolor=COLORS["block_edge"], zorder=1,
        )
    )  # fmt: skip

    labelled: set[frozenset[int]] = set()
    for u, v, w in graph.edges():
        points, (lx, ly) = _lane(u, v, two_way=graph.has_edge(v, u))
        if (u, v) in path_arcs:
            style = {"linewidth": 3.5, "color": COLORS["path"], "zorder": 3.5, "mutation_scale": 16}
        elif (u, v) in tree:
            style = {"linewidth": 2.4, "color": "tab:red", "zorder": 3, "mutation_scale": 12}
        else:
            style = {"linewidth": 1.1, "color": "#555555", "zorder": 2, "mutation_scale": 12}
        ax.add_patch(FancyArrowPatch(path=MplPath(points), arrowstyle="-|>", **style))
        # One weight label per street, even when it has two lanes
        if (pair := frozenset((u, v))) not in labelled:
            labelled.add(pair)
            ax.text(
                lx, ly, f"{w:g}",
                ha="center", va="center", fontsize=7, zorder=4,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
            )  # fmt: skip

    for v, (x, y) in enumerate(POSITIONS):
        color = viz._node_color(v, source, explored, frontier)
        on_path = v in path
        ax.add_patch(
            plt.Circle(
                (x, y), NODE_RADIUS, facecolor=color, zorder=5,
                edgecolor=COLORS["path"] if on_path else color, linewidth=3 if on_path else 0,
            )
        )  # fmt: skip
        ax.text(x, y, str(v + 1), ha="center", va="center", color="white",
                fontsize=8, fontweight="bold", zorder=6)  # fmt: skip
        if distances is not None:
            ax.text(
                x + NODE_RADIUS, y - NODE_RADIUS, viz._fmt(distances[v]),
                ha="left", va="bottom", fontsize=7, color="tab:green", fontweight="bold", zorder=6,
                bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.85},
            )  # fmt: skip

    ax.set_xlim(-2, WIDTH + 2)
    ax.set_ylim(HEIGHT + 2, -2)  # y grows downward, like the sketch
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title or f"City ({len(graph)} intersections, {graph.edge_count} lanes)")
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-s", "--source", type=int, help="1-indexed start")
    parser.add_argument("-t", "--target", type=int, help="1-indexed destination (needs -s)")
    parser.add_argument("-o", "--output", type=Path, help="save image instead of showing")
    args = parser.parse_args()
    if args.target is not None and args.source is None:
        parser.error("-t needs -s")

    graph = load()
    if args.source is None:
        ax = draw(graph)
    else:
        source = args.source - 1
        distances, parent = shortest_paths(graph, source)
        tree = [(p, v) for v, p in enumerate(parent) if p is not None]
        if args.target is None:
            title = f"Shortest paths from {args.source}"
            ax = draw(graph, title, source=source, distances=distances, tree_edges=tree)
        else:
            dist, path = shortest_path(graph, source, args.target - 1)
            route = " → ".join(str(v + 1) for v in path)
            title = (
                f"{args.source} → {args.target}: {dist:g} m\n{route}"
                if path
                else f"{args.target} is unreachable from {args.source}"
            )
            print(title.replace("\n", "  |  "))
            ax = draw(graph, title, source=source, distances=distances, path=path)
    if args.output:
        ax.figure.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        plt.show()
