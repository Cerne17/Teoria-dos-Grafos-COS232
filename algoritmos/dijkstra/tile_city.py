"""Draws a tile city (see tiles.py) and shortest paths on it.

Every road tile is a vertex. Routes go between places (houses, malls, parks...,
see places.py) or road tiles. An endpoint is a place name ("Joe's house"), a
place number (P12), a cell as row,col (0-indexed, row 0 at the top: a road
tile or any cell of a place) or a road vertex number (see --labels).

    python tile_city.py city/maps/sample2.txt --places        # list and number places
    python tile_city.py city/maps/sample2.txt -s "Joe's house" -t "shopping mall"
    python tile_city.py -s 0,5 -t 9,13           # between two road cells
    python tile_city.py -s P3                    # routes from a place to everywhere
    python tile_city.py --labels                 # road vertex numbers
    python tile_city.py -s P3 -t P9 -o r.png     # save instead of showing
"""

import argparse
from collections.abc import Collection, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

import places as places_mod
import tiles
import viz
from city import COLORS, shortest_path, shortest_paths
from places import Place
from tiles import CityGraph

DEFAULT_MAP = tiles.MAPS / "sample.txt"


def parse_vertex(text: str, city: CityGraph) -> int:
    """'row,col' -> the road tile's vertex; 'n' -> vertex n (1-indexed)."""
    if "," in text:
        r, c = (int(x) for x in text.split(","))
        if (r, c) not in city.vertex_at:
            raise ValueError(f"cell ({r}, {c}) is not a road tile")
        return city.vertex_at[(r, c)]
    v = int(text) - 1
    if not 0 <= v < len(city.graph):
        raise ValueError(f"vertex {text} out of range 1..{len(city.graph)}")
    return v


# How road sockets are marked: (marker, colour) per socket kind and side.
# Triangles point the way traffic flows: into the tile (entrance) or out (exit).
_INWARD = {"N": "v", "S": "^", "E": "<", "W": ">"}
_OUTWARD = {"N": "^", "S": "v", "E": ">", "W": "<"}
SOCKET_COLORS = {"R": "#29a8e0", "I": "#2ecc40", "O": "#ff4136"}


def draw_sockets(ax: plt.Axes, grid: Sequence[Sequence[int]], size: float = 6) -> None:
    """Marks every road side: dot = two-way, green triangle in = entrance, red out = exit."""
    groups: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for r, row in enumerate(grid):
        for c, t in enumerate(row):
            for side, sock in zip(tiles.SIDES, tiles.TILES[t].sockets):
                if sock not in SOCKET_COLORS:
                    continue
                marker = "o" if sock == "R" else (_INWARD if sock == "I" else _OUTWARD)[side]
                dr, dc = tiles.DELTA[side]
                xs, ys = groups.setdefault((marker, sock), ([], []))
                xs.append(c + 0.5 + dc * 0.36)
                ys.append(r + 0.5 + dr * 0.36)
    for (marker, sock), (xs, ys) in groups.items():
        ax.plot(xs, ys, linestyle="none", marker=marker, markersize=size,
                color=SOCKET_COLORS[sock], markeredgecolor="black", markeredgewidth=0.6,
                zorder=8)  # fmt: skip


def socket_legend(ax: plt.Axes, **kwargs) -> None:
    from matplotlib.lines import Line2D

    style = {"linestyle": "none", "markeredgecolor": "black", "markersize": 8}
    handles = [
        Line2D([], [], marker="o", color=SOCKET_COLORS["R"], label="two-way", **style),
        Line2D([], [], marker="v", color=SOCKET_COLORS["I"], label="entrance (points in)", **style),
        Line2D([], [], marker="^", color=SOCKET_COLORS["O"], label="exit (points out)", **style),
    ]
    ax.legend(handles=handles, ncol=3, frameon=False, fontsize=8, **kwargs)


def _centre(city: CityGraph, v: int) -> tuple[float, float]:
    r, c = city.cells[v]
    return c + 0.5, r + 0.5


def _arc(ax: plt.Axes, city: CityGraph, u: int, v: int, **style) -> None:
    ax.add_patch(
        FancyArrowPatch(
            _centre(city, u), _centre(city, v), arrowstyle="-|>", shrinkA=6, shrinkB=6, **style
        )
    )


def draw(
    grid: Sequence[Sequence[int]],
    city: CityGraph | None = None,
    title: str = "",
    source: int | None = None,
    distances: Sequence[float] | None = None,
    tree_edges: Sequence[tuple[int, int]] = (),
    explored: Collection[int] = (),
    frontier: Collection[int] = (),
    path: Sequence[int] = (),
    labels: bool = False,
    bad_sides: Sequence[tuple[int, int, str]] = (),
    places: Sequence[Place] = (),
    place_numbers: Sequence[Place] = (),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draws the tiles, then the Dijkstra state on top (vertices 0-indexed).

    Only vertices that Dijkstra has reached get a marker; the rest of the road
    is just the tile art. bad_sides marks sockets that don't fit in red.
    """
    rows, cols = len(grid), len(grid[0])
    if ax is None:
        _, ax = plt.subplots(figsize=(min(16, cols * 0.8), min(12, rows * 0.8) + 0.6))
    if city is None:
        city = tiles.to_graph(grid)

    ax.imshow(tiles.render(grid), extent=(0, cols, rows, 0), interpolation="nearest")

    for u, v in tree_edges:
        _arc(ax, city, u, v, color="tab:red", linewidth=1.8, mutation_scale=9, zorder=3)
    for u, v in zip(path, path[1:]):
        _arc(ax, city, u, v, color=COLORS["path"], linewidth=3.5, mutation_scale=12, zorder=4)

    reached = set(explored) | set(frontier) | ({source} if source is not None else set())
    if distances is not None:
        reached |= {v for v, d in enumerate(distances) if d != float("inf")}
    on_path = set(path)
    for v in reached | on_path:
        x, y = _centre(city, v)
        color = viz._node_color(v, source, explored, frontier)
        if v not in explored and v not in frontier and v != source:
            color = "tab:green"  # finished run: every reached vertex is final
        ax.add_patch(
            plt.Circle(
                (x, y), 0.27, facecolor=color, alpha=0.92, zorder=5,
                edgecolor=COLORS["path"] if v in on_path else "white",
                linewidth=2.2 if v in on_path else 0.6,
            )
        )  # fmt: skip
        if distances is not None:
            ax.text(x, y, viz._fmt(distances[v]), ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold", zorder=6)  # fmt: skip

    if labels:
        for v, (r, c) in enumerate(city.cells):
            if v not in reached:
                ax.text(c + 0.5, r + 0.5, str(v + 1), ha="center", va="center", fontsize=6,
                        color="black", zorder=6,
                        bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8})  # fmt: skip

    for r, c, side in bad_sides:
        dr, dc = tiles.DELTA[side]
        x, y = c + 0.5 + dc * 0.45, r + 0.5 + dr * 0.45
        ax.plot(x, y, marker="X", color="red", markersize=9, markeredgecolor="white", zorder=7)

    for p in places:  # outlined, with their name
        for r, c in p.cells:
            ax.add_patch(Rectangle((c + 0.04, r + 0.04), 0.92, 0.92, fill=False,
                                   edgecolor=COLORS["path"], linewidth=2.5, zorder=4))  # fmt: skip
        r, c = p.center
        ax.text(c + 0.5, r + 0.02, p.name, ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="white", zorder=9,
                bbox={"boxstyle": "round,pad=0.2", "fc": COLORS["path"], "ec": "none"})  # fmt: skip
    for p in place_numbers:  # P-numbers, as in --places
        r, c = p.center
        ax.text(c + 0.5, r + 0.5, f"P{p.id + 1}", ha="center", va="center", fontsize=6,
                zorder=8, bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none",
                                "alpha": 0.85})  # fmt: skip

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    g = city.graph
    ax.set_title(title or f"Tile city ({rows}x{cols}, {len(g)} road tiles, {g.edge_count} arcs)")
    return ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=__doc__.split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("map", nargs="?", type=Path, default=DEFAULT_MAP, help="tile map file")
    parser.add_argument("-s", "--source", help="start: place name, P-number, row,col or vertex")
    parser.add_argument("-t", "--target", help="destination, same forms as -s (needs -s)")
    parser.add_argument("--places", action="store_true", help="list places and number them on the map")
    parser.add_argument("--labels", action="store_true", help="write vertex numbers on road tiles")
    parser.add_argument("-o", "--output", type=Path, help="save image instead of showing")
    args = parser.parse_args()
    if args.target is not None and args.source is None:
        parser.error("-t needs -s")

    grid = tiles.load_map(args.map)
    city = tiles.to_graph(grid)
    all_places = places_mod.load_for(args.map, grid)
    if bad := tiles.mismatches(grid):
        print(f"warning: {len(bad)} sides don't fit their neighbour (marked with X)")
    if args.places:
        print(places_mod.listing(all_places))

    kwargs = {"labels": args.labels, "bad_sides": bad,
              "place_numbers": all_places if args.places else ()}  # fmt: skip
    if args.source is None:
        kwargs["city"] = city
    else:
        try:
            src = places_mod.resolve(args.source, city, all_places)
            dst = None if args.target is None else places_mod.resolve(args.target, city, all_places)
        except ValueError as e:
            parser.error(str(e))
        routed, source, target = places_mod.with_places(city, src, dst)
        kwargs["city"] = routed
        kwargs["places"] = [p for p in (src, dst) if isinstance(p, Place)]
        distances, parent = shortest_paths(routed.graph, source)
        kwargs |= {"source": source, "distances": distances}
        if target is None:
            kwargs["title"] = f"Shortest paths from {routed.label(source)}"
            kwargs["tree_edges"] = [(p, v) for v, p in enumerate(parent) if p is not None]
        else:
            dist, path = shortest_path(routed.graph, source, target)
            if path:
                kwargs["title"] = f"{routed.label(source)} → {routed.label(target)}: {dist:g} m"
                kwargs["path"] = path
                # Mark only the route's tiles, so the map stays readable
                on_path = set(path)
                kwargs["distances"] = [d if v in on_path else float("inf") for v, d in enumerate(distances)]
            else:
                kwargs["title"] = f"{routed.label(target)} is unreachable from {routed.label(source)}"
            print(kwargs["title"])

    ax = draw(grid, **kwargs)
    if args.output:
        ax.figure.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        plt.show()
