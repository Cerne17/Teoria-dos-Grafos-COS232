"""Dijkstra step by step on a tile city.

    python tile_city_steps.py city/maps/sample2.txt -s "Joe's house" -t "shopping mall"
    python tile_city_steps.py -s 5,0 -t 9,13           # between road cells
    python tile_city_steps.py my.txt -s P3 -o run.png  # 12 evenly spaced steps in one image
    python tile_city_steps.py city/maps/sample2.txt -s "Joe's house" -t P4 --gif run.gif

Endpoints: place name, place number (P12, see tile_city.py --places), row,col
(road tile or any cell of a place) or road vertex number. Window keys:
right/left = next/previous step, up/down = 10 steps, home/end = first/last.
Colours: orange = picked, green = final, yellow = discovered; red = parent tree.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import places as places_mod
import tile_city
import tiles
from city_steps import browse, route, save_grid
from steps import Step, record_steps
from tiles import CityGraph


def draw_step(
    grid: list[list[int]],
    city: CityGraph,
    steps: list[Step],
    i: int,
    ax: plt.Axes,
    target: int | None = None,
    places: list | None = None,
    disabled: list | None = None,
) -> None:
    """places: start/destination places to outline and name; disabled: places
    no driveway reaches, greyed out (see places.py)."""
    s = steps[i]
    path = route(s, target) if target is not None and target in s.explored else []
    tile_city.draw(
        grid,
        city,
        title=f"step {i + 1}/{len(steps)}: pick {city.label(s.picked)}, "
        f"{s.distances[s.picked]:g} m",
        source=s.picked,
        distances=s.distances,
        tree_edges=s.tree_edges,
        explored=s.explored,
        frontier=s.frontier,
        path=path,
        places=places or (),
        disabled=disabled or (),
        ax=ax,
    )


def short_label(city: CityGraph, v: int) -> str:
    if v in city.names:
        return city.names[v]
    r, c = city.cells[v]
    return f"{v + 1} ({r},{c})"


def export_gif(
    path: Path,
    grid: list[list[int]],
    city: CityGraph,
    steps: list[Step],
    source: int,
    target: int | None = None,
    fps: float = 4.0,
    title: str = "",
    places: list | None = None,
    disabled: list | None = None,
) -> Path:
    """The run as an animated GIF, one frame per step (see gif.py)."""
    import gif

    rows, cols = len(grid), len(grid[0])
    return gif.export(
        path,
        len(steps),
        lambda i, ax: draw_step(grid, city, steps, i, ax, target, places, disabled),
        lambda i: gif.describe(steps, i, source, lambda v: short_label(city, v), " m",
                               target, lambda s: route(s, target)),  # fmt: skip
        figsize=(14, max(6.5, 10 * rows / cols + 1)),
        fps=fps,
        title=title,
    )


def evenly_spaced(count: int, panels: int) -> list[int]:
    """panels step indices from first to last, both included."""
    if count <= panels:
        return list(range(count))
    return sorted({round(k * (count - 1) / (panels - 1)) for k in range(panels)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("map", nargs="?", type=Path, default=tile_city.DEFAULT_MAP)
    parser.add_argument("-s", "--source", required=True, help="start: place name, P-number, row,col or vertex")
    parser.add_argument("-t", "--target", help="destination, same forms as -s")
    parser.add_argument("-o", "--output", type=Path, help="save some steps as a grid image")
    parser.add_argument("--panels", type=int, default=12, help="steps drawn with -o (default 12)")
    parser.add_argument("--gif", type=Path, help="save the run as an animated GIF")
    parser.add_argument("--fps", type=float, default=4.0, help="GIF steps per second")
    args = parser.parse_args()

    grid = tiles.load_map(args.map)
    roads = tiles.to_graph(grid)
    all_places = places_mod.load_for(args.map, grid)
    try:
        src = places_mod.resolve(args.source, roads, all_places)
        dst = None if args.target is None else places_mod.resolve(args.target, roads, all_places)
    except ValueError as e:
        parser.error(str(e))
    # Adds vertices for start/destination places (driveways only out / only in)
    city, source, target = places_mod.with_places(roads, src, dst, grid)
    endpoints = [p for p in (src, dst) if isinstance(p, places_mod.Place)]
    disabled = [p for p in all_places if not p.accessible]

    steps = record_steps(city.graph, source)
    for i, s in enumerate(steps, 1):
        print(f"step {i}: pick {city.label(s.picked)}  d = {s.distances[s.picked]:g} m  "
              f"frontier = {len(s.frontier)}")  # fmt: skip
    unreached = len(city.graph) - len(steps)
    print(f"\n{len(steps)} vertices reached, {unreached} unreachable from {city.label(source)}")

    if target is not None:
        path = route(steps[-1], target)
        if path:
            print(f"route: {' -> '.join(short_label(city, v) for v in path)}  "
                  f"({steps[-1].distances[target]:g} m)")  # fmt: skip
        else:
            print(f"{city.label(target)} is unreachable")

    def draw_panel(i: int, ax: plt.Axes) -> None:
        draw_step(grid, city, steps, i, ax, target, endpoints, disabled)

    suptitle = f"Dijkstra on {args.map.stem} from {city.label(source)}"
    rows, cols = len(grid), len(grid[0])
    panel = (8, 8 * rows / cols + 0.5)
    if args.gif:
        export_gif(args.gif, grid, city, steps, source, target, args.fps, suptitle, endpoints,
                   disabled)  # fmt: skip
        print(f"saved {args.gif}")
    elif args.output:
        indices = evenly_spaced(len(steps), args.panels)
        save_grid(indices, draw_panel, suptitle, args.output, panel_size=panel, cols=3)
    else:
        browse(len(steps), draw_panel, suptitle, figsize=(12, 12 * rows / cols + 0.8))
