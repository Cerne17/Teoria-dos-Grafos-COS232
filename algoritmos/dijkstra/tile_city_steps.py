"""Dijkstra step by step on a tile city.

    python tile_city_steps.py -s 5,0                   # sample map, interactive window
    python tile_city_steps.py -s 5,0 -t 9,13           # also prints and draws the route
    python tile_city_steps.py my.txt -s 3 -o run.png   # 12 evenly spaced steps in one image

Vertices: number (see tile_city.py --labels) or row,col. Window keys:
right/left = next/previous step, up/down = 10 steps, home/end = first/last.
Colours: orange = picked, green = final, yellow = discovered; red = parent tree.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

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
) -> None:
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
        ax=ax,
    )


def evenly_spaced(count: int, panels: int) -> list[int]:
    """panels step indices from first to last, both included."""
    if count <= panels:
        return list(range(count))
    return sorted({round(k * (count - 1) / (panels - 1)) for k in range(panels)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("map", nargs="?", type=Path, default=tile_city.DEFAULT_MAP)
    parser.add_argument("-s", "--source", required=True, help="start: vertex number or row,col")
    parser.add_argument("-t", "--target", help="destination: vertex number or row,col")
    parser.add_argument("-o", "--output", type=Path, help="save some steps as a grid image")
    parser.add_argument("--panels", type=int, default=12, help="steps drawn with -o (default 12)")
    args = parser.parse_args()

    grid = tiles.load_map(args.map)
    city = tiles.to_graph(grid)
    try:
        source = tile_city.parse_vertex(args.source, city)
        target = None if args.target is None else tile_city.parse_vertex(args.target, city)
    except ValueError as e:
        parser.error(str(e))

    steps = record_steps(city.graph, source)
    for i, s in enumerate(steps, 1):
        print(f"step {i}: pick {city.label(s.picked)}  d = {s.distances[s.picked]:g} m  "
              f"frontier = {len(s.frontier)}")  # fmt: skip
    unreached = len(city.graph) - len(steps)
    print(f"\n{len(steps)} road tiles reached, {unreached} unreachable from {city.label(source)}")

    if target is not None:
        path = route(steps[-1], target)
        if path:
            print(f"route: {' -> '.join(str(v + 1) for v in path)}  "
                  f"({steps[-1].distances[target]:g} m)")  # fmt: skip
        else:
            print(f"{city.label(target)} is unreachable")

    def draw_panel(i: int, ax: plt.Axes) -> None:
        draw_step(grid, city, steps, i, ax, target)

    suptitle = f"Dijkstra on {args.map.stem} from {city.label(source)}"
    rows, cols = len(grid), len(grid[0])
    panel = (8, 8 * rows / cols + 0.5)
    if args.output:
        indices = evenly_spaced(len(steps), args.panels)
        save_grid(indices, draw_panel, suptitle, args.output, panel_size=panel, cols=3)
    else:
        browse(len(steps), draw_panel, suptitle, figsize=(12, 12 * rows / cols + 0.8))
