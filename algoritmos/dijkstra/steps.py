"""Draws Dijkstra step by step, one panel per vertex picked.

    python steps.py graphs/graph_02.txt                  # source 1, opens a window
    python steps.py graphs/graph_03.txt -s 3 -o g03.png  # source 3, saves to file
    python steps.py graphs/graph_03.txt --gif g03.gif    # animated, with a details panel
    python steps.py graphs/graph_03.txt -s 1 -t 5 --gif g03.gif   # + route to 5 in purple

Colours: orange = vertex picked this step, green = explored (final),
yellow = discovered (tentative), blue = unknown. Red arcs = parent tree so far.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

import viz
from adjacency_list_weighted import AdjacencyList
from dijkstra import Dijkstra


@dataclass
class Step:
    picked: int
    distances: list[float]
    explored: set[int]
    frontier: list[int]
    parent: list[int | None]

    @property
    def tree_edges(self) -> list[tuple[int, int]]:
        return [(p, v) for v, p in enumerate(self.parent) if p is not None]


def record_steps(graph: AdjacencyList, source: int) -> list[Step]:
    """Runs Dijkstra and snapshots its state after every vertex is picked."""
    steps: list[Step] = []
    explored: set[int] = set()

    def on_step(u, distances, discovered, parent):
        explored.add(u)
        # Copy: Dijkstra keeps mutating these lists after the callback returns
        steps.append(Step(u, distances[:], set(explored), discovered[:], parent[:]))

    Dijkstra(graph, source, on_step)
    return steps


def print_steps(steps: list[Step]) -> None:
    """Text table of the run, 1-indexed like the graph files."""
    for i, s in enumerate(steps, 1):
        dist = " ".join(viz._fmt(d).rjust(3) for d in s.distances)
        frontier = sorted(v + 1 for v in s.frontier)
        print(f"step {i}: pick {s.picked + 1}  d = [{dist} ]  frontier = {frontier}")


def route(step: Step, target: int) -> list[int]:
    """Route to target from the parent pointers at this step ([] if not reached)."""
    if step.distances[target] == float("inf"):
        return []
    path = [target]
    while (p := step.parent[path[-1]]) is not None:
        path.append(p)
    return path[::-1]


def draw_step(graph: AdjacencyList, steps: list[Step], i: int, ax: plt.Axes,
              target: int | None = None) -> None:  # fmt: skip
    """Step i with networkx; the route to target appears once target is final."""
    s = steps[i]
    path = route(s, target) if target is not None and target in s.explored else []
    viz.draw(graph, title=f"step {i + 1}: pick {s.picked + 1} (d = {viz._fmt(s.distances[s.picked])})",
             source=s.picked,
             distances=s.distances, tree_edges=s.tree_edges, explored=s.explored,
             frontier=s.frontier, path=path, ax=ax)  # fmt: skip


def export_gif(
    path: Path,
    graph: AdjacencyList,
    source: int,
    target: int | None = None,
    fps: float = 1.0,
    title: str = "",
) -> Path:
    """Dijkstra from source as an animated GIF in the networkx view (see gif.py)."""
    import gif

    steps = record_steps(graph, source)
    return gif.export(
        path,
        len(steps),
        lambda i, ax: draw_step(graph, steps, i, ax, target),
        lambda i: gif.describe(steps, i, source, target=target,
                               route=lambda s: route(s, target)),  # fmt: skip
        figsize=(11, 6.5),
        fps=fps,
        title=title,
    )


def draw_steps(graph: AdjacencyList, steps: list[Step], cols: int = 3,
               target: int | None = None) -> plt.Figure:  # fmt: skip
    rows = math.ceil(len(steps) / cols)
    # Constrained layout leaves room for the suptitle both on screen and on save
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False, layout="constrained"
    )
    for i, ax in enumerate(axes.flat[: len(steps)]):
        draw_step(graph, steps, i, ax, target)
    for ax in axes.flat[len(steps):]:
        ax.set_axis_off()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="graph file: n, then 'u v w' lines")
    parser.add_argument("-s", "--source", type=int, default=1, help="1-indexed source")
    parser.add_argument("-t", "--target", type=int, help="1-indexed destination: route in purple")
    parser.add_argument("-o", "--output", type=Path, help="save image instead of showing")
    parser.add_argument("--gif", type=Path, help="save the run as an animated GIF")
    parser.add_argument("--fps", type=float, default=1.0, help="GIF steps per second")
    args = parser.parse_args()

    graph = AdjacencyList.from_file(args.path)
    steps = record_steps(graph, args.source - 1)
    print_steps(steps)

    target = None if args.target is None else args.target - 1
    if target is not None:
        path = route(steps[-1], target)
        print(f"\nroute {args.source} -> {args.target}: "
              + (" -> ".join(str(v + 1) for v in path) + f"  ({steps[-1].distances[target]:g})"
                 if path else "unreachable"))  # fmt: skip

    if args.gif:
        export_gif(args.gif, graph, args.source - 1, target, args.fps,
                   f"Dijkstra on {args.path.stem} from {args.source}")  # fmt: skip
        print(f"saved {args.gif}")
        raise SystemExit

    fig = draw_steps(graph, steps, target=target)
    fig.suptitle(f"Dijkstra on {args.path.stem} from {args.source}", fontsize=14)
    if args.output:
        fig.savefig(args.output, dpi=120, bbox_inches="tight")
    else:
        plt.show()
