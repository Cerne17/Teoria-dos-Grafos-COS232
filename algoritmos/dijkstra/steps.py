"""Draws Dijkstra step by step, one panel per vertex picked.

    python steps.py graphs/graph_02.txt                  # source 1, opens a window
    python steps.py graphs/graph_03.txt -s 3 -o g03.png  # source 3, saves to file

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


def draw_steps(graph: AdjacencyList, steps: list[Step], cols: int = 3) -> plt.Figure:
    rows = math.ceil(len(steps) / cols)
    # Constrained layout leaves room for the suptitle both on screen and on save
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False, layout="constrained"
    )
    for ax, (i, s) in zip(axes.flat, enumerate(steps, 1)):
        viz.draw(
            graph,
            title=f"step {i}: pick {s.picked + 1} (d = {viz._fmt(s.distances[s.picked])})",
            source=s.picked,
            distances=s.distances,
            tree_edges=s.tree_edges,
            explored=s.explored,
            frontier=s.frontier,
            ax=ax,
        )
    for ax in axes.flat[len(steps):]:
        ax.set_axis_off()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", type=Path, help="graph file: n, then 'u v w' lines")
    parser.add_argument("-s", "--source", type=int, default=1, help="1-indexed source")
    parser.add_argument("-o", "--output", type=Path, help="save image instead of showing")
    args = parser.parse_args()

    graph = AdjacencyList.from_file(args.path)
    steps = record_steps(graph, args.source - 1)
    print_steps(steps)

    fig = draw_steps(graph, steps)
    fig.suptitle(f"Dijkstra on {args.path.stem} from {args.source}", fontsize=14)
    if args.output:
        fig.savefig(args.output, dpi=120, bbox_inches="tight")
    else:
        plt.show()
