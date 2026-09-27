"""Dijkstra step by step on the city map.

    python city_steps.py                  # from intersection 1, interactive window
    python city_steps.py -s 19 -t 4       # from 19, prints and highlights the route to 4
    python city_steps.py -s 19 -o run.png # every step in one grid image

Window keys: right/left = next/previous step, up/down = 10 steps, home/end = first/last.
Colours as in steps.py: orange = picked, green = final, yellow = discovered.
"""

import argparse
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt

import city
from adjacency_list_weighted import AdjacencyList
from steps import Step, print_steps, record_steps


def route(step: Step, target: int) -> list[int]:
    """Route to target as known at this step. Empty if target is unreachable."""
    if step.distances[target] == float("inf"):
        return []
    return city.path_from_parents(step.parent, target)


def draw_step(
    graph: AdjacencyList, steps: list[Step], i: int, ax: plt.Axes, target: int | None = None
) -> None:
    """Once target is final (explored), its route is drawn in purple."""
    s = steps[i]
    path = route(s, target) if target is not None and target in s.explored else []
    city.draw(
        graph,
        title=f"step {i + 1}/{len(steps)}: pick {s.picked + 1} ({s.distances[s.picked]:g} m)",
        source=s.picked,
        distances=s.distances,
        tree_edges=s.tree_edges,
        explored=s.explored,
        frontier=s.frontier,
        path=path,
        ax=ax,
    )


# Draws panel i (a step index) on the given axes
DrawPanel = Callable[[int, plt.Axes], None]


def browse(
    count: int,
    draw_panel: DrawPanel,
    suptitle: str,
    figsize: tuple[float, float] = (8, 8.4),
    block: bool = True,
) -> plt.Figure:
    """One panel at a time, driven by the arrow keys.

    block=False opens the window without taking over the event loop, for
    callers that already run one (e.g. the city builder).
    """
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    fig.suptitle(f"{suptitle}   (← → step, ↑ ↓ ten steps)")
    current = 0

    def render() -> None:
        ax.clear()
        draw_panel(current, ax)
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        nonlocal current
        moves = {
            "right": current + 1, "left": current - 1,
            "up": current + 10, "down": current - 10,
            "home": 0, "end": count - 1,
        }  # fmt: skip
        if event.key in moves:
            current = min(max(moves[event.key], 0), count - 1)
            render()

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    if block:
        plt.show()
    else:
        fig.show()
    return fig


def save_grid(
    indices: Sequence[int],
    draw_panel: DrawPanel,
    suptitle: str,
    output: Path,
    panel_size: tuple[float, float] = (5, 5),
    cols: int = 4,
) -> None:
    """Draws the panels listed in indices side by side and saves one image."""
    rows = math.ceil(len(indices) / cols)
    w, h = panel_size
    fig, axes = plt.subplots(
        rows, cols, figsize=(w * cols, h * rows), squeeze=False, layout="constrained"
    )
    for ax in axes.flat[len(indices):]:
        ax.set_axis_off()
    for i, ax in zip(indices, axes.flat):
        draw_panel(i, ax)
    fig.suptitle(suptitle, fontsize=16)
    fig.savefig(output, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-s", "--source", type=int, default=1, help="1-indexed start")
    parser.add_argument("-t", "--target", type=int, help="1-indexed destination: prints and highlights its route")
    parser.add_argument("-o", "--output", type=Path, help="save all steps as a grid image")
    args = parser.parse_args()

    graph = city.load()
    steps = record_steps(graph, args.source - 1)
    print_steps(steps)

    if args.target is not None:
        path = route(steps[-1], args.target - 1)
        if path:
            dist = steps[-1].distances[args.target - 1]
            print(f"\nroute {args.source} -> {args.target}: "
                  f"{' -> '.join(str(v + 1) for v in path)}  ({dist:g} m)")  # fmt: skip
        else:
            print(f"\n{args.target} is unreachable from {args.source}")

    suptitle = f"Dijkstra on the city from intersection {args.source}"
    target = None if args.target is None else args.target - 1

    def draw_panel(i: int, ax: plt.Axes) -> None:
        draw_step(graph, steps, i, ax, target)

    if args.output:
        save_grid(range(len(steps)), draw_panel, suptitle, args.output)
    else:
        browse(len(steps), draw_panel, suptitle)
