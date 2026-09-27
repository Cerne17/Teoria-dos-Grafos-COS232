"""Exports a Dijkstra run as an animated GIF: one frame per step, with a side
panel listing what happened (vertex picked, distances lowered, frontier...).

Used by the step viewers:

    python steps.py graphs/graph_03.txt --gif g03.gif
    python city_steps.py -s 19 -t 4 --gif city.gif
    python tile_city_steps.py city/maps/sample2.txt -s 0,2 -t 9,14 --gif tiles.gif

and by the builder (key e / "Export GIF" button, after choosing a start with d).
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from steps import Step
from viz import _fmt

INF = float("inf")
DrawPanel = Callable[[int, plt.Axes], None]


def describe(
    steps: Sequence[Step],
    i: int,
    source: int,
    label: Callable[[int], str] = lambda v: str(v + 1),
    unit: str = "",
    target: int | None = None,
    route: Callable[[Step], list[int]] | None = None,
    max_rows: int = 12,
) -> str:
    """The side-panel text for step i."""
    s = steps[i]
    before = steps[i - 1].distances if i else [0 if v == source else INF for v in range(len(s.distances))]
    d = lambda x: _fmt(x) + (unit if x != INF else "")  # noqa: E731
    lines = [
        f"step {i + 1} / {len(steps)}",
        "",
        f"picked {label(s.picked)}",
        f"  distance {d(s.distances[s.picked])} is now final",
        "",
    ]
    lowered = [v for v, (a, b) in enumerate(zip(before, s.distances)) if b < a]
    lines.append("lowered this step:" if lowered else "lowered this step: none")
    for v in lowered[:max_rows]:
        lines.append(f"  {label(v)}: {d(before[v])} → {d(s.distances[v])}"
                     f"  (via {label(s.parent[v])})")  # fmt: skip
    if len(lowered) > max_rows:
        lines.append(f"  … {len(lowered) - max_rows} more")
    lines.append("")
    frontier = sorted(s.frontier, key=lambda v: s.distances[v])
    lines.append(f"frontier ({len(frontier)}), next pick first:" if frontier else "frontier: empty — done")
    for v in frontier[:max_rows]:
        lines.append(f"  {label(v)}: {d(s.distances[v])}")
    if len(frontier) > max_rows:
        lines.append(f"  … {len(frontier) - max_rows} more")
    lines += ["", f"final: {len(s.explored)} of {len(s.distances)}"]
    if target is not None:
        lines.append("")
        if target in s.explored and route is not None:
            path = route(s)
            lines.append(f"route to {label(target)}: {d(s.distances[target])}")
            lines.append("  " + " → ".join(label(v) for v in path) if len(path) <= 8 else
                         f"  {len(path) - 1} moves")  # fmt: skip
        else:
            lines.append(f"route to {label(target)}: not final yet")
    return "\n".join(lines)


def export(
    path: str | Path,
    count: int,
    draw_panel: DrawPanel,
    text: Callable[[int], str],
    figsize: tuple[float, float] = (12, 8),
    fps: float = 1.5,
    hold_last: float = 4.0,
    dpi: int = 80,
    title: str = "",
) -> Path:
    """Renders panels 0..count-1 (map + text panel) into an animated GIF.

    The last frame stays on screen for hold_last seconds before looping.
    """
    fig, (ax, info) = plt.subplots(
        1, 2, figsize=figsize, width_ratios=[figsize[0] - 4, 4], layout="constrained"
    )
    frames = []
    for i in range(count):
        ax.clear()
        info.clear()
        draw_panel(i, ax)
        info.set_axis_off()
        info.text(0, 1, text(i), va="top", ha="left", family="monospace", fontsize=9,
                  transform=info.transAxes)  # fmt: skip
        if title:
            fig.suptitle(title)
        fig.canvas.draw()
        frame = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                                 fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)  # fmt: skip
        frames.append(frame.convert("RGB"))
    plt.close(fig)

    # One shared 256-colour palette (taken from the first and last frames, which
    # between them show every colour used): frames then differ only where the
    # search changed something, and the GIF stores just those differences
    sample = Image.new("RGB", (frames[0].width, frames[0].height * 2))
    sample.paste(frames[0], (0, 0))
    sample.paste(frames[-1], (0, frames[0].height))
    palette = sample.quantize(256, method=Image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]

    path = Path(path)
    durations = [int(1000 / fps)] * count
    durations[-1] = int(1000 * hold_last)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True)  # fmt: skip
    return path
