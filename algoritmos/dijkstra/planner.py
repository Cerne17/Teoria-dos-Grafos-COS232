"""Plans realistic tile cities: streets first, then scenery with WFC.

Plain WFC only sees neighbouring tiles, so it can't make a street network look
planned. Here the streets come from a recursive subdivision of the map into
blocks (like a city grid that grew over time):

  1. optional ring road around the map;
  2. split the largest region with a straight street, recursively, until every
     block is between min_block and max_block cells wide;
  3. optional cul-de-sacs: dead-end stubs into big blocks;
  4. some streets become one-way (only where the tile set has junction tiles
     for it, and only if every road tile stays reachable from every other);
  5. some junctions become roundabouts, some straights near junctions crosswalks;
  6. WFC fills everything else with scenery (tiles.py sockets still hold).

Every knob lives in a style (city/styles/*.json); tune them there or with --set.
Areas: a zones file (made by the builder) gives cells their own style. Streets
are laid along area borders first, then each area uses its own knobs.

    python planner.py                              # 'mixed' style, 12x20
    python planner.py 14 24 --style suburb --seed 3
    python planner.py --style downtown --set oneway=0.6 --set max_block=2
    python planner.py -o city/maps/planned.txt --png planned.png
    python planner.py 16 24 --zones city/maps/custom.zones.json
"""

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

import networkx as nx

import tiles
import wfc
from tiles import DELTA, OPPOSITE, SIDES, TILES
from viz import to_networkx

STYLES = tiles.ASSETS / "styles"
Cell = tuple[int, int]


@dataclass
class Style:
    """Knobs for the planner. Fractions are probabilities in 0..1."""

    ring: bool = True  # road around the whole map
    min_block: int = 2  # smallest block side, in cells (>= 1)
    max_block: int = 4  # regions larger than this get split again
    oneway: float = 0.2  # chance each street is tried as one-way
    roundabout: float = 0.15  # chance a two-way junction becomes a roundabout
    crosswalk: float = 0.3  # chance a straight next to a junction gets a crosswalk
    culdesac: float = 0.0  # chance a big block gets a dead-end stub
    # Scenery tile weights by base name (see wfc.DEFAULT_WEIGHTS); roads are ignored
    scenery: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, name_or_path: str) -> "Style":
        path = Path(name_or_path)
        if not path.exists():
            path = STYLES / f"{name_or_path}.json"
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if not k.startswith("_")})

    def with_overrides(self, assignments: list[str]) -> "Style":
        """Applies 'knob=value' strings, e.g. 'oneway=0.5' or 'scenery.park_center=4'."""
        types = {f.name: f.type for f in fields(self)}
        for item in assignments:
            key, value = item.split("=", 1)
            if key.startswith("scenery."):
                self.scenery[key.removeprefix("scenery.")] = float(value)
            elif key == "ring":
                self.ring = value.lower() in ("1", "true", "yes")
            elif key in types:
                setattr(self, key, int(value) if types[key] in (int, "int") else float(value))
            else:
                raise ValueError(f"unknown knob {key!r}; knobs: {', '.join(types)}")
        return self

    def scenery_weights(self) -> dict[str, float]:
        return wfc.scenery_weights({**wfc.DEFAULT_WEIGHTS, **self.scenery})


# Road tiles by their exact sockets, e.g. "R.R." -> [road_straight_r0, road_crosswalk_r0]
_BY_SOCKETS: dict[str, list[int]] = {}
for _t in TILES:
    if _t.is_road:
        _BY_SOCKETS.setdefault(_t.sockets, []).append(_t.index)


def _pick(sockets: str, want: str) -> int | None:
    """The tile with these sockets whose base starts with `want`, if any."""
    return next((t for t in _BY_SOCKETS.get(sockets, []) if TILES[t].base.startswith(want)), None)


Zones = Sequence[Sequence["Style | None"]]


class Planner:
    """style is the default; zones (rows x cols, None = default) overrides it per cell."""

    def __init__(
        self, rows: int, cols: int, style: Style, rng: random.Random, zones: Zones | None = None
    ):
        self.rows, self.cols, self.style, self.rng = rows, cols, style, rng
        self.zones = [[(zones[r][c] if zones else None) or style for c in range(cols)]
                      for r in range(rows)]  # fmt: skip
        self.road: set[Cell] = set()
        self.blocks: list[tuple[int, int, int, int]] = []  # r0, c0, r1, c1 (inclusive)
        # Socket per road cell and side: R two-way, I/O one-way in/out
        self.sockets: dict[Cell, dict[str, str]] = {}

    # ---- areas -------------------------------------------------------------------

    def style_at(self, cell: Cell) -> Style:
        return self.zones[cell[0]][cell[1]]

    def _region_styles(self, r0: int, c0: int, r1: int, c1: int) -> list[Style]:
        """Styles in the region, most common first."""
        counts: dict[int, list] = {}
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                st = self.zones[r][c]
                counts.setdefault(id(st), [st, 0])[1] += 1
        return [st for st, _ in sorted(counts.values(), key=lambda x: -x[1])]

    def _border_split(self, r0: int, c0: int, r1: int, c1: int, lo: int):
        """(vertical, k) for the street that best follows an area border, or None.

        A street at column k separates areas well when, row after row, the
        cells on its left and right belong to different areas.
        """
        z = self.zones
        best, best_score = None, 0.5  # must follow the border on at least half its length
        for k in range(c0 + lo, c1 - lo + 1):
            score = sum(z[r][k - 1] is not z[r][k + 1] for r in range(r0, r1 + 1)) / (r1 - r0 + 1)
            if score > best_score or (score == best_score and best and self.rng.random() < 0.5):
                best, best_score = (True, k), score
        for k in range(r0 + lo, r1 - lo + 1):
            score = sum(z[k - 1][c] is not z[k + 1][c] for c in range(c0, c1 + 1)) / (c1 - c0 + 1)
            if score > best_score or (score == best_score and best and self.rng.random() < 0.5):
                best, best_score = (False, k), score
        return best

    # ---- 1-3: street layout ------------------------------------------------------

    def _line(self, cells) -> None:
        self.road.update(cells)

    def _split(self, r0: int, c0: int, r1: int, c1: int) -> None:
        if r1 < r0 or c1 < c0:
            return
        h, w = r1 - r0 + 1, c1 - c0 + 1
        styles = self._region_styles(r0, c0, r1, c1)
        # The region follows its most common area; mixed regions split on borders first
        lo, hi = styles[0].min_block, styles[0].max_block
        border = None
        if len(styles) > 1:
            border = self._border_split(r0, c0, r1, c1, max(1, min(st.min_block for st in styles)))
        can_v, can_h = w >= 2 * lo + 1, h >= 2 * lo + 1  # room for a street + 2 blocks
        if border is None and ((h <= hi and w <= hi) or not (can_v or can_h)):
            self.blocks.append((r0, c0, r1, c1))
            return
        if border is not None:
            vertical, k = border
        else:
            # Split across the longer side, so blocks stay roughly square
            vertical = can_v and (not can_h or w > h or (w == h and self.rng.random() < 0.5))
            k = self.rng.randint(c0 + lo, c1 - lo) if vertical else self.rng.randint(r0 + lo, r1 - lo)
        if vertical:
            self._line((r, k) for r in range(r0, r1 + 1))
            self._split(r0, c0, r1, k - 1)
            self._split(r0, k + 1, r1, c1)
        else:
            self._line((k, c) for c in range(c0, c1 + 1))
            self._split(r0, c0, k - 1, c1)
            self._split(k + 1, c0, r1, c1)

    def _culdesacs(self) -> None:
        for r0, c0, r1, c1 in self.blocks:
            h, w = r1 - r0 + 1, c1 - c0 + 1
            style = self._region_styles(r0, c0, r1, c1)[0]
            if min(h, w) < 3 or self.rng.random() >= style.culdesac:
                continue
            # A stub down the middle, from a side that touches a road, leaving
            # at least one cell of block between its tip and the far side
            options = []
            mid_c, mid_r = (c0 + c1) // 2, (r0 + r1) // 2
            if (r0 - 1, mid_c) in self.road:
                options.append([(r, mid_c) for r in range(r0, r1 - 1)])
            if (r1 + 1, mid_c) in self.road:
                options.append([(r, mid_c) for r in range(r1, r0 + 1, -1)])
            if (mid_r, c0 - 1) in self.road:
                options.append([(mid_r, c) for c in range(c0, c1 - 1)])
            if (mid_r, c1 + 1) in self.road:
                options.append([(mid_r, c) for c in range(c1, c0 + 1, -1)])
            if options:
                stub = self.rng.choice(options)
                self._line(stub[: self.rng.randint(1, len(stub))])

    # ---- 4: sockets, streets and one-way ---------------------------------------------

    def _neighbour(self, cell: Cell, side: str) -> Cell:
        return cell[0] + DELTA[side][0], cell[1] + DELTA[side][1]

    def _init_sockets(self) -> None:
        for cell in self.road:
            self.sockets[cell] = {
                s: "R" if self._neighbour(cell, s) in self.road else "." for s in SIDES
            }

    def _degree(self, cell: Cell) -> int:
        return sum(v != "." for v in self.sockets[cell].values())

    def _streets(self) -> list[tuple[Cell, str, list[Cell], Cell, str]]:
        """Chains of degree-2 cells between junctions:
        (start junction, side it leaves by, inner cells, end junction, side it arrives by)."""
        junctions = [c for c in self.road if self._degree(c) != 2]
        seen: set[tuple[Cell, str]] = set()
        streets = []
        for j in junctions:
            for side in SIDES:
                if self.sockets[j][side] == "." or (j, side) in seen:
                    continue
                inner, prev, cell = [], j, self._neighbour(j, side)
                came_from = OPPOSITE[side]
                while self._degree(cell) == 2:
                    inner.append(cell)
                    out = next(s for s in SIDES if s != came_from and self.sockets[cell][s] != ".")
                    prev, cell = cell, self._neighbour(cell, out)
                    came_from = OPPOSITE[out]
                seen.add((j, side))
                seen.add((cell, came_from))
                streets.append((j, side, inner, cell, came_from))
        return streets

    def _side_to(self, a: Cell, b: Cell) -> str:
        return next(s for s in SIDES if self._neighbour(a, s) == b)

    def _valid(self, sockets, cells) -> bool:
        """Every cell's socket combination exists as a tile."""
        return all("".join(sockets[c][s] for s in SIDES) in _BY_SOCKETS for c in cells)

    def _connected(self, sockets) -> bool:
        grid = self._tile_grid(sockets, decorate=False)
        g = to_networkx(tiles.to_graph(grid).graph)
        return g.number_of_nodes() == 0 or nx.is_strongly_connected(g)

    def _oneways(self) -> None:
        streets = self._streets()
        attached: dict[tuple[Cell, str], int] = {}  # (junction, side) -> street
        for k, (start, side, _, end, end_side) in enumerate(streets):
            attached[(start, side)] = attached[(end, end_side)] = k

        def path(k: int, forward: bool) -> list[Cell]:
            start, _, inner, end, _ = streets[k]
            cells = [start, *inner, end]
            return cells if forward else cells[::-1]

        order = list(range(len(streets)))
        self.rng.shuffle(order)
        for k in order:
            start, _, _, end, _ = streets[k]
            middle = path(k, True)[len(path(k, True)) // 2]
            if self.rng.random() >= self.style_at(middle).oneway or start == end:
                continue
            if self._degree(start) == 1 or self._degree(end) == 1:
                continue  # a one-way dead end would trap cars
            # A one-way street keeps its direction straight through 4-way
            # crossings: the street across the crossing joins the group
            group = {k: self.rng.random() < 0.5}
            todo, ok = [k], True
            while todo and ok:
                i = todo.pop()
                p = path(i, group[i])
                links = []
                last, first = p[-1], p[0]
                if self._degree(last) == 4:  # continue out the far side
                    far = OPPOSITE[self._side_to(last, p[-2])]
                    links.append(((last, far), True))
                if self._degree(first) == 4:  # and come in from the far side
                    far = OPPOSITE[self._side_to(first, p[1])]
                    links.append(((first, far), False))
                for (j, side), leaves_j in links:
                    o = attached.get((j, side))
                    if o is None:
                        continue
                    # Orient o so it leaves j (if leaves_j) or arrives at j
                    o_fwd = (streets[o][0], streets[o][1]) == (j, side)
                    o_fwd = o_fwd if leaves_j else not o_fwd
                    if o in group:
                        ok = ok and group[o] == o_fwd
                    else:
                        group[o] = o_fwd
                        todo.append(o)
            if not ok:
                continue
            trial = {c: dict(v) for c, v in self.sockets.items()}
            touched: set[Cell] = set()
            for i, fwd in group.items():
                p = path(i, fwd)
                for a, b in zip(p, p[1:]):
                    s = self._side_to(a, b)
                    if trial[a][s] != "R":
                        ok = False  # already one-way: leave it alone
                    trial[a][s], trial[b][OPPOSITE[s]] = "O", "I"
                touched.update(p)
            if ok and self._valid(trial, touched) and self._connected(trial):
                self.sockets = trial

    # ---- 5-6: tiles -------------------------------------------------------------

    def _tile_grid(self, sockets, decorate: bool = True) -> list[list[int]]:
        grid = tiles.blank(self.rows, self.cols)
        roundabouts: set[Cell] = set()
        for cell in sorted(sockets):
            key = "".join(sockets[cell][s] for s in SIDES)
            degree = sum(v != "." for v in key)
            choice = None
            if decorate and set(key) <= {"R", "."} and degree >= 3:
                near = any(self._neighbour(cell, s) in roundabouts for s in SIDES)
                if not near and self.rng.random() < self.style_at(cell).roundabout:
                    choice = _pick(key, "roundabout")
                    if choice is not None:
                        roundabouts.add(cell)
            if decorate and choice is None and key in ("R.R.", ".R.R"):
                next_to_junction = any(
                    self._neighbour(cell, s) in sockets
                    and sum(v != "." for v in sockets[self._neighbour(cell, s)].values()) >= 3
                    for s in SIDES
                )
                if next_to_junction and self.rng.random() < self.style_at(cell).crosswalk:
                    choice = _pick(key, "road_crosswalk")
            if choice is None:
                choice = _pick(key, "road") or _pick(key, "oneway") or _pick(key, "mixed")
            if choice is None:
                choice = _BY_SOCKETS[key][0]
            grid[cell[0]][cell[1]] = choice
        return grid

    def roads(self) -> list[list[int]]:
        """Just the planned streets, on empty pavement."""
        top, left = (1, 1) if self.style.ring else (0, 0)
        bottom, right = self.rows - 1 - top, self.cols - 1 - left
        if self.style.ring:
            self._line((0, c) for c in range(self.cols))
            self._line((self.rows - 1, c) for c in range(self.cols))
            self._line((r, 0) for r in range(self.rows))
            self._line((r, self.cols - 1) for r in range(self.rows))
        self._split(top, left, bottom, right)
        if not self.style.ring:
            self._trim_edges()
        self._culdesacs()
        self._init_sockets()
        self._oneways()
        return self._tile_grid(self.sockets)

    def _trim_edges(self) -> None:
        """Without a ring, streets end at the map edge: stop them one cell short
        so they end in a dead end instead of running off the map."""
        edge = {c for c in self.road
                if c[0] in (0, self.rows - 1) or c[1] in (0, self.cols - 1)}  # fmt: skip
        self.road -= edge


def plan(
    rows: int,
    cols: int,
    style: Style | None = None,
    seed: int | None = None,
    zones: Zones | None = None,
) -> list[list[int]]:
    """A planned city: streets from the planner, scenery from WFC.

    zones (rows x cols of Style or None) gives areas their own style; None
    cells use `style`.
    """
    style = style or Style.load("mixed")
    rng = random.Random(seed)
    planner = Planner(rows, cols, style, rng, zones)
    grid = planner.roads()
    # Roads stay, but each may take a curb cut where a building's entrance meets it
    fixed = {
        (r, c): tiles.with_driveways(t)
        for r, row in enumerate(grid)
        for c, t in enumerate(row)
        if TILES[t].is_road
    }
    # One weight table per style, shared by its cells
    tables: dict[int, dict[str, float]] = {}
    cell_weights = [
        [tables.setdefault(id(st), st.scenery_weights()) for st in row] for row in planner.zones
    ]
    return wfc.generate(rows, cols, rng.randrange(2**32), style.scenery_weights(), fixed,
                        cell_weights=cell_weights)  # fmt: skip


def style_names() -> list[str]:
    return sorted(p.stem for p in STYLES.glob("*.json"))


# ---- zone files: which style each cell uses (null = the default style) ----------


def load_zones(path: str | Path) -> list[list[str | None]]:
    return json.loads(Path(path).read_text())["zones"]


def save_zones(zones: Sequence[Sequence[str | None]], path: str | Path) -> None:
    rows = ",\n    ".join(json.dumps(list(row)) for row in zones)
    Path(path).write_text(f'{{"zones": [\n    {rows}\n]}}\n')


def zone_styles(names: Sequence[Sequence[str | None]]) -> list[list[Style | None]]:
    """Zone names -> Style objects (one object per name, so cells share it)."""
    loaded: dict[str, Style] = {}
    return [[None if n is None else loaded.setdefault(n, Style.load(n)) for n in row]
            for row in names]  # fmt: skip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=__doc__.split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("rows", nargs="?", type=int, default=12)
    parser.add_argument("cols", nargs="?", type=int, default=20)
    parser.add_argument("--style", default="mixed", help=f"one of {style_names()} or a .json path")
    parser.add_argument("--set", action="append", default=[], metavar="KNOB=VALUE",
                        help="override a style knob, e.g. oneway=0.5 or scenery.park_center=4")  # fmt: skip
    parser.add_argument("--seed", type=int)
    parser.add_argument("--zones", type=Path, help="zones file from the builder (sets rows/cols)")
    parser.add_argument("-o", "--output", type=Path, help="save the map (text)")
    parser.add_argument("--png", type=Path, help="save a picture of the map")
    args = parser.parse_args()

    style = Style.load(args.style).with_overrides(args.set)
    zones = None
    if args.zones:
        names = load_zones(args.zones)
        args.rows, args.cols = len(names), len(names[0])
        zones = zone_styles(names)
    grid = plan(args.rows, args.cols, style, args.seed, zones)
    assert not tiles.mismatches(grid)
    roads, scc = wfc.connectivity(grid)
    print(f"road tiles {roads}, largest strongly connected {scc}")
    if args.output:
        tiles.save_map(grid, args.output)
        print(f"saved {args.output}")
    if args.png or not args.output:
        import matplotlib.pyplot as plt

        import tile_city

        ax = tile_city.draw(grid, title=f"{args.style} ({args.rows}x{args.cols})")
        if args.png:
            ax.figure.savefig(args.png, dpi=120, bbox_inches="tight")
        else:
            plt.show()
