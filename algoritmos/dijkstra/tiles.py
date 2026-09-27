"""Tile catalogue for tile-based cities, and how a tile map becomes a graph.

The atlas (city/atlas@5x.png) is a 10-column grid of 32 px tiles drawn at 5x.
Each tile has a socket on every side (N, E, S, W):

    .  nothing crosses this side          P  park continues
    R  two-way road                       B  city block continues
    I  one-way road entering the tile     O  one-way road leaving the tile
    M  inside a multi-cell building (skyscraper, mall...)
    D  driveway: a building's entrance meeting a road's curb cut (D-D, and only
       between a road and a non-road tile: never building to building)

Two neighbours fit when their facing sockets match: R-R, O-I, I-O, P-P, B-B, .-.
M-M fits only between parts of the same building in their right places, e.g.
mall_3x2_1_0 must sit east of mall_3x2_0_0 (parts are named base_x_y).

A tile map is a grid of tile indices. Every road tile is a vertex; an arc goes
from a road tile to its neighbour when the road may be driven that way, with
weight TILE_METERS. Vertices are numbered in row-major order.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import json

import numpy as np
from PIL import Image

from adjacency_list_weighted import AdjacencyList

ASSETS = Path(__file__).parent / "city"
ATLAS = ASSETS / "atlas@5x.png"
CATALOGUE = ASSETS / "tiles.json"  # name, base, sockets and building part of every tile
MAPS = ASSETS / "maps"
ATLAS_COLS, ATLAS_SCALE, TILE_PX = 10, 5, 32
TILE_METERS = 10

SIDES = "NESW"
DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}  # (row, col)
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
MATCH = {"R": "R", "O": "I", "I": "O", "P": "P", "B": "B", "D": "D", ".": "."}

@dataclass(frozen=True)
class Tile:
    index: int
    name: str
    base: str
    sockets: str  # N E S W
    part: tuple[int, int, int, int] | None = None  # x, y, width, height in a building

    def socket(self, side: str) -> str:
        return self.sockets[SIDES.index(side)]

    @property
    def is_road(self) -> bool:
        return any(s in "RIO" for s in self.sockets)

    def describe(self) -> str:
        """Sides in words, e.g. 'N exit, S entrance' (sides with nothing are left out)."""
        words = {"R": "two-way", "I": "entrance", "O": "exit", "P": "park", "B": "block",
                 "M": "same building", "D": "driveway"}  # fmt: skip
        parts = [f"{side} {words[s]}" for side, s in zip(SIDES, self.sockets) if s != "."]
        return ", ".join(parts) or "no connections"


def _build() -> list[Tile]:
    """The catalogue from city/tiles.json (index, name, base, sockets, part), in atlas order."""
    entries = json.loads(CATALOGUE.read_text())
    tiles = []
    for i, e in enumerate(entries):
        if e["index"] != i:
            raise ValueError(f"{CATALOGUE}: entry {i} has index {e['index']}")
        part = tuple(e["part"]) if e.get("part") else None
        tiles.append(Tile(i, e["name"], e["base"], e["sockets"], part))
    # Plain pavement is not in the atlas; the builder and the generator need it
    tiles.append(Tile(len(tiles), "empty", "empty", "...."))
    return tiles


TILES = _build()
EMPTY = TILES[-1].index
BY_NAME = {t.name: t for t in TILES}


def fits(a: int, side: str, b: int) -> bool:
    """Can tile b sit on the given side of tile a?"""
    ta, tb = TILES[a], TILES[b]
    if ta.socket(side) == "M" or tb.socket(OPPOSITE[side]) == "M":
        # Both parts of the same building, b exactly one step from a
        if not (ta.part and tb.part and ta.base == tb.base):
            return False
        dr, dc = DELTA[side]
        return (tb.part[0], tb.part[1]) == (ta.part[0] + dc, ta.part[1] + dr)
    if ta.socket(side) == "D":
        # A driveway links a place to the street: exactly one of the two is a road
        return tb.socket(OPPOSITE[side]) == "D" and ta.is_road != tb.is_road
    return MATCH[ta.socket(side)] == tb.socket(OPPOSITE[side])


def rotations_of(index: int) -> list[int]:
    """All tiles sharing index's base, in atlas order (r0, r90, ...)."""
    t = TILES[index]
    if t.part:
        return [index]  # building parts don't rotate
    # Same base and same number of driveways: r0/r90/... or drive_n/drive_e/...
    return [u.index for u in TILES
            if u.base == t.base and not u.part and u.sockets.count("D") == t.sockets.count("D")]  # fmt: skip


# ---- images -----------------------------------------------------------------


@cache
def tile_images() -> list[np.ndarray]:
    """Every tile as a 32x32 RGB array, indexed like TILES."""
    atlas = Image.open(ATLAS).convert("RGB")
    # The atlas is pixel art at 5x: sampling one pixel per 5x5 block is lossless
    small = np.asarray(atlas)[ATLAS_SCALE // 2 :: ATLAS_SCALE, ATLAS_SCALE // 2 :: ATLAS_SCALE]
    images = []
    for t in TILES[:EMPTY]:
        r, c = divmod(t.index, ATLAS_COLS)
        images.append(small[r * TILE_PX : (r + 1) * TILE_PX, c * TILE_PX : (c + 1) * TILE_PX])
    # Empty pavement: the top 8 rows of the horizontal road are pure pavement,
    # and the pattern repeats every 4 px, so stacking them makes a seamless tile
    pavement = images[BY_NAME["road_straight_r90"].index][:8]
    images.append(np.tile(pavement, (TILE_PX // 8, 1, 1)))
    return images


def render(grid: Sequence[Sequence[int]]) -> np.ndarray:
    """The whole map as one RGB image, TILE_PX pixels per cell."""
    images = tile_images()
    return np.vstack([np.hstack([images[i] for i in row]) for row in grid])


# ---- map files ----------------------------------------------------------------


def load_map(path: str | Path) -> list[list[int]]:
    """One row per line, a tile index per cell, '.' for empty; '#' starts a comment."""
    grid = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            grid.append([EMPTY if tok == "." else int(tok) for tok in line.split()])
    if not grid or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError(f"{path}: rows must all have the same number of cells")
    return grid


def save_map(grid: Sequence[Sequence[int]], path: str | Path) -> None:
    lines = ["# tile city: tile index per cell (see tiles.py), '.' = empty"]
    lines += [" ".join(" ." if i == EMPTY else f"{i:2d}" for i in row) for row in grid]
    Path(path).write_text("\n".join(lines) + "\n")


def from_image(path: str | Path) -> list[list[int]]:
    """Recovers the tile map of an image drawn with the atlas (e.g. sample_city.png)."""
    atlas = np.asarray(Image.open(ATLAS).convert("RGB")).astype(np.int16)
    image = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    size = TILE_PX * ATLAS_SCALE
    refs = [
        atlas[(t // ATLAS_COLS) * size : (t // ATLAS_COLS + 1) * size,
              (t % ATLAS_COLS) * size : (t % ATLAS_COLS + 1) * size]
        for t in range(EMPTY)
    ]  # fmt: skip
    grid = []
    for r in range(image.shape[0] // size):
        row = []
        for c in range(image.shape[1] // size):
            cell = image[r * size : (r + 1) * size, c * size : (c + 1) * size]
            row.append(int(np.argmin([np.abs(cell - ref).mean() for ref in refs])))
        grid.append(row)
    return grid


# ---- validation and graph -------------------------------------------------------


def mismatches(grid: Sequence[Sequence[int]]) -> list[tuple[int, int, str]]:
    """(row, col, side) of every side whose socket does not fit its neighbour.

    The outside of the map counts as empty pavement, so roads, parks and
    blocks may not run off the edge.
    """
    rows, cols = len(grid), len(grid[0])
    bad = []
    for r in range(rows):
        for c in range(cols):
            for side in SIDES:
                dr, dc = DELTA[side]
                nr, nc = r + dr, c + dc
                inside = 0 <= nr < rows and 0 <= nc < cols
                neighbour = grid[nr][nc] if inside else EMPTY
                if not fits(grid[r][c], side, neighbour):
                    bad.append((r, c, side))
    return bad


@dataclass
class CityGraph:
    graph: AdjacencyList
    cells: list[tuple[int, int]]  # vertex -> (row, col)
    vertex_at: dict[tuple[int, int], int]  # (row, col) -> road vertex
    names: dict[int, str] = field(default_factory=dict)  # place vertices (see places.py)

    def label(self, v: int) -> str:
        if v in self.names:
            return self.names[v]
        r, c = self.cells[v]
        return f"{v + 1} (row {r}, col {c})"


def to_graph(grid: Sequence[Sequence[int]]) -> CityGraph:
    """Every road tile becomes a vertex; arcs follow the drivable sockets.

    A two-way side gives arcs both ways (each tile adds its own), a one-way
    side 'O' gives a single arc out of the tile. Sides that don't fit their
    neighbour give no arc. O(rows * cols)
    """
    cells = [
        (r, c) for r, row in enumerate(grid) for c, t in enumerate(row) if TILES[t].is_road
    ]
    vertex_at = {cell: v for v, cell in enumerate(cells)}
    graph = AdjacencyList(len(cells))
    for v, (r, c) in enumerate(cells):
        for side in SIDES:
            if TILES[grid[r][c]].socket(side) not in "RO":
                continue
            dr, dc = DELTA[side]
            u = vertex_at.get((r + dr, c + dc))
            if u is not None and fits(grid[r][c], side, grid[r + dr][c + dc]):
                graph.add_edge(v, u, TILE_METERS)
    return CityGraph(graph, cells, vertex_at)


def building_cells(grid: Sequence[Sequence[int]], r: int, c: int) -> list[tuple[int, int, int]]:
    """(row, col, tile) of every part of the building whose part sits at (r, c).

    Parts that are missing or out of place are left out.
    """
    t = TILES[grid[r][c]]
    if not t.part:
        return [(r, c, grid[r][c])]
    x, y, w, h = t.part
    top, left = r - y, c - x
    cells = []
    for py in range(h):
        for px in range(w):
            rr, cc = top + py, left + px
            if not (0 <= rr < len(grid) and 0 <= cc < len(grid[0])):
                continue
            other = TILES[grid[rr][cc]]  # any variant of the right part (e.g. with a driveway)
            if other.base == t.base and other.part == (px, py, w, h):
                cells.append((rr, cc, grid[rr][cc]))
    return cells


def without_driveways(index: int) -> int:
    """The plain version of a driveway variant (the tile itself if it has none)."""
    t = TILES[index]
    plain = t.sockets.replace("D", ".")
    return next(u.index for u in TILES
                if u.base == t.base and u.part == t.part and u.sockets == plain)  # fmt: skip


def with_driveways(index: int) -> list[int]:
    """index plus its variants that add driveways (D) on some of its empty sides."""
    t = TILES[index]
    return [
        u.index for u in TILES
        if u.base == t.base and u.part == t.part
        and all(a == b or (a == "." and b == "D") for a, b in zip(t.sockets, u.sockets))
    ]  # fmt: skip


def blank(rows: int, cols: int) -> list[list[int]]:
    return [[EMPTY] * cols for _ in range(rows)]


def count_bases(grid: Iterable[Iterable[int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in grid:
        for t in row:
            counts[TILES[t].base] = counts.get(TILES[t].base, 0) + 1
    return counts
