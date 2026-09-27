"""Tile catalogue for tile-based cities, and how a tile map becomes a graph.

The atlas (city/atlas@5x.png) is a 10-column grid of 32 px tiles drawn at 5x.
Each tile has a socket on every side (N, E, S, W):

    .  nothing crosses this side          P  park continues
    R  two-way road                       B  city block continues
    I  one-way road entering the tile     O  one-way road leaving the tile

Two neighbours fit when their facing sockets match: R-R, O-I, I-O, P-P, B-B, .-.

A tile map is a grid of tile indices. Every road tile is a vertex; an arc goes
from a road tile to its neighbour when the road may be driven that way, with
weight TILE_METERS. Vertices are numbered in row-major order.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image

from adjacency_list_weighted import AdjacencyList

ASSETS = Path(__file__).parent / "city"
ATLAS = ASSETS / "atlas@5x.png"
MAPS = ASSETS / "maps"
ATLAS_COLS, ATLAS_SCALE, TILE_PX = 10, 5, 32
TILE_METERS = 10

SIDES = "NESW"
DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}  # (row, col)
OPPOSITE = {"N": "S", "E": "W", "S": "N", "W": "E"}
MATCH = {"R": "R", "O": "I", "I": "O", "P": "P", "B": "B", ".": "."}

# (base name, sockets N E S W at r0, rotations present in the atlas), in atlas order.
# rotation k turns the tile 90*k degrees clockwise.
_BASES: list[tuple[str, str, Sequence[int]]] = [
    ("road_straight", "R.R.", (0, 1)),
    ("road_crosswalk", "R.R.", (0, 1)),
    ("road_curve", ".RR.", (0, 1, 2, 3)),
    ("road_t", ".RRR", (0, 1, 2, 3)),
    ("road_cross", "RRRR", (None,)),
    ("road_deadend", "..R.", (0, 1, 2, 3)),
    ("roundabout_4", "RRRR", (None,)),
    ("roundabout_3", ".RRR", (0, 1, 2, 3)),
    ("oneway_straight", "O.I.", (0, 1, 2, 3)),  # drives S -> N
    ("oneway_curve_a", ".OI.", (0, 1, 2, 3)),  # drives S -> E
    ("oneway_curve_b", ".IO.", (0, 1, 2, 3)),  # drives E -> S
    ("oneway_cross", "OOII", (0, 1, 2, 3)),  # W -> E crossing S -> N
    ("mixed_t_in", ".RIR", (0, 1, 2, 3)),  # one-way stem entering
    ("mixed_t_out", ".ROR", (0, 1, 2, 3)),  # one-way stem leaving
    ("mixed_cross", "RORI", (0, 1, 2, 3)),  # W -> E crossing a two-way N-S
    ("park_single", "....", (None,)),
    ("park_cap", "P...", (0, 1, 2, 3)),
    ("park_corner", "PP..", (0, 1, 2, 3)),
    ("park_strip", "P.P.", (0, 1)),
    ("park_edge", "PPP.", (0, 1, 2, 3)),
    ("park_center", "PPPP", (None,)),
    ("block_single", "....", (None,)),
    ("block_cap", "B...", (0, 1, 2, 3)),
    ("block_corner", "BB..", (0, 1, 2, 3)),
    ("block_strip", "B.B.", (0, 1)),
    ("block_edge", "BBB.", (0, 1, 2, 3)),
    ("block_center", "BBBB", (None,)),
    ("parking_lot", "....", (0, 1)),
    ("plaza", "....", (None,)),
    ("tower", "....", (None,)),
]


@dataclass(frozen=True)
class Tile:
    index: int
    name: str
    base: str
    sockets: str  # N E S W

    def socket(self, side: str) -> str:
        return self.sockets[SIDES.index(side)]

    @property
    def is_road(self) -> bool:
        return any(s in "RIO" for s in self.sockets)

    def describe(self) -> str:
        """Sides in words, e.g. 'N exit, S entrance' (sides with nothing are left out)."""
        words = {"R": "two-way", "I": "entrance", "O": "exit", "P": "park", "B": "block"}
        parts = [f"{side} {words[s]}" for side, s in zip(SIDES, self.sockets) if s != "."]
        return ", ".join(parts) or "no connections"


def _rotate(sockets: str, k: int) -> str:
    """Sockets after turning the tile 90*k degrees clockwise: N moves to E, and so on."""
    k %= 4
    return sockets[-k:] + sockets[:-k] if k else sockets


def _build() -> list[Tile]:
    tiles = []
    for base, sockets, rotations in _BASES:
        for k in rotations:
            name = base if k is None else f"{base}_r{90 * k}"
            tiles.append(Tile(len(tiles), name, base, _rotate(sockets, k or 0)))
    # Plain pavement is not in the atlas; the builder and the generator need it
    tiles.append(Tile(len(tiles), "empty", "empty", "...."))
    return tiles


TILES = _build()
EMPTY = TILES[-1].index
BY_NAME = {t.name: t for t in TILES}


def fits(a: int, side: str, b: int) -> bool:
    """Can tile b sit on the given side of tile a?"""
    return MATCH[TILES[a].socket(side)] == TILES[b].socket(OPPOSITE[side])


def rotations_of(index: int) -> list[int]:
    """All tiles sharing index's base, in atlas order (r0, r90, ...)."""
    base = TILES[index].base
    return [t.index for t in TILES if t.base == base]


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
    vertex_at: dict[tuple[int, int], int]  # (row, col) -> vertex

    def label(self, v: int) -> str:
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


def blank(rows: int, cols: int) -> list[list[int]]:
    return [[EMPTY] * cols for _ in range(rows)]


def count_bases(grid: Iterable[Iterable[int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in grid:
        for t in row:
            counts[TILES[t].base] = counts.get(TILES[t].base, 0) + 1
    return counts
