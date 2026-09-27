"""Generates tile cities with wave function collapse (simple tiled model).

Each cell starts as the set of every tile (its "domain"). Repeatedly:
  1. observe: pick the undecided cell with the lowest entropy (fewest weighted
     options) and collapse it to one tile, drawn at random by weight;
  2. propagate: remove from the neighbours every tile whose socket no longer
     fits, and keep spreading while domains shrink.
If some domain becomes empty (a contradiction), start over. The map border
behaves like empty pavement, so no road, park or block runs off the edge.

    python wfc.py                          # 10 rows x 16 cols, random seed
    python wfc.py 12 20 --seed 7 -o city/maps/gen.txt --png gen.png
    python wfc.py --tries 30               # keep the best connected of 30 cities
    python wfc.py --learn city/maps/sample.txt   # weights = tile counts in a map
"""

import argparse
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import networkx as nx

import tiles
from tiles import DELTA, EMPTY, SIDES, TILES
from viz import to_networkx

# Relative weight of each tile *base*; rotations share it. Tuned so a city has
# long streets, a few junctions and roundabouts, and blocks/parks in between.
DEFAULT_WEIGHTS: dict[str, float] = {
    "road_straight": 10, "road_crosswalk": 1.5, "road_curve": 3, "road_t": 1.2,
    "road_cross": 0.6, "road_deadend": 0.15, "roundabout_4": 0.4, "roundabout_3": 0.3,
    "oneway_straight": 2, "oneway_curve_a": 0.4, "oneway_curve_b": 0.4,
    "oneway_cross": 0.3, "mixed_t_in": 0.4, "mixed_t_out": 0.4, "mixed_cross": 0.3,
    "park_single": 0.5, "park_cap": 0.4, "park_corner": 0.5, "park_strip": 0.3,
    "park_edge": 0.4, "park_center": 0.4,
    "block_single": 2, "block_cap": 0.6, "block_corner": 0.8, "block_strip": 0.4,
    "block_edge": 0.8, "block_center": 0.8,
    "parking_lot": 0.4, "plaza": 0.3, "tower": 0.3, "empty": 3,
    "house_pool": 0.4, "house_pair": 0.4, "house_garden": 0.4, "house_row": 0.3,
    "apartment_a": 0.3, "apartment_b": 0.3, "office_glass": 0.3, "shops": 0.3,
    # Chance-like weight of a curb cut on a kept road, vs 1 for no cut (see generate)
    "driveway": 0.5,
    # Multi-cell buildings: the weight applies to each part
    "skyscraper_2x2": 0.15, "slab_2x1": 0.2, "slab_1x2": 0.2, "mall_3x2": 0.1, "mall_2x3": 0.1,
}  # fmt: skip


class Contradiction(Exception):
    pass


class Unsatisfiable(Contradiction):
    """The fixed tiles conflict before anything is chosen: retrying can't help."""


def scenery_weights(weights: Mapping[str, float] = DEFAULT_WEIGHTS) -> dict[str, float]:
    """weights without any road tile: parks, blocks, parking lots, plazas, pavement.

    Roads can still be kept as fixed cells; see generate(fixed=...).
    """
    road_bases = {t.base for t in TILES if t.is_road}
    return {base: 0.0 if base in road_bases else w for base, w in weights.items()}


class WaveFunctionCollapse:
    """Domains are bitmasks: bit t set = tile t still possible in that cell.

    cell_weights optionally gives every cell its own weights (e.g. downtown vs
    suburb areas); cells then only draw from, and start with, their own tiles.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        weights: Mapping[str, float],
        rng: random.Random,
        cell_weights: Sequence[Sequence[Mapping[str, float]]] | None = None,
    ):
        self.rows, self.cols, self.rng = rows, cols, rng
        # Distinct weight tables ("profiles") and which one each cell uses
        profiles: dict[int, int] = {}
        self.profiles: list[list[float]] = []
        self.kept_weights: list[list[float]] = []  # for fixed cells with several options
        self.profile_of: list[int] = []
        for r in range(rows):
            for c in range(cols):
                table = cell_weights[r][c] if cell_weights else weights
                if id(table) not in profiles:
                    profiles[id(table)] = len(self.profiles)
                    self.profiles.append([table.get(t.base, 0.0) for t in TILES])
                    # How likely a kept road is to get a curb cut, among its options
                    driveway = table.get("driveway", DEFAULT_WEIGHTS["driveway"])
                    self.kept_weights.append(
                        [driveway ** t.sockets.count("D") for t in TILES]
                    )
                self.profile_of.append(profiles[id(table)])
        self.all_tiles = [
            sum(1 << t.index for t in TILES if w[t.index] > 0) for w in self.profiles
        ]
        # compatible[d][t]: tiles that may sit on side d of tile t
        self.compatible = {
            d: [sum(1 << b.index for b in TILES if tiles.fits(a.index, d, b.index)) for a in TILES]
            for d in SIDES
        }
        self._allowed_cache: dict[tuple[int, str], int] = {}
        self._entropy_cache: dict[tuple[int, int], float] = {}

    # ---- helpers ---------------------------------------------------------------

    @staticmethod
    def _bits(mask: int) -> list[int]:
        out = []
        while mask:
            low = mask & -mask
            out.append(low.bit_length() - 1)
            mask ^= low
        return out

    def _allowed(self, mask: int, side: str) -> int:
        """Union of the tiles that fit on `side` of any tile in mask."""
        key = (mask, side)
        if key not in self._allowed_cache:
            allowed = 0
            for t in self._bits(mask):
                allowed |= self.compatible[side][t]
            self._allowed_cache[key] = allowed
        return self._allowed_cache[key]

    def _entropy(self, cell: int, mask: int) -> float:
        key = (self.profile_of[cell], mask)
        if key not in self._entropy_cache:
            weights = self.profiles[key[0]]
            # max(): a fixed zero-weight tile never needs choosing, but must not log(0)
            ws = [max(weights[t], 1e-9) for t in self._bits(mask)]
            total = sum(ws)
            self._entropy_cache[key] = math.log(total) - sum(w * math.log(w) for w in ws) / total
        return self._entropy_cache[key]

    def _neighbours(self, cell: int):
        r, c = divmod(cell, self.cols)
        for side in SIDES:
            dr, dc = DELTA[side]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                yield side, nr * self.cols + nc

    # ---- the algorithm -----------------------------------------------------------

    def _propagate(self, domains: list[int], stack: list[int]) -> None:
        while stack:
            cell = stack.pop()
            for side, nb in self._neighbours(cell):
                narrowed = domains[nb] & self._allowed(domains[cell], side)
                if narrowed != domains[nb]:
                    if not narrowed:
                        raise Contradiction(divmod(nb, self.cols))
                    domains[nb] = narrowed
                    stack.append(nb)

    def _initial(self, fixed: Mapping[tuple[int, int], "int | Iterable[int]"]) -> list[int]:
        outside_ok = {d: self._allowed(1 << EMPTY, d) for d in SIDES}
        domains = []
        for cell in range(self.rows * self.cols):
            r, c = divmod(cell, self.cols)
            # A fixed tile is allowed even if its weight is 0 (e.g. roads when
            # only scenery is being generated around them)
            if (r, c) in fixed:
                pinned = fixed[(r, c)]
                pinned = [pinned] if isinstance(pinned, int) else pinned
                domain = sum(1 << t for t in pinned)
            else:
                domain = self.all_tiles[self.profile_of[cell]]
            # Sockets facing the border must accept empty pavement outside
            for side in SIDES:
                dr, dc = DELTA[side]
                if not (0 <= r + dr < self.rows and 0 <= c + dc < self.cols):
                    domain &= outside_ok[tiles.OPPOSITE[side]]
            if not domain:
                raise Unsatisfiable(f"({r}, {c}) runs off the map")
            domains.append(domain)
        try:
            self._propagate(domains, list(range(len(domains))))
        except Contradiction as e:
            raise Unsatisfiable(f"nothing fits at {e.args[0]}") from None
        return domains

    def run(self, fixed: Mapping[tuple[int, int], "int | Iterable[int]"] = {}) -> list[list[int]]:
        domains = self._initial(fixed)
        while True:
            open_cells = [i for i, m in enumerate(domains) if m & (m - 1)]
            if not open_cells:
                break
            # Lowest entropy first; the tiny noise breaks ties at random
            cell = min(open_cells, key=lambda i: self._entropy(i, domains[i]) + self.rng.random() * 1e-6)
            options = self._bits(domains[cell])
            weights = self.profiles[self.profile_of[cell]]
            ws = [weights[t] for t in options]
            if not any(ws):  # a fixed cell whose options all weigh 0 (e.g. kept roads)
                ws = [self.kept_weights[self.profile_of[cell]][t] for t in options]
                if not any(ws):  # e.g. driveway=0 but a painted entrance forces a curb cut
                    ws = [1.0] * len(options)
            choice = self.rng.choices(options, weights=ws)[0]
            domains[cell] = 1 << choice
            self._propagate(domains, [cell])
        return [
            [domains[r * self.cols + c].bit_length() - 1 for c in range(self.cols)]
            for r in range(self.rows)
        ]


def generate(
    rows: int,
    cols: int,
    seed: int | None = None,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    fixed: Mapping[tuple[int, int], "int | Iterable[int]"] = {},
    attempts: int = 100,
    cell_weights: Sequence[Sequence[Mapping[str, float]]] | None = None,
) -> list[list[int]]:
    """A city whose sockets all fit. fixed pins (row, col) -> a tile, or a set of
    allowed tiles (e.g. a road and its driveway variants) beforehand;
    cell_weights (rows x cols of weight tables) overrides weights per cell.

    Raises Unsatisfiable at once if the fixed tiles can't be completed with
    the weighted tiles, Contradiction if `attempts` random tries all dead-end.
    """
    rng = random.Random(seed)
    wfc = WaveFunctionCollapse(rows, cols, weights, rng, cell_weights)
    last: Contradiction | None = None
    for _ in range(attempts):
        try:
            return wfc.run(fixed)
        except Unsatisfiable:
            raise
        except Contradiction as e:
            last = e
    raise Contradiction(f"no city after {attempts} attempts; last conflict at cell {last}")


def connectivity(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    """(road tiles, size of the largest strongly connected group of them)."""
    g = to_networkx(tiles.to_graph(grid).graph)
    if g.number_of_nodes() == 0:
        return 0, 0
    return g.number_of_nodes(), max(len(c) for c in nx.strongly_connected_components(g))


def learned_weights(grid: Sequence[Sequence[int]]) -> dict[str, float]:
    """Weights proportional to how often each base appears in grid (+1 smoothing)."""
    counts = tiles.count_bases(grid)
    return {base: counts.get(base, 0) + 1 for base in DEFAULT_WEIGHTS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", nargs="?", type=int, default=10)
    parser.add_argument("cols", nargs="?", type=int, default=16)
    parser.add_argument("--seed", type=int, help="random seed (same seed, same city)")
    parser.add_argument("--tries", type=int, default=5,
                        help="generate this many and keep the most connected")  # fmt: skip
    parser.add_argument("--learn", type=Path, help="take tile weights from this map")
    parser.add_argument("-o", "--output", type=Path, help="save the map (text)")
    parser.add_argument("--png", type=Path, help="save a picture of the map")
    args = parser.parse_args()

    weights = learned_weights(tiles.load_map(args.learn)) if args.learn else DEFAULT_WEIGHTS
    seeds = random.Random(args.seed)
    best, best_score = None, (-1, -1)
    for k in range(args.tries):
        seed = seeds.randrange(2**32)
        grid = generate(args.rows, args.cols, seed, weights)
        roads, scc = connectivity(grid)
        print(f"try {k + 1}: seed {seed}  road tiles {roads}  largest strongly connected {scc}")
        if (scc, roads) > best_score:
            best, best_score = grid, (scc, roads)

    assert not tiles.mismatches(best)
    if args.output:
        tiles.save_map(best, args.output)
        print(f"saved {args.output}")
    if args.png or not args.output:
        import matplotlib.pyplot as plt

        import tile_city

        ax = tile_city.draw(best, labels=False)
        if args.png:
            ax.figure.savefig(args.png, dpi=120, bbox_inches="tight")
        else:
            plt.show()
