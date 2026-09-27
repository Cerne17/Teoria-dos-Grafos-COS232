"""Places in a tile city (houses, malls, parks...) and routes between them.

A place is one building: a single tile, every part of a multi-cell building
(mall, skyscraper, slab), or a whole connected park / grey office block. Its
driveways are the road tiles its drawn driveways (D sockets) meet; places
without one fall back to the nearest road tiles: the ones touching it, or the
first ones reached walking over non-road cells.

Names: houses get an owner ("Joe's house"); other places are named after their
district ("Downtown's shopping mall"), which comes from the areas file made in
the builder (<map>.zones.json), or else from the map's compass region
("North-west's park"). Rename places in <map>.places.json:

    {"4,13": "Joe's house", "0,0": "City hall"}      # "row,col" of any of its cells

For a route, the start place gets arcs out to its driveways and the destination
gets arcs in from its driveways, so no route can cut through a building. A
driveway costs DRIVEWAY_METERS plus TILE_METERS per cell walked to reach it.
Everything else is the usual road graph.
"""

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import tiles
from adjacency_list_weighted import AdjacencyList
from tiles import DELTA, EMPTY, SIDES, TILES, CityGraph

DRIVEWAY_METERS = tiles.TILE_METERS / 2
Cell = tuple[int, int]

KINDS = {
    "house_pool": "house", "house_pair": "house", "house_garden": "house",
    "house_row": "row houses", "apartment_a": "apartments", "apartment_b": "apartments",
    "office_glass": "office", "shops": "shops", "block_single": "corner shops",
    "skyscraper_2x2": "skyscraper", "slab_2x1": "warehouse", "slab_1x2": "warehouse",
    "mall_3x2": "shopping mall", "mall_2x3": "shopping mall", "tower": "helipad",
    "plaza": "plaza", "parking_lot": "parking lot", "park": "park", "block": "office block",
}  # fmt: skip
OWNERS = ["Joe", "Ana", "Maria", "Pedro", "Lucas", "Julia", "Rafa", "Bia", "Leo", "Clara",
          "Miguel", "Sofia", "Davi", "Laura", "Theo", "Alice", "Gabi", "Enzo", "Lara", "Nina"]  # fmt: skip


@dataclass
class Place:
    id: int
    name: str
    kind: str
    cells: list[Cell]

    @property
    def center(self) -> Cell:
        """The place's cell nearest its middle (where its marker is drawn)."""
        mr = sum(r for r, _ in self.cells) / len(self.cells)
        mc = sum(c for _, c in self.cells) / len(self.cells)
        return min(self.cells, key=lambda rc: (rc[0] - mr) ** 2 + (rc[1] - mc) ** 2)


def _group(grid, r: int, c: int, socket: str) -> list[Cell]:
    """Cells connected to (r, c) through `socket` sides (a whole park or block)."""
    seen, todo = {(r, c)}, [(r, c)]
    while todo:
        cr, cc = todo.pop()
        for side in SIDES:
            if TILES[grid[cr][cc]].socket(side) != socket:
                continue
            nr, nc = cr + DELTA[side][0], cc + DELTA[side][1]
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                todo.append((nr, nc))
    return sorted(seen)


def _district(cell: Cell, rows: int, cols: int, zones) -> str:
    r, c = cell
    if zones and zones[r][c]:
        return zones[r][c].replace("_", " ").title()
    ns = "North" if r < rows / 3 else "South" if r >= 2 * rows / 3 else ""
    ew = "west" if c < cols / 3 else "east" if c >= 2 * cols / 3 else ""
    return f"{ns}-{ew}" if ns and ew else (ns or ew.title() or "Central")


def find_places(
    grid: Sequence[Sequence[int]],
    zones: Sequence[Sequence[str | None]] | None = None,
    names: dict[str, str] | None = None,
) -> list[Place]:
    """Every place on the map, in reading order, with names."""
    rows, cols = len(grid), len(grid[0])
    taken: set[Cell] = set()
    places: list[Place] = []
    for r in range(rows):
        for c in range(cols):
            t = TILES[grid[r][c]]
            if (r, c) in taken or t.index == EMPTY or t.is_road:
                continue
            if t.part:
                cells = [(rr, cc) for rr, cc, _ in tiles.building_cells(grid, r, c)]
                kind = KINDS.get(t.base, t.base)
            elif "P" in t.sockets or t.base.startswith("park_"):
                cells, kind = _group(grid, r, c, "P"), "park"
            elif "B" in t.sockets or t.base.startswith("block_") and t.base != "block_single":
                cells, kind = _group(grid, r, c, "B"), "office block"
            else:
                cells, kind = [(r, c)], KINDS.get(t.base, t.base.replace("_", " "))
            taken.update(cells)
            places.append(Place(len(places), "", kind, cells))

    # Houses: owners, in a fixed shuffled order so names are stable for a map
    owners = OWNERS[:]
    random.Random(rows * 1000 + cols).shuffle(owners)
    counts: dict[str, int] = {}
    house = 0
    for p in places:
        if p.kind == "house":
            n = house // len(owners)
            p.name = f"{owners[house % len(owners)]}{' ' + str(n + 1) if n else ''}'s house"
            house += 1
        else:
            base = f"{_district(p.center, rows, cols, zones)}'s {p.kind}"
            counts[base] = counts.get(base, 0) + 1
            p.name = base if counts[base] == 1 else f"{base} {counts[base]}"
    # Numbers only where needed: the first of a repeated name also gets "1"
    for p in places:
        if p.kind != "house" and counts.get(p.name, 0) > 1:
            p.name += " 1"

    for key, name in (names or {}).items():
        cell = tuple(int(x) for x in key.split(","))
        for p in places:
            if cell in p.cells:
                p.name = name
    return places


def driveways(city: CityGraph, place: Place, grid=None) -> dict[int, float]:
    """Road vertices the place connects to -> metres to reach them.

    With the grid given, the place's own driveways (a D side meeting a road's
    D side) win. Otherwise: breadth-first walk from the place over non-road
    cells; the first ring of road tiles reached are the driveways (all at the
    same, smallest distance). Empty if nothing can be reached.
    """
    if grid is not None:
        drawn = {}
        for r, c in place.cells:
            for side in SIDES:
                dr, dc = DELTA[side]
                v = city.vertex_at.get((r + dr, c + dc))
                if (v is not None and TILES[grid[r][c]].socket(side) == "D"
                        and tiles.fits(grid[r][c], side, grid[r + dr][c + dc])):  # fmt: skip
                    drawn[v] = DRIVEWAY_METERS
        if drawn:
            return dict(sorted(drawn.items()))
    frontier, seen, walked = list(place.cells), set(place.cells), 0
    while frontier:
        found = {}
        for r, c in frontier:
            for dr, dc in DELTA.values():
                if (v := city.vertex_at.get((r + dr, c + dc))) is not None:
                    found[v] = DRIVEWAY_METERS + walked * tiles.TILE_METERS
        if found:
            return dict(sorted(found.items()))
        nxt = []
        for r, c in frontier:
            for dr, dc in DELTA.values():
                cell = (r + dr, c + dc)
                if cell not in seen and cell in city.walkable:
                    seen.add(cell)
                    nxt.append(cell)
        frontier, walked = nxt, walked + 1
    return {}


def find(places: Sequence[Place], text: str) -> Place:
    """A place by number ('12' or 'P12') or by name (case-insensitive; a unique
    part of the name is enough)."""
    key = text.strip().lower().removeprefix("p")
    if key.isdigit():
        i = int(key) - 1
        if 0 <= i < len(places):
            return places[i]
    exact = [p for p in places if p.name.lower() == text.strip().lower()]
    if exact:
        return exact[0]
    partial = [p for p in places if text.strip().lower() in p.name.lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"no place called {text!r} (list them with --places)")
    names = ", ".join(p.name for p in partial[:6])
    raise ValueError(f"{text!r} matches {len(partial)} places: {names}...")


def with_places(
    city: CityGraph, source: "int | Place", target: "int | Place | None", grid=None
) -> tuple[CityGraph, int, int | None]:
    """The road graph plus vertices for a start and/or destination place.

    A road endpoint stays as it is. A start place only gets arcs out to its
    driveways, a destination place only arcs in, so routes can't pass through
    buildings. Returns (graph, source vertex, target vertex).
    """
    extra = [p for p in (source, target) if isinstance(p, Place)]
    n = len(city.graph)
    graph = AdjacencyList(n + len(extra))
    for u, v, w in city.graph.edges():
        graph.add_edge(u, v, w)
    cells = city.cells + [p.center for p in extra]
    names = dict(city.names)
    ids: list[int | None] = []
    for endpoint, outgoing in ((source, True), (target, False)):
        if not isinstance(endpoint, Place):
            ids.append(endpoint)
            continue
        v = n + extra.index(endpoint)
        names[v] = endpoint.name
        for road, meters in driveways(city, endpoint, grid).items():
            if outgoing:
                graph.add_edge(v, road, meters)
            else:
                graph.add_edge(road, v, meters)
        ids.append(v)
    return CityGraph(graph, cells, city.vertex_at, names, city.walkable), ids[0], ids[1]


def load_names(map_path: Path) -> dict[str, str]:
    path = map_path.with_suffix(".places.json")
    return json.loads(path.read_text()) if path.exists() else {}


def save_names(names: dict[str, str], map_path: Path) -> None:
    map_path.with_suffix(".places.json").write_text(json.dumps(names, indent=2) + "\n")


def load_for(map_path: Path, grid) -> list[Place]:
    """Places of a saved map, using its areas and name overrides if present."""
    zones_path = map_path.with_suffix(".zones.json")
    zones = json.loads(zones_path.read_text())["zones"] if zones_path.exists() else None
    if zones and (len(zones), len(zones[0])) != (len(grid), len(grid[0])):
        zones = None
    return find_places(grid, zones, load_names(map_path))


def resolve(text: str, city: CityGraph, places: Sequence[Place]) -> "int | Place":
    """A route endpoint: 'row,col' (a road tile or any cell of a place), a road
    vertex number ('17'), a place number ('P12') or a place name ('Joe's house')."""
    text = text.strip()
    if "," in text and all(part.strip().isdigit() for part in text.split(",")):
        cell = tuple(int(x) for x in text.split(","))
        if cell in city.vertex_at:
            return city.vertex_at[cell]
        for p in places:
            if cell in p.cells:
                return p
        raise ValueError(f"cell {cell} is empty pavement: neither a road nor a place")
    if text.isdigit():
        v = int(text) - 1
        if not 0 <= v < len(city.cells):
            raise ValueError(f"road vertex {text} out of range 1..{len(city.cells)}")
        return v
    return find(places, text)


def listing(places: Sequence[Place]) -> str:
    return "\n".join(f"P{p.id + 1:<4}{p.name:34s}{p.kind:15s}cell {p.center[0]},{p.center[1]}"
                     for p in places)  # fmt: skip

