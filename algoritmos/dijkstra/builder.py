"""City builder: paint a tile city, fill it with wave function collapse, run Dijkstra.

    python builder.py                             # new 10x16 city -> city/maps/custom.txt
    python builder.py city/maps/sample.txt        # edit an existing map
    python builder.py my.txt --size 12x20         # new map of that size, saved to my.txt
    python builder.py --style downtown            # start with another style

Mouse
    palette (right)       left click: pick a tile; ◀ ▶ buttons or [ ] change page
                          (roads, curb cuts, parks & blocks, buildings, entrances)
    map (left)            left click / drag: paint     right click / drag: erase
                          Delete / Backspace: erase the tile under the mouse
                          Eraser button: left click / drag erases until turned off
                          middle click: pick the tile under the cursor
                          buildings (skyscraper, mall...): a click places the whole
                          building, the clicked cell being the picked part; erasing
                          any part erases the whole building
Buttons / keys
    p    Plan city: planned streets + scenery in the current style (realistic)
    y    next style (city/styles/*.json: tune a city by editing these)
    g    Fill all: keep what you painted, generate the rest with plain WFC (roads too)
    f    Fill scenery: keep every road, generate parks, blocks, lots... in the style's mix
         Press p, g or f again to re-roll: generated cells are redone, painted ones kept.
    z    Areas: drag a rectangle to give it the current style (right-drag clears);
         Plan city and Fill scenery then use each area's style, the rest the current one
    Size box: type ROWSxCOLS and press Enter to resize (content is kept / cropped)
    x    clear generated cells              o    show road sides on the map
    r    rotate the picked tile             n    new random city
    c    clear the map                      u    undo
    s    save                               l    reload from file
    d    Dijkstra: click a start, then a destination (d again or esc to leave).
         Click a building/park for a place (routes go place to place, e.g. from
         Joe's house to the mall) or a road tile. The Name box renames the last
         place clicked (saved in <map>.places.json).
    t    step through Dijkstra from the chosen start (new window)
    e    export the Dijkstra run as an animated GIF (<map>_dijkstra.gif)

Road sides (palette, and map with o): blue dot = two-way, green triangle
pointing in = entrance, red triangle pointing out = exit.
Red X = a side whose socket doesn't fit its neighbour (or runs off the map).
Areas are saved next to the map as <map>.zones.json.
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle
from matplotlib.widgets import Button, TextBox

import places as places_mod
import planner
import tile_city
import tile_city_steps
import tiles
import wfc
from city import shortest_path, shortest_paths
from city_steps import browse
from steps import record_steps
from tiles import EMPTY, TILES

PALETTE_COLS = tiles.ATLAS_COLS
KEYS = set("rgfxoncusldtpyze[]") | {"escape", "backspace", "delete"}
Cell = tuple[int, int]
Zones = list[list[str | None]]
MAX_SIZE = 60
ERASER_OFF, ERASER_ON = "#f2f2f2", "#ffb3b3"
AREA_COLORS = {"downtown": "#7b61ff", "mixed": "#ff9f1c", "suburb": "#2ecc40"}
OTHER_COLORS = ["#e84393", "#00b8d4", "#8d6e63", "#607d8b"]


def _free_keys() -> None:
    """Matplotlib binds many single letters (s = save, g = grid, ...); unbind ours."""
    for name, keys in plt.rcParams.items():
        if name.startswith("keymap."):
            plt.rcParams[name] = [k for k in keys if k not in KEYS]


class Builder:
    def __init__(self, grid: list[list[int]], path: Path, style: str = "mixed"):
        self.grid, self.path = grid, path
        self.styles = planner.style_names()
        self.style_name = style
        self.style = planner.Style.load(style)
        self.rows, self.cols = len(grid), len(grid[0])
        zones_file = self.zones_path()
        self.zones: Zones = (
            planner.load_zones(zones_file)
            if zones_file.exists()
            else [[None] * self.cols for _ in range(self.rows)]
        )
        if (len(self.zones), len(self.zones[0])) != (self.rows, self.cols):
            self.zones = _resized(self.zones, self.rows, self.cols, None)
        self.zone_mode = False
        self.zone_drag: tuple[int, int, int] | None = None  # row, col, mouse button
        self.zone_preview: Rectangle | None = None
        self.selected = tiles.BY_NAME["road_straight_r90"].index
        # Cells the generator filled; re-filling redoes these, never painted ones
        self.generated: set[Cell] = set()
        self.history: list[tuple[list[list[int]], set[Cell], Zones]] = []
        self.flagged: list[tuple[int, int, str]] = []  # sides blamed by a failed fill
        self.show_sockets = False
        self.eraser = False  # while on, left click / drag erases instead of painting
        self.hovered: Cell | None = None  # map cell under the mouse, for Delete
        self.painting: int | None = None  # tile being dragged, if any
        self.dijkstra_mode = False
        # Route endpoints: a road vertex or a place (see places.py)
        self.source: int | places_mod.Place | None = None
        self.target: int | places_mod.Place | None = None
        self.place_names = places_mod.load_names(path)  # "row,col" -> custom name
        self.last_place: places_mod.Place | None = None
        self.message = "pick a tile on the right, paint on the left"
        self.rng = random.Random()
        self.step_windows: list[plt.Figure] = []  # keep open viewers alive

        self.pages = palette_pages()
        self.page = 0
        self.palette: list[list[int | None]] = []  # current page as a grid; None = padding

        map_w = 11
        self.fig = plt.figure(
            figsize=(map_w + 5.5, max(8, map_w * self.rows / self.cols + 1.2)),
            layout="constrained",
        )
        axes = self.fig.subplot_mosaic(
            [["map", "pal"], ["map", "buttons"]], width_ratios=[map_w, 5], height_ratios=[4, 3.4]
        )
        self.map_ax, self.pal_ax = axes["map"], axes["pal"]
        self.buttons = self._make_buttons(axes["buttons"])
        self.status = self.fig.text(0.01, 0.005, "", fontsize=9, family="monospace")
        for event, handler in [
            ("button_press_event", self.on_press),
            ("button_release_event", self.on_release),
            ("motion_notify_event", self.on_motion),
            ("key_press_event", self.on_key),
        ]:
            self.fig.canvas.mpl_connect(event, handler)
        self.draw_palette()
        self.redraw()

    def _make_buttons(self, ax: plt.Axes) -> list[Button]:
        ax.set_axis_off()
        specs = [
            ("◀ tiles  ([)", lambda: self.turn_page(-1), "#f2f2f2"),
            ("tiles ▶  (])", lambda: self.turn_page(1), "#f2f2f2"),
            ("Plan city  (p)", self.plan_city, "#ffe3b3"),
            (self._style_label(), self.next_style, "#ffe3b3"),
            ("Fill all  (g)", self.fill_all, "#cfe8ff"),
            ("Fill scenery  (f)", self.fill_scenery, "#d8f0c8"),
            ("Clear generated  (x)", self.clear_generated, "#f2f2f2"),
            ("Road sides on map  (o)", self.toggle_sockets, "#f2f2f2"),
            ("Areas  (z)", self.toggle_zones, "#ffe3b3"),
            ("Undo  (u)", self.undo, "#f2f2f2"),
            ("Save  (s)", self.save, "#f2f2f2"),
            ("Export GIF  (e)", self.export_gif, "#e8d8ff"),
            (self._eraser_label(), self.toggle_eraser, ERASER_OFF),
        ]
        buttons = []
        # Two per row; the size box takes the slot after the last button
        for k, (label, action, color) in enumerate(specs):
            row, col = divmod(k, 2)
            bax = ax.inset_axes([col * 0.51, 0.9 - row * 0.127, 0.49, 0.1])
            button = Button(bax, label, color=color, hovercolor="white")
            button.label.set_fontsize(9)
            button.on_clicked(lambda _event, action=action: action())
            buttons.append(button)  # keep a reference or the button stops working
            if action == self.next_style:
                self.style_button = button
            if action == self.toggle_eraser:
                self.eraser_button = button
        row = -(-len(specs) // 2)  # the row after the buttons
        self.name_box = TextBox(ax.inset_axes([0.62, 0.9 - row * 0.127, 0.38, 0.1]), "Name  ",
                                initial="(click a place in d mode)")  # fmt: skip
        self.name_box.on_submit(self.rename_place)
        self.size_box = TextBox(ax.inset_axes([0.11, 0.9 - row * 0.127, 0.38, 0.1]),
                                "Size  ",
                                initial=f"{self.rows}x{self.cols}")  # fmt: skip
        self.size_box.on_submit(self.resize)
        return buttons

    # ---- drawing -------------------------------------------------------------------

    def draw_palette(self) -> None:
        ax = self.pal_ax
        ax.clear()
        title, indices = self.pages[self.page]
        rows = -(-len(indices) // PALETTE_COLS)
        self.palette = [[indices[k] if (k := r * PALETTE_COLS + c) < len(indices) else None
                         for c in range(PALETTE_COLS)] for r in range(rows)]  # fmt: skip
        shown = [[EMPTY if t is None else t for t in row] for row in self.palette]
        ax.imshow(tiles.render(shown), extent=(0, PALETTE_COLS, rows, 0), interpolation="nearest")
        tile_city.draw_sockets(ax, shown, size=4.5)
        # Separators, so each marker clearly belongs to one tile
        ax.hlines(range(1, rows), 0, PALETTE_COLS, color="white", linewidth=2)
        ax.vlines(range(1, PALETTE_COLS), 0, rows, color="white", linewidth=2)
        if self.selected in indices:
            r, c = divmod(indices.index(self.selected), PALETTE_COLS)
            ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor="red", linewidth=2.5))
        ax.set_xlim(0, PALETTE_COLS)
        ax.set_ylim(rows, 0)
        ax.set_axis_off()
        t = TILES[self.selected]
        ax.set_title(f"page {self.page + 1}/{len(self.pages)}: {title}   ◀ ▶ or [ ]\n"
                     f"picked {t.index}: {t.name} — {t.describe()}", fontsize=9)  # fmt: skip
        tile_city.socket_legend(ax, loc="upper center", bbox_to_anchor=(0.5, 0.0))

    def places(self) -> list[places_mod.Place]:
        return places_mod.find_places(self.grid, self.zones, self.place_names)

    def routed(self):
        """(graph with the chosen place endpoints, source vertex, target vertex, places)."""
        city = tiles.to_graph(self.grid)
        routed, source, target = places_mod.with_places(city, self.source, self.target, self.grid)
        endpoints = [p for p in (self.source, self.target) if isinstance(p, places_mod.Place)]
        return routed, source, target, endpoints

    def select(self, index: int) -> None:
        """Picks a tile and shows the palette page it is on."""
        self.selected = index
        self.page = next(i for i, (_, ids) in enumerate(self.pages) if index in ids)
        self.draw_palette()
        self.update_status()

    def turn_page(self, step: int) -> None:
        self.page = (self.page + step) % len(self.pages)
        self.draw_palette()
        self.fig.canvas.draw_idle()

    def redraw(self) -> None:
        self.map_ax.clear()
        city = tiles.to_graph(self.grid)
        kwargs: dict = {"city": city, "bad_sides": tiles.mismatches(self.grid) + self.flagged}
        title = f"{self.path.name} — {self.rows}x{self.cols}"
        if self.dijkstra_mode:
            kwargs["disabled"] = [p for p in self.places() if not p.accessible]
        if self.dijkstra_mode and self.source is not None:
            city, source, target, endpoints = self.routed()
            distances, parent = shortest_paths(city.graph, source)
            kwargs |= {"city": city, "source": source, "distances": distances,
                       "places": endpoints}  # fmt: skip
            if target is None:
                kwargs["tree_edges"] = [(p, v) for v, p in enumerate(parent) if p is not None]
                title = f"shortest paths from {city.label(source)} — click a destination"
            else:
                dist, path = shortest_path(city.graph, source, target)
                if path:
                    on_path = set(path)
                    kwargs["path"] = path
                    kwargs["distances"] = [d if v in on_path else float("inf")
                                           for v, d in enumerate(distances)]  # fmt: skip
                    title = f"{city.label(source)} → {city.label(target)}: {dist:g} m"
                else:
                    title = f"{city.label(target)} is unreachable from {city.label(source)}"
        elif self.dijkstra_mode:
            title = "Dijkstra mode — click a start: a building, park or road tile (grey = no access)"
        tile_city.draw(self.grid, title=title, ax=self.map_ax, **kwargs)
        if self.show_sockets:
            tile_city.draw_sockets(self.map_ax, self.grid)
        if self.zone_mode:
            self.draw_zones()
        self.update_status()

    def _area_color(self, name: str) -> str:
        others = [n for n in self.styles if n not in AREA_COLORS]
        return AREA_COLORS.get(name) or OTHER_COLORS[others.index(name) % len(OTHER_COLORS)
                                                     if name in others else 0]  # fmt: skip

    def draw_zones(self) -> None:
        """Tints each area with its style's colour; untinted cells use the current style."""
        overlay = [[to_rgba(self._area_color(z), 0.38) if z else (0, 0, 0, 0) for z in row]
                   for row in self.zones]  # fmt: skip
        self.map_ax.imshow(overlay, extent=(0, self.cols, self.rows, 0),
                           interpolation="nearest", zorder=9)  # fmt: skip
        used = sorted({z for row in self.zones for z in row if z})
        handles = [Patch(color=self._area_color(n), alpha=0.6, label=n) for n in used]
        handles.append(Patch(facecolor="none", edgecolor="gray",
                             label=f"no area: current style ({self.style_name})"))  # fmt: skip
        self.map_ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                           ncol=len(handles), fontsize=8, frameon=False)  # fmt: skip

    def update_status(self, hover: str = "") -> None:
        city = tiles.to_graph(self.grid)
        roads, scc = wfc.connectivity(self.grid)
        bad = len(tiles.mismatches(self.grid))
        self.status.set_text(
            f"road tiles {roads}  arcs {city.graph.edge_count}  largest strongly connected "
            f"{scc}  mismatched sides {bad}   |  {self.message}"
            + (f"   |  {hover}" if hover else "")
        )
        self.fig.canvas.draw_idle()

    # ---- editing -------------------------------------------------------------------

    def snapshot(self) -> None:
        self.history.append(
            ([row[:] for row in self.grid], set(self.generated), [row[:] for row in self.zones])
        )
        del self.history[:-100]

    def set_grid(
        self, grid: list[list[int]], message: str, generated: set[Cell] | None = None
    ) -> None:
        self.snapshot()
        self.grid = grid
        self.generated = generated or set()
        self.flagged = []
        self.source = self.target = None
        self.message = message
        self.redraw()

    def place_building(self, r: int, c: int, part: int) -> None:
        """Places every part of part's building so that part lands on (r, c)."""
        t = TILES[part]
        x, y, w, h = t.part
        top, left = r - y, c - x
        if top < 0 or left < 0 or top + h > self.rows or left + w > self.cols:
            self.message = f"{t.base} ({w}x{h}) doesn't fit there"
            self.update_status()
            return
        for py in range(h):
            for px in range(w):
                cell = (top + py, left + px)
                # Replacing part of another building removes that building
                if TILES[self.grid[cell[0]][cell[1]]].part:
                    self._erase_building(*cell)
                self.grid[cell[0]][cell[1]] = tiles.BY_NAME[f"{t.base}_{px}_{py}"].index
                self.generated.discard(cell)
        self.flagged = []
        self.source = self.target = None
        self.message = f"placed {t.base}"
        self.redraw()

    def _erase_building(self, r: int, c: int) -> None:
        for rr, cc, _ in tiles.building_cells(self.grid, r, c):
            self.grid[rr][cc] = EMPTY
            self.generated.discard((rr, cc))

    def paint(self, r: int, c: int, tile: int) -> None:
        if TILES[self.grid[r][c]].part and self.grid[r][c] != tile:
            self._erase_building(r, c)  # never leave half a building behind
        # Painting a generated cell makes it yours: re-fills won't touch it
        self.generated.discard((r, c))
        if self.grid[r][c] != tile or self.flagged:
            self.grid[r][c] = tile
            self.flagged = []
            self.source = self.target = None
            self.redraw()

    def _painted(self) -> dict[Cell, int]:
        """Cells the user placed: not empty and not produced by the generator."""
        return {
            (r, c): t
            for r, row in enumerate(self.grid)
            for c, t in enumerate(row)
            if t != EMPTY and (r, c) not in self.generated
        }

    def _style_label(self) -> str:
        return f"Style: {self.style_name}  (y)"

    def next_style(self) -> None:
        i = self.styles.index(self.style_name) if self.style_name in self.styles else -1
        self.style_name = self.styles[(i + 1) % len(self.styles)]
        self.style = planner.Style.load(self.style_name)
        self.style_button.label.set_text(self._style_label())
        self.message = f"style {self.style_name} — press p to plan a city, f for its scenery"
        self.update_status()

    def _zone_styles(self) -> list[list[planner.Style | None]] | None:
        if not any(z for row in self.zones for z in row):
            return None
        loaded = {self.style_name: self.style}  # the current style keeps its session tweaks
        return [[None if z is None else loaded.setdefault(z, planner.Style.load(z)) for z in row]
                for row in self.zones]  # fmt: skip

    def plan_city(self) -> None:
        grid = planner.plan(self.rows, self.cols, self.style, self.rng.randrange(2**32),
                            self._zone_styles())  # fmt: skip
        cells = {(r, c) for r in range(self.rows) for c in range(self.cols)}
        self.set_grid(grid, f"planned a {self.style_name} city — press again to re-roll", cells)

    def _generate(self, fixed: dict, what: str, scenery_only: bool = False) -> None:
        weights = self.style.scenery_weights() if scenery_only else wfc.DEFAULT_WEIGHTS
        cell_weights = None
        if scenery_only and (zones := self._zone_styles()):
            tables: dict[int, dict] = {id(self.style): weights}
            cell_weights = [[tables.setdefault(id(st or self.style), (st or self.style).scenery_weights())
                             for st in row] for row in zones]  # fmt: skip
        try:
            grid = wfc.generate(self.rows, self.cols, self.rng.randrange(2**32), weights, fixed,
                                cell_weights=cell_weights)  # fmt: skip
        except wfc.Contradiction as e:
            self.flagged = self._loose_ends(fixed) if scenery_only else []
            hint = (f"{len(self.flagged)} road ends lead nowhere (X) — finish them or use Fill all"
                    if self.flagged else f"kept tiles conflict: {e}")  # fmt: skip
            self.message = f"can't {what}: {hint}"
            self.redraw()
            return
        cells = {(r, c) for r in range(self.rows) for c in range(self.cols)} - fixed.keys()
        self.set_grid(grid, f"{what}: {len(cells)} cells generated — press again to re-roll", cells)

    def fill_all(self) -> None:
        self._generate(self._painted(), "fill all")

    def fill_scenery(self) -> None:
        # Roads stay, but may gain or lose curb cuts to match the new entrances
        roads = {
            (r, c): tiles.with_driveways(tiles.without_driveways(t))
            for r, row in enumerate(self.grid)
            for c, t in enumerate(row)
            if TILES[t].is_road
        }
        painted = {cell: t for cell, t in self._painted().items() if not TILES[t].is_road}
        self._generate(painted | roads, "fill scenery", scenery_only=True)

    def _loose_ends(self, fixed: dict) -> list[tuple[int, int, str]]:
        """Road sides of kept tiles that face a cell only scenery could fill (or the edge)."""
        loose = []
        for (r, c), t in fixed.items():
            t = t if isinstance(t, int) else t[0]  # a kept road with its driveway variants
            for side, sock in zip(tiles.SIDES, TILES[t].sockets):
                if sock in "RIO":
                    dr, dc = tiles.DELTA[side]
                    if (r + dr, c + dc) not in fixed:
                        loose.append((r, c, side))
        return loose

    def clear_generated(self) -> None:
        grid = [row[:] for row in self.grid]
        for r, c in self.generated:
            grid[r][c] = EMPTY
        self.set_grid(grid, f"cleared {len(self.generated)} generated cells")

    def _eraser_label(self) -> str:
        return f"Eraser: {'ON — click/drag erases' if self.eraser else 'off'}  (⌫ erases hovered)"

    def toggle_eraser(self) -> None:
        self.eraser = not self.eraser
        self.eraser_button.label.set_text(self._eraser_label())
        color = ERASER_ON if self.eraser else ERASER_OFF
        self.eraser_button.color = color  # colour while not hovered
        self.eraser_button.ax.set_facecolor(color)
        self.message = ("eraser on: left click or drag erases (buildings go whole)"
                        if self.eraser else "eraser off: left click paints the picked tile")  # fmt: skip
        self.update_status()

    def erase_at(self, r: int, c: int) -> None:
        """Deletes the tile at (r, c) — the whole building, if it is part of one."""
        if self.grid[r][c] == EMPTY:
            return
        name = TILES[self.grid[r][c]].base
        self.snapshot()
        self.paint(r, c, EMPTY)
        self.message = f"erased {name} at {r},{c} (u to undo)"
        self.update_status()

    def toggle_sockets(self) -> None:
        self.show_sockets = not self.show_sockets
        self.redraw()

    def undo(self) -> None:
        if not self.history:
            self.message = "nothing to undo"
            self.update_status()
            return
        self.grid, self.generated, self.zones = self.history.pop()
        self.rows, self.cols = len(self.grid), len(self.grid[0])
        self._show_size()
        self.flagged = []
        self.source = self.target = None
        self.message = "undone"
        self.redraw()

    def zones_path(self) -> Path:
        return self.path.with_suffix(".zones.json")

    def save(self) -> None:
        tiles.save_map(self.grid, self.path)
        saved = str(self.path)
        if self.place_names:
            places_mod.save_names(self.place_names, self.path)
            saved += f", {self.path.with_suffix('.places.json').name}"
        if any(z for row in self.zones for z in row):
            planner.save_zones(self.zones, self.zones_path())
            saved += f" and {self.zones_path().name}"
        self.message = f"saved {saved}"
        self.update_status()

    # ---- areas and size ----------------------------------------------------------

    def toggle_zones(self) -> None:
        self.zone_mode = not self.zone_mode
        self.dijkstra_mode = False
        self.message = (f"areas: drag to set {self.style_name} (y picks the style), "
                        "right-drag to clear" if self.zone_mode else "paint mode")  # fmt: skip
        self.redraw()

    def _zone_bounds(self, r: int, c: int) -> tuple[int, int, int, int]:
        r0, c0, _ = self.zone_drag
        return min(r0, r), min(c0, c), max(r0, r), max(c0, c)

    def _zone_drag_to(self, r: int, c: int) -> None:
        r0, c0, r1, c1 = self._zone_bounds(r, c)
        if self.zone_preview is None:
            self.zone_preview = Rectangle((0, 0), 1, 1, fill=False, linewidth=2.5,
                                          linestyle="--", edgecolor="black", zorder=10)  # fmt: skip
            self.map_ax.add_patch(self.zone_preview)
        self.zone_preview.set_bounds(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
        self.fig.canvas.draw_idle()

    def _zone_apply(self, r: int, c: int) -> None:
        r0, c0, r1, c1 = self._zone_bounds(r, c)
        name = self.style_name if self.zone_drag[2] == 1 else None
        self.snapshot()
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                self.zones[rr][cc] = name
        size = (r1 - r0 + 1) * (c1 - c0 + 1)
        self.message = (f"{size} cells set to {name} — press p to plan" if name
                        else f"cleared the area of {size} cells")  # fmt: skip
        self.zone_drag = self.zone_preview = None
        self.redraw()

    def resize(self, text: str) -> None:
        if self._syncing_size:
            return
        try:
            rows, cols = parse_size(text)
            if not (3 <= rows <= MAX_SIZE and 3 <= cols <= MAX_SIZE):
                raise ValueError
        except ValueError:
            self.message = f"size must look like 12x20, each between 3 and {MAX_SIZE}"
            self._show_size()
            self.update_status()
            return
        if (rows, cols) == (self.rows, self.cols):
            return
        self.snapshot()
        self.grid = _resized(self.grid, rows, cols, EMPTY)
        self.zones = _resized(self.zones, rows, cols, None)
        self.generated = {(r, c) for r, c in self.generated if r < rows and c < cols}
        self.rows, self.cols = rows, cols
        self.source = self.target = None
        self.flagged = []
        self.message = f"resized to {rows}x{cols} (cropped cells can be undone)"
        self.redraw()

    _syncing_size = False

    def _show_size(self) -> None:
        """Shows the current size in the box without triggering a resize."""
        self._syncing_size = True
        try:
            self.size_box.set_val(f"{self.rows}x{self.cols}")
        finally:
            self._syncing_size = False

    # ---- events --------------------------------------------------------------------

    def _cell(self, event, ax) -> tuple[int, int] | None:
        if event.inaxes is not ax or event.xdata is None:
            return None
        r, c = int(event.ydata), int(event.xdata)
        rows = self.rows if ax is self.map_ax else len(self.palette)
        cols = self.cols if ax is self.map_ax else PALETTE_COLS
        return (r, c) if 0 <= r < rows and 0 <= c < cols else None

    def on_press(self, event) -> None:
        if (cell := self._cell(event, self.pal_ax)) is not None:
            index = self.palette[cell[0]][cell[1]]
            if index is not None:
                self.selected = index
                self.draw_palette()
                self.update_status()
            return
        if (cell := self._cell(event, self.map_ax)) is None:
            return
        r, c = cell
        if self.zone_mode:
            if event.button in (1, 3):
                self.zone_drag = (r, c, event.button)
                self._zone_drag_to(r, c)
            return
        if self.dijkstra_mode:
            self.pick_endpoint(r, c)
        elif event.button == 2:
            self.select(self.grid[r][c])
            self.update_status()
        else:
            self.snapshot()
            erase = event.button == 3 or self.eraser
            if not erase and TILES[self.selected].part:
                self.place_building(r, c, self.selected)  # one building per click, no drag
                return
            self.painting = EMPTY if erase else self.selected
            self.paint(r, c, self.painting)

    def on_release(self, event) -> None:
        self.painting = None
        if self.zone_drag is not None:
            # Released outside the map: use the last cell the preview reached
            cell = self._cell(event, self.map_ax)
            if cell is None and self.zone_preview is not None:
                x, y = self.zone_preview.get_xy()
                w, h = self.zone_preview.get_width(), self.zone_preview.get_height()
                r0, c0, _ = self.zone_drag
                cell = (int(y + h - 1) if r0 == int(y) else int(y),
                        int(x + w - 1) if c0 == int(x) else int(x))  # fmt: skip
            self._zone_apply(*(cell or self.zone_drag[:2]))

    def on_motion(self, event) -> None:
        self.hovered = self._cell(event, self.map_ax)
        if (cell := self.hovered) is not None:
            r, c = cell
            if self.painting is not None:
                self.paint(r, c, self.painting)
            if self.zone_drag is not None:
                self._zone_drag_to(r, c)
            t = TILES[self.grid[r][c]]
            origin = " (generated)" if (r, c) in self.generated else ""
            if self.zones[r][c]:
                origin += f" [area: {self.zones[r][c]}]"
            self.update_status(f"cell {r},{c}{origin}: {t.name} — {t.describe()}")
        elif (cell := self._cell(event, self.pal_ax)) is not None:
            index = self.palette[cell[0]][cell[1]]
            if index is not None:
                t = TILES[index]
                self.update_status(f"tile {t.index} {t.name} — {t.describe()}")

    def pick_endpoint(self, r: int, c: int) -> None:
        try:
            endpoint = places_mod.resolve(f"{r},{c}", tiles.to_graph(self.grid), self.places())
        except ValueError as e:
            # Empty pavement, or a disabled place that no driveway reaches
            self.message = str(e)
            self.update_status()
            return
        if isinstance(endpoint, places_mod.Place):
            self.last_place = endpoint
            self.name_box.set_val(endpoint.name)
        if self.source is None or self.target is not None:
            self.source, self.target = endpoint, None
            self.message = "start set — click a destination; t steps through, e exports a GIF"
        else:
            self.target = endpoint
            self.message = "click another tile to start over; t steps through, e exports a GIF"
        self.redraw()

    def rename_place(self, name: str) -> None:
        name = name.strip()
        if self.last_place is None or not name or name == self.last_place.name:
            return
        r, c = self.last_place.cells[0]
        self.place_names[f"{r},{c}"] = name
        self.message = f"renamed to {name!r} (saved with s)"
        # Endpoints are Place objects: refresh them so the new name shows
        for attr in ("source", "target"):
            p = getattr(self, attr)
            if isinstance(p, places_mod.Place) and p.cells == self.last_place.cells:
                setattr(self, attr, next(q for q in self.places() if q.cells == p.cells))
        self.last_place = next(q for q in self.places() if q.cells == self.last_place.cells)
        self.redraw()

    def _run(self):
        """Dijkstra from the chosen start: (grid, graph, steps, target, places) or None."""
        if self.source is None:
            self.message = "press d and click a start (a place or road tile) first"
            self.update_status()
            return None
        grid = [row[:] for row in self.grid]
        city, source, target, endpoints = self.routed()
        self.disabled_now = [p for p in self.places() if not p.accessible]
        return grid, city, record_steps(city.graph, source), source, target, endpoints

    def open_steps(self) -> None:
        if (run := self._run()) is None:
            return
        grid, city, steps, source, target, endpoints = run
        disabled = self.disabled_now

        def draw_panel(i: int, ax: plt.Axes) -> None:
            tile_city_steps.draw_step(grid, city, steps, i, ax, target, endpoints, disabled)

        fig = browse(len(steps), draw_panel, f"Dijkstra from {city.label(source)}",
                     figsize=(12, 12 * self.rows / self.cols + 0.8), block=False)  # fmt: skip
        self.step_windows.append(fig)

    def export_gif(self) -> None:
        if (run := self._run()) is None:
            return
        grid, city, steps, source, target, endpoints = run
        out = self.path.with_name(f"{self.path.stem}_dijkstra.gif")
        try:
            tile_city_steps.export_gif(out, grid, city, steps, source, target,
                                       title=f"Dijkstra on {self.path.stem} from {city.label(source)}",
                                       places=endpoints, disabled=self.disabled_now)  # fmt: skip
        except Exception as e:  # keep the builder alive whatever goes wrong
            self.message = f"GIF export failed: {type(e).__name__}: {e}"
        else:
            self.message = f"saved {out} ({len(steps)} frames)"
        self.update_status()

    def on_key(self, event) -> None:
        if self.size_box.capturekeystrokes or self.name_box.capturekeystrokes:
            return  # typing in a text box, not a shortcut
        key = event.key
        if key == "r":
            options = tiles.rotations_of(self.selected)
            self.select(options[(options.index(self.selected) + 1) % len(options)])
        elif key in ("backspace", "delete"):
            if self.hovered is None:
                self.message = "point at a map cell, then press Delete to erase it"
                self.update_status()
            else:
                self.erase_at(*self.hovered)
        elif key in ("[", "]"):
            self.turn_page(-1 if key == "[" else 1)
        elif key == "p":
            self.plan_city()
        elif key == "z":
            self.toggle_zones()
        elif key == "y":
            self.next_style()
        elif key == "g":
            self.fill_all()
        elif key == "f":
            self.fill_scenery()
        elif key == "x":
            self.clear_generated()
        elif key == "o":
            self.toggle_sockets()
        elif key == "n":
            self._generate({}, "new random city")
        elif key == "c":
            self.set_grid(tiles.blank(self.rows, self.cols), "cleared")
        elif key == "u":
            self.undo()
        elif key == "s":
            self.save()
        elif key == "l" and self.path.exists():
            grid = tiles.load_map(self.path)
            self.set_grid(grid, f"reloaded {self.path}")
            self.rows, self.cols = len(grid), len(grid[0])
            zones_file = self.zones_path()
            self.zones = (planner.load_zones(zones_file) if zones_file.exists()
                          else [[None] * self.cols for _ in range(self.rows)])  # fmt: skip
            self._show_size()
            self.redraw()
        elif key == "d" or (key == "escape" and self.dijkstra_mode):
            self.dijkstra_mode = key == "d" and not self.dijkstra_mode
            self.zone_mode = False
            self.source = self.target = None
            self.message = "Dijkstra mode" if self.dijkstra_mode else "paint mode"
            self.redraw()
        elif key == "t":
            self.open_steps()
        elif key == "e":
            self.export_gif()


def palette_pages() -> list[tuple[str, list[int]]]:
    """The palette split into pages, so 200+ tiles stay big enough to see."""
    def has_d(t):
        return "D" in t.sockets

    scenery = ("park_", "block_", "parking_lot", "plaza", "tower")
    pages = [
        ("roads", [t.index for t in TILES if t.is_road and not has_d(t)]),
        ("roads with curb cuts", [t.index for t in TILES if t.is_road and has_d(t)]),
        ("parks, blocks, lots, plazas",
         [t.index for t in TILES if not t.is_road and not has_d(t)
          and (t.base.startswith(scenery) or t.index == EMPTY)]),
        ("buildings",
         [t.index for t in TILES if not t.is_road and not has_d(t)
          and not t.base.startswith(scenery) and t.index != EMPTY]),
        ("with entrances (driveways)", [t.index for t in TILES if not t.is_road and has_d(t)]),
    ]  # fmt: skip
    assert sorted(i for _, ids in pages for i in ids) == list(range(len(TILES)))
    return pages


def _resized(grid: list[list], rows: int, cols: int, fill) -> list[list]:
    """grid cropped or padded with `fill` to rows x cols, keeping the top-left."""
    return [[grid[r][c] if r < len(grid) and c < len(grid[0]) else fill for c in range(cols)]
            for r in range(rows)]  # fmt: skip


def parse_size(text: str) -> tuple[int, int]:
    rows, cols = (int(x) for x in text.lower().split("x"))
    return rows, cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=__doc__.split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("map", nargs="?", type=Path, default=tiles.MAPS / "custom.txt",
                        help="map to edit; created on save if missing")  # fmt: skip
    parser.add_argument("--size", type=parse_size, default=(10, 16),
                        help="ROWSxCOLS for a new map (default 10x16)")  # fmt: skip
    parser.add_argument("--style", default="mixed",
                        help=f"planner style: {planner.style_names()} or a .json path")  # fmt: skip
    args = parser.parse_args()

    _free_keys()
    grid = tiles.load_map(args.map) if args.map.exists() else tiles.blank(*args.size)
    builder = Builder(grid, args.map, args.style)
    plt.show()
