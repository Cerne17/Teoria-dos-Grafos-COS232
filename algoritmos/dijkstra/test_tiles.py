"""Run with: python -m unittest -v test_tiles"""

import unittest

import networkx as nx

import planner
import tiles
import wfc
from dijkstra import Dijkstra
from tiles import BY_NAME, EMPTY, TILES
from viz import to_networkx

SAMPLE = tiles.MAPS / "sample.txt"


def tile(name: str) -> int:
    return BY_NAME[name].index


class TestCatalogue(unittest.TestCase):
    def test_atlas_order(self):
        # Spot checks against city/preview.png
        for index, name in [(0, "road_straight_r0"), (17, "roundabout_4"),
                            (34, "oneway_cross_r0"), (49, "mixed_cross_r270"),
                            (65, "park_center"), (81, "block_center"), (85, "tower")]:  # fmt: skip
            self.assertEqual(TILES[index].name, name)
        self.assertEqual(TILES[EMPTY].name, "empty")

    def test_rotating_clockwise_moves_north_to_east(self):
        self.assertEqual(BY_NAME["road_curve_r0"].sockets, ".RR.")
        self.assertEqual(BY_NAME["road_curve_r90"].sockets, "..RR")
        # N E S W: enters on W (I), leaves on E (O), so it drives W -> E like tile 23's arrows
        self.assertEqual(BY_NAME["oneway_straight_r90"].sockets, ".O.I")

    def test_one_way_ends_only_fit_opposite_ends(self):
        up = tile("oneway_straight_r0")  # S -> N
        self.assertTrue(tiles.fits(up, "N", up))
        self.assertFalse(tiles.fits(up, "N", tile("oneway_straight_r180")))
        self.assertFalse(tiles.fits(up, "N", tile("road_straight_r0")))

    def test_every_tile_image_is_32px(self):
        self.assertEqual(len(tiles.tile_images()), len(TILES))
        for image in tiles.tile_images():
            self.assertEqual(image.shape, (32, 32, 3))


class TestSampleCity(unittest.TestCase):
    def setUp(self):
        self.grid = tiles.load_map(SAMPLE)
        self.city = tiles.to_graph(self.grid)

    def test_map_file_matches_the_picture(self):
        self.assertEqual(tiles.from_image(tiles.ASSETS / "sample_city.png"), self.grid)

    def test_every_side_fits(self):
        self.assertEqual(tiles.mismatches(self.grid), [])

    def test_graph(self):
        self.assertEqual(len(self.city.graph), 103)
        self.assertTrue(nx.is_strongly_connected(to_networkx(self.city.graph)))
        for u, v, w in self.city.graph.edges():
            (r0, c0), (r1, c1) = self.city.cells[u], self.city.cells[v]
            self.assertEqual(abs(r0 - r1) + abs(c0 - c1), 1)  # neighbours only
            self.assertEqual(w, tiles.TILE_METERS)

    def test_dijkstra_matches_networkx(self):
        g = to_networkx(self.city.graph)
        for source in range(0, len(self.city.graph), 10):
            expected = nx.single_source_dijkstra_path_length(g, source + 1)
            self.assertEqual(
                Dijkstra(self.city.graph, source),
                [expected.get(v + 1, float("inf")) for v in range(len(self.city.graph))],
            )


class TestGraph(unittest.TestCase):
    def test_one_way_street_has_one_direction(self):
        up = tile("oneway_straight_r0")
        city = tiles.to_graph([[up], [up]])
        top, bottom = city.vertex_at[(0, 0)], city.vertex_at[(1, 0)]
        self.assertTrue(city.graph.has_edge(bottom, top))
        self.assertFalse(city.graph.has_edge(top, bottom))

    def test_two_way_street_goes_both_ways(self):
        city = tiles.to_graph([[tile("road_straight_r90")] * 3])
        self.assertEqual(sorted((u, v) for u, v, _ in city.graph.edges()),
                         [(0, 1), (1, 0), (1, 2), (2, 1)])  # fmt: skip

    def test_mismatched_sides_give_no_arc(self):
        city = tiles.to_graph([[tile("road_straight_r90"), tile("road_straight_r0")]])
        self.assertEqual(city.graph.edge_count, 0)

    def test_save_and_load_round_trip(self):
        path = tiles.MAPS / "_test_roundtrip.txt"
        grid = [[EMPTY, 12], [3, EMPTY]]
        try:
            tiles.save_map(grid, path)
            self.assertEqual(tiles.load_map(path), grid)
        finally:
            path.unlink(missing_ok=True)


class TestWaveFunctionCollapse(unittest.TestCase):
    def test_generated_cities_always_fit(self):
        for seed in range(15):
            for rows, cols in [(4, 4), (8, 12)]:
                with self.subTest(seed=seed, size=(rows, cols)):
                    grid = wfc.generate(rows, cols, seed)
                    self.assertEqual(len(grid), rows)
                    self.assertEqual(tiles.mismatches(grid), [])

    def test_same_seed_same_city(self):
        self.assertEqual(wfc.generate(6, 9, 42), wfc.generate(6, 9, 42))

    def test_fixed_cells_are_kept(self):
        road = tile("road_straight_r90")
        fixed = {(3, c): road for c in range(2, 8)}
        grid = wfc.generate(7, 10, 1, fixed=fixed)
        for (r, c), t in fixed.items():
            self.assertEqual(grid[r][c], t)
        self.assertEqual(tiles.mismatches(grid), [])

    def test_impossible_fixed_cells_raise(self):
        # A road pointing off the map edge can never fit
        with self.assertRaises(wfc.Contradiction):
            wfc.generate(3, 3, 0, fixed={(0, 1): tile("road_straight_r0")}, attempts=3)


class TestPlanner(unittest.TestCase):
    def test_every_style_plans_fitting_fully_connected_cities(self):
        for name in planner.style_names():
            style = planner.Style.load(name)
            for seed in range(4):
                with self.subTest(style=name, seed=seed):
                    grid = planner.plan(12, 18, style, seed)
                    self.assertEqual(tiles.mismatches(grid), [])
                    roads, largest = wfc.connectivity(grid)
                    self.assertGreater(roads, 0)
                    self.assertEqual(roads, largest)  # every road tile reaches every other

    def test_same_seed_same_city(self):
        style = planner.Style.load("mixed")
        self.assertEqual(planner.plan(10, 14, style, 5), planner.plan(10, 14, style, 5))

    def test_oneway_knob_creates_one_way_streets(self):
        style = planner.Style.load("downtown").with_overrides(["oneway=1"])
        grid = planner.plan(12, 18, style, 1)
        self.assertTrue(any("I" in TILES[t].sockets for row in grid for t in row))
        none = planner.Style.load("downtown").with_overrides(["oneway=0"])
        grid = planner.plan(12, 18, none, 1)
        self.assertFalse(any("I" in TILES[t].sockets for row in grid for t in row))

    def test_overrides(self):
        style = planner.Style.load("mixed").with_overrides(
            ["max_block=6", "ring=false", "scenery.park_center=9"]
        )
        self.assertEqual((style.max_block, style.ring), (6, False))
        self.assertEqual(style.scenery["park_center"], 9.0)
        with self.assertRaises(ValueError):
            style.with_overrides(["nonsense=1"])

    def test_without_ring_roads_stay_inside_the_map(self):
        style = planner.Style.load("mixed").with_overrides(["ring=false"])
        for seed in range(3):
            self.assertEqual(tiles.mismatches(planner.plan(10, 14, style, seed)), [])


class TestAreas(unittest.TestCase):
    ROWS, COLS, BORDER = 14, 24, 10

    def zones(self) -> list[list[str | None]]:
        # Downtown left of column BORDER, suburb to the right
        return [["downtown" if c < self.BORDER else "suburb" for c in range(self.COLS)]
                for _ in range(self.ROWS)]  # fmt: skip

    def test_cell_weights_limit_each_cell_to_its_own_tiles(self):
        parks = {base: w for base, w in wfc.DEFAULT_WEIGHTS.items() if base.startswith("park")}
        blocks = {base: w for base, w in wfc.DEFAULT_WEIGHTS.items() if base.startswith("block")}
        cell_weights = [[parks if c < 4 else blocks for c in range(8)] for _ in range(5)]
        grid = wfc.generate(5, 8, 1, cell_weights=cell_weights)
        for row in grid:
            self.assertTrue(all(TILES[t].base.startswith("park") for t in row[:4]))
            self.assertTrue(all(TILES[t].base.startswith("block") for t in row[4:]))

    def test_areas_plan_fitting_connected_cities_with_a_street_on_the_border(self):
        styles = planner.zone_styles(self.zones())
        for seed in range(4):
            with self.subTest(seed=seed):
                grid = planner.plan(self.ROWS, self.COLS, planner.Style.load("mixed"), seed, styles)
                self.assertEqual(tiles.mismatches(grid), [])
                roads, largest = wfc.connectivity(grid)
                self.assertEqual(roads, largest)
                # The border between the areas is followed by a street (one of
                # the two columns next to it, for most of the map's height)
                on_border = max(
                    sum(TILES[grid[r][k]].is_road for r in range(self.ROWS))
                    for k in (self.BORDER - 1, self.BORDER)
                )
                self.assertGreaterEqual(on_border, self.ROWS - 2)

    def test_each_area_uses_its_own_knobs(self):
        oneway_all = planner.Style.load("downtown").with_overrides(["oneway=1"])
        oneway_none = planner.Style.load("suburb").with_overrides(["oneway=0", "culdesac=0"])
        zones = [[oneway_all if c < self.BORDER else oneway_none for c in range(self.COLS)]
                 for _ in range(self.ROWS)]  # fmt: skip
        grid = planner.plan(self.ROWS, self.COLS, oneway_none, 3, zones)
        one_way = [c for row in grid for c, t in enumerate(row) if "I" in TILES[t].sockets]
        self.assertTrue(one_way)
        # One-way streets start in downtown; the border street may carry them a bit further
        self.assertLess(sum(c > self.BORDER + 1 for c in one_way), len(one_way) / 2)

    def test_zone_file_round_trip(self):
        path = tiles.MAPS / "_test.zones.json"
        try:
            planner.save_zones(self.zones(), path)
            self.assertEqual(planner.load_zones(path), self.zones())
        finally:
            path.unlink(missing_ok=True)


class TestBuildings(unittest.TestCase):
    def test_parts_fit_only_in_their_place(self):
        a, b, c = tile("mall_3x2_0_0"), tile("mall_3x2_1_0"), tile("mall_3x2_2_0")
        self.assertTrue(tiles.fits(a, "E", b))
        self.assertFalse(tiles.fits(a, "E", c))  # skipped a part
        self.assertFalse(tiles.fits(b, "E", a))  # wrong order
        self.assertFalse(tiles.fits(a, "E", tile("skyscraper_2x2_1_0")))  # other building
        self.assertTrue(tiles.fits(a, "S", tile("mall_3x2_0_1")))
        self.assertTrue(tiles.fits(tile("tower"), "E", a))  # outer side: pavement

    def test_second_sample_uses_one_way_tiles_and_buildings_and_fits(self):
        grid = tiles.load_map(tiles.MAPS / "sample2.txt")
        self.assertEqual(tiles.from_image(tiles.ASSETS / "sample_city_2.png"), grid)
        self.assertEqual(tiles.mismatches(grid), [])
        bases = tiles.count_bases(grid)
        self.assertIn("mall_2x3", bases)
        self.assertIn("oneway_straight", bases)
        roads, largest = wfc.connectivity(grid)
        self.assertEqual(roads, largest)

    def test_generated_buildings_are_whole(self):
        style = planner.Style.load("downtown")
        for seed in range(4):
            grid = planner.plan(12, 18, style, seed)
            self.assertEqual(tiles.mismatches(grid), [])  # M sides only fit their own parts


class TestPlaces(unittest.TestCase):
    def setUp(self):
        import places

        self.places = places
        self.grid = tiles.load_map(tiles.MAPS / "sample2.txt")
        self.city = tiles.to_graph(self.grid)
        self.all = places.find_places(self.grid)

    def test_every_building_is_one_place_and_reaches_a_road(self):
        cells = [cell for p in self.all for cell in p.cells]
        self.assertEqual(len(cells), len(set(cells)))  # no cell in two places
        mall = self.places.find(self.all, "shopping mall")
        self.assertEqual(len(mall.cells), 6)
        for p in self.all:
            self.assertTrue(self.places.driveways(self.city, p), p.name)

    def test_find_by_name_number_and_cell(self):
        joe = self.places.find(self.all, "joe's house")
        self.assertEqual(self.places.find(self.all, f"P{joe.id + 1}"), joe)
        r, c = joe.cells[0]
        self.assertEqual(self.places.resolve(f"{r},{c}", self.city, self.all), joe)
        with self.assertRaises(ValueError):
            self.places.find(self.all, "house")  # ambiguous

    def test_route_between_places_matches_networkx_and_never_crosses_buildings(self):
        joe = self.places.find(self.all, "Joe's house")
        mall = self.places.find(self.all, "shopping mall")
        routed, s, t = self.places.with_places(self.city, joe, mall)
        self.assertEqual(Dijkstra(routed.graph, s)[t],
                         nx.dijkstra_path_length(to_networkx(routed.graph), s + 1, t + 1))  # fmt: skip
        # The start only has arcs out, the destination only arcs in
        self.assertFalse(any(v == s for _, v, _ in routed.graph.edges()))
        self.assertEqual(routed.graph.out_degree(t), 0)

    def test_renames(self):
        joe = self.places.find(self.all, "Joe's house")
        r, c = joe.cells[0]
        renamed = self.places.find_places(self.grid, names={f"{r},{c}": "City hall"})
        self.assertEqual(self.places.find(renamed, "City hall").cells, joe.cells)


class TestGif(unittest.TestCase):
    def test_export_writes_one_frame_per_step(self):
        from PIL import Image

        import gif
        from steps import record_steps
        from tile_city_steps import export_gif

        grid = tiles.load_map(SAMPLE)
        city = tiles.to_graph(grid)
        steps = record_steps(city.graph, 0)[:5]
        path = tiles.MAPS / "_test.gif"
        try:
            export_gif(path, grid, city, steps, 0, fps=10)
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 5)
        finally:
            path.unlink(missing_ok=True)
        text = gif.describe(steps, 1, 0)
        self.assertIn("step 2 / 5", text)
        self.assertIn("frontier", text)


if __name__ == "__main__":
    unittest.main()
