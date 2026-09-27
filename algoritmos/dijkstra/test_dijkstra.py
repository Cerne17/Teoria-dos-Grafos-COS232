"""Run with: python -m unittest -v test_dijkstra"""

import random
import unittest
from pathlib import Path

import networkx as nx

import city
from adjacency_list_weighted import AdjacencyList
from city_steps import route
from dijkstra import Dijkstra
from steps import record_steps
from viz import to_networkx

GRAPHS = Path(__file__).parent / "graphs"
INF = float("inf")


def load(name: str) -> AdjacencyList:
    return AdjacencyList.from_file(GRAPHS / f"{name}.txt")


def random_graph(rng: random.Random, n: int, p: float) -> AdjacencyList:
    """Each ordered pair (u, v), u != v, becomes an arc with probability p."""
    g = AdjacencyList(n)
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                g.add_edge(u, v, rng.randint(0, 20))
    return g


class TestKnownGraphs(unittest.TestCase):
    """Answers computed by hand from the graph files (vertex 1 = index 0)."""

    def test_graph_01_indirect_path_beats_direct_arc(self):
        # 1 -> 3 -> 2 costs 5 + 2 = 7, the direct 1 -> 2 costs 10
        self.assertEqual(Dijkstra(load("graph_01"), 0), [0, 7, 5])

    def test_graph_02(self):
        # 2 is reached through 4 (5 + 3 = 8), not directly (10)
        self.assertEqual(Dijkstra(load("graph_02"), 0), [0, 8, 9, 5, 7])

    def test_graph_03(self):
        self.assertEqual(Dijkstra(load("graph_03"), 0), [0, 7, 9, 20, 20, 11, 23])

    def test_graph_03_from_another_source_uses_the_cycle_back(self):
        # From 3, vertex 1 is only reachable through 7 -> 1
        self.assertEqual(Dijkstra(load("graph_03"), 2), [18, 25, 0, 11, 11, 2, 14])


class TestEdgeCases(unittest.TestCase):
    def test_single_vertex(self):
        self.assertEqual(Dijkstra(AdjacencyList(1), 0), [0])

    def test_unreachable_vertices_stay_infinite(self):
        g = AdjacencyList(4, [(0, 1, 3), (2, 3, 1)])
        self.assertEqual(Dijkstra(g, 0), [0, 3, INF, INF])

    def test_arcs_are_one_way(self):
        # 1 -> 0 exists, 0 -> 1 does not
        g = AdjacencyList(2, [(1, 0, 5)])
        self.assertEqual(Dijkstra(g, 0), [0, INF])
        self.assertEqual(Dijkstra(g, 1), [5, 0])

    def test_zero_weight_arcs(self):
        g = AdjacencyList(3, [(0, 1, 0), (1, 2, 0)])
        self.assertEqual(Dijkstra(g, 0), [0, 0, 0])

    def test_many_hops_beat_one_expensive_arc(self):
        n = 6
        g = AdjacencyList(n, [(i, i + 1, 1) for i in range(n - 1)])
        g.add_edge(0, n - 1, 100)
        self.assertEqual(Dijkstra(g, 0)[n - 1], n - 1)


class TestSteps(unittest.TestCase):
    def test_each_vertex_is_finalized_once_in_nondecreasing_order(self):
        g = load("graph_03")
        picked: list[int] = []
        final: list[float] = []

        def on_step(u, distances, discovered, parent):
            picked.append(u)
            final.append(distances[u])

        Dijkstra(g, 0, on_step)
        self.assertEqual(sorted(picked), list(range(len(g))))
        self.assertEqual(final, sorted(final))

    def test_parent_pointers_rebuild_shortest_paths(self):
        g = load("graph_03")
        last_parent: list[int | None] = []

        def on_step(u, distances, discovered, parent):
            last_parent[:] = parent

        distances = Dijkstra(g, 0, on_step)
        for v, p in enumerate(last_parent):
            if p is not None:
                self.assertEqual(distances[v], distances[p] + g.weight(p, v))


class TestCity(unittest.TestCase):
    def setUp(self):
        self.city = city.load()

    def test_every_lane_is_a_straight_street_as_long_as_its_weight(self):
        for u, v, w in self.city.edges():
            (x0, y0), (x1, y1) = city.POSITIONS[u], city.POSITIONS[v]
            with self.subTest(arc=(u + 1, v + 1)):
                self.assertTrue(x0 == x1 or y0 == y1)
                self.assertEqual(abs(x1 - x0) + abs(y1 - y0), w)

    def test_every_intersection_reaches_every_other(self):
        for source in self.city:
            self.assertNotIn(INF, Dijkstra(self.city, source))

    def test_matches_networkx_from_every_intersection(self):
        g = to_networkx(self.city)
        for source in self.city:
            expected = nx.single_source_dijkstra_path_length(g, source + 1)
            with self.subTest(source=source + 1):
                self.assertEqual(
                    Dijkstra(self.city, source), [expected[v + 1] for v in self.city]
                )

    def test_route_19_to_4(self):
        steps = record_steps(self.city, 18)
        self.assertEqual([v + 1 for v in route(steps[-1], 3)], [19, 16, 17, 18, 13, 10, 3, 4])
        self.assertEqual(steps[-1].distances[3], 230)

    def test_shortest_path_goes_round_the_roundabout(self):
        # The only lane into 8 comes from 9, so 5 -> 8 circles 6 -> 7 -> 9 -> 8
        self.assertEqual(city.shortest_path(self.city, 4, 7), (80, [4, 5, 6, 8, 7]))

    def test_shortest_path_to_itself_and_to_unreachable(self):
        self.assertEqual(city.shortest_path(self.city, 0, 0), (0, [0]))
        g = AdjacencyList(2)
        self.assertEqual(city.shortest_path(g, 0, 1), (INF, []))


class TestAgainstNetworkx(unittest.TestCase):
    def test_random_graphs(self):
        rng = random.Random(0)
        for trial in range(200):
            n = rng.randint(1, 12)
            g = random_graph(rng, n, p=rng.uniform(0.1, 0.6))
            source = rng.randrange(n)

            expected = nx.single_source_dijkstra_path_length(to_networkx(g), source + 1)
            expected = [expected.get(v + 1, INF) for v in range(n)]

            with self.subTest(trial=trial, n=n, source=source):
                self.assertEqual(Dijkstra(g, source), expected)


if __name__ == "__main__":
    unittest.main()
