from collections.abc import Callable

from adjacency_list_weighted import AdjacencyList

# on_step(u, distances, discovered, parent), called after u's arcs are relaxed
StepCallback = Callable[[int, list[float], list[int], list[int | None]], None]


def Dijkstra(
    graph: AdjacencyList, source: int, on_step: StepCallback | None = None
) -> list[float]:
    distances = [0 if i == source else float("inf") for i in range(len(graph))]
    # parent[v] = vertex before v on the best path found so far
    parent: list[int | None] = [None] * len(graph)
    discovered = [source]
    while len(discovered) != 0:
        u = discovered[0]
        d_u = float("inf")
        for v_k in discovered:
            if (d_v_k := distances[v_k]) < d_u:
                u = v_k
                d_u = d_v_k

        discovered.remove(u)

        for v, w in graph.neighbors(u):
            if distances[v] > distances[u] + w:
                distances[v] = distances[u] + w
                parent[v] = u
                if v not in discovered:
                    discovered.append(v)

        if on_step is not None:
            on_step(u, distances, discovered, parent)

    return distances
