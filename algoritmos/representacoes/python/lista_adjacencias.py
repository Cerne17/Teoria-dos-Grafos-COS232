from collections.abc import Iterable, Iterator
from itertools import combinations

from grafo import Grafo, Vertice, Aresta


class ListaAdjacencias(Grafo):
    def __init__(
        self, vertices: Iterable[Vertice] = (), arestas: Iterable[Aresta] = ()
    ) -> None:
        # Aqui implementamos a lista de adjacencias como um dicionario ja que queremos
        # chaves como strings, caso usassemos ints, poderiamos fazer a implementacao
        # padrao
        self._listaAdjacencias: dict[Vertice, list[Vertice]] = {v: [] for v in vertices}
        for u, w in arestas:
            self.inserir_aresta(u, w)

    def vertices(self) -> set[Vertice]:
        """O(V) no caso medio | O(V^2) no pior caso (colisao para toda insercao)"""
        return set(self._listaAdjacencias)

    def __len__(self) -> int:
        """O(1) o python na implementacao do dicionario tem o atributo de tamanho"""
        return len(self._listaAdjacencias)

    def __contains__(self, v: Vertice) -> bool:
        """O(1)"""
        return v in self._listaAdjacencias

    def __iter__(self) -> Iterator[Vertice]:
        """O(V)"""
        return iter(self._listaAdjacencias)

    def __getitem__(self, v: Vertice) -> set[Vertice] | None:
        """O(g_max) no caso medio para converter lista em conjunto | O(g_max^2) no pior caso"""
        return set(self._listaAdjacencias[v])

    def existe_aresta(self, v: Vertice, w: Vertice) -> bool:
        """O(g_max)"""
        return w in self._listaAdjacencias[v] or v in self._listaAdjacencias[w]

    def inserir_aresta(self, v: Vertice, w: Vertice) -> None:
        if v not in self._listaAdjacencias[w]:
            self._listaAdjacencias[w].append(v)
        if w not in self._listaAdjacencias[v]:
            self._listaAdjacencias[v].append(w)
