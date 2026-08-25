from abc import ABC, abstractmethod
from collections.abc import Iterable
from itertools import combinations


Vertice = str
Aresta = tuple[Vertice, Vertice]


class Grafo(ABC):
    """Contrato basico que qualquer representacao de grafo tem que seguir"""

    @property
    @abstractmethod
    def vertices(self) -> set[Vertice]: ...

    @abstractmethod
    def __len__(self) -> int:
        """Devolve o total de vertices em um grafo"""
        ...

    @abstractmethod
    def __contains__(self, v: Vertice) -> bool: ...

    @abstractmethod
    def __iter__(self) -> Iterable[Vertice]: ...

    @abstractmethod
    def __getitem__(self, v: Vertice) -> set[Vertice] | None: ...

    @abstractmethod
    def existe_aresta(self, v: Vertice, w: Vertice) -> bool: ...

    @abstractmethod
    def inserir_aresta(self, v: Vertice, w: Vertice) -> None: ...

    @abstractmethod
    def remover_aresta(self, v: Vertice, w: Vertice) -> None: ...

    @abstractmethod
    def obter_vertice_maior_grau(self) -> tuple[Vertice, int]: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(|V| = {len(self)})"

    def checar_clique(self, S: set[Vertice] | None = None) -> bool:
        """Resolve se S ou self eh ou nao uma clique"""
        if S is None:
            S = self.vertices
        return all(self.existe_aresta(u, v) for u, v in combinations(S, 2))
