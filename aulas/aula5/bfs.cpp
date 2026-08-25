/*
 * BFS_EXPLORATION(s)
 * 1. desmarcar todos os vértices    O(n)
 * 2. marcar s                       O(1)
 * 3. incluir s em Q                 O(1)
 * 4. enquanto Q não estiver vazia   O(n) ou O(V)
 * 5.   extrair u de Q               O(1)
 * 6.   para cada w vizinho de u     O(g_max)
 * 7.     se w n marcado             O(1)
 * 8.       adicionar w em Q         O(1)
 * 9.       marcar w                 O(1)
 *
 * COMPLEXIDADE TOTAL DE TEMPO: O(V + E)
 */
#include "bfs.hpp"
#include "graph.hpp"

#include <queue>

void bfsExploration(const Graph &g, const int source) {
  std::queue<int> q;
  std::vector<bool> visited(g.n, false);

  q.push(source);
  visited[source] = true;

  while (!q.empty()) {
    int u = q.front();
    q.pop();

    for (auto w : g.neighbors(u)) {
      if (!visited[w]) {
        visited[w] = true;
        q.push(w);
      }
    }
  }
}

/* DFS_EXPLORATION_RECURSIVE(s)
 * 1. marcar s
 * 2. para cada aresta (s, u) incidente a u
 * 3.   se u n estiver marcado
 * 4.     DFS_EXPLORATION(u) // chamada recursiva
 * 5. desmarcar todos vertices
 * 6. escolher vertice inicial v
 * 7. DFS_EXPLORATION_RECURSIVE(v)
 */
int n = 100000000;
AdjacencyList g(n);

std::vector<int> visited(n, false);
void dfsExplorationRecursive(const int source) {
  visited[source] = true;
  for (auto v : g.neighbors(source)) {
    if (!visited[v]) {
      dfsExplorationRecursive(v);
    }
  }
}
for (int j = 0; j < n; j++) {
  visited[j] = false;
}
