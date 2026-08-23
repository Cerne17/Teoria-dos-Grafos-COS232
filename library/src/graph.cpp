#include "graph.hpp"
#include <cassert>

AdjacencyList::AdjacencyList(int vertexCount)
    : Graph(vertexCount), adj(vertexCount + 1) {}

void AdjacencyList::addEdge(int u, int v) {
  assert(u >= 1 && u <= n && "addEdge: u out of range");
  assert(v >= 1 && v <= n && "addEdge: v out of range");
  adj[u].push_back(v);
  adj[v].push_back(u);
  m++;
}

int AdjacencyList::degree(int u) const {
  assert(u >= 1 && u <= n && "degree: u out of range");
  return static_cast<int>(adj[u].size());
}

std::vector<int> AdjacencyList::neighbors(int u) const {
  assert(u >= 1 && u <= n && "neighbors: u out of range");
  return adj[u];
}

AdjacencyMatrix::AdjacencyMatrix(int vertexCount)
    : Graph(vertexCount), matrix(vertexCount + 1) {}

void AdjacencyMatrix::addEdge(int u, int v) {
  assert(u >= 1 && u <= n && "addEdge: u out of range");
  assert(v >= 1 && v <= n && "addEdge: v out of range");
  matrix[u][v] = '1';
  matrix[v][u] = '1';
  m++;
}

int AdjacencyMatrix::degree(int u) const {
  assert(u >= 1 && u <= n && "degree: u out of range");
  int counter = 0;
  for (char i : matrix[u]) {
    if (i) {
      counter++;
    }
  }
  return counter;
}

std::vector<int> AdjacencyMatrix::neighbors(int u) const {
  assert(u >= 1 && u <= n && "neighbors: u out of range");
  std::vector<int> result;
  for (int v = 1; v <= n; ++v)
    if (matrix[u][v])
      result.push_back(v);
  return result;
}
