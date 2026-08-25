#include <cassert>
#include "graph.hpp"

Graph::Graph(int vertexCount): n(vertexCount), m(0), adj(vertexCount+1) { }

void Graph::addEdge(int u, int v) {
  assert(u >= 1 && u <= n && "addEdge: u out of range");
  assert(v >= 1 && v <= n && "addEdge: v out of range");
  adj[u].push_back(v);
  adj[v].push_back(u);
  m++;
}

int Graph::degree(int u) const {
  assert(u >= 1 && u <= n && "degree: u out of range");
  return static_cast<int>(adj[u].size());
}

