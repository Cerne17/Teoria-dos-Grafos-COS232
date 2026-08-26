#include "graph.hpp"
#include <cassert>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

SearchTree bfs(const Graph &g, int source) {
  assert(source > 0 && source <= g.n && "source is not a valid index.");

  std::vector<bool> visited(g.n + 1, false);
  std::vector<int> parent(g.n + 1, 0);
  std::vector<int> level(g.n + 1, 0);
  std::queue<int> q;

  visited[source] = true;
  q.push(source);

  while (!q.empty()) {
    int current = q.front();
    q.pop();

    for (int u : g.neighbors(current)) {
      if (!visited[u]) {
        visited[u] = true;
        q.push(u);
        parent[u] = current;
        level[u] = level[current] + 1;
      }
    }
  }
  return SearchTree{parent, level};
}

void bfsExploration(const Graph &g, int source) {
  std::queue<int> q;
  std::vector<bool> visited(g.n + 1, false);

  q.push(source);
  visited[source] = true;

  while (!q.empty()) {
    int current = q.front();
    q.pop();

    for (auto v : g.neighbors(current)) {
      if (!visited[v]) {
        q.push(v);
        visited[v] = true;
      }
    }
  }
}

SearchTree dfs(const Graph &g, int source) {
  assert(source > 0 && source <= g.n && "source is not a valid index.");

  std::vector<bool> visited(g.n + 1, false);
  std::vector<int> parent(g.n + 1, 0);
  std::vector<int> level(g.n + 1, 0);
  std::stack<std::pair<int, int>> s;

  s.push({source, 0});

  while (!s.empty()) {
    auto top = s.top();
    s.pop();

    int u = top.first;  // current vertex
    int p = top.second; // current vertex's parent

    if (visited[u])
      continue;

    visited[u] = true;
    parent[u] = p;
    level[u] = (u == source) ? 0 : level[p] + 1;

    std::vector<int> nbrs =
        g.neighbors(u); // avoids calling neighbors twice and make sure the
                        // vector is not destroyed on the iterations end
    for (auto it = nbrs.rbegin(); it != nbrs.rend(); ++it) {
      int v = *it;
      if (!visited[v])
        s.push({v, u});
    }
  }
  return SearchTree{parent, level};
}

void dfsExploration(const Graph &g, int source) {
  std::stack<int> s;
  std::vector<bool> visited(g.n + 1, false);

  s.push(source);

  while (!s.empty()) {
    int current = s.top();
    s.pop();
    if (!visited[current]) {
      visited[current] = true;
      for (auto v : g.neighbors(current)) {
        if (!visited[v])
          s.push(v);
      }
    }
  }
}
