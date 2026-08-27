#pragma once
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Undirected graph library.
//
// Vertices are 1-indexed. Every internal vector is sized n+1 and index 0 is
// unused padding, so a loop written as `for (int i = 0; ...)` reads garbage.
//
// Cost notation used throughout this header:
//   n        vertex count
//   m        edge count as read from the file (a repeated line is counted
//            again; neither representation deduplicates)
//   deg(u)   number of entries stored for u
//   k        number of connected components
//
// Costs are quoted separately for AdjacencyList and AdjacencyMatrix wherever
// they differ. Three of them are easy to miss when reading the call sites:
//
//   1. neighbors() returns a fresh std::vector by value, so every call
//      allocates and copies — even on the list, where the row already exists.
//   2. neighbors() on the matrix scans a whole row, O(n), no matter how few
//      neighbours u actually has. Any traversal that is O(n + m) on a list is
//      therefore O(n^2) on a matrix.
//   3. degree() is O(1) on the list but O(n) on the matrix, so a loop over
//      every vertex's degree costs O(n) against a list and O(n^2) against a
//      matrix.
// ---------------------------------------------------------------------------

// Abstract interface both representations implement. Algorithms take a
// `const Graph &` and pay whichever cost the concrete type imposes.
class Graph {
public:
  int n; // # of vertexes
  int m; // # of edges

  // Args:    vertexCount  number of vertices; valid ids are 1..vertexCount
  // Cost:    O(1) here; the derived constructors do the allocation
  explicit Graph(int vertexCount) : n(vertexCount), m(0) {}
  virtual ~Graph() = default;

  // Adds the undirected edge {u, v} and increments m.
  // Args:    u, v  endpoints, both in [1, n]
  // Returns: nothing
  virtual void addEdge(int u, int v) = 0;

  // Args:    u  vertex in [1, n]
  // Returns: degree of u
  virtual int degree(int u) const = 0;

  // Args:    u  vertex in [1, n]
  // Returns: u's neighbours, ascending, as a new vector (by value)
  virtual std::vector<int> neighbors(int u) const = 0;

  // Puts the structure in its queryable form. Call once after the last
  // addEdge; algorithms assume neighbours come back in ascending order.
  // Returns: nothing
  virtual void finalize() = 0;
};

// Adjacency list: one vector of neighbours per vertex.
// Space: O(n + m) ints.
class AdjacencyList : public Graph {
public:
  std::vector<std::vector<int>> adj;

  // Args:    vertexCount  number of vertices
  // Cost:    O(n) time and space — allocates n+1 empty rows
  explicit AdjacencyList(int vertexCount);

  // Appends v to u's row and u to v's. A repeated edge is stored again, so
  // this representation behaves as a multigraph and a self-loop adds 2 to
  // deg(u), keeping the handshake lemma (sum of degrees == 2m) true.
  // Args:    u, v  endpoints in [1, n]
  // Returns: nothing
  // Cost:    O(1) amortised
  void addEdge(int u, int v) override;

  // Args:    u  vertex in [1, n]
  // Returns: adj[u].size(), counting multiplicity
  // Cost:    O(1)
  int degree(int u) const override;

  // Args:    u  vertex in [1, n]
  // Returns: a copy of u's row
  // Cost:    O(deg(u)) time and space — copies the row into a new vector
  std::vector<int> neighbors(int u) const override;

  // Sorts every row so neighbours come back ascending, which is what makes
  // traversal output reproducible.
  // Returns: nothing
  // Cost:    O(sum of deg(u) log deg(u)), bounded by O(m log n)
  void finalize() override;
};

// Adjacency matrix: one byte per ordered pair.
// Space: O(n^2) bytes — 10k vertices is ~100 MB.
class AdjacencyMatrix : public Graph {
public:
  std::vector<std::vector<char>> matrix;

  // Args:    vertexCount  number of vertices
  // Cost:    O(n^2) time and space — allocates and zeroes (n+1)^2 bytes
  explicit AdjacencyMatrix(int vertexCount);

  // Sets both cells to 1. m still counts every call, but the write is
  // idempotent: a repeated edge changes no degree, and a self-loop adds only 1
  // to deg(u). Degrees here therefore describe the underlying simple graph
  // while m describes the file, so the handshake lemma fails and the degree
  // statistics disagree with the list's whenever the input repeats an edge.
  // Args:    u, v  endpoints in [1, n]
  // Returns: nothing
  // Cost:    O(1)
  void addEdge(int u, int v) override;

  // Args:    u  vertex in [1, n]
  // Returns: number of set cells in u's row
  // Cost:    O(n) — scans the row; NOT the O(1) the list gives you
  int degree(int u) const override;

  // Args:    u  vertex in [1, n]
  // Returns: u's neighbours, ascending, as a new vector
  // Cost:    O(n) time regardless of deg(u), O(deg(u)) space. This single call
  //          is why every traversal below degrades to O(n^2) here.
  std::vector<int> neighbors(int u) const override;

  // No-op: cells are already written in ascending column order.
  // Returns: nothing
  // Cost:    O(1)
  void finalize() override;
};

enum class Representation { List, Matrix };

// Reads a graph file: first line is n, every line after that is one edge as a
// pair of vertex ids. A trailing lone integer is silently dropped.
// Args:    path  file to read
//          rep   which representation to build
// Returns: owning pointer to a finalized graph
// Throws:  std::runtime_error if the file will not open, if n cannot be read,
//          or if n < 0
// Cost:    list   O(n + m) to build plus O(m log n) in finalize; O(n + m) space
//          matrix O(n^2 + m) time, O(n^2) space
std::unique_ptr<Graph> readGraph(const std::string &path, Representation rep);

struct GraphStats {
  int n; // # of vertexes
  int m; // # of edges

  int minDegree;
  int maxDegree;
  double avgDegree; // 2m/n, derived from m rather than from the degree array
  double medianDegree;
  int componentsAmount;
  int largestComponentSize;
  int smallestComponentSize;
  int exactDiameter;       // -1 when skipped by exactDiameterLimit
  int approximateDiameter; // lower bound; see getApproximateDiameter
};

// Computes every statistic the assignment asks for.
// Args:    g                   graph to measure
//          exactDiameterLimit  skip the exact diameter when n exceeds it; pass
//                              a non-positive value to always compute it
// Returns: a fully populated GraphStats; exactDiameter is -1 when it was
//          skipped, the same "undefined" convention used by level and
//          getDistance
// Cost:    dominated by the exact diameter when it runs —
//            list   O(n*(n + m))       matrix O(n^3)
//          when it is skipped, the degree loop and the sort dominate —
//            list   O(n log n + m)     matrix O(n^2)
//          Space: O(n + k)
GraphStats computeStats(const Graph &g, int exactDiameterLimit = 0);

// Result of a traversal rooted at one vertex. Both vectors are sized n+1.
struct SearchTree {
  std::vector<int> parent; // 0 for the root AND for unreached vertices
  std::vector<int> level;  // -1 for unreached vertices; use this, not parent,
                           // to tell an unreached vertex from the root
};

// Breadth-first search. level[v] is the true graph distance from source to v.
// Args:    g       graph to traverse
//          source  root, in [1, n] (asserted)
// Returns: SearchTree where level[source] == 0 and level[v] == -1 for every v
//          outside source's component
// Cost:    list O(n + m) | matrix O(n^2); space O(n)
SearchTree bfs(const Graph &g, int source);

// BFS that visits every reachable vertex and keeps nothing. Currently unused.
// Args:    g, source  as in bfs
// Returns: nothing
// Cost:    list O(n + m) | matrix O(n^2); space O(n)
void bfsExploration(const Graph &g, int source);

// Depth-first search, iterative. Pushes (vertex, parent) pairs and settles a
// vertex when it is popped, which reproduces the recursive DFS tree exactly.
// Args:    g       graph to traverse
//          source  root, in [1, n] (asserted)
// Returns: SearchTree where level[v] is v's DEPTH IN THE DFS TREE, not its
//          distance from source — use bfs or getDistance for distances
// Cost:    list O(n + m) | matrix O(n^2)
//          Space: O(n + m), because the stack holds up to one entry per edge
//          endpoint, unlike the O(n) a recursive DFS would use
SearchTree dfs(const Graph &g, int source);

// DFS that visits every reachable vertex and keeps nothing. Currently unused.
// It pushes neighbours in ascending order while dfs() pushes them descending,
// so the two visit a component in different orders.
// Args:    g, source  as in dfs
// Returns: nothing
// Cost:    list O(n + m) | matrix O(n^2); space O(n + m)
void dfsExploration(const Graph &g, int source);

// Appends one statistics row, writing the header first if the file is new.
// Args:    path, graphName  destination and the row's label
//          stats            row contents
// Returns: nothing
// Throws:  std::runtime_error if the file will not open
// Cost:    O(1) plus I/O
void writeStats(const std::string &path, const std::string &graphName,
                const GraphStats &stats);

// Appends one row per vertex: graphName, node, parent, level.
// Args:    path, graphName  destination and the rows' label
//          tree             traversal result to dump
// Returns: nothing
// Throws:  std::runtime_error if the file will not open
// Cost:    O(n) plus I/O
void writeSearchTree(const std::string &path, const std::string &graphName,
                     const SearchTree &tree);

// Appends one row per vertex: graphName, node, componentId, where componentId
// is the vertex's index in `components`.
// Args:    path, graphName  destination and the rows' label
//          components       partition to dump
// Returns: nothing
// Throws:  std::runtime_error if the file will not open
// Cost:    O(n) plus I/O
void writeComponents(const std::string &path, const std::string &graphName,
                     const std::vector<std::vector<int>> &components);

// Connected components, found by running a BFS from each unvisited vertex.
// Args:    g  graph to partition
// Returns: one vector of vertex ids per component, in BFS discovery order
//          within a component, the components themselves sorted by size
//          descending. That sort is not stable, so equal-sized components can
//          swap places between runs and their ids are not reproducible.
// Cost:    list O(n + m + k log k) | matrix O(n^2); space O(n)
std::vector<std::vector<int>> getComponentsByBreadth(const Graph &g);

// Same partition as getComponentsByBreadth, found with DFS instead. Only the
// order of the vertices inside each component differs.
// Args:    g  graph to partition
// Returns: as getComponentsByBreadth, but in DFS discovery order
// Cost:    list O(n + m + k log k) | matrix O(n^2); space O(n + m)
std::vector<std::vector<int>> getComponentsByDepth(const Graph &g);

// Shortest-path distance, via a full BFS from u.
// Args:    g     graph to search
//          u, v  endpoints in [1, n]
// Returns: number of edges on a shortest u-v path, or -1 when v is not
//          reachable from u
// Cost:    list O(n + m) | matrix O(n^2); space O(n)
int getDistance(const Graph &g, const int u, const int v);

// Exact diameter, by running a BFS from every vertex and keeping the largest
// finite level.
// Args:    g  graph to measure
// Returns: max over all reachable pairs of d(u, v). For a disconnected graph
//          the textbook answer is infinite; because unreachable pairs are
//          skipped, this returns the largest diameter among the components.
// Cost:    list O(n*(n + m)) | matrix O(n^3) — the most expensive call in the
//          library, and the reason computeStats takes a limit; space O(n)
int getExactDiameter(const Graph &g);

// Approximate diameter by double sweep: BFS from a vertex, then BFS from the
// farthest vertex that found, and report that vertex's eccentricity.
// Args:    g  graph to estimate
// Returns: a LOWER bound on the diameter — exact on trees, an underestimate on
//          general graphs. It sweeps only the largest component, so on a
//          disconnected graph it can also fall below getExactDiameter for a
//          second reason: a smaller component may hold the real diameter.
// Cost:    list O(n + m) | matrix O(n^2); space O(n)
int getApproximateDiameter(const Graph &g);

// Overload for callers that already hold the components, so the BFS sweep that
// builds them is not repeated.
// Args:    g           graph to estimate
//          components  output of getComponentsByBreadth for g; components[0]
//                      must be the largest and must not be empty
// Returns: as the one-argument overload
// Cost:    as above, minus the component sweep
int getApproximateDiameter(const Graph &g,
                           const std::vector<std::vector<int>> &components);
