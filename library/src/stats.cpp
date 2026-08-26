#include "graph.hpp"
#include <algorithm>
#include <vector>

GraphStats computeStats(const Graph &g) {
  int n = g.n;
  int m = g.m;

  // TODO: ENABLE CHOICE FOR USER: BREADTH VS. DEPTH
  int componentsAmount = size(getComponentsByBreadth(g));

  std::vector<int> degrees;
  degrees.reserve(n);

  for (int i = 1; i <= n; i++) {
    degrees.push_back(g.degree(i));
  }
  std::sort(degrees.begin(), degrees.end());

  int minDegree = degrees.front();
  int maxDegree = degrees.back();
  double avgDegree = 2.0 * m / n;
  double medianDegree;
  if (n % 2 == 1) {
    medianDegree = degrees[n / 2];
  } else {
    medianDegree = (degrees[n / 2 - 1] + degrees[n / 2]) / 2.0;
  }

  return GraphStats{
      n, m, minDegree, maxDegree, avgDegree, medianDegree, componentsAmount};
}
