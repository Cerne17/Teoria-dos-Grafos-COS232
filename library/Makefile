CXX := g++
CXXFLAGS := -std=c++17 -Iinclude -Wall -Wextra -g
RELFLAGS := -std=c++17 -Iinclude -Wall -Wextra -O2

OBJS := src/graph.o src/io.o src/search.o src/stats.o src/components.o src/distance.o app/main.o
LIB_SRCS := src/graph.cpp src/io.cpp src/search.cpp src/stats.cpp src/components.cpp src/distance.cpp

.PHONY: all release clean

all: app/graphs

# Debug build: -g, no optimisation. The default while developing.
app/graphs: $(OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Optimised build of the same program, for when the code is already debugged
# and you just want the results. Compiled straight from the sources so it can
# never pick up a stale debug object.
release: app/graphs_release

app/graphs_release: app/main.cpp $(LIB_SRCS) include/graph.hpp
	$(CXX) $(RELFLAGS) app/main.cpp $(LIB_SRCS) -o $@


benchmark: app/benchmark.cpp $(LIB_SRCS)
	$(CXX) -std=c++17 -Iinclude -O2 $^ -o app/benchmark

%.o: %.cpp include/graph.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) app/graphs app/graphs_release
