# Minimal Metal reprexes. Requires macOS + Xcode command-line tools (`xcode-select --install`).
# Shaders are compiled at runtime from inline source, so there is no metallib step.
CXX   = xcrun clang++
FLAGS = -std=c++17 -fobjc-arc -O2 -framework Metal -framework Foundation

all: dispatch_overhead grid_barrier

dispatch_overhead: dispatch_overhead.mm
	$(CXX) $(FLAGS) $< -o $@

grid_barrier: grid_barrier.mm
	$(CXX) $(FLAGS) $< -o $@

run: all
	./dispatch_overhead
	@echo
	./grid_barrier

clean:
	rm -f dispatch_overhead grid_barrier

.PHONY: all run clean
