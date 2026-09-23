# Performance Tracing & Profiling Guide

TinyMatrix includes built-in, zero-dependency performance tracing and timeline profiling tools to inspect execution bottlenecks, memory footprint, and call graphs.

---

## 1. Quick Tracing with `PerformanceTracer`

Enable tracing globally to measure wall time, CPU time, and memory allocation across matrix operations:

```python
from tinymatrix import Matrix, enable_tracing, disable_tracing

# Start recording trace spans and memory tracking
tracer = enable_tracing(track_memory=True)

A = Matrix.random(50, 50)
B = Matrix.random(50, 50)

# Run operations
C = A @ B
P, L, U = A.lu()
Q, R = A.qr()
U_svd, S, Vt = A.svd()

# Stop tracing
disable_tracing()

# Print formatted summary table to console
print(tracer.format_summary_table())

# Export timeline to Chrome Tracing / Perfetto format
tracer.export_chrome_trace("trace.json")
```

### Visualizing Traces in Perfetto / Chrome
1. Open [https://ui.perfetto.dev/](https://ui.perfetto.dev/) or navigate to `chrome://tracing` in any Chromium browser.
2. Drag and drop `trace.json`.
3. Interactively inspect flame charts, execution timelines, durations, and memory deltas per operation.

---

## 2. Granular Tracing: `@traced` and `trace_op`

You can trace custom functions or blocks of code:

```python
from tinymatrix import trace_op, traced

# Using decorator
@traced(name="custom_pipeline", category="pipeline")
def run_pipeline(A):
    return (A.T @ A).inv()

# Using context manager
with trace_op("feature_engineering", category="prep", matrix_shape=(100, 100)):
    # your code here
    pass
```

---

## 3. Call-Tree Profiling with `cProfile`

Profile function call stacks and identify top hotspots:

```python
from tinymatrix import Matrix, profile_code

A = Matrix.random(60, 60)

# Profile execution
result, stats = profile_code(lambda: A.svd())

# Print top 15 cumulative time consumers
stats.strip_dirs().sort_stats("cumulative").print_stats(15)
```

---

## 4. Benchmarking CLI Tool

TinyMatrix provides a dedicated tracing CLI in `benchmarks/trace.py`:

```bash
# Run tracing suite on default matrix sizes (20x20 and 40x40)
uv run python benchmarks/trace.py

# Trace specific matrix sizes and export Perfetto JSON
uv run python benchmarks/trace.py --size 30,60 --save-trace trace.json

# Profile a specific operation with cProfile
uv run python benchmarks/trace.py --profile svd --size 50
```
