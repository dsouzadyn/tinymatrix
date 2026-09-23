"""Performance tracing, memory tracking, and profiling tools for TinyMatrix.

Provides:
- PerformanceTracer: Records timing, memory, and spans exportable to Chrome Tracing / Perfetto.
- trace_op: Context manager and decorator for tracing operations.
- profile_code: cProfile wrapper for call-tree analysis.
- Memory tracking via tracemalloc.
"""

import cProfile
import json
import os
import pstats
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


@dataclass
class TraceSpan:
    """A recorded trace span compatible with Chrome Tracing / Perfetto format."""

    name: str
    cat: str
    ph: str  # Phase: 'X' for complete span, 'B' for begin, 'E' for end
    ts: float  # Microseconds timestamp
    dur: float  # Duration in microseconds
    pid: int = 1
    tid: int = 1
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceTracer:
    """Records performance traces, memory metrics, and execution timelines.

    Traces can be exported to Chrome Tracing / Perfetto JSON format and viewed
    interactively at https://ui.perfetto.dev/ or chrome://tracing.
    """

    def __init__(self, track_memory: bool = True) -> None:
        self.track_memory = track_memory
        self.spans: List[TraceSpan] = []
        self._enabled = False
        self._start_time_us: float = 0.0

    def start(self) -> "PerformanceTracer":
        """Start recording trace events."""
        self._enabled = True
        self._start_time_us = time.perf_counter() * 1_000_000.0
        if self.track_memory and not tracemalloc.is_tracing():
            tracemalloc.start()
        return self

    def stop(self) -> None:
        """Stop recording trace events."""
        self._enabled = False

    def clear(self) -> None:
        """Clear all recorded spans."""
        self.spans.clear()

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @contextmanager
    def trace(
        self, name: str, category: str = "op", **metadata: Any
    ) -> Iterator[Dict[str, Any]]:
        """Context manager to measure and record a named operation span."""
        if not self._enabled:
            yield metadata
            return

        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        start_mem = (
            tracemalloc.get_traced_memory()[0]
            if (self.track_memory and tracemalloc.is_tracing())
            else 0
        )

        span_args = dict(metadata)
        try:
            yield span_args
        finally:
            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            wall_us = (end_wall - start_wall) * 1_000_000.0
            cpu_us = (end_cpu - start_cpu) * 1_000_000.0

            if self.track_memory and tracemalloc.is_tracing():
                current_mem, peak_mem = tracemalloc.get_traced_memory()
                span_args["mem_delta_kb"] = round((current_mem - start_mem) / 1024.0, 3)
                span_args["peak_mem_kb"] = round(peak_mem / 1024.0, 3)

            span_args["cpu_time_ms"] = round(cpu_us / 1000.0, 4)
            span_args["wall_time_ms"] = round(wall_us / 1000.0, 4)

            span = TraceSpan(
                name=name,
                cat=category,
                ph="X",
                ts=round((start_wall * 1_000_000.0) - self._start_time_us, 2),
                dur=round(wall_us, 2),
                args=span_args,
            )
            self.spans.append(span)

    def export_chrome_trace(self, file_path: str) -> None:
        """Export recorded spans to a Chrome Tracing / Perfetto JSON file."""
        os.makedirs(
            os.path.dirname(os.path.abspath(file_path)), exist_ok=True
        ) if os.path.dirname(file_path) else None
        data = {
            "traceEvents": [s.to_dict() for s in self.spans],
            "displayTimeUnit": "ms",
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def summary(self) -> List[Dict[str, Any]]:
        """Compute aggregated performance metrics per operation name."""
        stats: Dict[str, Dict[str, Any]] = {}
        for s in self.spans:
            wall_ms = s.dur / 1000.0
            peak_kb = s.args.get("peak_mem_kb", 0.0)
            if s.name not in stats:
                stats[s.name] = {
                    "name": s.name,
                    "count": 0,
                    "total_ms": 0.0,
                    "min_ms": wall_ms,
                    "max_ms": wall_ms,
                    "peak_mem_kb": peak_kb,
                }
            item = stats[s.name]
            item["count"] += 1
            item["total_ms"] += wall_ms
            item["min_ms"] = min(item["min_ms"], wall_ms)
            item["max_ms"] = max(item["max_ms"], wall_ms)
            item["peak_mem_kb"] = max(item["peak_mem_kb"], peak_kb)

        result = []
        for item in stats.values():
            item["avg_ms"] = round(item["total_ms"] / item["count"], 4)
            item["total_ms"] = round(item["total_ms"], 4)
            item["min_ms"] = round(item["min_ms"], 4)
            item["max_ms"] = round(item["max_ms"], 4)
            result.append(item)

        result.sort(key=lambda x: x["total_ms"], reverse=True)
        return result

    def format_summary_table(self) -> str:
        """Return a formatted ASCII table of the trace summary."""
        items = self.summary()
        if not items:
            return "No trace data recorded."

        headers = [
            "Operation",
            "Calls",
            "Total (ms)",
            "Avg (ms)",
            "Min (ms)",
            "Max (ms)",
            "Peak Mem (KB)",
        ]
        col_widths = [len(h) for h in headers]
        for it in items:
            col_widths[0] = max(col_widths[0], len(str(it["name"])))
            col_widths[1] = max(col_widths[1], len(str(it["count"])))
            col_widths[2] = max(col_widths[2], len(f"{it['total_ms']:.3f}"))
            col_widths[3] = max(col_widths[3], len(f"{it['avg_ms']:.3f}"))
            col_widths[4] = max(col_widths[4], len(f"{it['min_ms']:.3f}"))
            col_widths[5] = max(col_widths[5], len(f"{it['max_ms']:.3f}"))
            col_widths[6] = max(col_widths[6], len(f"{it['peak_mem_kb']:.1f}"))

        line = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        header_row = (
            "| "
            + " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
            + " |"
        )

        rows = [line, header_row, line]
        for it in items:
            row = (
                f"| {it['name']:<{col_widths[0]}} "
                f"| {it['count']:>{col_widths[1]}} "
                f"| {it['total_ms']:>{col_widths[2]}.3f} "
                f"| {it['avg_ms']:>{col_widths[3]}.3f} "
                f"| {it['min_ms']:>{col_widths[4]}.3f} "
                f"| {it['max_ms']:>{col_widths[5]}.3f} "
                f"| {it['peak_mem_kb']:>{col_widths[6]}.1f} |"
            )
            rows.append(row)
        rows.append(line)
        return "\n".join(rows)


# Global tracer singleton
_GLOBAL_TRACER = PerformanceTracer()


def get_global_tracer() -> PerformanceTracer:
    """Return the global PerformanceTracer instance."""
    return _GLOBAL_TRACER


def enable_tracing(track_memory: bool = True) -> PerformanceTracer:
    """Enable global performance tracing."""
    _GLOBAL_TRACER.track_memory = track_memory
    return _GLOBAL_TRACER.start()


def disable_tracing() -> None:
    """Disable global performance tracing."""
    _GLOBAL_TRACER.stop()


def is_tracing_enabled() -> bool:
    """Check if global performance tracing is currently enabled."""
    return _GLOBAL_TRACER.is_enabled


@contextmanager
def trace_op(
    name: str, category: str = "op", **metadata: Any
) -> Iterator[Dict[str, Any]]:
    """Context manager for tracing an operation block using the global tracer."""
    with _GLOBAL_TRACER.trace(name, category=category, **metadata) as span_args:
        yield span_args


def traced(name: Optional[str] = None, category: str = "function"):
    """Decorator to trace function execution with the global tracer."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        op_name = name or fn.__qualname__

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_op(op_name, category=category):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def profile_code(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Tuple[Any, pstats.Stats]:
    """Profile a function execution using cProfile and return (result, pstats.Stats)."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        result = fn(*args, **kwargs)
    finally:
        profiler.disable()
    stats = pstats.Stats(profiler)
    return result, stats
