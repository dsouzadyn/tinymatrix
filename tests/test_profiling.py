import json
import os
import tempfile


from tinymatrix import (
    Matrix,
    PerformanceTracer,
    disable_tracing,
    enable_tracing,
    get_global_tracer,
    is_tracing_enabled,
    profile_code,
    trace_op,
    traced,
)


def test_tracer_lifecycle():
    tracer = PerformanceTracer(track_memory=True)
    assert not tracer.is_enabled
    assert len(tracer.spans) == 0

    tracer.start()
    assert tracer.is_enabled

    with tracer.trace("test_op", category="unit_test", meta="val"):
        _ = [i * i for i in range(1000)]

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.name == "test_op"
    assert span.cat == "unit_test"
    assert span.dur > 0
    assert span.args["meta"] == "val"
    assert "wall_time_ms" in span.args

    tracer.stop()
    assert not tracer.is_enabled

    # Trace while disabled does not record
    with tracer.trace("disabled_op"):
        pass
    assert len(tracer.spans) == 1

    tracer.clear()
    assert len(tracer.spans) == 0


def test_tracer_export_chrome_trace():
    tracer = PerformanceTracer(track_memory=False)
    tracer.start()

    with tracer.trace("step_1"):
        pass

    with tracer.trace("step_2", category="test"):
        pass

    tracer.stop()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name

    try:
        tracer.export_chrome_trace(path)
        assert os.path.exists(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "traceEvents" in data
        assert len(data["traceEvents"]) == 2
        assert data["traceEvents"][0]["name"] == "step_1"
        assert data["traceEvents"][1]["name"] == "step_2"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_tracer_summary_and_table():
    tracer = PerformanceTracer(track_memory=False)

    # Empty summary
    assert tracer.summary() == []
    assert "No trace data recorded" in tracer.format_summary_table()

    tracer.start()
    for _ in range(3):
        with tracer.trace("repeat_op"):
            pass
    with tracer.trace("single_op"):
        pass
    tracer.stop()

    summary = tracer.summary()
    assert len(summary) == 2
    repeat_stat = next(s for s in summary if s["name"] == "repeat_op")
    assert repeat_stat["count"] == 3
    assert repeat_stat["total_ms"] >= 0

    table = tracer.format_summary_table()
    assert "repeat_op" in table
    assert "single_op" in table


def test_global_tracing_toggle_and_context():
    assert get_global_tracer() is not None

    disable_tracing()
    assert not is_tracing_enabled()

    tracer = enable_tracing(track_memory=True)
    assert is_tracing_enabled()
    tracer.clear()

    with trace_op("global_op", category="global"):
        pass

    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "global_op"

    disable_tracing()
    assert not is_tracing_enabled()


def test_traced_decorator():
    tracer = enable_tracing(track_memory=False)
    tracer.clear()

    @traced(name="custom_math", category="test")
    def compute(a, b):
        return a + b

    res = compute(3, 4)
    assert res == 7
    assert len(tracer.spans) == 1
    assert tracer.spans[0].name == "custom_math"
    assert tracer.spans[0].cat == "test"

    @traced()
    def auto_named():
        return 42

    assert auto_named() == 42
    assert len(tracer.spans) == 2
    assert "auto_named" in tracer.spans[1].name

    disable_tracing()


def test_profile_code():
    def slow_loop():
        return sum(i for i in range(5000))

    result, stats = profile_code(slow_loop)
    assert result == sum(range(5000))
    assert stats.total_calls > 0


def test_matrix_operations_record_traces():
    tracer = enable_tracing(track_memory=True)
    tracer.clear()

    A = Matrix.random(4, 4)
    B = Matrix.random(4, 4)

    _ = A @ B
    _ = A.lu()
    _ = A.qr()
    _ = (A.T @ A + Matrix.identity(4) * 0.1).cholesky()
    _ = (A.T @ A).eig()
    _ = A.svd()
    _ = A.solve([1, 1, 1, 1])
    _ = A.lstsq([1, 1, 1, 1])

    names = [s.name for s in tracer.spans]
    assert "matmul" in names
    assert "lu" in names
    assert "qr" in names
    assert "cholesky" in names
    assert "eig" in names
    assert "svd" in names
    assert "solve" in names
    assert "lstsq" in names

    disable_tracing()
