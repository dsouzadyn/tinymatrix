import math
import pytest

from tinymatrix import Matrix, ShapeError
from tinymatrix.optimization import (
    from_numpy,
    matmul_blocked,
    matmul_strassen,
    to_numpy,
)


def _assert_matrices_close(A: Matrix, B: Matrix, tol: float = 1e-6):
    assert A.shape() == B.shape()
    m, n = A.shape()
    for r in range(m):
        for c in range(n):
            assert math.isclose(float(A[r, c]), float(B[r, c]), abs_tol=tol)


# =============================================================================
# Blocked & Strassen Matmul Tests
# =============================================================================


def test_matmul_blocked():
    A = Matrix.random(8, 6)
    B = Matrix.random(6, 10)

    standard = A @ B
    blocked = matmul_blocked(A, B, block_size=4)
    blocked_method = A.matmul_blocked(B, block_size=4)

    _assert_matrices_close(standard, blocked)
    _assert_matrices_close(standard, blocked_method)


def test_matmul_blocked_shape_error():
    A = Matrix(2, 3)
    B = Matrix(4, 5)
    with pytest.raises(ShapeError):
        _ = matmul_blocked(A, B)


def test_matmul_strassen():
    # Square 8x8 matrix
    A = Matrix.random(8, 8)
    B = Matrix.random(8, 8)

    standard = A @ B
    strassen = matmul_strassen(A, B, threshold=2)
    strassen_method = A.matmul_strassen(B, threshold=2)

    _assert_matrices_close(standard, strassen)
    _assert_matrices_close(standard, strassen_method)


def test_matmul_strassen_non_power_of_2():
    # Non-power-of-2 dimensions: 6x6
    A = Matrix.random(6, 6)
    B = Matrix.random(6, 6)

    standard = A @ B
    strassen = matmul_strassen(A, B, threshold=2)
    _assert_matrices_close(standard, strassen)


def test_matmul_strassen_shape_error():
    A = Matrix(2, 3)
    B = Matrix(4, 5)
    with pytest.raises(ShapeError):
        _ = matmul_strassen(A, B)


# =============================================================================
# Lazy Matrix Evaluation Tests
# =============================================================================
def test_lazy_matrix_basic():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    lazy_A = A.lazy()
    assert lazy_A.shape() == (2, 2)
    assert lazy_A.evaluate() == A
    assert lazy_A.eval() == A


def test_lazy_matrix_chained():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[5, 6], [7, 8]])
    C = Matrix(matrix=[[2, 0], [0, 2]])

    # Expression: (A + B) @ C - A * 2
    lazy_expr = (A.lazy() + B.lazy()) @ C.lazy() - A.lazy() * 2
    actual = lazy_expr.evaluate()

    expected = (A + B) @ C - A * 2
    assert actual == expected


def test_lazy_matrix_rmul_and_transpose():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    lazy_expr = 3 * A.lazy().T
    actual = lazy_expr.eval()
    expected = 3 * A.T
    assert actual == expected


def test_lazy_matrix_with_concrete_matrices():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[5, 6], [7, 8]])

    # Interop between LazyMatrix and Matrix
    lazy_expr = A.lazy() + B - A
    assert lazy_expr.eval() == B


# =============================================================================
# NumPy Interoperability Tests
# =============================================================================


def test_numpy_interop():
    class DummyArray:
        def __init__(self, data, dtype="float64"):
            self._data = data
            self.dtype = dtype

        def tolist(self):
            return self._data

    arr = DummyArray([[1, 2], [3, 4]], dtype="int32")
    mat = from_numpy(arr)
    assert mat.shape() == (2, 2)
    assert mat.dtype == "int"
    assert mat.M == [[1, 2], [3, 4]]

    # Test complex array conversion
    arr_c = DummyArray([[1 + 1j]], dtype="complex128")
    mat_c = Matrix.from_numpy(arr_c)
    assert mat_c.dtype == "complex"

    try:
        import numpy  # noqa: F401

        res = to_numpy(mat)
        assert res.shape == (2, 2)
        res_class = mat.to_numpy()
        assert res_class.shape == (2, 2)
        mat_from_res = Matrix.from_numpy(res)
        assert mat_from_res.shape == (2, 2)
    except ImportError:
        with pytest.raises(ImportError, match="NumPy is not installed"):
            to_numpy(mat)
        with pytest.raises(ImportError, match="NumPy is not installed"):
            mat.to_numpy()


def test_strassen_threshold_base_case():
    A = Matrix.random(4, 4)
    B = Matrix.random(4, 4)
    # Strassen rec hits n <= threshold
    res = matmul_strassen(A, B, threshold=4)
    _assert_matrices_close(A @ B, res)


def test_benchmark_script_runs():
    from benchmarks.benchmark import run_benchmarks

    run_benchmarks(sizes=(2,))
