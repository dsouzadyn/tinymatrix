import math
from decimal import Decimal

import pytest

from tinymatrix import Matrix
from tinymatrix.exceptions import ShapeError


# =============================================================================
# Strict Validation & Empty Matrix Tests
# =============================================================================


def test_empty_matrix_data():
    empty1 = Matrix(matrix=[])
    assert empty1.shape() == (0, 0)
    assert empty1.M == []
    assert len(empty1) == 0

    empty2 = Matrix(matrix=[[], []])
    assert empty2.shape() == (2, 0)
    assert empty2.M == [[], []]
    assert len(empty2) == 2


def test_invalid_dimension_types():
    with pytest.raises(TypeError, match="Dimensions m and n must be integers"):
        Matrix(m=2.5, n=3)

    with pytest.raises(TypeError, match="Dimensions m and n must be integers"):
        Matrix(m=True, n=3)

    with pytest.raises(ValueError, match="Dimensions m and n must be non-negative"):
        Matrix(m=-1, n=3)

    with pytest.raises(ValueError, match="Dimensions m and n must be non-negative"):
        Matrix(m=3, n=-2)


def test_invalid_matrix_data_structure():
    with pytest.raises(TypeError, match="Matrix data must be a list of lists"):
        Matrix(matrix="invalid")

    with pytest.raises(TypeError, match="Each row must be a list or tuple"):
        Matrix(matrix=[1, 2, 3])


def test_identity_validation():
    with pytest.raises(TypeError, match="Dimension n must be an integer"):
        Matrix.identity(2.5)

    with pytest.raises(TypeError, match="Dimension n must be an integer"):
        Matrix.identity(True)

    with pytest.raises(ValueError, match="Dimension n must be non-negative"):
        Matrix.identity(-1)


# =============================================================================
# Python Protocols: __len__, __iter__
# =============================================================================


def test_len_protocol():
    A = Matrix(3, 4)
    assert len(A) == 3


def test_iter_protocol():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    rows = list(A)
    assert len(rows) == 2
    assert all(isinstance(r, Matrix) for r in rows)
    assert rows[0].M == [[1.0, 2.0, 3.0]]
    assert rows[1].M == [[4.0, 5.0, 6.0]]


# =============================================================================
# Division & In-Place Operators
# =============================================================================


def test_scalar_truediv():
    A = Matrix(matrix=[[2, 4], [6, 8]])
    B = A / 2
    assert B.M == [[1.0, 2.0], [3.0, 4.0]]

    with pytest.raises(ZeroDivisionError):
        _ = A / 0


def test_matrix_truediv():
    A = Matrix(matrix=[[6, 8], [10, 12]])
    B = Matrix(matrix=[[2, 4], [5, 3]])
    C = A / B
    assert C.M == [[3.0, 2.0], [2.0, 4.0]]

    Z = Matrix(matrix=[[1, 0], [1, 1]])
    with pytest.raises(ZeroDivisionError):
        _ = A / Z


def test_rtruediv():
    A = Matrix(matrix=[[1, 2], [4, 5]])
    B = 20 / A
    assert B.M == [[20.0, 10.0], [5.0, 4.0]]

    Z = Matrix(matrix=[[0, 2], [4, 5]])
    with pytest.raises(ZeroDivisionError):
        _ = 20 / Z

    with pytest.raises(TypeError):
        _ = "invalid" / A


def test_inplace_arithmetic():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[5, 6], [7, 8]])

    A += B
    assert A.M == [[6.0, 8.0], [10.0, 12.0]]

    A -= B
    assert A.M == [[1.0, 2.0], [3.0, 4.0]]

    A *= 2
    assert A.M == [[2.0, 4.0], [6.0, 8.0]]

    A /= 2
    assert A.M == [[1.0, 2.0], [3.0, 4.0]]


def test_radd_and_rsub():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = 5 + A
    assert B.M == [[6.0, 7.0], [8.0, 9.0]]

    C = 10 - A
    assert C.M == [[9.0, 8.0], [7.0, 6.0]]


def test_arithmetic_invalid_operands():
    A = Matrix(2, 2)
    with pytest.raises(TypeError):
        _ = A + "str"

    with pytest.raises(TypeError):
        _ = A - "str"

    with pytest.raises(TypeError):
        _ = A @ "str"

    with pytest.raises(TypeError):
        _ = A / "str"


# =============================================================================
# Matrix Power
# =============================================================================


def test_matrix_power():
    A = Matrix(matrix=[[1, 2], [3, 4]])

    P0 = A**0
    assert P0 == Matrix.identity(2)

    P1 = A**1
    assert P1 == A

    P2 = A**2
    assert P2 == A @ A

    P3 = A**3
    assert P3 == A @ A @ A

    with pytest.raises(ValueError):
        _ = A ** (-1)

    with pytest.raises(ValueError):
        _ = A**1.5

    non_square = Matrix(2, 3)
    with pytest.raises(ShapeError):
        _ = non_square**2


# =============================================================================
# Helpers: row, col, flatten, diagonal, trace, is_square, tolist
# =============================================================================


def test_row_and_col_accessors():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])

    r0 = A.row(0)
    assert r0.shape() == (1, 3)
    assert r0.M == [[1.0, 2.0, 3.0]]

    r1 = A.row(-1)
    assert r1.shape() == (1, 3)
    assert r1.M == [[4.0, 5.0, 6.0]]

    c0 = A.col(0)
    assert c0.shape() == (2, 1)
    assert c0.M == [[1.0], [4.0]]

    c2 = A.col(-1)
    assert c2.shape() == (2, 1)
    assert c2.M == [[3.0], [6.0]]

    with pytest.raises(IndexError):
        _ = A.row(5)

    with pytest.raises(IndexError):
        _ = A.col(5)

    with pytest.raises(TypeError):
        _ = A.row("0")


def test_flatten_diagonal_trace():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A.flatten() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert A.diagonal() == [1.0, 5.0, 9.0]
    assert A.trace() == 15.0
    assert A.is_square() is True
    assert A.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]

    non_square = Matrix(2, 3)
    assert non_square.is_square() is False
    with pytest.raises(ShapeError, match="Trace is only defined for square"):
        _ = non_square.trace()

    empty = Matrix(0, 0)
    assert empty.trace() == 0.0


# =============================================================================
# Reductions: sum, mean, min, max, std
# =============================================================================


def test_reductions_axis_none():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    assert A.sum() == 10.0
    assert A.mean() == 2.5
    assert A.min() == 1.0
    assert A.max() == 4.0
    assert math.isclose(A.std(), math.sqrt(1.25))


def test_reductions_axis_0():
    A = Matrix(matrix=[[1, 2], [3, 4]])

    s0 = A.sum(axis=0)
    assert s0.shape() == (1, 2)
    assert s0.M == [[4.0, 6.0]]

    m0 = A.mean(axis=0)
    assert m0.shape() == (1, 2)
    assert m0.M == [[2.0, 3.0]]

    min0 = A.min(axis=0)
    assert min0.shape() == (1, 2)
    assert min0.M == [[1.0, 2.0]]

    max0 = A.max(axis=0)
    assert max0.shape() == (1, 2)
    assert max0.M == [[3.0, 4.0]]

    std0 = A.std(axis=0)
    assert std0.shape() == (1, 2)
    assert std0.M == [[1.0, 1.0]]


def test_reductions_axis_1():
    A = Matrix(matrix=[[1, 2], [3, 4]])

    s1 = A.sum(axis=1)
    assert s1.shape() == (2, 1)
    assert s1.M == [[3.0], [7.0]]

    m1 = A.mean(axis=1)
    assert m1.shape() == (2, 1)
    assert m1.M == [[1.5], [3.5]]

    min1 = A.min(axis=1)
    assert min1.shape() == (2, 1)
    assert min1.M == [[1.0], [3.0]]

    max1 = A.max(axis=1)
    assert max1.shape() == (2, 1)
    assert max1.M == [[2.0], [4.0]]


def test_reductions_invalid_axis():
    A = Matrix(2, 2)
    with pytest.raises(ValueError):
        _ = A.sum(axis=2)

    with pytest.raises(ValueError):
        _ = A.mean(axis=-1)


def test_reductions_empty_matrix():
    E = Matrix(0, 0)
    assert E.sum() == 0.0

    with pytest.raises(ValueError):
        _ = E.mean()

    with pytest.raises(ValueError):
        _ = E.min()

    with pytest.raises(ValueError):
        _ = E.max()

    with pytest.raises(ValueError):
        _ = E.std()


def test_reductions_complex_unsupported():
    C = Matrix(matrix=[[1 + 1j]], dtype="complex")
    with pytest.raises(TypeError):
        _ = C.min()

    with pytest.raises(TypeError):
        _ = C.max()

    with pytest.raises(TypeError):
        _ = C.std()


# =============================================================================
# Transformations: reshape, squeeze, expand_dims
# =============================================================================


def test_reshape():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    R = A.reshape(3, 2)
    assert R.shape() == (3, 2)
    assert R.M == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    with pytest.raises(ShapeError):
        _ = A.reshape(2, 2)

    with pytest.raises(TypeError):
        _ = A.reshape(2.5, 3)

    with pytest.raises(ValueError):
        _ = A.reshape(-1, 6)


def test_squeeze():
    row_mat = Matrix(matrix=[[1, 2, 3]])
    assert row_mat.squeeze() == [1.0, 2.0, 3.0]
    assert row_mat.squeeze(axis=0) == [1.0, 2.0, 3.0]

    with pytest.raises(ValueError):
        _ = row_mat.squeeze(axis=1)

    col_mat = Matrix(matrix=[[1], [2], [3]])
    assert col_mat.squeeze() == [1.0, 2.0, 3.0]
    assert col_mat.squeeze(axis=1) == [1.0, 2.0, 3.0]

    single = Matrix(matrix=[[42]])
    assert single.squeeze() == 42.0

    square = Matrix(matrix=[[1, 2], [3, 4]])
    assert square.squeeze() == square


def test_expand_dims():
    v = [1, 2, 3]
    r = Matrix.expand_dims(v, axis=0)
    assert r.shape() == (1, 3)
    assert r.M == [[1.0, 2.0, 3.0]]

    c = Matrix.expand_dims(v, axis=1)
    assert c.shape() == (3, 1)
    assert c.M == [[1.0], [2.0], [3.0]]

    with pytest.raises(ValueError):
        _ = Matrix.expand_dims(v, axis=2)


# =============================================================================
# Apply & Element-wise Math Functions
# =============================================================================


def test_apply():
    A = Matrix(matrix=[[1, 2], [3, 4]], dtype="int")
    B = A.apply(lambda x: x * 10)
    assert B.M == [[10, 20], [30, 40]]
    assert B.dtype == "int"


def test_elementwise_math():
    A = Matrix(matrix=[[-1, 4], [-9, 16]])

    assert A.abs().M == [[1.0, 4.0], [9.0, 16.0]]
    assert abs(A).M == [[1.0, 4.0], [9.0, 16.0]]

    P = Matrix(matrix=[[4, 9], [16, 25]])
    assert P.sqrt().M == [[2.0, 3.0], [4.0, 5.0]]

    Z = Matrix.zeroes(1, 2)
    assert Z.exp().M == [[1.0, 1.0]]
    assert Z.sin().M == [[0.0, 0.0]]
    assert Z.cos().M == [[1.0, 1.0]]

    E = Matrix(matrix=[[math.e]])
    assert math.isclose(E.log()[0, 0], 1.0)


def test_elementwise_math_complex():
    C = Matrix(matrix=[[1 + 2j]], dtype="complex")
    assert C.abs().dtype == "complex"
    assert C.sqrt().dtype == "complex"
    assert C.exp().dtype == "complex"
    assert C.log().dtype == "complex"
    assert C.sin().dtype == "complex"
    assert C.cos().dtype == "complex"


def test_decimal_with_core_ops():
    A = Matrix(
        matrix=[[Decimal("1.5"), Decimal("2.5")], [Decimal("3.5"), Decimal("4.5")]],
        dtype="decimal",
    )
    assert A.sum() == Decimal("12.0")
    assert A.trace() == Decimal("6.0")
    assert A.row(0).M == [[Decimal("1.5"), Decimal("2.5")]]
    assert A.flatten() == [
        Decimal("1.5"),
        Decimal("2.5"),
        Decimal("3.5"),
        Decimal("4.5"),
    ]


# =============================================================================
# Random Constructors
# =============================================================================


def test_random_constructors():
    U = Matrix.random(3, 4, low=2.0, high=5.0)
    assert U.shape() == (3, 4)
    assert all(2.0 <= x < 5.0 for x in U.flatten())

    U2 = Matrix.uniform(2, 3, low=-1.0, high=1.0)
    assert U2.shape() == (2, 3)
    assert all(-1.0 <= x < 1.0 for x in U2.flatten())

    N = Matrix.normal(4, 4, mean=0.0, std=1.0)
    assert N.shape() == (4, 4)
    assert len(N.flatten()) == 16


# =============================================================================
# Broadcasting Tests
# =============================================================================


def test_broadcasting_row_vector():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])  # 2x3
    row = Matrix(matrix=[[10, 20, 30]])  # 1x3

    C = A + row
    assert C.shape() == (2, 3)
    assert C.M == [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]

    M = A * row
    assert M.shape() == (2, 3)
    assert M.M == [[10.0, 40.0, 90.0], [40.0, 100.0, 180.0]]


def test_broadcasting_col_vector():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])  # 2x3
    col = Matrix(matrix=[[10], [20]])  # 2x1

    C = A + col
    assert C.shape() == (2, 3)
    assert C.M == [[11.0, 12.0, 13.0], [24.0, 25.0, 26.0]]

    D = A / col
    assert D.shape() == (2, 3)
    assert D.M == [[0.1, 0.2, 0.3], [0.2, 0.25, 0.3]]


def test_broadcasting_shape_mismatch():
    A = Matrix(2, 3)
    B = Matrix(3, 2)
    with pytest.raises(ShapeError, match="Matrix size mismatch for addition"):
        _ = A + B


# =============================================================================
# Large Matrix String Representation
# =============================================================================


def test_large_matrix_repr():
    L = Matrix.ones(15, 15)
    r = repr(L)
    assert "..." in r
    assert r.count("\n") == 6  # 3 top rows + ... + 3 bottom rows = 7 lines (6 newlines)

    wide = Matrix.ones(2, 15)
    rw = repr(wide)
    assert "..." in rw


def test_scalar_validation_containers():
    A = Matrix(2, 2)
    assert A._is_valid_scalar([1, 2]) is False
    assert A._is_valid_scalar({"a": 1}) is False
    assert A._is_valid_scalar(A) is False
    assert A._is_valid_scalar({1, 2}) is False
    assert A._is_valid_scalar((1, 2)) is False


def test_expand_dims_from_matrix():
    A = Matrix(2, 3)
    r = Matrix.expand_dims(A, axis=0)
    assert r.shape() == (1, 6)
    c = Matrix.expand_dims(A, axis=1)
    assert c.shape() == (6, 1)

    with pytest.raises(TypeError, match="Expected Matrix or sequence"):
        _ = Matrix.expand_dims(12345, axis=0)


def test_reduction_axis_edge_cases():
    A = Matrix(2, 2)
    with pytest.raises(ValueError):
        _ = A.min(axis=2)
    with pytest.raises(ValueError):
        _ = A.max(axis=2)
    with pytest.raises(ValueError):
        _ = A.std(axis=2)
    with pytest.raises(ValueError, match="Degrees of freedom"):
        _ = Matrix(matrix=[[42.0]]).std(ddof=1)

    E_cols = Matrix(0, 3)
    assert E_cols.sum(axis=0).shape() == (1, 3)
    E_rows = Matrix(3, 0)
    assert E_rows.sum(axis=1).shape() == (3, 1)


def test_rsub_invalid():
    A = Matrix(2, 2)
    with pytest.raises(TypeError):
        _ = "str" - A


def test_zero_dimension_broadcasting_mismatch():
    A = Matrix(0, 2)
    B = Matrix(1, 2)
    with pytest.raises(ShapeError):
        _ = A + B


def test_remaining_edge_cases_100_percent():
    A = Matrix(2, 3)
    with pytest.raises(TypeError):
        _ = A.col("0")

    with pytest.raises(ValueError):
        _ = A.squeeze(axis=2)

    with pytest.raises(ValueError, match="Cannot squeeze axis 0"):
        _ = A.squeeze(axis=0)

    std1 = A.std(axis=1)
    assert std1.shape() == (2, 1)

    tall = Matrix.ones(15, 2)
    rtall = repr(tall)
    assert "..." in rtall

    with pytest.raises(ShapeError):
        _ = Matrix(2, 3) + Matrix(2, 2)

    with pytest.raises(TypeError, match="Unsupported operand type"):
        _ = A._elementwise_op(object(), lambda a, b: a, "test")
