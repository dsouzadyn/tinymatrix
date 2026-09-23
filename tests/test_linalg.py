import math
from decimal import Decimal

import pytest

from tinymatrix import (
    Matrix,
    ShapeError,
    SingularMatrixError,
    det,
    inv,
    kron,
    matrix_power,
    norm,
    rank,
)


# =============================================================================
# Determinant Tests
# =============================================================================


def test_det_empty():
    A = Matrix(0, 0)
    assert A.det() == 1.0
    assert det(A) == 1.0


def test_det_1x1():
    A = Matrix(matrix=[[5.0]])
    assert A.det() == 5.0
    assert det(A) == 5.0


def test_det_2x2():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    assert A.det() == -2.0
    assert det(A) == -2.0


def test_det_3x3():
    # Known 3x3 matrix
    A = Matrix(matrix=[[6, 1, 1], [4, -2, 5], [2, 8, 7]])
    assert math.isclose(A.det(), -306.0)
    assert math.isclose(det(A), -306.0)


def test_det_4x4():
    # Diagonal matrix det is product of diagonals
    A = Matrix(
        matrix=[
            [2, 0, 0, 0],
            [0, 3, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 5],
        ]
    )
    assert math.isclose(A.det(), 120.0)


def test_det_singular():
    # Row 2 is multiple of row 1
    A = Matrix(matrix=[[1, 2, 3], [2, 4, 6], [7, 8, 9]])
    assert math.isclose(A.det(), 0.0, abs_tol=1e-12)


def test_det_non_square():
    A = Matrix(2, 3)
    with pytest.raises(ShapeError, match="Determinant is only defined for square"):
        _ = A.det()


def test_det_with_dtypes():
    A_int2 = Matrix(matrix=[[1, 2], [3, 4]], dtype="int")
    assert A_int2.det() == -2
    assert isinstance(A_int2.det(), int)

    A_int3 = Matrix(matrix=[[1, 2, 3], [0, 4, 5], [1, 0, 6]], dtype="int")
    assert A_int3.det() == 22
    assert isinstance(A_int3.det(), int)
    A_dec = Matrix(
        matrix=[[Decimal("2.0"), Decimal("0.0")], [Decimal("0.0"), Decimal("3.0")]],
        dtype="decimal",
    )
    assert A_dec.det() == Decimal("6.0")


# =============================================================================
# Matrix Inverse Tests
# =============================================================================


def test_inv_empty():
    A = Matrix(0, 0)
    assert A.inv() == Matrix(0, 0)
    assert inv(A) == Matrix(0, 0)


def test_inv_1x1():
    A = Matrix(matrix=[[4.0]])
    inv_A = A.inv()
    assert inv_A.M == [[0.25]]
    assert inv(A).M == [[0.25]]


def test_inv_2x2():
    A = Matrix(matrix=[[4, 7], [2, 6]])
    inv_A = A.inv()
    product = A @ inv_A

    for r in range(2):
        for c in range(2):
            expected = 1.0 if r == c else 0.0
            assert math.isclose(product[r, c], expected, abs_tol=1e-12)


def test_inv_3x3():
    A = Matrix(matrix=[[1, 2, 3], [0, 1, 4], [5, 6, 0]])
    inv_A = A.inv()
    product = A @ inv_A

    for r in range(3):
        for c in range(3):
            expected = 1.0 if r == c else 0.0
            assert math.isclose(product[r, c], expected, abs_tol=1e-12)


def test_inv_singular():
    A = Matrix(matrix=[[1, 2], [2, 4]])
    with pytest.raises(SingularMatrixError, match="Matrix is singular"):
        _ = A.inv()

    with pytest.raises(SingularMatrixError, match="Matrix is singular"):
        _ = inv(A)


def test_inv_non_square():
    A = Matrix(2, 3)
    with pytest.raises(ShapeError, match="Matrix inverse is only defined for square"):
        _ = A.inv()


# =============================================================================
# Matrix Rank Tests
# =============================================================================


def test_rank_empty():
    assert Matrix(0, 0).rank() == 0
    assert Matrix(0, 3).rank() == 0
    assert Matrix(3, 0).rank() == 0
    assert rank(Matrix(0, 0)) == 0


def test_rank_full():
    A = Matrix(matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert A.rank() == 3
    assert rank(A) == 3


def test_rank_deficient():
    # Row 3 is sum of rows 1 and 2
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [5, 7, 9]])
    assert A.rank() == 2


def test_rank_zero_matrix():
    Z = Matrix.zeroes(3, 4)
    assert Z.rank() == 0


def test_rank_single_row_and_column():
    row = Matrix(matrix=[[1, 2, 3]])
    assert row.rank() == 1

    col = Matrix(matrix=[[1], [2], [3]])
    assert col.rank() == 1


def test_rank_with_complex():
    C = Matrix(matrix=[[1 + 1j, 0], [0, 2 + 2j]], dtype="complex")
    assert C.rank() == 2


# =============================================================================
# Matrix Norm Tests
# =============================================================================


def test_norm_empty():
    assert Matrix(0, 0).norm() == 0.0
    assert norm(Matrix(0, 0)) == 0.0


def test_norm_frobenius():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    assert math.isclose(A.norm(), math.sqrt(30.0))
    assert math.isclose(norm(A, "fro"), math.sqrt(30.0))


def test_norm_1():
    # Max column sum
    A = Matrix(matrix=[[1, -5], [3, 2]])
    # col 0 sum: 1+3=4, col 1 sum: 5+2=7 -> 7.0
    assert A.norm(ord=1) == 7.0
    assert norm(A, ord=1) == 7.0


def test_norm_inf():
    # Max row sum
    A = Matrix(matrix=[[1, -5], [3, 2]])
    # row 0 sum: 1+5=6, row 1 sum: 3+2=5 -> 6.0
    assert A.norm(ord=float("inf")) == 6.0
    assert A.norm(ord="inf") == 6.0


def test_norm_neg_1():
    # Min column sum
    A = Matrix(matrix=[[1, -5], [3, 2]])
    assert A.norm(ord=-1) == 4.0


def test_norm_neg_inf():
    # Min row sum
    A = Matrix(matrix=[[1, -5], [3, 2]])
    assert A.norm(ord=float("-inf")) == 5.0
    assert A.norm(ord="-inf") == 5.0


def test_norm_spectral_2():
    # Vector Euclidean norm
    v_row = Matrix(matrix=[[3, 4]])
    assert math.isclose(v_row.norm(ord=2), 5.0)

    v_col = Matrix(matrix=[[3], [4]])
    assert math.isclose(v_col.norm(ord=2), 5.0)

    # Diagonal matrix: spectral norm is largest diagonal absolute value
    D = Matrix(matrix=[[2, 0], [0, -5]])
    assert math.isclose(D.norm(ord=2), 5.0, abs_tol=1e-5)

    # Zero matrix norm 2
    Z = Matrix.zeroes(2, 2)
    assert Z.norm(ord=2) == 0.0


def test_norm_invalid():
    A = Matrix(2, 2)
    with pytest.raises(ValueError, match="Invalid norm order"):
        _ = A.norm(ord=99)


# =============================================================================
# Kronecker Product Tests
# =============================================================================


def test_kron_identity():
    I2 = Matrix.identity(2)
    K = I2.kron(I2)
    assert K.shape() == (4, 4)
    assert K == Matrix.identity(4)
    assert kron(I2, I2) == Matrix.identity(4)


def test_kron_general():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[0, 5], [6, 7]])

    K = A.kron(B)
    assert K.shape() == (4, 4)
    expected = [
        [0.0, 5.0, 0.0, 10.0],
        [6.0, 7.0, 12.0, 14.0],
        [0.0, 15.0, 0.0, 20.0],
        [18.0, 21.0, 24.0, 28.0],
    ]
    assert K.M == expected


def test_kron_non_square():
    A = Matrix(matrix=[[1, 2]])  # 1x2
    B = Matrix(matrix=[[3], [4]])  # 2x1

    K = A.kron(B)  # (1*2) x (2*1) = 2x2
    assert K.shape() == (2, 2)
    assert K.M == [[3.0, 6.0], [4.0, 8.0]]


def test_kron_invalid():
    A = Matrix(2, 2)
    with pytest.raises(TypeError, match="Kronecker product requires another Matrix"):
        _ = A.kron("invalid")

    with pytest.raises(TypeError):
        _ = kron(A, [1, 2])


def test_matrix_power_linalg():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    assert matrix_power(A, 2) == A @ A
