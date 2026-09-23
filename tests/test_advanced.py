import math
import pytest

from tinymatrix import (
    CSCMatrix,
    CSRMatrix,
    Matrix,
    ShapeError,
    block,
    cond,
    hstack,
    pinv,
    vstack,
)


def _assert_matrices_close(A: Matrix, B: Matrix, tol: float = 1e-6):
    assert A.shape() == B.shape()
    m, n = A.shape()
    for r in range(m):
        for c in range(n):
            assert math.isclose(float(A[r, c]), float(B[r, c]), abs_tol=tol)


# =============================================================================
# Sparse Matrix (CSR and CSC) Tests
# =============================================================================


def test_csr_basic():
    dense = Matrix(matrix=[[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    csr = CSRMatrix.from_dense(dense)

    assert csr.shape() == (3, 3)
    assert csr.nnz() == 5
    assert csr[0, 0] == 1.0
    assert csr[0, 1] == 0.0
    assert csr[0, 2] == 2.0
    assert csr[2, 1] == 5.0

    with pytest.raises(IndexError):
        _ = csr[10, 0]

    reconstructed = csr.to_dense()
    assert reconstructed == dense


def test_csc_basic():
    dense = Matrix(matrix=[[1, 0, 2], [0, 0, 3], [4, 5, 0]])
    csc = CSCMatrix.from_dense(dense)

    assert csc.shape() == (3, 3)
    assert csc.nnz() == 5
    assert csc[0, 0] == 1.0
    assert csc[0, 1] == 0.0
    assert csc[2, 0] == 4.0

    with pytest.raises(IndexError):
        _ = csc[0, 10]

    reconstructed = csc.to_dense()
    assert reconstructed == dense


def test_sparse_conversions_and_matmul():
    dense1 = Matrix(matrix=[[1, 0], [0, 2]])
    dense2 = Matrix(matrix=[[3, 4], [5, 6]])

    csr1 = CSRMatrix.from_dense(dense1)
    csc1 = csr1.to_csc()
    csr_back = csc1.to_csr()
    assert csr_back.to_dense() == dense1

    # CSR @ dense
    res = csr1 @ dense2
    assert res == dense1 @ dense2

    # CSR @ CSR
    csr2 = CSRMatrix.from_dense(dense2)
    res_sparse = csr1 @ csr2
    assert res_sparse == dense1 @ dense2

    # Shape mismatch in matmul
    with pytest.raises(ShapeError):
        _ = csr1 @ Matrix(3, 2)


# =============================================================================
# Immutability (frozen=True) Tests
# =============================================================================


def test_frozen_matrix():
    A = Matrix(matrix=[[1, 2], [3, 4]], frozen=True)
    assert A.is_frozen() is True

    with pytest.raises(TypeError, match="Cannot modify frozen/immutable Matrix"):
        A[0, 0] = 99

    with pytest.raises(TypeError, match="Cannot modify frozen/immutable Matrix"):
        A[0] = [10, 20]

    with pytest.raises(TypeError, match="Cannot modify frozen/immutable Matrix"):
        A[:, :] = Matrix(2, 2)

    # Copy preserves frozen state
    B = A.copy()
    assert B.is_frozen() is True

    # Unfreeze returns a mutable copy
    C = A.unfreeze()
    assert C.is_frozen() is False
    C[0, 0] = 99
    assert C[0, 0] == 99.0
    assert A[0, 0] == 1.0  # Original unchanged

    # Freeze an existing matrix
    D = Matrix(matrix=[[1, 2]])
    assert D.is_frozen() is False
    D.freeze()
    assert D.is_frozen() is True
    with pytest.raises(TypeError):
        D[0, 0] = 5


# =============================================================================
# Stacking Operations Tests
# =============================================================================


def test_vstack():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[5, 6]])

    V = vstack([A, B])
    assert V.shape() == (3, 2)
    assert V.M == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    with pytest.raises(ValueError):
        _ = vstack([])

    with pytest.raises(ShapeError):
        _ = vstack([A, Matrix(1, 3)])

    with pytest.raises(TypeError):
        _ = vstack([A, "invalid"])


def test_hstack():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[5], [6]])

    H = hstack([A, B])
    assert H.shape() == (2, 3)
    assert H.M == [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]

    with pytest.raises(ValueError):
        _ = hstack([])

    with pytest.raises(ShapeError):
        _ = hstack([A, Matrix(3, 1)])

    with pytest.raises(TypeError):
        _ = hstack([A, "invalid"])


def test_block():
    A = Matrix.identity(2)
    B = Matrix.zeroes(2, 2)
    C = Matrix.zeroes(2, 2)
    D = Matrix.identity(2) * 2

    # Block matrix:
    # [I2  0]
    # [0  2*I2]
    BLK = block([[A, B], [C, D]])
    assert BLK.shape() == (4, 4)
    expected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
    ]
    assert BLK.M == expected

    with pytest.raises(ValueError):
        _ = block([])

    with pytest.raises(ValueError):
        _ = block([[]])


# =============================================================================
# Advanced Indexing & Boolean Masks Tests
# =============================================================================


def test_fancy_indexing():
    A = Matrix(matrix=[[1, 2], [3, 4], [5, 6]])

    # Select rows 0 and 2
    sub = A[[0, 2]]
    assert sub.shape() == (2, 2)
    assert sub.M == [[1.0, 2.0], [5.0, 6.0]]


def test_boolean_masks():
    A = Matrix(matrix=[[1, 2], [3, 4]])

    # Comparison operators
    gt = A > 2
    assert gt.M == [[0.0, 0.0], [1.0, 1.0]]

    ge = A >= 2
    assert ge.M == [[0.0, 1.0], [1.0, 1.0]]

    lt = A < 3
    assert lt.M == [[1.0, 1.0], [0.0, 0.0]]

    le = A <= 3
    assert le.M == [[1.0, 1.0], [1.0, 0.0]]

    # Boolean mask indexing
    vals = A[A > 2]
    assert vals == [3.0, 4.0]

    vals_le = A[A <= 2]
    assert vals_le == [1.0, 2.0]


# =============================================================================
# Condition Number Tests
# =============================================================================


def test_cond():
    I3 = Matrix.identity(3)
    assert math.isclose(I3.cond(), 1.0)
    assert math.isclose(cond(I3), 1.0)
    assert math.isclose(I3.cond(p=1), 1.0)

    # Singular matrix has inf condition number
    Z = Matrix(matrix=[[1, 2], [2, 4]])
    assert Z.cond() == float("inf")
    assert Z.cond(p=1) == float("inf")

    non_square = Matrix(2, 3)
    with pytest.raises(ShapeError):
        _ = non_square.cond()


# =============================================================================
# Pseudo-Inverse (pinv) Tests
# =============================================================================


def test_pinv_invertible():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    inv_A = A.inv()
    pinv_A = A.pinv()
    _assert_matrices_close(inv_A, pinv_A)
    _assert_matrices_close(pinv(A), inv_A)


def test_pinv_tall():
    # 3x2 matrix
    A = Matrix(matrix=[[1, 0], [0, 1], [0, 0]])
    A_pinv = A.pinv()
    assert A_pinv.shape() == (2, 3)

    # Moore-Penrose condition 1: A @ A_pinv @ A = A
    _assert_matrices_close(A @ A_pinv @ A, A)

    # Moore-Penrose condition 2: A_pinv @ A @ A_pinv = A_pinv
    _assert_matrices_close(A_pinv @ A @ A_pinv, A_pinv)


def test_pinv_wide():
    # 2x3 matrix
    A = Matrix(matrix=[[1, 0, 0], [0, 1, 0]])
    A_pinv = pinv(A)
    assert A_pinv.shape() == (3, 2)
    _assert_matrices_close(A @ A_pinv @ A, A)


def test_pinv_empty():
    E = Matrix(0, 0)
    assert E.pinv() == Matrix(0, 0)

    Z = Matrix.zeroes(2, 2)
    _assert_matrices_close(Z.pinv(), Matrix.zeroes(2, 2))
