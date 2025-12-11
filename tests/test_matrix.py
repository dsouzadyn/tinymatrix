import pytest

from tinymatrix import Matrix
from tinymatrix.exceptions import ShapeError


# --- Arithmetic operations ---
def test_add():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[5, 6], [7, 8]])
    C = A + B
    assert C.M == [[6, 8], [10, 12]]


def test_sub():
    A = Matrix(matrix=[[5, 6], [7, 8]])
    B = Matrix(matrix=[[1, 2], [3, 4]])
    C = A - B
    assert C.M == [[4, 4], [4, 4]]


def test_scalar_mul():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = A * 2
    C = 3 * A
    assert B.M == [[2, 4], [6, 8]]
    assert C.M == [[3, 6], [9, 12]]


def test_neg():
    A = Matrix(matrix=[[1, -2], [3, 4]])
    B = -A
    assert B.M == [[-1, 2], [-3, -4]]


def test_eq():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[1, 2], [3, 4]])
    C = Matrix(matrix=[[4, 3], [2, 1]])
    assert A == B
    assert A != C


# --- Matrix multiplication ---
def test_matmul():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    B = Matrix(matrix=[[7, 8], [9, 10], [11, 12]])
    C = A @ B
    assert C.M == [[58, 64], [139, 154]]


def test_matmul_shape_error():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[1, 2, 3]])
    with pytest.raises(ShapeError):
        _ = A @ B


# --- Transpose ---
def test_transpose():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    T = A.T
    assert T.M == [[1, 4], [2, 5], [3, 6]]


# --- Slicing and indexing ---
def test_getitem():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert A[0, 0] == 1
    assert A[1, 2] == 6
    # Row slice
    sub = A[0:2, 1:3]
    assert sub.M == [[2, 3], [5, 6]]
    # Single row slice
    assert A[1, :].M == [[4, 5, 6]]
    # Single column slice
    assert A[:, 2].M == [[3], [6], [9]]


def test_setitem_single():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    A[0, 1] = 10
    assert A.M == [[1, 10], [3, 4]]


def test_setitem_row():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    A[1] = [5, 6]
    assert A.M == [[1, 2], [5, 6]]


def test_setitem_slice():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = Matrix(matrix=[[10, 11], [12, 13]])
    A[0:2, 1:3] = B
    assert A.M == [[1, 10, 11], [4, 12, 13], [7, 8, 9]]


def test_setitem_shape_error():
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[1, 2, 3]])
    with pytest.raises(ShapeError):
        A[0:1, 0:2] = B


# --- Identity, zeroes, ones ---
def test_static_constructors():
    I = Matrix.identity(3)
    expected_I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert I.M == expected_I

    Z = Matrix.zeroes(2, 3)
    assert Z.M == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    O = Matrix.ones(2, 2)
    assert O.M == [[1.0, 1.0], [1.0, 1.0]]
    O = Matrix.ones(2, 2)
    assert O.M == [[1.0, 1.0], [1.0, 1.0]]
