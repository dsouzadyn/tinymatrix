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


# --- Robustness and Error Handling ---

def test_init_invalid_inputs():
    """Test initialization with invalid arguments."""
    with pytest.raises(ValueError, match="Provide either matrix or"):
        Matrix()
    with pytest.raises(ValueError, match="Provide either matrix or"):
        Matrix(m=5)
    with pytest.raises(ValueError, match="Provide either matrix or"):
        Matrix(n=5)


def test_init_ragged_matrix():
    """Test initialization with consistent row lengths."""
    ragged = [[1, 2], [3, 4, 5]]
    with pytest.raises(ShapeError, match="All rows must have equal length"):
        Matrix(matrix=ragged)


def test_eq_robustness():
    """Test equality against different types and shapes."""
    A = Matrix(3, 3)
    assert A != "string"
    assert A != 123
    assert A != Matrix(2, 3)
    assert A != Matrix(3, 2)


def test_getitem_errors():
    """Test getitem with invalid indices and types."""
    A = Matrix.identity(3)
    
    # Index out of bounds
    with pytest.raises(IndexError):
        _ = A[10, 0]
    with pytest.raises(IndexError):
        _ = A[0, 10]
        
    # Invalid index type
    with pytest.raises(TypeError, match="Invalid index type"):
        _ = A["invalid"]
    with pytest.raises(TypeError, match="Invalid index type"):
        _ = A[1.5]


def test_getitem_slice_step():
    """Test slicing with steps."""
    data = [[1, 2, 3, 4], 
            [5, 6, 7, 8], 
            [9, 10, 11, 12]]
    A = Matrix(matrix=data)
    
    # Slice rows with step 2
    sub = A[::2, :]
    assert sub.M == [[1, 2, 3, 4], [9, 10, 11, 12]]
    
    # Slice cols with step 2
    sub = A[:, ::2]
    assert sub.M == [[1, 3], [5, 7], [9, 11]]


def test_setitem_errors():
    """Test setitem error conditions."""
    A = Matrix(2, 2)
    
    # 1. Invalid index type
    with pytest.raises(TypeError, match="Invalid index type"):
        A["key"] = 0
        
    # 2. Slice assignment errors
    # Assigning scalar to slice (requires Matrix)
    with pytest.raises(TypeError, match="Slice assignment requires a Matrix"):
        A[:, :] = 5
        
    # Shape mismatch in slice assignment
    with pytest.raises(ShapeError, match="Slice shape mismatch"):
        A[:, :] = Matrix(3, 3) # Target is 2x2, Source is 3x3
        
    # 3. Single element assignment errors
    # Assigning non-numeric
    with pytest.raises(TypeError, match="Single element assignment requires a number"):
        A[0, 0] = "string"
        
    # 4. Row assignment errors
    # Assigning non-list/tuple
    with pytest.raises(TypeError, match="Row assignment requires a list or tuple"):
        A[0] = 5
    
    # Row length mismatch
    with pytest.raises(ShapeError, match="Row length mismatch"):
        A[0] = [1, 2, 3] # Matrix is 2x2


def test_arithmetic_errors():
    """Test arithmetic operations with invalid types/shapes."""
    A = Matrix(2, 2)
    B = Matrix(3, 3)
    
    with pytest.raises(ShapeError, match="Matrix size mismatch for addition"):
        _ = A + B
        
    with pytest.raises(ShapeError, match="Matrix size mismatch for subtraction"):
        _ = A - B
        
    with pytest.raises(TypeError, match="Matrix can only be multiplied by a scalar"):
        _ = A * "string"


def test_copy_independence():
    """Test that copy creates a deep copy of the data."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = A.copy()
    
    # Modify B, A should remain unchanged
    B[0, 0] = 99
    assert A[0, 0] == 1
    assert B[0, 0] == 99.0


def test_empty_matrix():
    """Test behavior with zero dimensions."""
    # 0x0 Matrix
    E = Matrix(0, 0)
    assert E.shape() == (0, 0)
    assert str(E) == "" # Should be empty string or harmless
    
    # Nx0 Matrix
    E2 = Matrix(2, 0)
    assert E2.shape() == (2, 0)
    assert E2.M == [[], []]
    
    # 0xN Matrix
    E3 = Matrix(0, 2)
    assert E3.shape() == (0, 2)
    assert E3.M == []

