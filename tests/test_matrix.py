from decimal import Decimal

import pytest

from tinymatrix import Matrix
from tinymatrix.exceptions import ShapeError

# =============================================================================
# Initialization Tests
# =============================================================================


def test_init_with_matrix():
    """Test initialization with explicit matrix data."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    assert A.shape() == (2, 2)
    assert A.M == [[1.0, 2.0], [3.0, 4.0]]


def test_init_with_dimensions():
    """Test initialization with dimensions only."""
    A = Matrix(3, 4)
    assert A.shape() == (3, 4)
    assert all(all(x == 0.0 for x in row) for row in A.M)


def test_init_invalid_inputs():
    """Test initialization with invalid arguments."""
    with pytest.raises(ValueError, match="Provide either matrix or"):
        Matrix()
    with pytest.raises(ValueError, match="Provide either matrix or"):
        Matrix(m=5)
    with pytest.raises(ValueError, match="Provide either matrix or"):
        Matrix(n=5)


def test_init_ragged_matrix():
    """Test initialization with inconsistent row lengths."""
    with pytest.raises(ShapeError, match="All rows must have equal length"):
        Matrix(matrix=[[1, 2], [3, 4, 5]])


def test_init_empty_matrix():
    """Test initialization with zero dimensions."""
    E1 = Matrix(0, 0)
    assert E1.shape() == (0, 0)
    assert E1.M == []

    E2 = Matrix(2, 0)
    assert E2.shape() == (2, 0)
    assert E2.M == [[], []]

    E3 = Matrix(0, 2)
    assert E3.shape() == (0, 2)
    assert E3.M == []


def test_init_single_element():
    """Test 1x1 matrix."""
    A = Matrix(matrix=[[42]])
    assert A.shape() == (1, 1)
    assert A[0, 0] == 42.0


# =============================================================================
# Data Type Tests
# =============================================================================


def test_dtype_int():
    """Test integer dtype."""
    A = Matrix(matrix=[[1, 2], [3, 4]], dtype="int")
    assert A.dtype == "int"
    assert all(isinstance(x, int) for row in A.M for x in row)


def test_dtype_float():
    """Test float dtype (default)."""
    A = Matrix(matrix=[[1, 2], [3, 4]], dtype="float")
    assert A.dtype == "float"
    assert all(isinstance(x, float) for row in A.M for x in row)


def test_dtype_decimal():
    """Test Decimal dtype."""
    A = Matrix(matrix=[[1, 2], [3, 4]], dtype="decimal")
    assert A.dtype == "decimal"
    assert all(isinstance(x, Decimal) for row in A.M for x in row)


def test_dtype_complex():
    """Test complex dtype."""
    A = Matrix(matrix=[[1, 2], [3, 4]], dtype="complex")
    assert A.dtype == "complex"
    assert all(isinstance(x, complex) for row in A.M for x in row)


def test_dtype_invalid():
    """Test invalid dtype."""
    with pytest.raises(ValueError, match="Unsupported dtype"):
        Matrix(2, 2, dtype="invalid")


def test_astype_conversion():
    """Test converting between dtypes."""
    A = Matrix(matrix=[[1, 2], [3, 4]], dtype="int")
    B = A.astype("float")

    assert B.dtype == "float"
    assert all(isinstance(x, float) for row in B.M for x in row)
    assert A.dtype == "int"  # Original unchanged


def test_astype_invalid():
    """Test astype with invalid dtype."""
    A = Matrix(2, 2)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        A.astype("invalid")


# =============================================================================
# Static Constructor Tests
# =============================================================================


def test_zeroes():
    """Test zeroes constructor."""
    Z = Matrix.zeroes(2, 3)
    assert Z.shape() == (2, 3)
    assert Z.M == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_zeroes_with_dtype():
    """Test zeroes with different dtypes."""
    Z = Matrix.zeroes(2, 2, dtype="int")
    assert all(isinstance(x, int) for row in Z.M for x in row)


def test_ones():
    """Test ones constructor."""
    O = Matrix.ones(2, 3)
    assert O.shape() == (2, 3)
    assert O.M == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_ones_with_dtype():
    """Test ones with different dtypes."""
    O = Matrix.ones(2, 2, dtype="decimal")
    assert all(isinstance(x, Decimal) for row in O.M for x in row)
    assert all(x == Decimal(1) for row in O.M for x in row)


def test_identity():
    """Test identity matrix constructor."""
    I = Matrix.identity(3)
    expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert I.M == expected


def test_identity_1x1():
    """Test 1x1 identity matrix."""
    I = Matrix.identity(1)
    assert I.M == [[1.0]]


def test_identity_with_dtype():
    """Test identity with different dtypes."""
    I = Matrix.identity(2, dtype="int")
    assert I.M == [[1, 0], [0, 1]]


# =============================================================================
# Copy Tests
# =============================================================================


def test_copy_independence():
    """Test that copy creates a deep copy."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = A.copy()

    B[0, 0] = 99
    assert A[0, 0] == 1.0
    assert B[0, 0] == 99.0


def test_copy_preserves_dtype():
    """Test that copy preserves dtype."""
    A = Matrix(matrix=[[1, 2]], dtype="decimal")
    B = A.copy()
    assert B.dtype == "decimal"


# =============================================================================
# Indexing and Slicing Tests
# =============================================================================


def test_getitem_single_element():
    """Test getting a single element."""
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    assert A[0, 0] == 1.0
    assert A[1, 2] == 6.0


def test_getitem_row():
    """Test getting a row."""
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    row = A[0]
    assert row.M == [[1.0, 2.0, 3.0]]
    assert row.shape() == (1, 3)


def test_getitem_row_slice():
    """Test getting a row slice."""
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sub = A[1, :]
    assert sub.M == [[4.0, 5.0, 6.0]]


def test_getitem_column_slice():
    """Test getting a column slice."""
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sub = A[:, 1]
    assert sub.M == [[2.0], [5.0], [8.0]]


def test_getitem_submatrix():
    """Test getting a submatrix."""
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    sub = A[0:2, 1:3]
    assert sub.M == [[2.0, 3.0], [5.0, 6.0]]


def test_getitem_slice_with_step():
    """Test slicing with steps."""
    A = Matrix(matrix=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    sub1 = A[::2, :]
    assert sub1.M == [[1.0, 2.0, 3.0, 4.0], [9.0, 10.0, 11.0, 12.0]]

    sub2 = A[:, ::2]
    assert sub2.M == [[1.0, 3.0], [5.0, 7.0], [9.0, 11.0]]


def test_getitem_negative_indices():
    """Test negative indexing."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    assert A[-1, -1] == 4.0
    assert A[-2, -1] == 2.0


def test_getitem_errors():
    """Test getitem with invalid indices."""
    A = Matrix.identity(3)

    with pytest.raises(IndexError):
        _ = A[10, 0]

    with pytest.raises(IndexError):
        _ = A[0, 10]

    with pytest.raises(TypeError, match="Invalid index type"):
        _ = A["invalid"]

    with pytest.raises(TypeError, match="Invalid index type"):
        _ = A[1.5]


def test_setitem_single_element():
    """Test setting a single element."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    A[0, 1] = 10
    assert A.M == [[1.0, 10.0], [3.0, 4.0]]


def test_setitem_with_type_conversion():
    """Test setitem converts to matrix dtype."""
    A = Matrix(matrix=[[1, 2]], dtype="int")
    A[0, 0] = 5.7
    assert A[0, 0] == 5  # Should be truncated to int


def test_setitem_row():
    """Test setting a row."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    A[1] = [5, 6]
    assert A.M == [[1.0, 2.0], [5.0, 6.0]]


def test_setitem_row_with_tuple():
    """Test setting a row with tuple."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    A[0] = (7, 8)
    assert A.M == [[7.0, 8.0], [3.0, 4.0]]


def test_setitem_submatrix():
    """Test setting a submatrix."""
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = Matrix(matrix=[[10, 11], [12, 13]])
    A[0:2, 1:3] = B
    assert A.M == [[1.0, 10.0, 11.0], [4.0, 12.0, 13.0], [7.0, 8.0, 9.0]]


def test_setitem_errors():
    """Test setitem error conditions."""
    A = Matrix(2, 2)

    with pytest.raises(TypeError, match="Invalid index type"):
        A["key"] = 0

    with pytest.raises(TypeError, match="Slice assignment requires a Matrix"):
        A[:, :] = 5

    with pytest.raises(ShapeError, match="Slice shape mismatch"):
        A[:, :] = Matrix(3, 3)

    with pytest.raises(TypeError, match="Single element assignment requires a number"):
        A[0, 0] = "string"

    with pytest.raises(TypeError, match="Row assignment requires a list or tuple"):
        A[0] = 5

    with pytest.raises(ShapeError, match="Row length mismatch"):
        A[0] = [1, 2, 3]


# =============================================================================
# Arithmetic Operations Tests
# =============================================================================


def test_add():
    """Test matrix addition."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[5, 6], [7, 8]])
    C = A + B
    assert C.M == [[6.0, 8.0], [10.0, 12.0]]


def test_add_preserves_dtype():
    """Test addition preserves left operand dtype."""
    A = Matrix(matrix=[[1, 2]], dtype="int")
    B = Matrix(matrix=[[3, 4]], dtype="float")
    C = A + B
    assert C.dtype == "int"


def test_add_shape_mismatch():
    """Test addition with shape mismatch."""
    A = Matrix(2, 2)
    B = Matrix(3, 3)
    with pytest.raises(ShapeError, match="Matrix size mismatch for addition"):
        _ = A + B


def test_sub():
    """Test matrix subtraction."""
    A = Matrix(matrix=[[5, 6], [7, 8]])
    B = Matrix(matrix=[[1, 2], [3, 4]])
    C = A - B
    assert C.M == [[4.0, 4.0], [4.0, 4.0]]


def test_sub_shape_mismatch():
    """Test subtraction with shape mismatch."""
    A = Matrix(2, 2)
    B = Matrix(3, 3)
    with pytest.raises(ShapeError, match="Matrix size mismatch for subtraction"):
        _ = A - B


def test_scalar_mul():
    """Test scalar multiplication."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = A * 2
    C = 3 * A
    assert B.M == [[2.0, 4.0], [6.0, 8.0]]
    assert C.M == [[3.0, 6.0], [9.0, 12.0]]


def test_scalar_mul_with_zero():
    """Test multiplication by zero."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = A * 0
    assert B.M == [[0.0, 0.0], [0.0, 0.0]]


def test_scalar_mul_with_negative():
    """Test multiplication by negative scalar."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = A * -2
    assert B.M == [[-2.0, -4.0], [-6.0, -8.0]]


def test_scalar_mul_with_float():
    """Test multiplication by float scalar."""
    A = Matrix(matrix=[[2, 4], [6, 8]])
    B = A * 0.5
    assert B.M == [[1.0, 2.0], [3.0, 4.0]]


def test_scalar_mul_invalid():
    """Test scalar multiplication with invalid type."""
    A = Matrix(2, 2)
    with pytest.raises(TypeError, match="Matrix can only be multiplied by a scalar"):
        _ = A * "string"


def test_neg():
    """Test matrix negation."""
    A = Matrix(matrix=[[1, -2], [3, 4]])
    B = -A
    assert B.M == [[-1.0, 2.0], [-3.0, -4.0]]


def test_matmul():
    """Test matrix multiplication."""
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    B = Matrix(matrix=[[7, 8], [9, 10], [11, 12]])
    C = A @ B
    assert C.M == [[58.0, 64.0], [139.0, 154.0]]


def test_matmul_identity():
    """Test matrix multiplication with identity."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    I = Matrix.identity(2)
    B = A @ I
    C = I @ A
    assert B == A
    assert C == A


def test_matmul_square_matrices():
    """Test multiplication of square matrices."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[5, 6], [7, 8]])
    C = A @ B
    assert C.M == [[19.0, 22.0], [43.0, 50.0]]


def test_matmul_non_square():
    """Test multiplication of non-square matrices."""
    A = Matrix(matrix=[[1, 2]])  # 1x2
    B = Matrix(matrix=[[3], [4]])  # 2x1
    C = A @ B  # Should be 1x1
    assert C.M == [[11.0]]


def test_matmul_shape_error():
    """Test matrix multiplication with incompatible shapes."""
    A = Matrix(matrix=[[1, 2], [3, 4]])  # 2x2
    B = Matrix(matrix=[[1, 2, 3]])  # 1x3
    with pytest.raises(ShapeError, match="Cannot multiply"):
        _ = A @ B


def test_matmul_with_different_dtypes():
    """Test matrix multiplication with different dtypes."""
    A = Matrix(matrix=[[1, 2]], dtype="int")
    B = Matrix(matrix=[[3], [4]], dtype="float")
    C = A @ B
    assert C.dtype == "int"  # Preserves left operand dtype


# =============================================================================
# Transpose Tests
# =============================================================================


def test_transpose():
    """Test matrix transpose."""
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    T = A.T
    assert T.M == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_transpose_square():
    """Test transpose of square matrix."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    T = A.T
    assert T.M == [[1.0, 3.0], [2.0, 4.0]]


def test_transpose_twice():
    """Test double transpose returns to original."""
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    assert A.T.T == A


def test_transpose_single_row():
    """Test transpose of single row."""
    A = Matrix(matrix=[[1, 2, 3]])
    T = A.T
    assert T.M == [[1.0], [2.0], [3.0]]


def test_transpose_single_column():
    """Test transpose of single column."""
    A = Matrix(matrix=[[1], [2], [3]])
    T = A.T
    assert T.M == [[1.0, 2.0, 3.0]]


# =============================================================================
# Equality Tests
# =============================================================================


def test_eq():
    """Test matrix equality."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[1, 2], [3, 4]])
    C = Matrix(matrix=[[4, 3], [2, 1]])
    assert A == B
    assert A != C


def test_eq_with_different_dtypes():
    """Test equality with different dtypes but same values."""
    A = Matrix(matrix=[[1, 2]], dtype="int")
    B = Matrix(matrix=[[1, 2]], dtype="float")
    assert A == B


def test_eq_against_non_matrix():
    """Test equality against non-Matrix types."""
    A = Matrix(2, 2)
    assert A != "string"
    assert A != 123
    assert A != [1, 2, 3]
    assert A != None


def test_eq_different_shapes():
    """Test equality with different shapes."""
    A = Matrix(2, 3)
    B = Matrix(3, 2)
    assert A != B


# =============================================================================
# String Representation Tests
# =============================================================================


def test_repr():
    """Test string representation."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    expected = "[1.0 2.0]\n[3.0 4.0]"
    assert repr(A) == expected


def test_str():
    """Test str() output."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    expected = "[1.0 2.0]\n[3.0 4.0]"
    assert str(A) == expected


def test_repr_empty():
    """Test repr of empty matrix."""
    E = Matrix(0, 0)
    assert str(E) == ""


# =============================================================================
# Edge Cases and Complex Scenarios
# =============================================================================


def test_chain_operations():
    """Test chaining multiple operations."""
    A = Matrix(matrix=[[1, 2], [3, 4]])
    B = Matrix(matrix=[[5, 6], [7, 8]])
    C = (A + B) * 2 - A
    expected = Matrix(matrix=[[11.0, 14.0], [17.0, 20.0]])
    assert C == expected


def test_matrix_with_very_small_values():
    """Test matrix with very small floating point values."""
    A = Matrix(matrix=[[1e-10, 2e-10], [3e-10, 4e-10]])
    B = A * 2
    assert B[0, 0] == pytest.approx(2e-10)


def test_matrix_with_very_large_values():
    """Test matrix with very large values."""
    A = Matrix(matrix=[[1e10, 2e10], [3e10, 4e10]])
    B = A * 2
    assert B[0, 0] == pytest.approx(2e10)


def test_complex_matrix_operations():
    """Test operations with complex numbers."""
    A = Matrix(matrix=[[1 + 2j, 3 + 4j]], dtype="complex")
    B = Matrix(matrix=[[5 + 6j, 7 + 8j]], dtype="complex")
    C = A + B
    assert C[0, 0] == 6 + 8j


def test_decimal_precision():
    """Test Decimal dtype maintains precision."""
    A = Matrix(matrix=[[Decimal("0.1"), Decimal("0.2")]], dtype="decimal")
    B = A + A
    # Decimal avoids floating point errors
    assert B[0, 0] == Decimal("0.2")
    assert B[0, 1] == Decimal("0.4")


def test_shape_method():
    """Test shape() method."""
    A = Matrix(3, 4)
    assert A.shape() == (3, 4)

    B = Matrix(matrix=[[1, 2, 3]])
    assert B.shape() == (1, 3)


def test_mixed_type_arithmetic():
    """Test arithmetic between matrices of different dtypes."""
    A = Matrix(matrix=[[1, 2]], dtype="int")
    B = Matrix(matrix=[[3.5, 4.5]], dtype="float")

    # Result should have dtype of left operand
    C = A + B
    assert C.dtype == "int"
    assert C[0, 0] == 4  # 1 + 3.5 truncated to int
