import math

import pytest

from tinymatrix import (
    Matrix,
    MatrixError,
    ShapeError,
    SingularMatrixError,
    cholesky,
    eig,
    lstsq,
    lu,
    qr,
    solve,
    svd,
)


def _assert_matrices_close(A: Matrix, B: Matrix, tol: float = 1e-7):
    assert A.shape() == B.shape()
    m, n = A.shape()
    for r in range(m):
        for c in range(n):
            assert math.isclose(float(A[r, c]), float(B[r, c]), abs_tol=tol)


# =============================================================================
# LU Decomposition Tests
# =============================================================================


def test_lu_empty():
    A = Matrix(0, 0)
    P, L, U = A.lu()
    assert P.shape() == (0, 0)
    assert L.shape() == (0, 0)
    assert U.shape() == (0, 0)


def test_lu_square_2x2():
    A = Matrix(matrix=[[4, 3], [6, 3]])
    P, L, U = A.lu()
    # P @ A == L @ U
    PA = P @ A
    LU = L @ U
    _assert_matrices_close(PA, LU)


def test_lu_square_3x3():
    A = Matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    P, L, U = lu(A)
    _assert_matrices_close(P @ A, L @ U)


def test_lu_rectangular():
    # Tall 4x2
    A_tall = Matrix(matrix=[[1, 2], [3, 4], [5, 6], [7, 8]])
    P, L, U = A_tall.lu()
    _assert_matrices_close(P @ A_tall, L @ U)

    # Wide 2x4
    A_wide = Matrix(matrix=[[1, 2, 3, 4], [5, 6, 7, 8]])
    P, L, U = lu(A_wide)
    _assert_matrices_close(P @ A_wide, L @ U)


# =============================================================================
# QR Decomposition Tests
# =============================================================================


def test_qr_empty():
    A = Matrix(0, 0)
    Q, R = A.qr()
    assert Q.shape() == (0, 0)
    assert R.shape() == (0, 0)


def test_qr_square():
    A = Matrix(matrix=[[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
    Q, R = A.qr()

    # Check A = Q @ R
    _assert_matrices_close(A, Q @ R)

    # Check Q.T @ Q = I
    QtQ = Q.T @ Q
    _assert_matrices_close(QtQ, Matrix.identity(3))


def test_qr_rectangular():
    # Tall 4x2
    A = Matrix(matrix=[[1, -1], [1, 2], [1, 1], [1, 0]])
    Q, R = qr(A)
    _assert_matrices_close(A, Q @ R)
    _assert_matrices_close(Q.T @ Q, Matrix.identity(2))

    # Wide 2x3
    A_wide = Matrix(matrix=[[1, 2, 3], [4, 5, 6]])
    Q_w, R_w = qr(A_wide)
    _assert_matrices_close(A_wide, Q_w @ R_w)


def test_qr_linearly_dependent():
    A = Matrix(matrix=[[1, 2], [2, 4]])
    Q, R = qr(A)
    _assert_matrices_close(A, Q @ R)


# =============================================================================
# Cholesky Decomposition Tests
# =============================================================================


def test_cholesky_empty():
    A = Matrix(0, 0)
    assert A.cholesky() == Matrix(0, 0)
    assert cholesky(A) == Matrix(0, 0)


def test_cholesky_2x2():
    # A = [[4, 2], [2, 5]]
    A = Matrix(matrix=[[4, 2], [2, 5]])
    L = A.cholesky()
    _assert_matrices_close(A, L @ L.T)
    assert math.isclose(L[0, 0], 2.0)
    assert math.isclose(L[0, 1], 0.0)
    assert math.isclose(L[1, 0], 1.0)
    assert math.isclose(L[1, 1], 2.0)


def test_cholesky_3x3():
    # A = [[25, 15, -5], [15, 18, 0], [-5, 0, 11]]
    A = Matrix(matrix=[[25, 15, -5], [15, 18, 0], [-5, 0, 11]])
    L = cholesky(A)
    _assert_matrices_close(A, L @ L.T)


def test_cholesky_errors():
    non_square = Matrix(2, 3)
    with pytest.raises(ShapeError):
        _ = non_square.cholesky()

    not_symmetric = Matrix(matrix=[[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Matrix is not symmetric"):
        _ = not_symmetric.cholesky()

    not_pd = Matrix(matrix=[[-1, 0], [0, -1]])
    with pytest.raises(MatrixError, match="Matrix is not positive-definite"):
        _ = not_pd.cholesky()


# =============================================================================
# Eigenvalues & Eigenvectors Tests
# =============================================================================


def test_eig_empty():
    evals, evecs = Matrix(0, 0).eig()
    assert evals == []
    assert evecs.shape() == (0, 0)


def test_eig_1x1():
    evals, evecs = Matrix(matrix=[[7.0]]).eig()
    assert evals == [7.0]
    assert evecs[0, 0] == 1.0


def test_eig_symmetric_2x2():
    A = Matrix(matrix=[[2, 1], [1, 2]])
    evals, evecs = A.eig()
    # Eigenvalues should be 3 and 1
    assert math.isclose(evals[0], 3.0, abs_tol=1e-7)
    assert math.isclose(evals[1], 1.0, abs_tol=1e-7)

    # A @ v = lambda * v
    for i in range(2):
        v = Matrix(matrix=[[evecs[0, i]], [evecs[1, i]]])
        Av = A @ v
        lv = v * evals[i]
        _assert_matrices_close(Av, lv)


def test_eig_symmetric_3x3():
    A = Matrix(matrix=[[4, 1, 2], [1, 3, 0], [2, 0, 5]])
    evals, evecs = eig(A)

    # Verify A @ v = lambda * v for each column
    for i in range(3):
        v = Matrix(matrix=[[evecs[0, i]], [evecs[1, i]], [evecs[2, i]]])
        Av = A @ v
        lv = v * evals[i]
        _assert_matrices_close(Av, lv, tol=1e-5)


def test_eig_non_symmetric():
    A = Matrix(matrix=[[2, 1], [0, 3]])
    evals, evecs = eig(A)
    # Upper triangular eigenvalues are 3 and 2
    assert any(math.isclose(x, 3.0, abs_tol=1e-4) for x in evals)
    assert any(math.isclose(x, 2.0, abs_tol=1e-4) for x in evals)


def test_eig_non_square():
    with pytest.raises(ShapeError):
        _ = Matrix(2, 3).eig()


# =============================================================================
# SVD Tests
# =============================================================================


def test_svd_empty():
    A = Matrix(0, 0)
    U, S, Vt = A.svd()
    assert U.shape() == (0, 0)
    assert S == []
    assert Vt.shape() == (0, 0)


def test_svd_square():
    A = Matrix(matrix=[[3, 2, 2], [2, 3, -2]])
    U, S, Vt = A.svd()

    # Reconstruct A = U @ diag(S) @ Vt
    k = len(S)
    Sigma = Matrix.zeroes(k, k)
    for i in range(k):
        Sigma[i, i] = S[i]

    reconstructed = U @ Sigma @ Vt
    _assert_matrices_close(A, reconstructed, tol=1e-5)


def test_svd_tall():
    A = Matrix(matrix=[[1, 2], [3, 4], [5, 6]])
    U, S, Vt = svd(A)

    Sigma = Matrix.zeroes(2, 2)
    Sigma[0, 0] = S[0]
    Sigma[1, 1] = S[1]
    reconstructed = U @ Sigma @ Vt
    _assert_matrices_close(A, reconstructed, tol=1e-5)


# =============================================================================
# Linear System Solver (solve) Tests
# =============================================================================


def test_solve_2x2():
    # 2x + y = 5
    # x + 3y = 5  => x = 2, y = 1
    A = Matrix(matrix=[[2, 1], [1, 3]])
    b = [5, 5]
    x = A.solve(b)
    assert math.isclose(x[0, 0], 2.0)
    assert math.isclose(x[1, 0], 1.0)
    _assert_matrices_close(A @ x, Matrix(matrix=[[5.0], [5.0]]))


def test_solve_multiple_rhs():
    A = Matrix(matrix=[[2, 1], [1, 3]])
    B = Matrix(matrix=[[5, 8], [5, 9]])
    X = solve(A, B)
    _assert_matrices_close(A @ X, B)


def test_solve_errors():
    non_square = Matrix(2, 3)
    with pytest.raises(ShapeError):
        _ = non_square.solve([1, 2])

    singular = Matrix(matrix=[[1, 2], [2, 4]])
    with pytest.raises(SingularMatrixError):
        _ = singular.solve([1, 2])

    A = Matrix(2, 2)
    with pytest.raises(ShapeError, match="Dimension mismatch"):
        _ = A.solve([1, 2, 3])

    with pytest.raises(TypeError, match="Expected Matrix or sequence"):
        _ = A.solve(12345)


# =============================================================================
# Least-Squares Solver (lstsq) Tests
# =============================================================================


def test_lstsq_exact():
    A = Matrix(matrix=[[1, 1], [1, 2], [1, 3]])
    b = [2, 3, 4]  # Perfectly on line y = 1 + 1*x => x = [1, 1]
    x, residual = A.lstsq(b)
    assert math.isclose(x[0, 0], 1.0, abs_tol=1e-6)
    assert math.isclose(x[1, 0], 1.0, abs_tol=1e-6)
    assert math.isclose(residual, 0.0, abs_tol=1e-6)


def test_lstsq_overdetermined():
    A = Matrix(matrix=[[1, 1], [1, 2], [1, 3]])
    b = Matrix(matrix=[[1], [2], [4]])
    x, residual = lstsq(A, b)
    assert residual > 0.0
    assert x.shape() == (2, 1)


def test_lstsq_errors():
    A = Matrix(3, 2)
    with pytest.raises(ShapeError, match="Dimension mismatch"):
        _ = A.lstsq([1, 2])

    with pytest.raises(TypeError, match="Expected Matrix or sequence"):
        _ = A.lstsq(None)


def test_lstsq_rank_deficient():
    A = Matrix(matrix=[[1, 1], [1, 1], [1, 1]])
    b = [1, 1, 1]
    x, residual = lstsq(A, b)
    assert x.shape() == (2, 1)
