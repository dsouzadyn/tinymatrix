"""Matrix decompositions and linear system solvers for TinyMatrix."""

import math
from typing import List, Sequence, Tuple, Union
from .exceptions import MatrixError, ShapeError, SingularMatrixError
from .matrix import Matrix
from .types import TinyMatrixNumeric


def lu(A: Matrix) -> Tuple[Matrix, Matrix, Matrix]:
    """Compute the pivoted LU decomposition: P @ A = L @ U.

    Returns:
        (P, L, U) where P is a permutation matrix, L is unit lower triangular,
        and U is upper triangular.
    """
    m, n = A.shape()
    if m == 0 or n == 0:
        return Matrix(m, m), Matrix(m, 0), Matrix(0, n)

    P_indices = list(range(m))
    L_data = [[1.0 if i == j else 0.0 for j in range(min(m, n))] for i in range(m)]
    U_data = [[float(x) for x in row] for row in A.M]

    k_max = min(m, n)
    for k in range(k_max):
        # Partial pivoting: find row with max absolute value in column k
        pivot_row = k
        max_val = abs(U_data[k][k])
        for i in range(k + 1, m):
            val = abs(U_data[i][k])
            if val > max_val:
                max_val = val
                pivot_row = i

        if pivot_row != k:
            # Swap rows in U
            U_data[k], U_data[pivot_row] = U_data[pivot_row], U_data[k]
            # Swap permutation indices
            P_indices[k], P_indices[pivot_row] = (
                P_indices[pivot_row],
                P_indices[k],
            )
            # Swap previously computed multipliers in L
            for j in range(k):
                L_data[k][j], L_data[pivot_row][j] = (
                    L_data[pivot_row][j],
                    L_data[k][j],
                )

        pivot = U_data[k][k]
        if abs(pivot) > 1e-15:
            for i in range(k + 1, m):
                factor = U_data[i][k] / pivot
                L_data[i][k] = factor
                U_data[i][k] = 0.0
                for j in range(k + 1, n):
                    U_data[i][j] -= factor * U_data[k][j]

    # Construct permutation matrix P
    P_data = [[1.0 if P_indices[i] == j else 0.0 for j in range(m)] for i in range(m)]

    P = Matrix(matrix=P_data, dtype=A.dtype)
    L = Matrix(matrix=L_data, dtype=A.dtype)
    U = Matrix(matrix=U_data[: min(m, n)], dtype=A.dtype)
    return P, L, U


def qr(A: Matrix, mode: str = "reduced") -> Tuple[Matrix, Matrix]:
    """Compute QR decomposition: A = Q @ R.

    Using Modified Gram-Schmidt with re-orthogonalization for numerical stability.

    Returns:
        (Q, R) where Q has orthonormal columns and R is upper triangular.
    """
    m, n = A.shape()
    if m == 0 or n == 0:
        return Matrix(m, 0), Matrix(0, n)

    k = min(m, n) if mode == "reduced" else m

    # Modified Gram-Schmidt on columns of A
    V = [[float(A.M[i][j]) for i in range(m)] for j in range(n)]
    Q_cols = []
    R_data = [[0.0 for _ in range(n)] for _ in range(k)]

    for j in range(min(m, n)):
        v = V[j][:]
        for i in range(j):
            q_i = Q_cols[i]
            rij = sum(q_i[r] * v[r] for r in range(m))
            R_data[i][j] = rij
            for r in range(m):
                v[r] -= rij * q_i[r]

        norm_v = math.sqrt(sum(x**2 for x in v))
        if norm_v > 1e-14:
            R_data[j][j] = norm_v
            Q_cols.append([x / norm_v for x in v])
        else:
            R_data[j][j] = 0.0
            # Construct orthogonal vector
            e = [0.0] * m
            e[j if j < m else 0] = 1.0
            for q_i in Q_cols:
                dot = sum(q_i[r] * e[r] for r in range(m))
                for r in range(m):
                    e[r] -= dot * q_i[r]
            norm_e = math.sqrt(sum(x**2 for x in e))
            Q_cols.append([x / norm_e for x in e] if norm_e > 1e-14 else e)

    # For any remaining columns when n > m
    for j in range(min(m, n), n):
        v = V[j][:]
        for i in range(min(m, n)):
            q_i = Q_cols[i]
            rij = sum(q_i[r] * v[r] for r in range(m))
            R_data[i][j] = rij

    Q_data = [[Q_cols[j][i] for j in range(len(Q_cols))] for i in range(m)]
    return Matrix(matrix=Q_data, dtype=A.dtype), Matrix(matrix=R_data, dtype=A.dtype)


def cholesky(A: Matrix) -> Matrix:
    """Compute the Cholesky decomposition of a symmetric positive-definite matrix.

    Returns:
        Lower-triangular matrix L such that A = L @ L.T.
    """
    if not A.is_square():
        raise ShapeError("Cholesky decomposition requires a square matrix")

    n = A.m
    if n == 0:
        return Matrix(0, 0, dtype=A.dtype)

    # Check symmetry
    for i in range(n):
        for j in range(i + 1, n):
            if not math.isclose(float(A.M[i][j]), float(A.M[j][i]), abs_tol=1e-10):
                raise ValueError("Matrix is not symmetric")

    L = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = float(A.M[i][i]) - s
                if val <= 1e-14:
                    raise MatrixError("Matrix is not positive-definite")
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (float(A.M[i][j]) - s) / L[j][j]

    return Matrix(matrix=L, dtype=A.dtype)


def eig(A: Matrix) -> Tuple[List[float], Matrix]:
    """Compute the eigenvalues and eigenvectors of a square matrix.

    Returns:
        (eigenvalues, eigenvectors) where eigenvectors columns correspond
        to eigenvalues.
    """
    if not A.is_square():
        raise ShapeError("Eigenvalue computation requires a square matrix")

    n = A.m
    if n == 0:
        return [], Matrix(0, 0, dtype=A.dtype)
    if n == 1:
        return [float(A.M[0][0])], Matrix.identity(1, dtype=A.dtype)

    # Check if symmetric
    is_sym = True
    for i in range(n):
        for j in range(i + 1, n):
            if not math.isclose(float(A.M[i][j]), float(A.M[j][i]), abs_tol=1e-10):
                is_sym = False
                break
        if not is_sym:
            break

    if is_sym:
        # Jacobi eigenvalue algorithm for symmetric matrices
        D = [[float(A.M[i][j]) for j in range(n)] for i in range(n)]
        V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        max_rotations = 100 * n * n
        for _ in range(max_rotations):
            # Find largest off-diagonal element
            p, q = 0, 1
            max_off = abs(D[0][1])
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(D[i][j]) > max_off:
                        max_off = abs(D[i][j])
                        p, q = i, j

            if max_off < 1e-14:
                break

            # Compute rotation angle
            diff = D[q][q] - D[p][p]
            if abs(D[p][q]) < 1e-15:
                t = 0.0
            else:
                phi = diff / (2.0 * D[p][q])
                t = 1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0))
                if phi < 0.0:
                    t = -t

            c = 1.0 / math.sqrt(t * t + 1.0)
            s = t * c
            tau = s / (1.0 + c)

            # Apply rotation to D
            app = D[p][p]
            aqq = D[q][q]
            apq = D[p][q]

            D[p][p] = app - t * apq
            D[q][q] = aqq + t * apq
            D[p][q] = 0.0
            D[q][p] = 0.0

            for i in range(n):
                if i != p and i != q:
                    aip = D[i][p]
                    aiq = D[i][q]
                    D[i][p] = aip - s * (aiq + tau * aip)
                    D[p][i] = D[i][p]
                    D[i][q] = aiq + s * (aip - tau * aiq)
                    D[q][i] = D[i][q]

            # Accumulate eigenvectors
            for i in range(n):
                vip = V[i][p]
                viq = V[i][q]
                V[i][p] = vip - s * (viq + tau * vip)
                V[i][q] = viq + s * (vip - tau * viq)

        evals = [D[i][i] for i in range(n)]
        # Sort eigenvalues descending
        idx = sorted(range(n), key=lambda i: evals[i], reverse=True)
        evals_sorted = [evals[i] for i in idx]
        V_sorted = [[V[r][c] for c in idx] for r in range(n)]
        return evals_sorted, Matrix(matrix=V_sorted, dtype=A.dtype)

    else:
        # General QR algorithm for non-symmetric matrices
        H = A.copy()
        V = Matrix.identity(n, dtype=A.dtype)

        for _ in range(100):
            # Wilkinson shift using bottom 2x2 corner
            d = (float(H.M[n - 2][n - 2]) - float(H.M[n - 1][n - 1])) / 2.0
            sign = 1.0 if d >= 0 else -1.0
            shift = float(H.M[n - 1][n - 1]) - (float(H.M[n - 1][n - 2]) ** 2) / (
                d + sign * math.sqrt(d * d + float(H.M[n - 1][n - 2]) ** 2 + 1e-15)
            )

            shifted = H - Matrix.identity(n, dtype=A.dtype) * shift
            Q, R = qr(shifted)
            H = R @ Q + Matrix.identity(n, dtype=A.dtype) * shift
            V = V @ Q

            # Check convergence of sub-diagonals
            sub_diag_max = max(abs(float(H.M[i][i - 1])) for i in range(1, n))
            if sub_diag_max < 1e-12:
                break

        evals = [float(H.M[i][i]) for i in range(n)]
        return evals, V


def svd(A: Matrix) -> Tuple[Matrix, List[float], Matrix]:
    """Compute the Singular Value Decomposition: A = U @ diag(S) @ Vt.

    Returns:
        (U, S, Vt) where U and Vt are orthogonal matrices and S is a list
        of singular values sorted in descending order.
    """
    m, n = A.shape()
    if m == 0 or n == 0:
        return Matrix(m, 0), [], Matrix(0, n)

    # Compute A.T @ A (n x n)
    AtA = A.T @ A
    eigenvalues, V_mat = eig(AtA)

    # Singular values are sqrt(max(0, lambda_i))
    s_vals = [math.sqrt(max(0.0, lam)) for lam in eigenvalues]

    # Compute U columns: u_i = A @ v_i / sigma_i
    U_cols = []
    tol = 1e-12
    for j in range(len(s_vals)):
        sigma = s_vals[j]
        v_col = [V_mat.M[r][j] for r in range(n)]
        # u = A @ v
        u = [sum(float(A.M[i][k]) * v_col[k] for k in range(n)) for i in range(m)]
        if sigma > tol:
            U_cols.append([x / sigma for x in u])
        else:
            U_cols.append([0.0] * m)

    # Complete orthonormal basis for U if needed
    for i in range(len(U_cols)):
        norm_u = math.sqrt(sum(x**2 for x in U_cols[i]))
        if norm_u < tol:
            e = [0.0] * m
            e[i % m] = 1.0
            for u_prev in U_cols[:i]:
                dot = sum(u_prev[r] * e[r] for r in range(m))
                for r in range(m):
                    e[r] -= dot * u_prev[r]
            norm_e = math.sqrt(sum(x**2 for x in e))
            U_cols[i] = [x / norm_e for x in e] if norm_e > tol else e

    # Transpose Q columns to matrix rows
    k = min(m, n)
    U_data = [[U_cols[c][r] for c in range(k)] for r in range(m)]
    U = Matrix(matrix=U_data, dtype=A.dtype)
    Vt = V_mat.T[:k, :]
    return U, s_vals[:k], Vt


def solve(A: Matrix, b: Union[Matrix, Sequence[TinyMatrixNumeric]]) -> Matrix:
    """Solve the linear system A @ x = b for square matrix A.

    Args:
        A: Square coefficient matrix (n x n).
        b: Right-hand side vector (n x 1) or matrix (n x k).

    Returns:
        Solution matrix x.
    """
    if not A.is_square():
        raise ShapeError("Coefficient matrix A must be square")

    n = A.m
    if isinstance(b, Matrix):
        b_mat = b
    elif isinstance(b, (list, tuple)):
        b_mat = Matrix(matrix=[[x] for x in b])
    else:
        raise TypeError("Expected Matrix or sequence for b")

    if b_mat.m != n:
        raise ShapeError(
            f"Dimension mismatch between A ({n}x{n}) and b ({b_mat.m}x{b_mat.n})"
        )

    # Use LU decomposition with partial pivoting
    P, L, U = lu(A)

    # Check for singularity
    for i in range(n):
        if abs(float(U.M[i][i])) < 1e-14:
            raise SingularMatrixError("Matrix A is singular and cannot be solved")

    # Pb = P @ b
    Pb = P @ b_mat
    k_cols = b_mat.n
    x_data = [[0.0 for _ in range(k_cols)] for _ in range(n)]

    for col in range(k_cols):
        # Forward substitution: L @ y = Pb
        y = [0.0] * n
        for i in range(n):
            s = sum(float(L.M[i][j]) * y[j] for j in range(i))
            y[i] = float(Pb.M[i][col]) - s

        # Back substitution: U @ x = y
        x_col = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = sum(float(U.M[i][j]) * x_col[j] for j in range(i + 1, n))
            x_col[i] = (y[i] - s) / float(U.M[i][i])

        for i in range(n):
            x_data[i][col] = x_col[i]

    return Matrix(matrix=x_data, dtype=A.dtype)


def lstsq(
    A: Matrix, b: Union[Matrix, Sequence[TinyMatrixNumeric]]
) -> Tuple[Matrix, float]:
    """Solve the linear least-squares problem: min ||A @ x - b||_2.

    Returns:
        (x, residual) where x is the least-squares solution.
    """
    m, n = A.shape()
    if isinstance(b, Matrix):
        b_mat = b
    elif isinstance(b, (list, tuple)):
        b_mat = Matrix(matrix=[[x] for x in b])
    else:
        raise TypeError("Expected Matrix or sequence for b")

    if b_mat.m != m:
        raise ShapeError(
            f"Dimension mismatch between A ({m}x{n}) and b ({b_mat.m}x{b_mat.n})"
        )

    # Solve using QR decomposition: A = Q @ R => R @ x = Q.T @ b
    Q, R = qr(A, mode="reduced")
    Qt_b = Q.T @ b_mat

    k = min(m, n)
    k_cols = b_mat.n
    x_data = [[0.0 for _ in range(k_cols)] for _ in range(n)]

    for col in range(k_cols):
        x_col = [0.0] * n
        for i in range(k - 1, -1, -1):
            pivot = float(R.M[i][i])
            if abs(pivot) < 1e-14:
                continue
            s = sum(float(R.M[i][j]) * x_col[j] for j in range(i + 1, n))
            x_col[i] = (float(Qt_b.M[i][col]) - s) / pivot
        for i in range(n):
            x_data[i][col] = x_col[i]

    x = Matrix(matrix=x_data, dtype=A.dtype)
    diff = A @ x - b_mat
    residual = diff.norm(ord="fro")
    return x, residual
