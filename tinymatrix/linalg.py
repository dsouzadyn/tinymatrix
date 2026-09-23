"""Linear algebra module for TinyMatrix."""

from typing import Optional, Union

from .decompositions import cholesky, eig, lstsq, lu, qr, solve, svd
from .matrix import Matrix
from .types import TinyMatrixNumeric


def det(A: Matrix) -> TinyMatrixNumeric:
    """Compute the determinant of a square matrix."""
    return A.det()


def inv(A: Matrix) -> Matrix:
    """Compute the (multiplicative) inverse of a square matrix."""
    return A.inv()


def rank(A: Matrix, tol: Optional[float] = None) -> int:
    """Compute the matrix rank."""
    return A.rank(tol=tol)


def norm(A: Matrix, ord: Union[int, float, str] = "fro") -> float:
    """Compute the matrix or vector norm."""
    return A.norm(ord=ord)


def kron(A: Matrix, B: Matrix) -> Matrix:
    """Compute the Kronecker product of two matrices."""
    return A.kron(B)


def matrix_power(A: Matrix, n: int) -> Matrix:
    """Raise a square matrix to the integer power n."""
    return A**n


__all__ = [
    "det",
    "inv",
    "rank",
    "norm",
    "kron",
    "matrix_power",
    "lu",
    "qr",
    "cholesky",
    "eig",
    "svd",
    "solve",
    "lstsq",
]
