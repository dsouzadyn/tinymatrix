"""Matrix stacking and block concatenation operations for TinyMatrix."""

from typing import Sequence

from .exceptions import ShapeError
from .matrix import Matrix


def vstack(matrices: Sequence[Matrix]) -> Matrix:
    """Stack matrices vertically (row-wise)."""
    if not matrices:
        raise ValueError("Need at least one matrix to vstack")

    n = matrices[0].n
    dtype = matrices[0].dtype

    for mat in matrices:
        if not isinstance(mat, Matrix):
            raise TypeError("All items must be Matrix instances")
        if mat.n != n:
            raise ShapeError("All matrices must have identical number of columns")

    new_data = []
    for mat in matrices:
        for row in mat.M:
            new_data.append(row[:])

    return Matrix(matrix=new_data, dtype=dtype)


def hstack(matrices: Sequence[Matrix]) -> Matrix:
    """Stack matrices horizontally (column-wise)."""
    if not matrices:
        raise ValueError("Need at least one matrix to hstack")

    m = matrices[0].m
    dtype = matrices[0].dtype

    for mat in matrices:
        if not isinstance(mat, Matrix):
            raise TypeError("All items must be Matrix instances")
        if mat.m != m:
            raise ShapeError("All matrices must have identical number of rows")

    new_data = [[] for _ in range(m)]
    for mat in matrices:
        for r in range(m):
            new_data[r].extend(mat.M[r])

    return Matrix(matrix=new_data, dtype=dtype)


def block(blocks: Sequence[Sequence[Matrix]]) -> Matrix:
    """Assemble a Matrix from a 2D nested sequence of matrix blocks."""
    if not blocks or not blocks[0]:
        raise ValueError("Need a non-empty 2D sequence of blocks")

    row_mats = []
    for block_row in blocks:
        row_mats.append(hstack(block_row))

    return vstack(row_mats)
