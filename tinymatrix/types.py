from decimal import Decimal
from typing import List, Tuple, Union

DTYPES = {
    "int": int,
    "float": float,
    "decimal": Decimal,
    "complex": complex,
}

type TinyMatrixNumeric = Union[int, float, Decimal, complex]
type TinyMatrixData = List[List[TinyMatrixNumeric]]
type TinyMatrixIndex = Union[int, slice]
type TinyMatrixIndexPair = Tuple[TinyMatrixIndex, TinyMatrixIndex]
