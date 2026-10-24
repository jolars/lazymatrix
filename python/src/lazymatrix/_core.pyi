import numpy as np
from numpy.typing import NDArray

class Matrix:
    @staticmethod
    def from_dense(
        values: NDArray[np.float64], normalization: tuple[bool, bool]
    ) -> Matrix: ...
    @staticmethod
    def from_csc(
        shape: tuple[int, int],
        values: NDArray[np.float64],
        indices: NDArray[np.int64],
        indptr: NDArray[np.int64],
        normalization: tuple[bool, bool],
    ) -> Matrix: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def centers(self) -> NDArray[np.float64] | None: ...
    @property
    def scales(self) -> NDArray[np.float64] | None: ...
    def matvec(
        self, values: NDArray[np.float64], transpose: bool
    ) -> NDArray[np.float64]: ...
    def matmat(
        self, values: NDArray[np.float64], transpose: bool
    ) -> NDArray[np.float64]: ...
    def column(self, index: int, transpose: bool) -> NDArray[np.float64]: ...
    def toarray(self, transpose: bool) -> NDArray[np.float64]: ...
