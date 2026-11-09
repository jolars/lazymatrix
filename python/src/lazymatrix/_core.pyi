import numpy as np
from numpy.typing import NDArray

class Params:
    def __init__(
        self,
        axis: int,
        dimension: int,
        centers: NDArray[np.float64] | None,
        scales: NDArray[np.float64] | None,
    ) -> None: ...
    @property
    def dimension(self) -> int: ...
    @property
    def axis(self) -> int: ...
    @property
    def centers(self) -> NDArray[np.float64] | None: ...
    @property
    def scales(self) -> NDArray[np.float64] | None: ...

Specification = tuple[int, str, str, int, tuple[float, float]]

class Matrix:
    @staticmethod
    def from_dense(
        values: NDArray[np.float64],
        specification: Specification | None,
        parameters: Params | None,
    ) -> Matrix: ...
    @staticmethod
    def from_sparse(
        shape: tuple[int, int],
        is_csc: bool,
        buffers: tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]],
        specification: Specification | None,
        parameters: Params | None,
    ) -> Matrix: ...
    @property
    def normalization(self) -> Params: ...
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
