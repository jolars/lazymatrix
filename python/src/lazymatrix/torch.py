"""PyTorch layers that normalize columns without materializing the input."""

import importlib
import operator
from typing import Any


def _require_torch() -> None:
    try:
        importlib.import_module("torch")
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        raise ModuleNotFoundError(
            "The PyTorch adapter requires torch. Install it with "
            '`pip install "lazymatrix[torch]"`.',
            name="torch",
        ) from error


_require_torch()

# Unconditional imports let static API extraction resolve PyTorch's names.
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

__all__ = ["NormalizedLinear"]


def _buffer(value: Any, name: str, weight: Tensor) -> Tensor | None:
    if value is None:
        return None
    values = torch.as_tensor(value)
    if values.layout != torch.strided or values.is_complex() or values.is_quantized:
        raise TypeError(f"{name} must contain dense real numeric values")
    if values.ndim != 1 or values.numel() != weight.shape[1]:
        raise ValueError(f"{name} must have one entry per input feature")
    # Copy even when dtype and device already match, so caller mutation is safe.
    values = values.detach().to(device=weight.device, dtype=weight.dtype, copy=True)
    if name == "scales" and torch.any(values == 0):
        index = torch.nonzero(values == 0)[0, 0].item()
        raise ValueError(f"scale at column {index} must be nonzero")
    return values


class NormalizedLinear(nn.Linear):
    """Apply a linear layer to lazily centered and scaled input features.

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    bias : bool, default=True
        Learn an additive bias in normalized coordinates. Disabling bias does
        not disable the centering correction.
    centers : real array-like or torch.Tensor, optional
        Fixed column centers, with one entry per input feature. None disables
        centering. Existing LazyMatrix centers can be passed directly.
    scales : real array-like or torch.Tensor, optional
        Fixed column divisors, with one entry per input feature. None disables
        scaling. Exact zeros, including negative zero and values that round to
        zero in the layer's dtype, are rejected. Negative and nonfinite values
        are retained. Computed LazyMatrix scales already replace zero with one.
    device : torch.device or str, optional
        CPU or CUDA device, following PyTorch's default device when omitted.
    dtype : torch.dtype, optional
        torch.float32 or torch.float64, following PyTorch's default floating
        dtype when omitted. Supplied statistics do not determine this default.

    Notes
    -----
    The layer accepts dense tensors with shape ``(*, in_features)`` and
    two-dimensional sparse COO or CSR tensors. Inputs must match the layer's
    dtype and device. Sparse inputs must not require gradients. COO duplicates
    are summed; CSR inputs must satisfy PyTorch's storage invariants. Other
    sparse layouts require explicit conversion to COO or CSR.

    Weights have shape ``(out_features, in_features)`` and remain trainable in
    normalized coordinates. Each forward pass derives ``A = weight / scales``
    and evaluates ``X @ A.T - A @ centers + bias``, omitting disabled components.
    The layer never constructs the normalized input. All forward and backward
    operations use PyTorch, with no Rust execution or custom autograd kernels.

    Centers and scales are copied, detached, persistent buffers. They move with
    the layer and are included in its state_dict. Checkpoint reconstruction must
    enable the same normalization components; saved centers and scales can be
    supplied to the constructor before loading the state_dict. Statistics are
    never fitted or updated by this layer, including in training mode.

    Factoring normalization changes floating-point evaluation order. Large
    offsets can lose centered variation, and sparse implicit zeros and
    nonfinite values can differ from explicit dense normalization. Mixed
    precision, autocast, and devices other than CPU and CUDA are unsupported.

    Examples
    --------
    >>> import torch
    >>> from lazymatrix.torch import NormalizedLinear
    >>> layer = NormalizedLinear(2, 3, centers=[1.0, 2.0], scales=[2.0, 4.0])
    >>> layer(torch.tensor([[1.0, 0.0]]).to_sparse_csr()).shape
    torch.Size([1, 3])
    """

    centers: Tensor | None
    scales: Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        centers: Any = None,
        scales: Any = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        in_features = operator.index(in_features)
        out_features = operator.index(out_features)
        if in_features < 0 or out_features < 0:
            raise ValueError("feature counts must be nonnegative")
        if (dtype or torch.get_default_dtype()) not in (torch.float32, torch.float64):
            raise TypeError("NormalizedLinear supports only float32 and float64")
        target = torch.get_default_device() if device is None else torch.device(device)
        if target.type not in ("cpu", "cuda"):
            raise ValueError("NormalizedLinear supports only CPU and CUDA devices")
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.register_buffer("centers", _buffer(centers, "centers", self.weight))
        self.register_buffer("scales", _buffer(scales, "scales", self.weight))

    def forward(self, input: Tensor) -> Tensor:
        """Transform raw features without constructing their normalized matrix.

        Returns a dense tensor with the input's leading dimensions and
        out_features as the last dimension. Weight and bias gradients use
        normalized coordinates; dense input gradients use raw coordinates.
        """
        if not isinstance(input, Tensor):
            raise TypeError("input must be a PyTorch tensor")
        if input.layout not in (torch.strided, torch.sparse_coo, torch.sparse_csr):
            raise TypeError("input must use dense, sparse COO, or sparse CSR storage")
        if input.dtype not in (torch.float32, torch.float64):
            raise TypeError("NormalizedLinear supports only float32 and float64")
        if input.dtype != self.weight.dtype or input.device != self.weight.device:
            raise ValueError("input and layer must have the same dtype and device")
        if input.device.type not in ("cpu", "cuda"):
            raise ValueError("NormalizedLinear supports only CPU and CUDA devices")
        if input.ndim == 0 or input.shape[-1] != self.in_features:
            raise ValueError("input's last dimension must match in_features")
        if input.layout != torch.strided:
            if input.ndim != 2:
                raise ValueError("sparse input must be two-dimensional")
            if input.layout == torch.sparse_coo and input.sparse_dim() != 2:
                raise TypeError("sparse COO input must have two sparse dimensions")
            if input.requires_grad:
                raise ValueError("sparse inputs must not require gradients")
            if input.layout == torch.sparse_coo:
                input = input.coalesce()

        weight = self.weight if self.scales is None else self.weight / self.scales
        if input.layout == torch.strided:
            result = F.linear(input, weight, self.bias)
        else:
            result = torch.sparse.mm(input, weight.T)
            if self.bias is not None:
                result = result + self.bias
        if self.centers is not None:
            result = result - F.linear(self.centers, weight)
        return result

    def extra_repr(self) -> str:
        """Describe the layer dimensions and enabled normalization components."""
        return (
            f"{super().extra_repr()}, center={self.centers is not None}, "
            f"scale={self.scales is not None}"
        )
