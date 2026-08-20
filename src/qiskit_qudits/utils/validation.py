"""Generic numeric-range validation helpers.

These always raise plain ``TypeError``/``ValueError``, mirroring what
``int()`` would do. Call sites needing a library-specific exception
(e.g. ``QuditCircuitError``) should catch and translate locally -
that keeps the translation explicit instead of smuggling an exception
class through a low-level helper.
"""

from __future__ import annotations

from math import isfinite
from typing import Any

import numpy as np

from qiskit_qudits.utils.consts import MIN_QUDIT_DIM
from qiskit_qudits.utils.typeguards import is_integral, is_real, is_vector


def validate_integer_range(
    value: Any,  # noqa: ANN401
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    name: str = "value",
) -> int:
    """Validate that ``value`` is an integer within ``[min, max]``.

    Args:
        value: Candidate value.
        minimum: Inclusive lower bound, or ``None`` for unbounded below.
        maximum: Inclusive upper bound, or ``None`` for unbounded above.
        name: Identifier used in error messages.

    Returns:
        ``value`` coerced to a plain ``int``.

    Raises:
        TypeError: If ``value`` is not integer-like.
        ValueError: If ``value`` falls outside the given bounds.
    """
    if not is_integral(value):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}.",
        )

    coerced = int(value)
    if minimum is not None and maximum is not None:
        if not (minimum <= coerced <= maximum):
            raise ValueError(
                f"{name} must be in [{minimum}, {maximum}], got {coerced}.",
            )
    elif minimum is not None and coerced < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {coerced}.")
    elif maximum is not None and coerced > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {coerced}.")
    return coerced


def validate_float_finite(
    value: Any,  # noqa: ANN401
    *,
    name: str = "value",
) -> float:
    """Validate that ``value`` is a finite float.

    Args:
        value: Candidate value.
        name: Identifier used in error messages.

    Returns:
        ``value`` coerced to a plain ``float``.

    Raises:
        TypeError: If ``value`` is not float-like.
        ValueError: If ``value`` is not finite.
    """
    if not is_real(value):
        raise TypeError(f"{name} must be a float, got {type(value).__name__}.")

    coerced = float(value)
    if not isfinite(coerced):
        raise ValueError(f"{name} must be finite, got {coerced}.")
    return coerced


def validate_vector(
    value: Any,  # noqa: ANN401
    *,
    dtype: np.typing.DTypeLike | None = np.complex128,
    name: str = "value",
) -> np.typing.NDArray[Any]:
    """Validate that ``value`` is a 1-D vector.

    Args:
        value: Candidate value.
        dtype: Optional dtype the result must be castable to. If
            ``None``, the array's natural dtype is kept.
        name: Identifier used in error messages.

    Returns:
        ``value`` coerced to a 1-D :class:`numpy.ndarray`, cast to
        ``dtype`` if given.

    Raises:
        TypeError: If ``value`` is not a 1-D vector of a numeric
            (non-object) dtype, or is not castable to ``dtype``.
    """
    if not is_vector(value):
        raise TypeError(
            f"{name} must be a 1-D vector, got {type(value).__name__}.",
        )

    try:
        return np.asarray(value, dtype=dtype).ravel()
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} could not be cast to dtype {dtype!r}.",
        ) from exc


def validate_dim(value: Any) -> int:  # noqa: ANN401
    """Validate that ``value`` is a valid qudit dimension.

    Args:
        value: Candidate qudit dimension.

    Returns:
        ``value`` coerced to a plain ``int``.
    """
    return validate_integer_range(value, minimum=MIN_QUDIT_DIM, name="dim")
