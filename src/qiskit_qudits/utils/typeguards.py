"""Generic, domain-agnostic type guards."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import FloatLike, IntLike, VectorLike


def is_integral(val: Any) -> TypeGuard[IntLike]:  # noqa: ANN401
    """Check if a value is strictly an integer-like type.

    Args:
        val: Any input value.

    Returns:
        ``True`` if *val* is a plain ``int`` or ``numpy.integer``
        (excluding ``bool``), ``False`` otherwise.
    """
    if isinstance(val, bool):
        return False
    return isinstance(val, (int, np.integer))


def is_real(val: Any) -> TypeGuard[FloatLike]:  # noqa: ANN401
    """Check if a value is strictly a real-like type.

    Args:
        val: Any input value.

    Returns:
        ``True`` if *val* is a plain ``int``, ``float``,
        ``numpy.integer`` or ``numpy.floating`` (excluding ``bool``),
        ``False`` otherwise.
    """
    if isinstance(val, bool):
        return False
    return isinstance(val, (int, float, np.integer, np.floating))


def is_vector(val: Any) -> TypeGuard[VectorLike]:  # noqa: ANN401
    """Check if a value is strictly a vector.

    Args:
        val: Any input value.

    Returns:
        ``True`` if *val* is castable to a :data:`numpy.ndarray` with
        **ndim=1**, ``False`` otherwise.
    """
    try:
        arr = np.asarray(val)
    except (ValueError, TypeError):
        return False

    return (arr.ndim == 1) and (arr.dtype != object)
