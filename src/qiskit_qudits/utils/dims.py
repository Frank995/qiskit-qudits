"""Pure dimension arithmetic for the qudit-in-qubit encoding scheme."""

from __future__ import annotations

from qiskit_qudits.utils.consts import MIN_QUDIT_DIM


def qubits_per_qudit(dim: int) -> int:
    r"""Return the number of qubits needed to encode a qudit dimension.

    Computes :math:`\lceil \log_2(d) \rceil` using exact integer
    arithmetic (``(dim - 1).bit_length()``), avoiding floating-point
    edge cases that ``math.log2`` could introduce for large inputs.

    Args:
        dim: Dimension :math:`d` of the qudit. Must be at least 2.

    Returns:
        The smallest integer *n* such that :math:`2^n \geq d`.

    Raises:
        ValueError: If ``dim < 2``.
    """
    if dim < MIN_QUDIT_DIM:
        raise ValueError(f"a qudit requires dimension at least 2, got {dim}.")
    return (dim - 1).bit_length()
