"""Implementation of the Z-Gate class for qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import override

from qiskit_qudits.gates.single.p import QuditPGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditZGate(QuditPGate):
    r"""Qudit Z-gate class.

    This gate applies the Z (clock) operation on qudit systems,
    rotating the phase around the Z-axis of each quantum state level by
    an amount proportional to the level index. The Z-gate is the qudit
    generalisation of the Pauli-Z gate for qubits.

    .. math::

        Z_d \lvert k \rangle = \omega^{k}
        \lvert k \rangle, \quad \omega = e^{\frac{i 2 \pi}{d}}, \quad k
        \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "Z"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit Z-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(
            dim,
            np.pi,
            label=label,
        )

    @override
    def inverse(self, annotated: bool = False) -> QuditZdgGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditZdgGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditZdgGate(self.dim)


class QuditZdgGate(QuditPGate):
    r"""Qudit Zdg-gate class.

    This gate applies the inverse Z (clock) operation on qudit systems,
    rotating the phase around the Z-axis of each quantum state level by
    an amount proportional to the level index. The inverse Z-gate is the
    qudit generalisation of the inverse Pauli-Z gate for qubits.

    .. math::

        Z_d^\dagger \lvert k \rangle = \omega^{-k}
        \lvert k \rangle, \quad \omega = e^{\frac{i 2 \pi}{d}}, \quad k
        \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "Zdg"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit Zdg-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(
            dim,
            -np.pi,
            label=label,
        )

    @override
    def inverse(self, annotated: bool = False) -> QuditZGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditZGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditZGate(self.dim)
