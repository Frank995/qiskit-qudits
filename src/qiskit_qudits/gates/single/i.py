"""Implementation of the I-Gate class for qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from qiskit.circuit import QuantumCircuit
from typing_extensions import override

from qiskit_qudits.gates.base.gate import QuditGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditIGate(QuditGate):
    r"""Qudit I-gate class.

    This gate applies the I (identity) operation on qudit systems,
    leaving the quantum state completely unchanged. The I-gate is the
    qudit generalisation of the identity gate for qubits.

    .. math::

        I_d \lvert k \rangle = \lvert k \rangle,
        \quad k \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.
    """

    gate_name: ClassVar[str] = "I"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit I-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(type(self).gate_name, dim, [], label=label)

    def _define(self) -> None:
        """Lazily build the qubit-level circuit decomposition."""
        qc = QuantumCircuit(
            self.num_qubits,
            name=self.label if self.label is not None else self.name,
        )

        for index in range(self.num_qubits):
            qc.id(index)

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the I-gate.

        .. math::

            I_{HD} = \left[
                \begin{array}{c|c}
                    I_d & \mathbf{0} \\ \hline
                    \mathbf{0} & I_{HD-d}
                \end{array}
            \right] \quad

            I_d = \begin{pmatrix}
                1 & 0 & \cdots & 0 \\
                0 & 1 & \cdots & 0 \\
                \vdots & \vdots & \ddots & \vdots \\
                0 & 0 & \cdots & 1 \\
            \end{pmatrix}

        Where :math:`HD = 2^n` is the full Hilbert-space dimension
        (``self.hilbert_dim``).

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`I_d`.
        """
        return np.eye(self.hilbert_dim, dtype=np.complex128)

    @override
    def inverse(self, annotated: bool = False) -> Self:
        """Return the inverse gate.

        The identity operation is self-inverse, so the inverse is a
        new :class:`QuditIGate` with identical dimension.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditIGate` with the same dimension.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return type(self)(self.dim)
