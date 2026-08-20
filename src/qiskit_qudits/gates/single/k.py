"""Implementation of the K-Gate class for qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from qiskit.circuit import QuantumCircuit
from typing_extensions import override

from qiskit_qudits.gates.base.gate import QuditGate
from qiskit_qudits.gates.single.not_ import QuditNOTGate
from qiskit_qudits.gates.single.x import QuditXGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditKGate(QuditGate):
    r"""Qudit K-gate class.

    This gate applies the K (parity) operation on qudit systems,
    which performs an index inversion transformation on the
    computational basis. This gate is equivalent to the I-gate
    for :math:`d=2`.

    .. math::

        K_d \lvert k \rangle = \lvert (d-k) \bmod d \rangle,
        \quad k \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "K"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit K-gate.

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

        if self.fills_hilbert_space:
            # Apply NOT gate first, then X gate.
            qc.append(QuditNOTGate(self.dim), list(range(self.num_qubits)))
            qc.append(QuditXGate(self.dim), list(range(self.num_qubits)))
        else:
            # Fall back to unitary synthesis.
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the K-gate.

        .. math::

            K_{HD} = \left[
                \begin{array}{c|c}
                    K_d & \mathbf{0} \\ \hline
                    \mathbf{0} & I_{HD-d}
                \end{array}
            \right] \quad

            K_d = \begin{pmatrix}
                1 & 0 & \cdots & 0 \\
                0 & 0 & \cdots & 1 \\
                \vdots & \vdots & \unicode{x22F0} & \vdots \\
                0 & 1 & \cdots & 0 \\
            \end{pmatrix}

        Where :math:`HD = 2^n` is the full Hilbert-space dimension
        (``self.hilbert_dim``).

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`K_d`.
        """
        d = self.dim
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        unitary[1:d, 1:d] = (
            np.flipud(  # pyright: ignore[reportUnknownMemberType]
                unitary[1:d, 1:d],
            )
        )
        return unitary

    @override
    def inverse(self, annotated: bool = False) -> Self:
        """Return the inverse gate.

        The parity operation is self-inverse, so the inverse is a
        new :class:`QuditKGate` with identical dimension.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditKGate` with the same dimension.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return type(self)(self.dim)
