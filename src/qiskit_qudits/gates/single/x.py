"""Implementation of the X-Gate class for qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from qiskit.circuit import QuantumCircuit
from typing_extensions import override

from qiskit_qudits.gates.base.gate import QuditGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditXGate(QuditGate):
    r"""Qudit X-gate class.

    This gate applies the X (shift) operation, circularly shifting the
    quantum state up by one level. The X-gate is the qudit
    generalisation of the Pauli-X gate for qubits.

    .. math::

        X_d \lvert k \rangle = \lvert (k+1) \bmod d \rangle,
        \quad k \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "X"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit X-gate.

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
            # Apply controlled X gates with all lower-index qubits
            # as controls.
            for index in reversed(range(self.num_qubits)):
                if index > 0:
                    qc.mcx(list(reversed(range(index))), index)
                else:
                    qc.x(index)
        else:
            # Fall back to unitary synthesis.
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the X-gate.

        .. math::

            X_{HD} = \left[
                \begin{array}{c|c}
                    X_d & \mathbf{0} \\ \hline
                    \mathbf{0} & I_{HD-d}
                \end{array}
            \right] \quad

            X_d = \begin{pmatrix}
                0 & \cdots & 0 & 1 \\
                1 & \cdots & 0 & 0 \\
                \vdots & \ddots & \vdots & \vdots \\
                0 & \cdots & 1 & 0 \\
            \end{pmatrix}

        Where :math:`HD = 2^n` is the full Hilbert-space dimension
        (``self.hilbert_dim``).

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`X_d`.
        """
        d = self.dim
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        unitary[:d, :d] = np.roll(  # pyright: ignore[reportUnknownMemberType]
            unitary[:d, :d],
            1,
            axis=0,
        )
        return unitary

    @override
    def inverse(self, annotated: bool = False) -> QuditXdgGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditXdgGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditXdgGate(self.dim)


class QuditXdgGate(QuditGate):
    r"""Qudit Xdg-gate class.

    This gate applies the inverse X (shift) operation, circularly
    shifting the quantum state down by one level. The Xdg-gate is the
    qudit generalisation of the inverse Pauli-X gate for qubits.

    .. math::

        X_d^\dagger \lvert k \rangle = \lvert (k-1) \bmod d \rangle,
        \quad k \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "Xdg"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit Xdg-gate.

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
            # Apply controlled X gates with all lower-index qubits
            # as controls.
            for index in range(self.num_qubits):
                if index > 0:
                    qc.mcx(list(reversed(range(index))), index)
                else:
                    qc.x(index)
        else:
            # Fall back to unitary synthesis.
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the inverse Xdg-gate.

        .. math::

            X_{HD}^\dagger = \left[
                \begin{array}{c|c}
                    X_d^\dagger & \mathbf{0} \\ \hline
                    \mathbf{0} & I_{HD-d}
                \end{array}
            \right] \quad

            X_d^\dagger = \begin{pmatrix}
                0 & 1 & \cdots & 0 \\
                \vdots & \vdots & \ddots & \vdots \\
                0 & 0 & \cdots & 1 \\
                1 & 0 & \cdots & 0 \\
            \end{pmatrix}

        Where :math:`HD = 2^n` is the full Hilbert-space dimension
        (``self.hilbert_dim``).

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`X_d^\dagger`.
        """
        d = self.dim
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        unitary[:d, :d] = np.roll(  # pyright: ignore[reportUnknownMemberType]
            unitary[:d, :d],
            -1,
            axis=0,
        )
        return unitary

    @override
    def inverse(self, annotated: bool = False) -> QuditXGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditXGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditXGate(self.dim)
