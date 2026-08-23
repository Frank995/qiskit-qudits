"""Implementation of the P-Gate class for qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from qiskit.circuit import QuantumCircuit
from typing_extensions import override

from qiskit_qudits.gates.base.gate import QuditGate
from qiskit_qudits.gates.base.mixins import QuditPhaseGateMixin

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import FloatLike, IntLike


class QuditPGate(QuditGate, QuditPhaseGateMixin):
    r"""Qudit P-gate class.

    This gate applies the P (phase) operation on qudit systems,
    rotating the phase around the Z-axis of each quantum state level by
    an amount proportional to the level index.

    .. math::

        P_d(\theta) \lvert k \rangle = \omega^{\frac{k \, \theta}{\pi}}
        \lvert k \rangle, \quad \omega = e^{\frac{i 2 \pi}{d}}, \quad k
        \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "P"

    def __init__(
        self,
        dim: IntLike,
        theta: FloatLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit P-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            theta: Rotation angle in radians.
            label: An optional label for the gate.
        """
        super().__init__(
            type(self).gate_name,
            dim,
            [self._validate_theta(theta)],
            label=label,
        )

    def _define(self) -> None:
        """Lazily build the qubit-level circuit decomposition."""
        qc = QuantumCircuit(
            self.num_qubits,
            name=self.label if self.label is not None else self.name,
        )

        if self.fills_hilbert_space:
            # Apply P gates whose angles are proportional to powers
            # of the root of unity.
            for index in range(self.num_qubits):
                qc.p(2 * self.theta * (1 << index) / self.hilbert_dim, index)
        else:
            # Fall back to unitary synthesis.
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the P-gate.

        .. math::

            P_{HD}(\theta) = \left[
                \begin{array}{c|c}
                    P_d(\theta) & \mathbf{0} \\ \hline
                    \mathbf{0} & I_{HD-d}
                \end{array}
            \right] \quad

            P_d(\theta) = \begin{pmatrix}
                1 & 0 & \cdots & 0 \\
                0 & \omega^{\frac{\theta}{\pi}} & \cdots & 0 \\
                \vdots & \vdots & \ddots & \vdots \\
                0 & 0 & \cdots & \omega^{\frac{(d-1)\theta}{\pi}}
            \end{pmatrix}

        Where :math:`HD = 2^n` is the full Hilbert-space dimension
        (``self.hilbert_dim``).

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`P_d(\theta)`.
        """
        d = self.dim
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        phases = np.exp(1j * 2 * np.arange(d) * self.theta / d)
        np.fill_diagonal(unitary[:d, :d], phases)
        return unitary

    @override
    def inverse(self, annotated: bool = False) -> Self:
        r"""Return the inverse gate.

        The inverse of the Z-rotation operation with rotation angle
        :math:`\theta` is a new :class:`QuditPGate` with angle
        :math:`-\theta`.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditPGate` with the same dimension
            and negated angle.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return type(self)(self.dim, -self.theta)
