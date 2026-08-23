"""Implementation of the SUMP-Gate class for controlled qudit operations."""  # noqa: W505

from __future__ import annotations

import itertools as it
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from qiskit.circuit import QuantumCircuit
from typing_extensions import override

from qiskit_qudits.gates.base.controlledgate import QuditControlledGate
from qiskit_qudits.gates.base.mixins import QuditPhaseGateMixin
from qiskit_qudits.gates.single.p import QuditPGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import FloatLike, IntLike, VectorLike


class QuditSUMPGate(QuditControlledGate, QuditPhaseGateMixin):
    r"""Qudit SUMP-gate.

    This gate applies the SUMP (controlled phase) operation on qudit
    systems, rotating the phase of the target qudit around the Z-axis
    by an amount proportional to the product of the sum of the control
    qudit levels and the target qudit level. The SUMP-gate is the qudit
    generalization of the CP gate for qubits.

    .. math::

        SUMP_d(\theta) \lvert j_0 \rangle \cdots \lvert j_{m-1} \rangle
        \lvert k \rangle
        = \omega^{\frac{(j_0 + \cdots + j_{m-1}) \, k \, \theta}{\pi}}
          \lvert j_0 \rangle \cdots \lvert j_{m-1} \rangle
          \lvert k \rangle, \quad \omega = e^{\frac{i 2 \pi}{d}},
          \quad k \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace of any qudit are
    fixed points, keeping the full :math:`2^n \times 2^n` matrix
    unitary.

    When every qudit dimension is a power of two, the decomposition
    uses a predefined controlled-phase cascade rather than unitary
    synthesis, so that transpiler and optimisation passes can recognise
    it.
    """

    gate_name: ClassVar[str] = "SUMP"

    def __init__(
        self,
        target_dim: IntLike,
        control_dims: VectorLike,
        theta: FloatLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit SUMP-gate.

        Args:
            target_dim: Dimension *d_t* of the target qudit.
            control_dims: Dimension of each control qudit, given
                in register order ``(d_{c_0}, d_{c_1}, ...)``.
            theta: Rotation angle in radians.
            label: An optional label for the gate.
        """
        super().__init__(
            type(self).gate_name,
            target_dim,
            control_dims,
            [self._validate_theta(theta)],
            base_gate=QuditPGate(target_dim, theta),
            label=label,
        )

    def _define(self) -> None:
        """Lazily build the qubit-level circuit decomposition."""
        qc = QuantumCircuit(
            self.num_qubits,
            name=self.label if self.label is not None else self.name,
        )

        if self.fills_hilbert_space:
            target_qubits = list(self._target_qudit_range())

            # Iterate over control qudits.
            for control_qudit in range(self.num_control_qudits):
                control_qubits = list(self._control_qudit_range(control_qudit))

                # Iterate over the control qubits of the current qudit
                # and apply phase rotations to each target qubit.
                for ctrl_idx, ctrl_qubit in enumerate(control_qubits):
                    # Each control qubit i of a control qudit
                    # contributes a weight 2^i to the control level.
                    # Each target qubit j contributes a weight 2^j to
                    # the target level. The phase on |c>|t> must be
                    # 2*theta*c*t/d_t, so for each (control_qubit,
                    # target_qubit) pair we apply a controlled phase
                    # gate with angle
                    # 2*theta*2^ctrl_idx*2^tgt_idx/d_t.
                    for tgt_idx, tgt_qubit in enumerate(target_qubits):
                        angle = (
                            2
                            * self.theta
                            * (2**ctrl_idx)
                            * (2**tgt_idx)
                            / self.target_hilbert_dim
                        )
                        qc.cp(angle, ctrl_qubit, tgt_qubit)
        else:
            # Fall back to unitary synthesis.
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the SUMP gate.

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`SUMP_d(\theta)`.
        """
        d_t = self.target_dim
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        control_strides, target_stride = self._compute_control_target_strides()
        k_range = np.arange(d_t)

        # Loop over controls configuration
        for j_controls in it.product(
            *(range(d_c) for d_c in self._control_dims),
        ):
            cumulative_sum = sum(j_controls)
            if cumulative_sum == 0:
                continue

            # Calculate all phases for this target block at once
            phases = np.exp(
                1j * 2 * cumulative_sum * k_range * self.theta / d_t,
            )

            # Get the flat indices for the diagonal
            # entries of this block
            control_offset = sum(
                j_c * s
                for j_c, s in zip(j_controls, control_strides, strict=True)
            )
            diag_indices = control_offset + k_range * target_stride

            # Update the diagonal in one shot
            unitary[diag_indices, diag_indices] = phases

        return unitary

    @override
    def inverse(self, annotated: bool = False) -> Self:
        r"""Return the inverse gate.

        The inverse of the Z-rotation operation with rotation angle
        :math:`\theta` is a new :class:`QuditSUMPGate` with angle
        :math:`-\theta`.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditSUMPGate` with the same number of
            levels and negated angle.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return type(self)(
            self.target_dim,
            self.control_dims,
            -self.theta,
        )
