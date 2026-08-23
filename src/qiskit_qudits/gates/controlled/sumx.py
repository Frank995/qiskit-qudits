"""Implementation of the SUMX and SUMXdg gate classes for controlled qudit operations."""  # noqa: E501, W505

from __future__ import annotations

import itertools as it
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from qiskit.circuit import QuantumCircuit
from typing_extensions import override

from qiskit_qudits.gates.base.controlledgate import QuditControlledGate
from qiskit_qudits.gates.single.x import QuditXdgGate, QuditXGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike, VectorLike


class QuditSUMXGate(QuditControlledGate):
    r"""Qudit SUMX-gate.

    This gate applies the SUMX (sum) operation on qudit systems,
    circularly shifting the target qudit up by the sum of the control
    qudit values. The SUMX-gate is the qudit generalization of the
    CX gate for qubits.

    .. math::

        SUMX_d \lvert j_0 \rangle \cdots \lvert j_{m-1} \rangle
        \lvert k \rangle
        = \lvert j_0 \rangle \cdots \lvert j_{m-1} \rangle
          \lvert\, (k + j_0 + \cdots + j_{m-1}) \bmod d_t \,\rangle

    Basis states outside the valid qudit subspace of any qudit are
    fixed points, keeping the full :math:`2^n \times 2^n` matrix
    unitary.

    When every qudit dimension is a power of two, the decomposition
    uses a predefined controlled-increment cascade rather than
    unitary synthesis, so that transpiler and optimisation passes can
    recognise it.
    """

    gate_name: ClassVar[str] = "SUMX"

    def __init__(
        self,
        target_dim: IntLike,
        control_dims: VectorLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit SUMX-gate.

        Args:
            target_dim: Dimension *d_t* of the target qudit.
            control_dims: Dimension of each control qudit, given
                in register order ``(d_{c_0}, d_{c_1}, ...)``.
            label: An optional label for the gate.
        """
        super().__init__(
            type(self).gate_name,
            target_dim,
            control_dims,
            [],
            base_gate=QuditXGate(target_dim),
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
                # and apply controlled increments to the target qubits.
                for ctrl_idx, ctrl_qubit in enumerate(control_qubits):
                    # Only target qubits at index >= ctrl_idx are
                    # affected: a control qubit with weight
                    # 2^ctrl_idx can only increment target bits of
                    # equal or higher significance.
                    affected_target_qubits = target_qubits[ctrl_idx:]

                    # From qubits with highest to lowest significance,
                    # apply controlled X-gate with all lower-index
                    # qubits as controls plus the additional control
                    # qudit.
                    for tgt_idx in reversed(
                        range(len(affected_target_qubits)),
                    ):
                        total_controls = [
                            ctrl_qubit,
                            *affected_target_qubits[:tgt_idx],
                        ]
                        qc.mcx(
                            total_controls,
                            affected_target_qubits[tgt_idx],
                        )
        else:
            # Fall back to unitary synthesis.
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the SUMX gate.

        Each valid control configuration
        :math:`(j_0, \dots, j_{m-1})` induces a cyclic permutation of
        the target block, shifting :math:`\lvert k \rangle \mapsto
        \lvert (k + j_0 + \cdots + j_{m-1}) \bmod d_t \rangle` while
        leaving all other (invalid) basis states untouched.

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`SUMX_d`.
        """
        d_t = self.target_dim
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        control_strides, target_stride = self._compute_control_target_strides()
        k_range = np.arange(d_t)

        # Loop over controls configuration
        for j_controls in it.product(
            *(range(d_c) for d_c in self._control_dims),
        ):
            cumulative_shift = sum(j_controls) % d_t
            if cumulative_shift == 0:
                continue

            # Calculate flat indices for the current target qudit block
            control_offset = sum(
                j_c * s
                for j_c, s in zip(j_controls, control_strides, strict=True)
            )

            # Current columns (where the 1s are in an identity matrix)
            cols = control_offset + k_range * target_stride
            # New rows (where the 1s should move for
            # the X^s permutation)
            rows = (
                control_offset
                + ((k_range + cumulative_shift) % d_t) * target_stride
            )

            # We clear the diagonal 1s and set the shifted 1s
            unitary[cols, cols] = 0
            unitary[rows, cols] = 1.0

        return unitary

    @override
    def inverse(self, annotated: bool = False) -> QuditSUMXdgGate:
        r"""Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditSUMXdgGate` with the same target and
            control qudit dimensions.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditSUMXdgGate(self.target_dim, self.control_dims)


class QuditSUMXdgGate(QuditControlledGate):
    r"""Qudit SUMXdg-gate.

    This gate applies the inverse SUMX (sum) operation on qudit
    systems, circularly shifting the target qudit down by the sum
    of the control qudit values. The SUMXdg-gate is the qudit
    generalization of the inverse CX gate for qubits.

    .. math::

        SUMX_d^\dagger \lvert j_0 \rangle \cdots \lvert j_{m-1} \rangle
        \lvert k \rangle
        = \lvert j_0 \rangle \cdots \lvert j_{m-1} \rangle
          \lvert\, (k - j_0 - \cdots - j_{m-1}) \bmod d_t \,\rangle

    Basis states outside the valid qudit subspace of any qudit are
    fixed points, keeping the full :math:`2^n \times 2^n` matrix
    unitary.

    When every qudit dimension is a power of two, the decomposition
    uses a predefined controlled-decrement cascade rather than
    unitary synthesis, so that transpiler and optimisation passes can
    recognise it.
    """

    gate_name: ClassVar[str] = "SUMXdg"

    def __init__(
        self,
        target_dim: IntLike,
        control_dims: VectorLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit SUMXdg-gate.

        Args:
            target_dim: Dimension *d_t* of the target qudit.
            control_dims: Dimension of each control qudit, given
                in register order ``(d_{c_0}, d_{c_1}, ...)``.
            label: An optional label for the gate.
        """
        super().__init__(
            type(self).gate_name,
            target_dim,
            control_dims,
            [],
            base_gate=QuditXdgGate(target_dim),
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
                # and apply controlled decrements to the target qubits.
                for ctrl_idx, ctrl_qubit in enumerate(control_qubits):
                    # Only target qubits at index >= ctrl_idx are
                    # affected: a control qubit with weight
                    # 2^ctrl_idx can only decrement target bits of
                    # equal or higher significance.
                    affected_target_qubits = target_qubits[ctrl_idx:]

                    # From qubits with lowest to highest significance
                    # (the reverse order of SUMX's increment cascade,
                    # since mcx is self-inverse), apply controlled
                    # X-gate with all lower-index qubits as controls
                    # plus the additional control qudit.
                    for tgt_idx in range(len(affected_target_qubits)):
                        total_controls = [
                            ctrl_qubit,
                            *affected_target_qubits[:tgt_idx],
                        ]
                        qc.mcx(
                            total_controls,
                            affected_target_qubits[tgt_idx],
                        )
        else:
            # Fall back to unitary synthesis.
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the SUMXdg gate.

        Each valid control configuration
        :math:`(j_0, \dots, j_{m-1})` induces a cyclic permutation of
        the target block, shifting :math:`\lvert k \rangle \mapsto
        \lvert (k - j_0 - \cdots - j_{m-1}) \bmod d_t \rangle` while
        leaving all other (invalid) basis states untouched.

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`SUMX_d^\dagger`.
        """
        d_t = self.target_dim
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        control_strides, target_stride = self._compute_control_target_strides()
        k_range = np.arange(d_t)

        # Loop over controls configuration
        for j_controls in it.product(
            *(range(d_c) for d_c in self._control_dims),
        ):
            cumulative_shift = sum(j_controls) % d_t
            if cumulative_shift == 0:
                continue

            # Calculate flat indices for the current target qudit block
            control_offset = sum(
                j_c * s
                for j_c, s in zip(j_controls, control_strides, strict=True)
            )

            # Current columns (where the 1s are in an identity matrix)
            cols = control_offset + k_range * target_stride
            # New rows (where the 1s should move for
            # the X^-s permutation)
            rows = (
                control_offset
                + ((k_range - cumulative_shift) % d_t) * target_stride
            )

            # We clear the diagonal 1s and set the shifted 1s
            unitary[cols, cols] = 0
            unitary[rows, cols] = 1.0

        return unitary

    @override
    def inverse(self, annotated: bool = False) -> QuditSUMXGate:
        r"""Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditSUMXGate` with the same target and
            control qudit dimensions.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditSUMXGate(self.target_dim, self.control_dims)
