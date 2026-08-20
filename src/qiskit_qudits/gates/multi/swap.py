"""Implementation of the SWAP-Gate class for qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from qiskit.circuit import QuantumCircuit
from typing_extensions import override

from qiskit_qudits.gates.base.multigate import QuditMultiGate
from qiskit_qudits.gates.controlled.sumx import QuditSUMXGate
from qiskit_qudits.gates.single.k import QuditKGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditSWAPGate(QuditMultiGate):
    r"""Qudit SWAP-gate.

    This gate swaps the states of two qudits of the same dimension
    :math:`d`.

    .. math::

        SWAP_d \lvert j \rangle \lvert k \rangle = \lvert k \rangle
        \lvert j \rangle, \quad j, k \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace of any qudit are
    fixed points, keeping the full :math:`2^n \times 2^n` matrix
    unitary.

    When every qudit dimension is a power of two, the decomposition
    uses a fixed sequence of SUMX and K gates (rather than unitary
    synthesis), so that transpiler and optimisation passes can
    recognise it.
    """

    gate_name: ClassVar[str] = "SWAP"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit SWAP-gate.

        Args:
            dim: The dimension of the qudits the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(
            type(self).gate_name,
            [dim] * 2,
            [],
            label=label,
        )

    def _define(self) -> None:
        """Lazily build the qubit-level circuit decomposition."""
        qc = QuantumCircuit(
            self.num_qubits,
            name=self.label if self.label is not None else self.name,
        )

        if self.fills_hilbert_space:
            d = self.dims[0]
            qubits_0 = list(self._qudit_range(0))
            qubits_1 = list(self._qudit_range(1))

            # SWAP is realised as SUMX† · SUMX · SUMX†, which maps
            # |a, b> -> |a-b, b> -> |a-b, a> -> |-b, a>, followed by a
            # K gate on qudit 0 to complement -b back into b, yielding
            # the swapped state |b, a>.

            # Apply SUMX† (target 0, control 1).
            qc.append(QuditSUMXGate(d, [d]).inverse(), qubits_1 + qubits_0)

            # Apply SUMX (target 1, control 0).
            qc.append(QuditSUMXGate(d, [d]), qubits_0 + qubits_1)

            # Apply SUMX† (target 0, control 1).
            qc.append(QuditSUMXGate(d, [d]).inverse(), qubits_1 + qubits_0)

            # Apply K to qudit 0 to complement its residual -b into b.
            qc.append(QuditKGate(d), qubits_0)
        else:
            # Fall back to unitary synthesis.
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the SWAP gate.

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`SWAP_d`.
        """
        d = self.dims[0]
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        strides = self._compute_strides()

        # Permute the valid (j, k) -> (k, j) subspace. States with
        # j == k are fixed points and are already handled by np.eye,
        # as are all states outside the valid qudit subspace.
        for j in range(d):
            for k in range(d):
                if j == k:
                    continue
                src_idx = j * strides[0] + k * strides[1]
                dst_idx = k * strides[0] + j * strides[1]
                unitary[src_idx, src_idx] = 0.0
                unitary[dst_idx, src_idx] = 1.0

        return unitary

    @override
    def inverse(self, annotated: bool = False) -> Self:
        r"""Return the inverse gate.

        The swap operation is self-inverse, so the inverse is a
        new :class:`QuditSWAPGate` with identical dimension.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditSWAPGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return type(self)(self.dims[0])
