"""Implementation of the QFT-Gate class for multi-qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from typing_extensions import override

from qiskit_qudits.gates.base.multigate import QuditMultiGate
from qiskit_qudits.gates.controlled.sump import QuditSUMPGate
from qiskit_qudits.gates.multi.swap import QuditSWAPGate
from qiskit_qudits.gates.single.h import QuditHdgGate
from qiskit_qudits.utils.validation import is_integral

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditQFTGate(QuditMultiGate):
    r"""Qudit QFT-gate.

    This gate applies the Quantum Fourier Transform (QFT) to a register
    of :math:`n` qudits sharing the same dimension :math:`d`, acting on
    the :math:`d^n`-dimensional computational subspace.

    .. math::

        QFT \lvert x \rangle = \frac{1}{\sqrt{d^n}}
            \sum_{y=0}^{d^n - 1} \omega^{xy} \lvert y \rangle,
            \quad \omega = e^{i 2\pi / d^n}

    where :math:`x = \sum_{i=0}^{n-1} x_i d^i` is the base-:math:`d`
    value encoded across the register, with qudit 0 holding the
    least-significant digit.

    Basis states outside the valid qudit subspace of any qudit are
    fixed points, keeping the full :math:`2^n \times 2^n` matrix
    unitary.

    When every qudit dimension is a power of two, the decomposition
    uses a predefined cascade of Hadamard, SUMP and SWAP gates rather
    than unitary synthesis, so that transpiler and optimisation passes
    can recognise it.
    """

    gate_name: ClassVar[str] = "QFT"

    def __init__(
        self,
        num_qudits: IntLike,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit QFT-gate.

        Args:
            num_qudits: The number of qudits the transform acts on.
            dim: The dimension of each qudit the gate operates on.
            label: An optional label for the gate.

        Raises:
            TypeError: If ``num_qudits`` is not an integer-like object.
        """
        if not is_integral(num_qudits):
            raise TypeError(
                f"num_qudits must be an integer, "
                f"got {type(num_qudits).__name__}.",
            )

        super().__init__(
            type(self).gate_name,
            [dim] * int(num_qudits),
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
            n = self.num_qudits

            # Iterate over target qudits from most to least significant.
            # Each target gets a Hadamard followed by controlled phase
            # rotations from all lower-index (less significant) qudits.
            for target in reversed(range(n)):
                target_qubits = list(self._qudit_range(target))
                qc.append(QuditHdgGate(d), target_qubits)

                for control in reversed(range(target)):
                    control_qubits = list(self._qudit_range(control))
                    # A SUMP between qudits 'distance' positions apart
                    # contributes the phase
                    # exp(i 2*pi c*t / d^(distance+1)).
                    distance = target - control
                    theta = np.pi / d**distance
                    qc.append(
                        QuditSUMPGate(d, [d], theta),
                        control_qubits + target_qubits,
                    )

            # The routine above outputs the transformed digits in
            # reversed significance, so restore the order with SWAPs.
            for low in range(n // 2):
                high = n - 1 - low
                qc.append(
                    QuditSWAPGate(d),
                    list(self._qudit_range(low))
                    + list(self._qudit_range(high)),
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
        r"""Construct the unitary matrix for the QFT gate.

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            the qudit QFT.
        """
        d = self.dims[0]
        n = self.num_qudits
        dim: int = d**n
        strides = self._compute_strides()

        # Flat Hilbert-space index of every valid base-d value, where
        # qudit 0 holds the least-significant digit.
        values = np.arange(dim)
        flat_indices = np.zeros(dim, dtype=int)
        for qudit_index in range(n):
            digit = (values // d**qudit_index) % d
            flat_indices += digit * strides[qudit_index]

        # DFT block over the valid d^n-dimensional subspace.
        exponents = np.outer(values, values)
        dft = np.exp(2j * np.pi * exponents / dim) / np.sqrt(dim)

        # Identity on invalid states keeps the full matrix unitary.
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        unitary[np.ix_(flat_indices, flat_indices)] = dft
        return unitary

    @override
    def inverse(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        annotated: bool = False,
    ) -> Gate:
        r"""Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`Gate` with the inverse QFT.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return cast("QuantumCircuit", self.definition).inverse().to_gate()
