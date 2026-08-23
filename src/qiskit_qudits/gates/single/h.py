"""Implementation of the H-Gate class for qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.qft import synth_qft_full
from scipy.linalg import dft
from typing_extensions import override

from qiskit_qudits.gates.base.gate import QuditGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditHGate(QuditGate):
    r"""Qudit H-gate class.

    This gate applies the H (Hadamard) operation on qudit systems,
    creating an equal superposition of all computational basis states
    with relative phases determined by the Discrete Fourier Transform
    (DFT) over the qudit dimension.

    .. math::

        H_d \lvert k \rangle = \frac{1}{\sqrt{d}} \sum_{j=0}^{d-1}
        \omega^{-k j} \lvert j \rangle, \quad \omega =
        e^{\frac{i 2 \pi}{d}}, \quad k \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "H"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit H-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(type(self).gate_name, dim, [], label=label)

    def _define(self) -> None:
        """Lazily build the qubit-level circuit decomposition."""
        if self.fills_hilbert_space:
            # Apply inverse QFT synthesis.
            qc = synth_qft_full(
                self.num_qubits,
                do_swaps=True,
                approximation_degree=0,
                inverse=True,
                name=self.label if self.label is not None else self.name,
            )
        else:
            # Fall back to unitary synthesis.
            qc = QuantumCircuit(
                self.num_qubits,
                name=self.label if self.label is not None else self.name,
            )
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the H-gate.

        .. math::

            H_{HD} = \left[
                \begin{array}{c|c}
                    H_d & \mathbf{0} \\ \hline
                    \mathbf{0} & I_{HD-d}
                \end{array}
            \right] \quad

            H_d = \frac{1}{\sqrt{d}}\begin{pmatrix}
                1 & 1 & \dots & 1 \\
                1 & \omega^{-1} & \dots & \omega^{-(d-1)} \\
                \vdots & \vdots & \ddots & \vdots \\
                1 & \omega^{-(d-1)} & \dots & \omega^{-(d-1)^2}
            \end{pmatrix}

        Where :math:`HD = 2^n` is the full Hilbert-space dimension
        (``self.hilbert_dim``).

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`H_d`.
        """
        d = self.dim
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        unitary[:d, :d] = dft(d, scale="sqrtn")
        return unitary

    @override
    def inverse(self, annotated: bool = False) -> QuditHdgGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditHdgGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditHdgGate(self.dim)


class QuditHdgGate(QuditGate):
    r"""Qudit Hdg-gate class.

    This gate applies the inverse H (Hadamard) operation on qudit
    systems, creating an equal superposition of all computational basis
    states with relative phases determined by the Discrete Fourier
    Transform (DFT) over the qudit dimension.

    .. math::

        H_d^\dagger \lvert k \rangle = \frac{1}{\sqrt{d}}
        \sum_{j=0}^{d-1} \omega^{k j} \lvert j \rangle, \quad \omega =
        e^{\frac{i 2 \pi}{d}}, \quad k \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "Hdg"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit Hdg-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(type(self).gate_name, dim, [], label=label)

    def _define(self) -> None:
        """Lazily build the qubit-level circuit decomposition."""
        if self.fills_hilbert_space:
            # Apply QFT synthesis.
            qc = synth_qft_full(
                self.num_qubits,
                do_swaps=True,
                approximation_degree=0,
                inverse=False,
                name=self.label if self.label is not None else self.name,
            )
        else:
            # Fall back to unitary synthesis.
            qc = QuantumCircuit(
                self.num_qubits,
                name=self.label if self.label is not None else self.name,
            )
            qc.unitary(
                self._build_unitary(),
                list(range(self.num_qubits)),
                label=type(self).gate_name,
            )

        self.definition = qc

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Construct the unitary matrix for the Hdg-gate.

        .. math::

            H_{HD}^\dagger = \left[
                \begin{array}{c|c}
                    H_d^\dagger & \mathbf{0} \\ \hline
                    \mathbf{0} & I_{HD-d}
                \end{array}
            \right] \quad

            H_d^\dagger = \frac{1}{\sqrt{d}}\begin{pmatrix}
                1 & 1 & \cdots & 1 \\
                1 & \omega^{1} & \cdots & \omega^{d-1} \\
                \vdots & \vdots & \ddots & \vdots \\
                1 & \omega^{d-1} & \cdots & \omega^{(d-1)^2}
            \end{pmatrix}

        Where :math:`HD = 2^n` is the full Hilbert-space dimension
        (``self.hilbert_dim``).

        Returns:
            A :math:`(2^n, 2^n)` complex unitary matrix representing
            :math:`H_d`.
        """
        d = self.dim
        unitary = np.eye(self.hilbert_dim, dtype=np.complex128)
        unitary[:d, :d] = dft(d, scale="sqrtn").conj().T
        return unitary

    @override
    def inverse(self, annotated: bool = False) -> QuditHGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditHGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditHGate(self.dim)
