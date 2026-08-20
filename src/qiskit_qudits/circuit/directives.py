"""Qudit-level, non-unitary circuit directives.

Every directive knows how to *expand itself* onto the encoded
:class:`~qiskit.circuit.QuantumCircuit` through
:meth:`QuditDirective.apply`.
That single dispatch point is what lets
:class:`~qiskit_qudits.circuit.QuditQuantumCircuit` treat gates and
directives uniformly in ``append``, ``copy``, ``compose``, ``inverse``
and the ideal circuit view.

Directives subclass :class:`~qiskit.circuit.Instruction` (whose base
``validate_parameter`` is permissive) so that they carry a ``name``,
``params`` and ``label`` and play nicely with ``count_ops``-style
inspection. They are *never* appended to a Qiskit circuit directly:
:meth:`~QuditDirective.apply` emits primitive Qiskit operations instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from qiskit.circuit.instruction import Instruction

from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.utils.dims import qubits_per_qudit
from qiskit_qudits.utils.encoding import (
    embed_state,
    level_to_bitstring,
    validate_basis_states,
)
from qiskit_qudits.utils.validation import (
    validate_dim,
    validate_vector,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from qiskit.circuit.quantumcircuit import QuantumCircuit

    from qiskit_qudits.circuit.clbyte import ClByte
    from qiskit_qudits.circuit.qudit import Qudit
    from qiskit_qudits.utils.consts import IntLike, VectorLike


class QuditDirective(Instruction, ABC):
    r"""Base class for non-unitary qudit-level operations.

    Subclasses must implement :meth:`apply`, which receives the encoded
    circuit together with the *resolved* target qudits and clbytes and
    is responsible for emitting the equivalent primitive Qiskit
    operations.

    **Subclassing contract** - :meth:`apply` must validate its arguments
    *before* mutating the circuit, so that a failed application never
    leaves a half-built circuit behind.
    """

    def __init__(
        self,
        name: str,
        dims: Sequence[IntLike],
        *,
        num_clbytes: int = 0,
        num_clbits: int = 0,
        params: Sequence[object] = (),
        label: str | None = None,
    ) -> None:
        r"""Create a directive.

        Args:
            name: Operation name, e.g. ``'measure'``.
            dims: Dimension of each target qudit, in target order.
            num_clbytes: Number of target clbytes.
            num_clbits: Total number of target clbits.
            params: Values stored in :attr:`params` for introspection.
            label: Optional display label.

        """
        self._dims: tuple[int, ...] = tuple(validate_dim(dim) for dim in dims)
        self._widths: tuple[int, ...] = tuple(
            qubits_per_qudit(dim) for dim in self._dims
        )
        self._num_clbytes: int = num_clbytes
        super().__init__(
            name,
            sum(self._widths),
            num_clbits,
            list(params),
            label=label,
        )

    @property
    def dims(self) -> tuple[int, ...]:
        """Dimension of each target qudit, in target order."""
        return self._dims

    @property
    def widths(self) -> tuple[int, ...]:
        """Number of encoding qubits of each target qudit."""
        return self._widths

    @property
    def num_qudits(self) -> int:
        """Number of target qudits."""
        return len(self._dims)

    @property
    def num_clbytes(self) -> int:
        """Number of target clbytes."""
        return self._num_clbytes

    def _check_targets(
        self,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Validate the arity and dimensions of the resolved targets.

        Args:
            qudits: Resolved target qudits.
            clbytes: Resolved target clbytes.

        Raises:
            QuditCircuitError: On an arity or dimension mismatch.
        """
        if len(qudits) != self.num_qudits:
            raise QuditCircuitError(
                f"'{self.name}' acts on {self.num_qudits} qudit(s), "
                f"got {len(qudits)}.",
            )
        if len(clbytes) != self.num_clbytes:
            raise QuditCircuitError(
                f"'{self.name}' acts on {self.num_clbytes} clbyte(s), "
                f"got {len(clbytes)}.",
            )
        for position, (qudit, dim) in enumerate(
            zip(qudits, self._dims, strict=True),
        ):
            if qudit.dim != dim:
                raise QuditCircuitError(
                    f"'{self.name}' expects a {dim}-level qudit at position "
                    f"{position}, got {qudit.dim} levels.",
                )

    @abstractmethod
    def apply(
        self,
        circuit: QuantumCircuit,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Emit the encoded implementation of this directive.

        Args:
            circuit: The encoded circuit to mutate.
            qudits: Resolved target qudits, in target order.
            clbytes: Resolved target clbytes, in target order.

        Raises:
            NotImplementedError: If not overridden.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement apply().",
        )


class QuditBarrier(QuditDirective):
    """A barrier spanning every encoding qubit of the target qudits."""

    def __init__(
        self,
        dims: Sequence[IntLike],
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit barrier.

        Args:
            dims: Dimension of each target qudit.
            label: Optional barrier label.

        """
        super().__init__("barrier", dims, label=label)
        # Mirrors Qiskit's `Barrier`: directives are skipped by
        # `size`/`depth`.
        self._directive = True

    def apply(
        self,
        circuit: QuantumCircuit,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Apply one Qiskit barrier over all encoding qubits.

        Args:
            circuit: The encoded circuit to mutate.
            qudits: Resolved target qudits, in target order.
            clbytes: Resolved target clbytes, in target order.

        """
        self._check_targets(qudits, clbytes)
        qubits = [qubit for qudit in qudits for qubit in qudit.qubits]
        if qubits:
            circuit.barrier(*qubits, label=self.label)


class QuditReset(QuditDirective):
    r"""Reset a qudit to :math:`\lvert 0 \rangle`.

    All encoding qubits are reset, which projects the qudit onto
    :math:`\lvert 0 \rangle` including any leakage state.
    """

    def __init__(self, dims: Sequence[IntLike]) -> None:
        """Create a qudit reset.

        Args:
            dims: Dimension of each target qudit.

        """
        super().__init__("reset", dims)

    def apply(
        self,
        circuit: QuantumCircuit,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Reset every encoding qubit of each target qudit.

        Args:
            circuit: The encoded circuit to mutate.
            qudits: Resolved target qudits, in target order.
            clbytes: Resolved target clbytes, in target order.

        """
        self._check_targets(qudits, clbytes)
        for qudit in qudits:
            for qubit in qudit.qubits:
                circuit.reset(qubit)


class QuditMeasure(QuditDirective):
    r"""Measure qudits into clbytes in the computational (level) basis.

    Qubit ``j`` of a qudit is measured into clbit ``j`` of its clbyte,
    so the recorded bits are the little-endian binary expansion of the
    measured level.
    """

    def __init__(self, dims: Sequence[IntLike]) -> None:
        """Create a qudit measurement.

        Args:
            dims: Dimension of each measured qudit.

        """
        widths = [qubits_per_qudit(validate_dim(dim)) for dim in dims]
        super().__init__(
            "measure",
            dims,
            num_clbytes=len(list(dims)),
            num_clbits=sum(widths),
        )

    def apply(
        self,
        circuit: QuantumCircuit,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Emit one primitive measurement per encoding qubit.

        Args:
            circuit: The encoded circuit to mutate.
            qudits: Resolved target qudits, in target order.
            clbytes: Resolved target clbytes, in target order.

        Raises:
            QuditCircuitError: If a clbyte is too narrow for its qudit,
                or on an arity or dimension mismatch.
        """
        self._check_targets(qudits, clbytes)
        for qudit, clbyte in zip(qudits, clbytes, strict=True):
            if clbyte.num_clbits != qudit.num_qubits:
                raise QuditCircuitError(
                    f"cannot measure a {qudit.dim}-level qudit "
                    f"({qudit.num_qubits} qubit(s)) into a clbyte of "
                    f"{clbyte.num_clbits} bit(s).",
                )
        for qudit, clbyte in zip(qudits, clbytes, strict=True):
            for qubit, clbit in zip(qudit.qubits, clbyte.clbits, strict=True):
                circuit.measure(qubit, clbit)


class QuditInitializeLevels(QuditDirective):
    r"""Initialise qudits to computational basis states.

    Like Qiskit's :meth:`~qiskit.circuit.QuantumCircuit.initialize`,
    this is **not** unitary: it resets the encoding qubits first.
    """

    def __init__(
        self,
        dims: Sequence[IntLike],
        states: Sequence[IntLike],
    ) -> None:
        r"""Create a basis-state initialisation instruction.

        Args:
            dims: Dimension of each target qudit, in target order.
            states: Target state index to prepare on each qudit,
                in target order.

        """
        dims = tuple(validate_dim(dim) for dim in dims)
        self._states: tuple[int, ...] = validate_basis_states(states, dims)
        super().__init__("initialize", dims, params=list(self._states))

    @property
    def values(self) -> tuple[int, ...]:
        """The prepared levels, in target order."""
        return self._states

    def apply(
        self,
        circuit: QuantumCircuit,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Prepare each target qudit in its basis state.

        Args:
            circuit: The encoded circuit to mutate.
            qudits: Resolved target qudits, in target order.
            clbytes: Resolved target clbytes, in target order.

        Raises:
            QuditCircuitError: On an arity or dimension mismatch.
        """
        self._check_targets(qudits, clbytes)
        for qudit, state in zip(qudits, self._states, strict=True):
            # Qiskit applies the leftmost character of the label to the
            # *last* qubit of the list, and `level_to_bitstring` is
            # MSB-first, so passing the qubits LSB-first is correct.
            circuit.initialize(
                level_to_bitstring(state, qudit.num_qubits),
                list(qudit.qubits),
            )


class QuditStatePreparation(QuditDirective):
    r"""Initialise qudits from an arbitrary qudit state-vector.

    The stored state is always the *logical* vector of length
    :math:`\prod_i d_i`; the embedding into the :math:`2^N` dimensional
    encoded space (zero amplitude on invalid basis states) happens in
    :meth:`apply`.
    """

    def __init__(
        self,
        dims: Sequence[IntLike],
        amplitudes: VectorLike,
    ) -> None:
        r"""Create a state preparation.

        Args:
            dims: Dimension of each target qudit, least significant
                (i.e. first target) first.
            amplitudes: :math:`\prod_i d_i` normalised amplitudes.

        """
        dims = tuple(validate_dim(dim) for dim in dims)
        vector = validate_vector(amplitudes)
        # `embed_state` performs all validation;
        # keep the logical vector.
        _ = embed_state(dims, vector)
        self._amplitudes = vector
        super().__init__("initialize", dims, params=[vector])

    @property
    def amplitudes(self) -> np.typing.NDArray[np.complex128]:
        """The logical qudit state-vector."""
        return self._amplitudes

    def apply(
        self,
        circuit: QuantumCircuit,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Prepare the encoded state on the target qudits.

        Args:
            circuit: The encoded circuit to mutate.
            qudits: Resolved target qudits, in target order.
            clbytes: Resolved target clbytes, in target order.

        """
        self._check_targets(qudits, clbytes)
        qubits = [qubit for qudit in qudits for qubit in qudit.qubits]
        circuit.initialize(
            embed_state(self._dims, self._amplitudes).tolist(),
            qubits,
        )
