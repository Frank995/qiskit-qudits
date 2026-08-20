"""Qudit-level circuit instructions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeAlias, final

if TYPE_CHECKING:
    from qiskit.circuit import Instruction

    from qiskit_qudits.circuit.clbyte import ClByte
    from qiskit_qudits.circuit.qudit import Qudit


@final
@dataclass(frozen=True, slots=True)
class QuditCircuitInstruction:
    """An operation together with the qudits and clbytes it acts on.

    This is the qudit analogue of
    :class:`~qiskit.circuit.CircuitInstruction`. It is immutable; use
    :meth:`replace` to derive a modified copy.

    Attributes:
        operation: The applied operation. Either a
            :class:`~qiskit_qudits.gates.base.gate.QuditGate` (or any
            other :class:`~qiskit.circuit.Instruction` acting on the
            encoding qubits) or a
            :class:`~qiskit_qudits.circuit.directives.QuditDirective`.
        qudits: The target qudits, in operand order.
        clbytes: The target clbytes, in operand order.
    """

    operation: Instruction
    qudits: tuple[Qudit, ...] = ()
    clbytes: tuple[ClByte, ...] = ()

    @property
    def name(self) -> str:
        """Name of the applied operation."""
        return self.operation.name

    @property
    def num_qudits(self) -> int:
        """Number of target qudits."""
        return len(self.qudits)

    @property
    def num_clbytes(self) -> int:
        """Number of target clbytes."""
        return len(self.clbytes)

    @property
    def dims(self) -> tuple[int, ...]:
        """Dimension of each target qudit, in operand order."""
        return tuple(qudit.dim for qudit in self.qudits)

    def replace(
        self,
        *,
        operation: Instruction | None = None,
        qudits: tuple[Qudit, ...] | None = None,
        clbytes: tuple[ClByte, ...] | None = None,
    ) -> QuditCircuitInstruction:
        """Return a copy with the given fields replaced.

        Args:
            operation: New operation, or ``None`` to keep the
                current one.
            qudits: New target qudits, or ``None`` to keep them.
            clbytes: New target clbytes, or ``None`` to keep them.

        Returns:
            The derived instruction.
        """
        return replace(
            self,
            operation=self.operation if operation is None else operation,
            qudits=self.qudits if qudits is None else qudits,
            clbytes=self.clbytes if clbytes is None else clbytes,
        )

    def __repr__(self) -> str:
        """Return a short, unambiguous representation."""
        return (
            f"QuditCircuitInstruction({self.name}, "
            f"qudits={self.qudits}, clbytes={self.clbytes})"
        )


#: Handle returned by the gate helpers of
#: :class:`~qiskit_qudits.circuit.QuditQuantumCircuit`, mirroring
#: Qiskit's :class:`~qiskit.circuit.InstructionSet` but immutable.
QuditInstructionSet: TypeAlias = tuple[QuditCircuitInstruction, ...]
