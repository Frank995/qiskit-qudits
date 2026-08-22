"""Unit tests for :mod:`qiskit_qudits.circuit.instruction`.

:class:`~qiskit_qudits.circuit.instruction.QuditCircuitInstruction` is
a frozen, slotted dataclass pairing an operation with its qudit and
cldigit operands. The operation used throughout is a plain Qiskit
:class:`~qiskit.circuit.Instruction`, which keeps the tests focused on
the container rather than on any gate implementation.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
from typing import TYPE_CHECKING, Any

import pytest
from qiskit.circuit import Instruction

from qiskit_qudits.circuit.cldigit import ClDigit
from qiskit_qudits.circuit.instruction import QuditCircuitInstruction
from qiskit_qudits.circuit.qudit import Qudit

if TYPE_CHECKING:
    from qiskit_qudits.circuit.qudit import QuditRegister


@pytest.fixture
def operation() -> Instruction:
    """A lightweight two-qubit Qiskit instruction."""
    return Instruction("dummy", 2, 0, [])


@pytest.fixture
def instruction(
    operation: Instruction,
    qutrit_register: QuditRegister,
) -> QuditCircuitInstruction:
    """One qutrit target and one qutrit-sized cldigit."""
    return QuditCircuitInstruction(
        operation,
        (qutrit_register[0],),
        (ClDigit(3),),
    )


class TestQuditCircuitInstructionBasics:
    """Construction, defaults and derived properties."""

    def test_targets_default_to_empty_tuples(
        self,
        operation: Instruction,
    ) -> None:
        """Only the operation is mandatory."""
        instruction = QuditCircuitInstruction(operation)
        assert instruction.operation is operation
        assert instruction.qudits == ()
        assert instruction.cldigits == ()
        assert instruction.num_qudits == 0
        assert instruction.num_cldigits == 0
        assert instruction.dims == ()

    @pytest.mark.parametrize("name", ["dummy", "measure", "qudit_x"])
    def test_name_delegates_to_the_operation(self, name: str) -> None:
        """``name`` is never stored, it is read off the operation."""
        instruction = QuditCircuitInstruction(Instruction(name, 1, 0, []))
        assert instruction.name == name
        assert instruction.name == instruction.operation.name

    def test_counts_and_dims_follow_the_targets(
        self,
        operation: Instruction,
        qutrit_register: QuditRegister,
    ) -> None:
        """``dims`` reports the target dimensions in operand order."""
        instruction = QuditCircuitInstruction(
            operation,
            (qutrit_register[0], Qudit(4), Qudit(2)),
            (ClDigit(3), ClDigit(4)),
        )
        assert instruction.num_qudits == 3
        assert instruction.num_cldigits == 2
        assert instruction.dims == (3, 4, 2)

    def test_it_is_a_slotted_dataclass(
        self,
        instruction: QuditCircuitInstruction,
    ) -> None:
        """Three fields, no ``__dict__`` and closed for subclassing."""
        assert is_dataclass(instruction)
        assert [field.name for field in fields(instruction)] == [
            "operation",
            "qudits",
            "cldigits",
        ]
        assert not hasattr(instruction, "__dict__")
        assert getattr(QuditCircuitInstruction, "__final__", False) is True

    def test_repr(
        self,
        operation: Instruction,
        qutrit_register: QuditRegister,
    ) -> None:
        """The representation lists the operation name and targets."""
        instruction = QuditCircuitInstruction(
            operation,
            (qutrit_register[0],),
        )
        assert repr(instruction) == (
            "QuditCircuitInstruction(dummy, "
            "qudits=(Qudit(qt[0], d=3),), cldigits=())"
        )


class TestReplace:
    """:meth:`QuditCircuitInstruction.replace` semantics."""

    def test_replacing_the_operation_keeps_the_targets(
        self,
        instruction: QuditCircuitInstruction,
    ) -> None:
        """Only the operation changes."""
        other = Instruction("other", 2, 0, [])
        derived = instruction.replace(operation=other)
        assert derived.operation is other
        assert derived.qudits is instruction.qudits
        assert derived.cldigits is instruction.cldigits

    def test_replacing_the_qudits_keeps_the_rest(
        self,
        instruction: QuditCircuitInstruction,
    ) -> None:
        """Only the qudit operands change."""
        qudits = (Qudit(4), Qudit(5))
        derived = instruction.replace(qudits=qudits)
        assert derived.qudits is qudits
        assert derived.dims == (4, 5)
        assert derived.operation is instruction.operation
        assert derived.cldigits is instruction.cldigits

    def test_replacing_the_cldigits_keeps_the_rest(
        self,
        instruction: QuditCircuitInstruction,
    ) -> None:
        """Only the cldigit operands change."""
        cldigits = (ClDigit(2),)
        derived = instruction.replace(cldigits=cldigits)
        assert derived.cldigits is cldigits
        assert derived.operation is instruction.operation
        assert derived.qudits is instruction.qudits

    def test_the_original_is_never_mutated(
        self,
        instruction: QuditCircuitInstruction,
    ) -> None:
        """``replace`` builds a new object."""
        original = (
            instruction.operation,
            instruction.qudits,
            instruction.cldigits,
        )
        derived = instruction.replace(qudits=(Qudit(2),))
        assert derived is not instruction
        assert instruction.operation is original[0]
        assert instruction.qudits is original[1]
        assert instruction.cldigits is original[2]

    def test_none_means_keep(
        self,
        instruction: QuditCircuitInstruction,
    ) -> None:
        """Passing ``None`` explicitly is the same as omitting it."""
        derived = instruction.replace(
            operation=None,
            qudits=None,
            cldigits=None,
        )
        assert derived == instruction
        assert derived is not instruction
        assert derived.qudits is instruction.qudits

    def test_an_empty_tuple_really_clears_the_targets(
        self,
        instruction: QuditCircuitInstruction,
    ) -> None:
        """An empty tuple is a value, not the keep sentinel."""
        derived = instruction.replace(qudits=(), cldigits=())
        assert derived.qudits == ()
        assert derived.cldigits == ()
        assert derived.num_qudits == 0
        assert derived.num_cldigits == 0


class TestImmutability:
    """The dataclass is frozen."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("operation", Instruction("other", 1, 0, [])),
            ("qudits", ()),
            ("cldigits", ()),
        ],
    )
    def test_fields_cannot_be_assigned(
        self,
        instruction: QuditCircuitInstruction,
        field: str,
        value: Any,
    ) -> None:
        """Assigning to any field raises ``FrozenInstanceError``."""
        with pytest.raises(FrozenInstanceError, match="cannot assign"):
            setattr(instruction, field, value)

    @pytest.mark.parametrize(
        "field",
        ["operation", "qudits", "cldigits"],
    )
    def test_fields_cannot_be_deleted(
        self,
        instruction: QuditCircuitInstruction,
        field: str,
    ) -> None:
        """Deleting any field raises ``FrozenInstanceError``."""
        with pytest.raises(FrozenInstanceError, match="cannot delete"):
            delattr(instruction, field)


class TestEquality:
    """Dataclass-generated ``__eq__``."""

    def test_equal_when_every_field_matches(
        self,
        operation: Instruction,
        qutrit_register: QuditRegister,
    ) -> None:
        """Same operation and same target objects compare equal."""
        qudits = (qutrit_register[0],)
        first = QuditCircuitInstruction(operation, qudits)
        second = QuditCircuitInstruction(operation, qudits)
        assert first == second

    def test_different_targets_are_not_equal(
        self,
        operation: Instruction,
        qutrit_register: QuditRegister,
    ) -> None:
        """Targets take part in the comparison."""
        first = QuditCircuitInstruction(operation, (qutrit_register[0],))
        second = QuditCircuitInstruction(operation, (qutrit_register[1],))
        assert first != second

    def test_different_operations_are_not_equal(
        self,
        operation: Instruction,
        qutrit_register: QuditRegister,
    ) -> None:
        """The operation takes part in the comparison."""
        qudits = (qutrit_register[0],)
        other = Instruction("other", 2, 0, [])
        first = QuditCircuitInstruction(operation, qudits)
        second = QuditCircuitInstruction(other, qudits)
        assert first != second

    @pytest.mark.parametrize("other", ["dummy", 42, None])
    def test_not_equal_to_unrelated_objects(
        self,
        instruction: QuditCircuitInstruction,
        other: Any,
    ) -> None:
        """Comparing against a foreign type returns ``False``."""
        assert instruction != other
