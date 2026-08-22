"""Structural operations, views, metrics and decoding.

Covers ``copy_empty_like`` / ``copy`` / ``compose`` / ``inverse``, the
``size`` / ``depth`` / ``width`` / ``count_ops`` metrics, the qubit and
ideal views together with the drawing entry points, the equality and
hashing contract and the counts-decoding helpers.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import pytest
from qiskit.circuit import Barrier, Measure, QuantumCircuit, Reset

from qiskit_qudits.circuit.cldigit import ClDigit, ClDigitRegister
from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.circuit.quantumcircuit import QuditQuantumCircuit
from qiskit_qudits.circuit.qudit import Qudit, QuditRegister

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit_qudits.utils.encoding import InvalidPolicy, Levels

# ------------------------------------------------------------------ #
# Factories
# ------------------------------------------------------------------ #
MIXED_DIMS = (2, 3, 5)


def mixed_circuit() -> QuditQuantumCircuit:
    """Return a heterogeneous circuit with matching cldigits."""
    return QuditQuantumCircuit(
        QuditRegister.from_dims(MIXED_DIMS, "mix"),
        ClDigitRegister.from_dims(MIXED_DIMS, "out"),
    )


def circuit_with_loose_wires() -> QuditQuantumCircuit:
    """Return a circuit mixing registers and loose objects."""
    circuit = QuditQuantumCircuit(
        QuditRegister(2, 3, "reg"),
        ClDigitRegister(1, 3, "creg"),
    )
    circuit.add_qudits([Qudit(3)])
    circuit.add_cldigits([ClDigit(3)])
    return circuit


def two_digit_circuit() -> QuditQuantumCircuit:
    """Return a circuit whose only wires are two qutrit cldigits."""
    return QuditQuantumCircuit(ClDigitRegister.from_dims([3, 3], "out"))


class TestCopyEmptyLike:
    """``copy_empty_like`` keeps the wires and drops the data."""

    def test_wires_and_registers_are_shared(self) -> None:
        """Bit-level objects are shared, not rebuilt."""
        original = circuit_with_loose_wires()
        original.x(0)

        empty = original.copy_empty_like()

        assert empty.qudits == original.qudits
        assert empty.cldigits == original.cldigits
        assert empty.qdregs == original.qdregs
        assert empty.cbregs == original.cbregs
        assert empty.qubits == original.qubits
        assert empty.clbits == original.clbits
        assert empty.name == original.name

    def test_no_instruction_is_carried_over(self) -> None:
        """Both views of the copy start out empty."""
        original = circuit_with_loose_wires()
        original.x(0)

        empty = original.copy_empty_like()

        assert len(empty.data) == 0
        assert len(empty.circuit.data) == 0
        assert len(original.data) == 1

    def test_phase_metadata_and_name_are_carried_over(self) -> None:
        """The phase is copied and the metadata duplicated."""
        original = QuditQuantumCircuit(
            1,
            dim=3,
            global_phase=0.25,
            metadata={"run": 1},
        )

        empty = original.copy_empty_like("blank")

        assert empty.name == "blank"
        assert empty.global_phase == 0.25
        assert empty.metadata == {"run": 1}
        assert empty.metadata is not original.metadata


class TestCopy:
    """``copy`` replays the whole instruction log."""

    def test_copy_replays_every_instruction(self) -> None:
        """Operations are shared; the two logs match."""
        original = QuditQuantumCircuit(2, 2, dim=3)
        original.h(0)
        original.measure(1, 1)

        clone = original.copy()

        assert [entry.name for entry in clone.data] == ["H", "measure"]
        assert all(
            copied.operation is source.operation
            for copied, source in zip(clone.data, original.data, strict=True)
        )
        assert len(clone.circuit.data) == len(original.circuit.data)
        assert clone.qudits == original.qudits

    def test_the_copy_has_an_independent_log(self) -> None:
        """Appending to the copy leaves the original alone."""
        original = QuditQuantumCircuit(1, dim=3)
        original.h(0)

        clone = original.copy("clone")
        clone.x(0)

        assert clone.name == "clone"
        assert len(clone.data) == 2
        assert len(original.data) == 1
        assert len(original.circuit.data) == 1


class TestCompose:
    """``compose`` inlines another qudit circuit."""

    def test_default_mapping_uses_the_first_qudits(self) -> None:
        """``other`` lands on qudits ``0..n-1``, in order."""
        base = QuditQuantumCircuit(3, dim=3)
        other = QuditQuantumCircuit(2, dim=3)
        other.x(0)
        other.h(1)

        composed = base.compose(other)

        assert composed is not None
        assert [entry.name for entry in composed.data] == ["X", "H"]
        assert [entry.qudits[0] for entry in composed.data] == [
            base.qudits[0],
            base.qudits[1],
        ]
        assert len(base.data) == 0

    def test_an_explicit_mapping_is_honoured(self) -> None:
        """``qudits``/``cldigits`` choose the destination wires."""
        base = QuditQuantumCircuit(3, 3, dim=3)
        other = QuditQuantumCircuit(1, 1, dim=3)
        other.measure(0, 0)

        composed = base.compose(other, qudits=[2], cldigits=[1])

        assert composed is not None
        assert composed.data[0].qudits == (base.qudits[2],)
        assert composed.data[0].cldigits == (base.cldigits[1],)

    def test_inplace_composition_returns_none(self) -> None:
        """``inplace=True`` mutates the receiver."""
        base = QuditQuantumCircuit(1, dim=3)
        other = QuditQuantumCircuit(1, dim=3)
        other.x(0)

        assert base.compose(other, inplace=True) is None
        assert [entry.name for entry in base.data] == ["X"]
        assert len(base.circuit.data) == 1

    def test_front_runs_other_first(self) -> None:
        """``front=True`` rebuilds with ``other`` at the front."""
        base = QuditQuantumCircuit(1, dim=3)
        base.x(0)
        other = QuditQuantumCircuit(1, dim=3)
        other.h(0)

        composed = base.compose(other, front=True)

        assert composed is not None
        assert [entry.name for entry in composed.data] == ["H", "X"]
        assert [entry.operation.name for entry in composed.circuit.data] == [
            "H",
            "X",
        ]
        assert [entry.name for entry in base.data] == ["X"]

    def test_front_composition_can_be_inplace(self) -> None:
        """The rebuilt circuit is adopted by the receiver."""
        base = QuditQuantumCircuit(1, dim=3)
        base.x(0)
        other = QuditQuantumCircuit(1, dim=3)
        other.h(0)

        assert base.compose(other, front=True, inplace=True) is None
        assert [entry.name for entry in base.data] == ["H", "X"]

    @pytest.mark.parametrize("front", [False, True], ids=["back", "front"])
    def test_global_phases_add_up(self, front: bool) -> None:
        """Composition accumulates the two global phases."""
        base = QuditQuantumCircuit(1, dim=3, global_phase=0.25)
        other = QuditQuantumCircuit(1, dim=3, global_phase=0.5)

        composed = base.compose(other, front=front)

        assert composed is not None
        assert composed.global_phase == pytest.approx(0.75)

    def test_a_qudit_count_mismatch_is_rejected(self) -> None:
        """``other`` cannot be wider than the destination."""
        base = QuditQuantumCircuit(1, dim=3)
        other = QuditQuantumCircuit(2, dim=3)

        with pytest.raises(QuditCircuitError, match="onto 1 target"):
            base.compose(other)

    def test_a_cldigit_count_mismatch_is_rejected(self) -> None:
        """Cldigits are mapped one-to-one as well."""
        base = QuditQuantumCircuit(1, dim=3)
        other = QuditQuantumCircuit(1, 1, dim=3)

        with pytest.raises(
            QuditCircuitError,
            match=r"1 cldigit.* onto 0 target",
        ):
            base.compose(other)

    def test_a_dimension_mismatch_is_rejected(self) -> None:
        """Mapped qudits must agree pairwise on their dimension."""
        base = QuditQuantumCircuit(1, dim=4)
        other = QuditQuantumCircuit(1, dim=3)

        with pytest.raises(QuditCircuitError, match="dimension mismatch"):
            base.compose(other)


class TestInverse:
    """``inverse`` builds the adjoint circuit."""

    def test_inverse_reverses_and_inverts_every_gate(self) -> None:
        """Order is reversed and each gate replaced by its adjoint."""
        circuit = QuditQuantumCircuit(1, dim=3)
        circuit.h(0)
        circuit.p(0.5, 0)
        circuit.x(0)

        inverted = circuit.inverse()

        assert [entry.name for entry in inverted.data] == ["Xdg", "P", "Hdg"]
        assert inverted.data[1].operation.params == [-0.5]
        assert inverted.name == f"{circuit.name}_dg"
        assert [entry.name for entry in circuit.data] == ["H", "P", "X"]

    def test_inverse_negates_the_global_phase(self) -> None:
        """The phase is negated (and re-normalised by Qiskit)."""
        circuit = QuditQuantumCircuit(1, dim=3, global_phase=0.5)

        inverted = circuit.inverse()

        assert (
            inverted.global_phase
            == QuantumCircuit(
                global_phase=-0.5,
            ).global_phase
        )

    @pytest.mark.parametrize(
        "prepare",
        [
            lambda qc: qc.measure(0, 0),
            lambda qc: qc.reset(0),
            lambda qc: qc.initialize_levels(1, 0),
            lambda qc: qc.barrier(),
        ],
        ids=["measure", "reset", "initialize", "barrier"],
    )
    def test_inverse_rejects_non_unitary_operations(
        self,
        prepare: Callable[[QuditQuantumCircuit], object],
    ) -> None:
        """Only :class:`~qiskit.circuit.Gate` instances can be undone."""
        circuit = QuditQuantumCircuit(1, 1, dim=3)
        prepare(circuit)

        # NOTE: unlike Qiskit's `QuantumCircuit.inverse`, a barrier is
        # refused too, because directives are not `Gate` instances.
        with pytest.raises(QuditCircuitError, match="cannot invert"):
            circuit.inverse()


class TestMetrics:
    """``size``, ``depth``, ``width`` and ``count_ops``."""

    def test_size_excludes_directives_by_default(self) -> None:
        """Barriers do not count, measurements do."""
        circuit = QuditQuantumCircuit(2, 2, dim=3)
        circuit.x(0)
        circuit.barrier()
        circuit.measure(0, 0)

        assert circuit.size() == 2

    def test_size_accepts_a_custom_filter(self) -> None:
        """Any predicate over the instructions can be used."""
        circuit = QuditQuantumCircuit(2, 2, dim=3)
        circuit.x(0)
        circuit.barrier()
        circuit.measure(0, 0)

        assert circuit.size(lambda _: True) == 3
        assert circuit.size(lambda entry: entry.name == "X") == 1

    def test_depth_follows_the_qudit_wires(self) -> None:
        """Independent qudits do not stack up."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.x(0)
        circuit.x(0)
        circuit.x(1)

        assert circuit.depth() == 2

    def test_depth_follows_the_cldigit_wires(self) -> None:
        """Two qudits sharing a cldigit are serialised by it."""
        circuit = QuditQuantumCircuit(2, 1, dim=3)
        circuit.measure(0, 0)
        circuit.measure(1, 0)

        assert circuit.depth() == 2

    def test_a_barrier_synchronises_without_adding_depth(self) -> None:
        """The barrier lifts wire 1 to the level of wire 0."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.x(0)
        circuit.barrier()
        circuit.x(1)

        assert circuit.depth() == 2

    @pytest.mark.parametrize(
        "build",
        [QuditQuantumCircuit, lambda: QuditQuantumCircuit(2, 2, dim=3)],
        ids=["no-wires", "no-instructions"],
    )
    def test_depth_of_an_empty_circuit_is_zero(
        self,
        build: Callable[[], QuditQuantumCircuit],
    ) -> None:
        """Without instructions the critical path is empty."""
        assert build().depth() == 0

    def test_width_counts_qudits_and_cldigits(self) -> None:
        """Width is measured in qudit wires, not qubit wires."""
        assert QuditQuantumCircuit(2, 3, dim=3).width() == 5
        assert mixed_circuit().width() == 6

    def test_count_ops_is_ordered_by_frequency_then_name(self) -> None:
        """Ties are broken alphabetically on the operation name."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.x(0)
        circuit.x(1)
        circuit.h(0)
        circuit.barrier()

        counts = circuit.count_ops()

        assert isinstance(counts, OrderedDict)
        assert list(counts.items()) == [("X", 2), ("H", 1), ("barrier", 1)]


class TestQubitView:
    """``to_qubit_circuit``."""

    def test_the_default_is_an_independent_copy(self) -> None:
        """A copy protects the live encoded circuit."""
        circuit = QuditQuantumCircuit(1, dim=3)
        circuit.x(0)

        copied = circuit.to_qubit_circuit()

        assert isinstance(copied, QuantumCircuit)
        assert copied is not circuit.circuit
        assert len(copied.data) == len(circuit.circuit.data)

    def test_copy_false_returns_the_live_circuit(self) -> None:
        """``copy=False`` hands out the object behind ``circuit``."""
        circuit = QuditQuantumCircuit(1, dim=3)

        assert circuit.to_qubit_circuit(copy=False) is circuit.circuit


class TestIdealView:
    """``to_ideal_circuit`` renders one wire per qudit."""

    def test_one_wire_per_qudit_and_cldigit(self) -> None:
        """Registers keep their qudit-level names and sizes."""
        circuit = mixed_circuit()

        ideal = circuit.to_ideal_circuit()

        assert ideal.num_qubits == 3
        assert ideal.num_clbits == 3
        assert [register.name for register in ideal.qregs] == ["mix"]
        assert [register.name for register in ideal.cregs] == ["out"]
        assert ideal.name == circuit.name

    def test_loose_objects_become_loose_wires(self) -> None:
        """A registerless qudit gets a registerless qubit."""
        circuit = QuditQuantumCircuit()
        circuit.add_qudits([Qudit(3)])
        circuit.add_cldigits([ClDigit(3)])

        ideal = circuit.to_ideal_circuit()

        assert ideal.qregs == []
        assert ideal.cregs == []
        assert ideal.num_qubits == 1
        assert ideal.num_clbits == 1

    def test_directives_become_native_qiskit_operations(self) -> None:
        """Barrier, reset and measure render as themselves."""
        circuit = QuditQuantumCircuit(2, 2, dim=3)
        circuit.barrier()
        circuit.reset(0)
        circuit.measure(0, 0)

        ideal = circuit.to_ideal_circuit()

        assert isinstance(ideal.data[0].operation, Barrier)
        assert isinstance(ideal.data[1].operation, Reset)
        assert isinstance(ideal.data[2].operation, Measure)
        assert len(ideal.data[0].qubits) == 2
        assert ideal.data[1].qubits == (ideal.qubits[0],)
        assert ideal.data[2].clbits == (ideal.clbits[0],)

    def test_gate_labels_carry_the_params_and_dimensions(self) -> None:
        """Everything an opaque box cannot show goes in the label."""
        circuit = QuditQuantumCircuit(1, dim=3)
        circuit.x(0)
        circuit.p(0.5, 0)

        ideal = circuit.to_ideal_circuit()

        assert ideal.data[0].operation.name == "X"
        assert ideal.data[0].operation.num_qubits == 1
        assert ideal.data[0].operation.label == "X(d=3)"
        assert ideal.data[1].operation.label == "P(0.5, d=3)"

    def test_annotation_can_be_switched_off(self) -> None:
        """``annotate_levels=False`` drops the ``d=`` suffix."""
        circuit = QuditQuantumCircuit(1, dim=3)
        circuit.x(0)

        ideal = circuit.to_ideal_circuit(annotate_levels=False)

        assert ideal.data[0].operation.label == "X"


class TestDrawing:
    """``draw``, ``decompose`` and the dunder renderings."""

    @pytest.mark.parametrize("view", ["ideal", "real", "decomposed"])
    def test_every_view_can_be_drawn_as_text(self, view: str) -> None:
        """All three renderings produce a text drawing."""
        circuit = QuditQuantumCircuit(1, 1, dim=2)
        circuit.x(0)
        circuit.measure(0, 0)

        drawing = circuit.draw(output="text", view=view)

        assert drawing is not None
        assert str(drawing)

    def test_an_unknown_view_is_rejected(self) -> None:
        """Only the three documented views exist."""
        circuit = QuditQuantumCircuit(1, dim=2)

        with pytest.raises(QuditCircuitError, match="unknown view"):
            circuit.draw(output="text", view="fancy")

    def test_decompose_unrolls_the_encoded_circuit(self) -> None:
        """The qudit gate is replaced by its qubit definition."""
        circuit = QuditQuantumCircuit(1, dim=2)
        circuit.x(0)

        decomposed = circuit.decompose()

        assert isinstance(decomposed, QuantumCircuit)
        assert [entry.operation.name for entry in decomposed.data] == ["x"]

    def test_decompose_accepts_several_repetitions(self) -> None:
        """``reps`` is forwarded to Qiskit."""
        circuit = QuditQuantumCircuit(1, dim=2)
        circuit.x(0)

        assert isinstance(circuit.decompose(reps=2), QuantumCircuit)

    def test_str_renders_the_ideal_view(self) -> None:
        """``str`` is the text drawing of the ideal circuit."""
        circuit = QuditQuantumCircuit(1, dim=3)
        circuit.x(0)

        text = str(circuit)

        assert "qd" in text
        assert "X" in text

    def test_repr_summarises_the_circuit(self) -> None:
        """``repr`` is a one-line, unambiguous summary."""
        circuit = QuditQuantumCircuit(2, 1, dim=3)

        assert repr(circuit) == (
            "<QuditQuantumCircuit 'quditcircuit-0': 2 qudit(s) "
            "dims=(3, 3), 1 cldigit(s), 0 instruction(s), 4 qubit(s)>"
        )


class TestEquality:
    """``__eq__`` and the (absent) ``__hash__``."""

    def test_identically_built_circuits_are_equal(self) -> None:
        """Equality compares the dimensions and the encoding."""
        first = QuditQuantumCircuit(2, dim=2)
        first.x(0)
        second = QuditQuantumCircuit(2, dim=2)
        second.x(0)

        assert first == second

    def test_a_different_operation_breaks_equality(self) -> None:
        """The encoded circuits must match too."""
        first = QuditQuantumCircuit(2, dim=2)
        first.x(0)
        second = QuditQuantumCircuit(2, dim=2)
        second.h(0)

        assert first != second

    def test_different_dimensions_are_never_equal(self) -> None:
        """The qudit dimensions are part of the identity."""
        assert QuditQuantumCircuit(1, dim=2) != QuditQuantumCircuit(1, dim=3)

    def test_a_foreign_object_is_not_equal(self) -> None:
        """``NotImplemented`` makes Python fall back to identity."""
        circuit = QuditQuantumCircuit(1, dim=2)

        assert circuit != object()
        assert circuit.__eq__(object()) is NotImplemented

    def test_the_class_is_unhashable(self) -> None:
        """Mutable circuits cannot be dictionary keys."""
        assert QuditQuantumCircuit.__hash__ is None

        with pytest.raises(TypeError, match="unhashable"):
            hash(QuditQuantumCircuit())


class TestDecoding:
    """``decode_bitstring`` and ``decode_counts``."""

    def test_decode_bitstring_uses_the_cldigit_layout(self) -> None:
        """The rightmost characters belong to cldigit ``0``."""
        circuit = two_digit_circuit()

        assert circuit.cldigit_widths == (2, 2)
        assert circuit.decode_bitstring("01 10") == (2, 1)

    def test_decode_counts_aggregates_by_levels(self) -> None:
        """Every key of the mapping is decoded and summed."""
        circuit = two_digit_circuit()

        decoded = circuit.decode_counts({"01 10": 5, "00 10": 3})

        assert decoded == {(2, 1): 5, (2, 0): 3}

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [
            ("keep", {(3,): 4, (2,): 6}),
            ("drop", {(2,): 6}),
        ],
        ids=["keep", "drop"],
    )
    def test_the_invalid_policy_is_passed_through(
        self,
        policy: InvalidPolicy,
        expected: dict[Levels, int],
    ) -> None:
        """Level 3 leaks out of a three-level qudit."""
        circuit = QuditQuantumCircuit(ClDigitRegister(1, 3, "out"))

        decoded = circuit.decode_counts({"11": 4, "10": 6}, on_invalid=policy)

        assert decoded == expected

    def test_leakage_can_be_turned_into_an_error(self) -> None:
        """``on_invalid='raise'`` reports the offending digit."""
        circuit = QuditQuantumCircuit(ClDigitRegister(1, 3, "out"))

        with pytest.raises(ValueError, match="outside the 3-level"):
            circuit.decode_counts({"11": 4}, on_invalid="raise")

    def test_a_dropped_shot_decodes_to_none(self) -> None:
        """A single leaked bit-string can be discarded."""
        circuit = QuditQuantumCircuit(ClDigitRegister(1, 3, "out"))

        assert circuit.decode_bitstring("11", on_invalid="drop") is None
