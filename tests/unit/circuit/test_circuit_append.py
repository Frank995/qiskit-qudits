"""Operand resolution and validation of ``QuditQuantumCircuit``.

Covers ``_qudit_argument_conversion`` / ``_clbyte_argument_conversion``
(directly and through ``append``), the duplicate-operand guard, every
branch of ``_validate_operands`` and ``_operation_dims``, the ``copy``
flag and the atomicity of a failed ``append``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.circuit import Instruction, Measure
from qiskit.circuit.library import CXGate, XGate

from qiskit_qudits.circuit.clbyte import ClByte, ClByteRegister
from qiskit_qudits.circuit.directives import QuditBarrier, QuditMeasure
from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.circuit.quantumcircuit import QuditQuantumCircuit
from qiskit_qudits.circuit.qudit import Qudit, QuditRegister
from qiskit_qudits.gates import QuditPGate, QuditXGate

if TYPE_CHECKING:
    from collections.abc import Callable


# ------------------------------------------------------------------ #
# Factories
# ------------------------------------------------------------------ #
@pytest.fixture
def circuit() -> QuditQuantumCircuit:
    """Return three qutrits with three matching clbytes."""
    return QuditQuantumCircuit(3, 3, dim=3)


def tagged_operation(num_qubits: int, **attributes: object) -> Instruction:
    """Return a bare instruction carrying qudit metadata.

    Args:
        num_qubits: Width of the instruction.
        attributes: Attributes to set on it, e.g. ``dims`` or ``dim``.

    Returns:
        The annotated instruction.
    """
    operation = Instruction("dummy", num_qubits, 0, [])
    for name, value in attributes.items():
        setattr(operation, name, value)
    return operation


class TestQuditArgumentConversion:
    """``_qudit_argument_conversion`` for every specifier form."""

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda qc: qc.qudits[1], [1]),
            (lambda qc: qc.qdregs[0], [0, 1, 2]),
            (lambda _: 2, [2]),
            (lambda _: -1, [2]),
            (lambda _: np.int64(1), [1]),
            (lambda _: np.array([0, 1]), [0, 1]),
            (lambda _: slice(0, 2), [0, 1]),
            (lambda _: slice(None, None, 2), [0, 2]),
            (lambda qc: [qc.qudits[2], 0], [2, 0]),
            (lambda _: (), []),
        ],
        ids=[
            "qudit",
            "register",
            "index",
            "negative-index",
            "numpy-index",
            "numpy-array",
            "slice",
            "strided-slice",
            "mixed-sequence",
            "empty-sequence",
        ],
    )
    def test_specifier_resolves_to_qudits(
        self,
        circuit: QuditQuantumCircuit,
        build: Callable[[QuditQuantumCircuit], object],
        expected: list[int],
    ) -> None:
        """Every accepted specifier resolves in the implied order."""
        resolved = circuit._qudit_argument_conversion(build(circuit))

        assert resolved == [circuit.qudits[index] for index in expected]

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (lambda _: "0", "strings are not valid qudit"),
            (lambda _: 3, "out of range"),
            (lambda _: -4, "out of range"),
            (lambda _: Qudit(3), "not in this circuit"),
            (
                lambda _: QuditRegister(1, 3, "foreign"),
                "'foreign' is not in this circuit",
            ),
            (lambda _: 1.5, "invalid qudit specifier"),
            (lambda _: [1.5], "invalid qudit specifier element"),
            (lambda qc: [qc.clbytes[0]], "invalid qudit specifier element"),
        ],
        ids=[
            "string",
            "index-too-large",
            "index-too-negative",
            "foreign-qudit",
            "foreign-register",
            "float",
            "float-element",
            "clbyte-element",
        ],
    )
    def test_invalid_specifier_raises(
        self,
        circuit: QuditQuantumCircuit,
        build: Callable[[QuditQuantumCircuit], object],
        message: str,
    ) -> None:
        """Unsupported specifiers raise a ``QuditCircuitError``."""
        with pytest.raises(QuditCircuitError, match=message):
            circuit._qudit_argument_conversion(build(circuit))

    def test_append_uses_the_conversion(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """``append`` resolves its ``qudits`` argument the same way."""
        instruction = circuit.append(QuditXGate(3), -1)

        assert instruction.qudits == (circuit.qudits[2],)

    def test_append_surfaces_conversion_errors(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """A bad specifier fails the whole ``append``."""
        with pytest.raises(QuditCircuitError, match="strings are not valid"):
            circuit.append(QuditXGate(3), "0")


class TestClbyteArgumentConversion:
    """``_clbyte_argument_conversion`` for every specifier form."""

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda qc: qc.clbytes[1], [1]),
            (lambda qc: qc.cbregs[0], [0, 1, 2]),
            (lambda _: 2, [2]),
            (lambda _: -1, [2]),
            (lambda _: np.int64(1), [1]),
            (lambda _: slice(0, 2), [0, 1]),
            (lambda qc: [qc.clbytes[2], 0], [2, 0]),
            (lambda _: (), []),
        ],
        ids=[
            "clbyte",
            "register",
            "index",
            "negative-index",
            "numpy-index",
            "slice",
            "mixed-sequence",
            "empty-sequence",
        ],
    )
    def test_specifier_resolves_to_clbytes(
        self,
        circuit: QuditQuantumCircuit,
        build: Callable[[QuditQuantumCircuit], object],
        expected: list[int],
    ) -> None:
        """Every accepted specifier resolves in the implied order."""
        resolved = circuit._clbyte_argument_conversion(build(circuit))

        assert resolved == [circuit.clbytes[index] for index in expected]

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (lambda _: "0", "strings are not valid clbyte"),
            (lambda _: 3, "out of range"),
            (lambda _: -4, "out of range"),
            (lambda _: ClByte(3), "not in this circuit"),
            (
                lambda _: ClByteRegister(1, 3, "foreign"),
                "'foreign' is not in this circuit",
            ),
            (lambda _: 1.5, "invalid clbyte specifier"),
            (lambda _: [1.5], "invalid clbyte specifier element"),
            (lambda qc: [qc.qudits[0]], "invalid clbyte specifier element"),
        ],
        ids=[
            "string",
            "index-too-large",
            "index-too-negative",
            "foreign-clbyte",
            "foreign-register",
            "float",
            "float-element",
            "qudit-element",
        ],
    )
    def test_invalid_specifier_raises(
        self,
        circuit: QuditQuantumCircuit,
        build: Callable[[QuditQuantumCircuit], object],
        message: str,
    ) -> None:
        """Unsupported specifiers raise a ``QuditCircuitError``."""
        with pytest.raises(QuditCircuitError, match=message):
            circuit._clbyte_argument_conversion(build(circuit))

    def test_append_uses_the_conversion(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """``append`` resolves its ``clbytes`` argument the same way."""
        instruction = circuit.append(QuditMeasure([3]), 0, -2)

        assert instruction.clbytes == (circuit.clbytes[1],)


class TestDuplicateOperands:
    """The ``_check_duplicates`` guard."""

    def test_duplicate_qudits_are_rejected(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """The same qudit cannot appear twice in one operation."""
        with pytest.raises(QuditCircuitError, match="duplicate qudit"):
            circuit.append(QuditBarrier([3, 3]), [0, 0])

    def test_duplicate_clbytes_are_rejected(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """The same clbyte cannot appear twice in one operation."""
        with pytest.raises(QuditCircuitError, match="duplicate clbyte"):
            circuit.append(QuditMeasure([3, 3]), [0, 1], [2, 2])


class TestAppendValidation:
    """``append`` refuses operations that do not fit their operands."""

    def test_a_non_instruction_is_rejected(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """Only :class:`~qiskit.circuit.Instruction` can be appended."""
        with pytest.raises(QuditCircuitError, match="expected an Instruction"):
            circuit.append("X", 0)

    def test_qudit_count_mismatch_is_rejected(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """A two-qudit directive needs two target qudits."""
        with pytest.raises(QuditCircuitError, match="acts on 2 qudit"):
            circuit.append(QuditBarrier([3, 3]), 0)

    def test_per_position_dimension_mismatch_is_rejected(self) -> None:
        """Dimensions are checked operand by operand."""
        mixed = QuditQuantumCircuit(QuditRegister.from_dims([3, 4], "mix"))

        with pytest.raises(
            QuditCircuitError,
            match="expects a 3-level qudit at position 0",
        ):
            mixed.append(QuditXGate(3), 1)

    def test_encoded_width_mismatch_is_rejected(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """A one-qubit gate does not fit a two-qubit qutrit."""
        with pytest.raises(
            QuditCircuitError,
            match=r"acts on 1 qubit.* but the target qudits encode into 2",
        ):
            circuit.append(XGate(), 0)

    def test_directive_clbyte_count_mismatch_is_rejected(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """A measure directive needs exactly one clbyte per qudit."""
        with pytest.raises(QuditCircuitError, match="acts on 1 clbyte"):
            circuit.append(QuditMeasure([3]), 0)

    def test_plain_clbit_count_mismatch_is_rejected(self) -> None:
        """A non-directive is checked against the clbyte widths."""
        narrow = QuditQuantumCircuit(
            QuditRegister(1, 2, "q"),
            ClByteRegister(1, 3, "c"),
        )

        with pytest.raises(
            QuditCircuitError,
            match=r"acts on 1 clbit.* but the target clbytes provide 2",
        ):
            narrow.append(Measure(), 0, 0)


class TestRawQiskitOperations:
    """Raw Qiskit gates over a qudit's encoding qubits."""

    def test_single_qubit_gate_on_a_two_level_qudit(self) -> None:
        """A qubit gate fits a ``d=2`` qudit exactly."""
        qudit_circuit = QuditQuantumCircuit(1, dim=2)

        instruction = qudit_circuit.append(XGate(), 0)

        assert instruction.operation.name == "x"
        assert instruction.dims == (2,)
        assert list(qudit_circuit.circuit.data[0].qubits) == list(
            qudit_circuit.qudits[0].qubits,
        )

    def test_two_qubit_gate_inside_one_ququart(self) -> None:
        """A two-qubit gate spans the encoding of a ``d=4`` qudit."""
        qudit_circuit = QuditQuantumCircuit(1, dim=4)

        qudit_circuit.append(CXGate(), 0)

        assert qudit_circuit.circuit.data[0].operation.name == "cx"
        assert list(qudit_circuit.circuit.data[0].qubits) == list(
            qudit_circuit.qudits[0].qubits,
        )

    def test_two_qubit_gate_across_two_qudits(self) -> None:
        """Operand qubits are concatenated in target order."""
        qudit_circuit = QuditQuantumCircuit(2, dim=2)

        qudit_circuit.append(CXGate(), [1, 0])

        assert list(qudit_circuit.circuit.data[0].qubits) == [
            qudit_circuit.qudits[1].qubits[0],
            qudit_circuit.qudits[0].qubits[0],
        ]


class TestCopyFlag:
    """The ``copy`` keyword of ``append``."""

    def test_copy_true_isolates_later_mutation(self) -> None:
        """The default copies an operation that carries params."""
        qudit_circuit = QuditQuantumCircuit(1, dim=3)
        gate = QuditPGate(3, 0.5)

        instruction = qudit_circuit.append(gate, 0)
        gate.params = [1.25]

        assert instruction.operation is not gate
        assert instruction.operation.params == [0.5]
        assert qudit_circuit.data[0].operation.params == [0.5]

    def test_copy_false_stores_the_same_object(self) -> None:
        """``copy=False`` records the argument itself."""
        qudit_circuit = QuditQuantumCircuit(1, dim=3)
        gate = QuditPGate(3, 0.5)

        instruction = qudit_circuit.append(gate, 0, copy=False)

        assert instruction.operation is gate

    def test_a_parameterless_operation_is_never_copied(self) -> None:
        """Without params there is nothing that could leak."""
        qudit_circuit = QuditQuantumCircuit(1, dim=3)
        gate = QuditXGate(3)

        instruction = qudit_circuit.append(gate, 0)

        assert instruction.operation is gate


class TestOperationDims:
    """``_operation_dims`` metadata extraction."""

    def test_a_directive_reports_its_own_dims(self) -> None:
        """Directives always know their target dimensions."""
        dims = QuditQuantumCircuit._operation_dims(QuditBarrier([3, 4]))

        assert dims == (3, 4)

    def test_a_dims_tuple_wins_over_a_dim_attribute(self) -> None:
        """Multi-qudit metadata takes precedence."""
        operation = tagged_operation(3, dims=(2, 3), dim=5)

        assert QuditQuantumCircuit._operation_dims(operation) == (2, 3)

    def test_a_lone_dim_reports_one_dimension(self) -> None:
        """A single-qudit gate exposes ``dim`` only."""
        operation = tagged_operation(2, dim=3)

        assert QuditQuantumCircuit._operation_dims(operation) == (3,)

    @pytest.mark.parametrize(
        "attributes",
        [{}, {"dims": ()}, {"dims": (2, "x")}],
        ids=["nothing", "empty-dims", "non-integer-dims"],
    )
    def test_missing_metadata_returns_none(
        self,
        attributes: dict[str, object],
    ) -> None:
        """Unusable metadata disables the dimension check."""
        operation = tagged_operation(1, **attributes)

        assert QuditQuantumCircuit._operation_dims(operation) is None

    def test_a_raw_qiskit_gate_has_no_dimensions(self) -> None:
        """Plain Qiskit gates carry no qudit metadata."""
        assert QuditQuantumCircuit._operation_dims(XGate()) is None


class TestAtomicity:
    """A failing ``append`` must not leave a half-built circuit."""

    def test_a_validation_failure_changes_nothing(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """Both views keep their length when validation fails."""
        circuit.append(QuditXGate(3), 0)
        before = (len(circuit.data), len(circuit.circuit.data))

        with pytest.raises(QuditCircuitError, match="encode into"):
            circuit.append(XGate(), 1)

        assert (len(circuit.data), len(circuit.circuit.data)) == before

    def test_a_failure_during_expansion_changes_nothing(self) -> None:
        """A directive that refuses to expand records nothing."""
        narrow = QuditQuantumCircuit(
            QuditRegister(1, 4, "q"),
            ClByteRegister(1, 2, "c"),
        )

        with pytest.raises(QuditCircuitError, match="cannot measure"):
            narrow.append(QuditMeasure([4]), 0, 0)

        assert len(narrow.data) == 0
        assert len(narrow.circuit.data) == 0


class TestBookkeeping:
    """What ``append`` records in the two parallel views."""

    def test_the_returned_instruction_records_the_operands(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """The resolved operands are stored on the instruction."""
        gate = QuditXGate(3)

        instruction = circuit.append(gate, 1, copy=False)

        assert instruction.operation is gate
        assert instruction.qudits == (circuit.qudits[1],)
        assert instruction.clbytes == ()
        assert circuit.data[-1] is instruction

    def test_a_zero_operand_directive_touches_only_the_log(self) -> None:
        """An empty barrier is logged but emits no encoded gate."""
        empty = QuditQuantumCircuit()

        empty.append(QuditBarrier([]), ())

        assert len(empty.data) == 1
        assert len(empty.circuit.data) == 0

    def test_data_grows_in_application_order(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """``data`` is the qudit-level log, newest last."""
        circuit.append(QuditXGate(3), 0)
        circuit.append(QuditMeasure([3]), 1, 1)

        assert [entry.name for entry in circuit.data] == ["X", "measure"]

    def test_the_encoded_circuit_receives_expanded_operations(
        self,
        circuit: QuditQuantumCircuit,
    ) -> None:
        """One qudit measure becomes one measure per encoding qubit."""
        circuit.append(QuditXGate(3), 1)
        circuit.append(QuditMeasure([3]), 0, 2)

        encoded = circuit.circuit.data
        assert [entry.operation.name for entry in encoded] == [
            "X",
            "measure",
            "measure",
        ]
        assert list(encoded[0].qubits) == list(circuit.qudits[1].qubits)
        assert list(encoded[1].qubits) == [circuit.qudits[0].qubits[0]]
        assert list(encoded[1].clbits) == [circuit.clbytes[2].clbits[0]]
        assert list(encoded[2].qubits) == [circuit.qudits[0].qubits[1]]
        assert list(encoded[2].clbits) == [circuit.clbytes[2].clbits[1]]
