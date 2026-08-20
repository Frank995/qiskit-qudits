"""Gate helpers and directives of ``QuditQuantumCircuit``.

Covers every single-qudit gate helper (including broadcasting, labels
and per-qudit dimensions), the controlled and multi-qudit helpers
(``sumx`` / ``sumxdg`` / ``sump`` / ``swap`` / ``qft``), the
``barrier`` / ``reset`` / ``measure`` / ``measure_all`` directives and
the two initialisation entry points.

Only bookkeeping is asserted here: what an operation *does* to a state
is the business of the ``tests/quantum`` suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from qiskit_qudits.circuit.clbyte import ClByteRegister
from qiskit_qudits.circuit.directives import (
    QuditBarrier,
    QuditInitializeLevels,
    QuditMeasure,
    QuditReset,
    QuditStatePreparation,
)
from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.circuit.instruction import QuditCircuitInstruction
from qiskit_qudits.circuit.quantumcircuit import QuditQuantumCircuit
from qiskit_qudits.circuit.qudit import QuditRegister
from qiskit_qudits.gates import (
    QuditHdgGate,
    QuditHGate,
    QuditIGate,
    QuditKGate,
    QuditNOTGate,
    QuditPGate,
    QuditQFTGate,
    QuditSdgGate,
    QuditSGate,
    QuditSUMPGate,
    QuditSUMXdgGate,
    QuditSUMXGate,
    QuditSWAPGate,
    QuditTdgGate,
    QuditTGate,
    QuditXdgGate,
    QuditXGate,
    QuditZdgGate,
    QuditZGate,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# ------------------------------------------------------------------ #
# Factories
# ------------------------------------------------------------------ #
#: Dimensions of the heterogeneous circuit used by several tests.
MIXED_DIMS = (2, 3, 5)

#: Encoding width of each dimension in :data:`MIXED_DIMS`.
MIXED_WIDTHS = (1, 2, 3)

#: ``(helper name, gate class, operation name)`` of every
#: parameter-free single-qudit helper.
GATE_HELPERS = [
    ("i", QuditIGate, "I"),
    ("h", QuditHGate, "H"),
    ("hdg", QuditHdgGate, "Hdg"),
    ("k", QuditKGate, "K"),
    ("not_", QuditNOTGate, "NOT"),
    ("x", QuditXGate, "X"),
    ("xdg", QuditXdgGate, "Xdg"),
    ("z", QuditZGate, "Z"),
    ("zdg", QuditZdgGate, "Zdg"),
    ("s", QuditSGate, "S"),
    ("sdg", QuditSdgGate, "Sdg"),
    ("t", QuditTGate, "T"),
    ("tdg", QuditTdgGate, "Tdg"),
]

parametrize_helpers = pytest.mark.parametrize(
    ("helper", "gate_class", "gate_name"),
    GATE_HELPERS,
    ids=[helper for helper, _, _ in GATE_HELPERS],
)

#: ``(helper name, gate class, operation name)`` of the parameter-free
#: controlled helpers.
CONTROLLED_HELPERS = [
    ("sumx", QuditSUMXGate, "SUMX"),
    ("sumxdg", QuditSUMXdgGate, "SUMXdg"),
]

parametrize_controlled_helpers = pytest.mark.parametrize(
    ("helper", "gate_class", "gate_name"),
    CONTROLLED_HELPERS,
    ids=[helper for helper, _, _ in CONTROLLED_HELPERS],
)

parametrize_controlled_helper_names = pytest.mark.parametrize(
    "helper",
    [helper for helper, _, _ in CONTROLLED_HELPERS],
    ids=[helper for helper, _, _ in CONTROLLED_HELPERS],
)


def mixed_circuit() -> QuditQuantumCircuit:
    """Return a heterogeneous three-qudit circuit."""
    return QuditQuantumCircuit(QuditRegister.from_dims(MIXED_DIMS, "mix"))


class TestParameterFreeGateHelpers:
    """The thirteen single-qudit helpers sharing one code path."""

    @parametrize_helpers
    def test_helper_records_its_gate_in_both_views(
        self,
        helper: str,
        gate_class: type,
        gate_name: str,
    ) -> None:
        """The helper appends its own gate class to the log."""
        circuit = QuditQuantumCircuit(2, dim=3)

        result = getattr(circuit, helper)(0)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], QuditCircuitInstruction)
        assert isinstance(result[0].operation, gate_class)
        assert result[0].operation.dim == 3
        assert result[0].qudits == (circuit.qudits[0],)
        assert [entry.name for entry in circuit.data] == [gate_name]
        assert [entry.operation.name for entry in circuit.circuit.data] == [
            gate_name,
        ]

    @parametrize_helpers
    def test_helper_forwards_the_label(
        self,
        helper: str,
        gate_class: type,
        gate_name: str,
    ) -> None:
        """``label`` reaches every gate the helper creates."""
        circuit = QuditQuantumCircuit(2, dim=3)

        result = getattr(circuit, helper)([0, 1], label="tag")

        assert all(isinstance(entry.operation, gate_class) for entry in result)
        assert [entry.operation.label for entry in result] == ["tag", "tag"]
        assert circuit.circuit.data[0].operation.name == gate_name

    @parametrize_helpers
    def test_gate_is_built_with_each_qudits_own_dimension(
        self,
        helper: str,
        gate_class: type,
        gate_name: str,
    ) -> None:
        """Mixed-dimension circuits get one gate per dimension."""
        circuit = mixed_circuit()

        result = getattr(circuit, helper)(slice(None))

        assert [entry.operation.dim for entry in result] == list(MIXED_DIMS)
        assert [entry.operation.num_qubits for entry in result] == list(
            MIXED_WIDTHS,
        )
        assert all(isinstance(entry.operation, gate_class) for entry in result)
        assert {entry.operation.name for entry in circuit.circuit.data} == {
            gate_name,
        }

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda _: 1, [1]),
            (lambda _: [0, 2], [0, 2]),
            (lambda _: slice(1, 3), [1, 2]),
            (lambda qc: qc.qdregs[0], [0, 1, 2]),
        ],
        ids=["int", "list", "slice", "register"],
    )
    def test_helpers_broadcast_over_the_specifier(
        self,
        build: Callable[[QuditQuantumCircuit], object],
        expected: list[int],
    ) -> None:
        """One instruction is produced per resolved target."""
        circuit = QuditQuantumCircuit(3, dim=3)

        result = circuit.x(build(circuit))

        assert [entry.qudits[0] for entry in result] == [
            circuit.qudits[index] for index in expected
        ]
        assert len(circuit.data) == len(expected)
        assert len(circuit.circuit.data) == len(expected)

    def test_qnot_is_the_very_same_function_as_not_(self) -> None:
        """``qnot`` is an alias, not a wrapper."""
        assert QuditQuantumCircuit.qnot is QuditQuantumCircuit.not_

    def test_qnot_appends_the_not_gate(self) -> None:
        """Calling the alias behaves like calling ``not_``."""
        circuit = QuditQuantumCircuit(1, dim=3)

        result = circuit.qnot(0)

        assert isinstance(result[0].operation, QuditNOTGate)


class TestPhaseGateHelper:
    """The one-angle helper ``p``."""

    def test_p_takes_the_angle_first(self) -> None:
        """The signature mirrors ``QuantumCircuit.p``."""
        circuit = QuditQuantumCircuit(2, dim=3)

        result = circuit.p(0.5, 1)

        assert isinstance(result[0].operation, QuditPGate)
        assert result[0].operation.params == [0.5]
        assert result[0].qudits == (circuit.qudits[1],)
        assert circuit.circuit.data[0].operation.name == "P"

    def test_p_broadcasts_and_forwards_the_label(self) -> None:
        """``p`` broadcasts exactly like the other helpers."""
        circuit = QuditQuantumCircuit(3, dim=3)

        result = circuit.p(0.25, [0, 2], label="phase")

        assert [entry.qudits[0] for entry in result] == [
            circuit.qudits[0],
            circuit.qudits[2],
        ]
        assert [entry.operation.label for entry in result] == [
            "phase",
            "phase",
        ]

    def test_p_uses_each_qudits_own_dimension(self) -> None:
        """The angle is shared, the dimension is not."""
        circuit = mixed_circuit()

        result = circuit.p(0.25, slice(None))

        assert [entry.operation.dim for entry in result] == list(MIXED_DIMS)
        assert [entry.operation.params for entry in result] == [[0.25]] * 3

    @pytest.mark.parametrize(
        "theta",
        [float("inf"), float("nan")],
        ids=["inf", "nan"],
    )
    def test_p_rejects_a_non_finite_angle(self, theta: float) -> None:
        """A non-finite angle never reaches the circuit."""
        circuit = QuditQuantumCircuit(1, dim=3)

        # NOTE: the angle is validated by the gate, so this is a plain
        # `ValueError` rather than a `QuditCircuitError`.
        with pytest.raises(ValueError, match="must be finite"):
            circuit.p(theta, 0)

        assert len(circuit.data) == 0


class TestControlledGateHelpers:
    """The two parameter-free controlled helpers."""

    @parametrize_controlled_helpers
    def test_helper_records_its_gate_in_both_views(
        self,
        helper: str,
        gate_class: type,
        gate_name: str,
    ) -> None:
        """The helper appends its own gate class to the log."""
        circuit = QuditQuantumCircuit(2, dim=3)

        result = getattr(circuit, helper)(0, 1)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], QuditCircuitInstruction)
        assert isinstance(result[0].operation, gate_class)
        assert result[0].operation.target_dim == 3
        assert result[0].operation.control_dims == (3,)
        assert result[0].qudits == (circuit.qudits[0], circuit.qudits[1])
        assert [entry.name for entry in circuit.data] == [gate_name]
        assert [entry.operation.name for entry in circuit.circuit.data] == [
            gate_name,
        ]

    @parametrize_controlled_helper_names
    def test_encoded_operands_are_controls_then_target(
        self,
        helper: str,
    ) -> None:
        """Qubits reach the encoding in the gate's register order."""
        circuit = QuditQuantumCircuit(2, dim=3)

        getattr(circuit, helper)(0, 1)

        assert list(circuit.circuit.data[0].qubits) == [
            *circuit.qudits[0].qubits,
            *circuit.qudits[1].qubits,
        ]

    @parametrize_controlled_helper_names
    def test_helper_broadcasts_over_the_targets(
        self,
        helper: str,
    ) -> None:
        """One instruction per target, all sharing the controls."""
        circuit = QuditQuantumCircuit(3, dim=3)

        result = getattr(circuit, helper)(0, [1, 2])

        assert len(result) == 2
        assert [entry.qudits for entry in result] == [
            (circuit.qudits[0], circuit.qudits[1]),
            (circuit.qudits[0], circuit.qudits[2]),
        ]
        assert len(circuit.circuit.data) == 2

    @parametrize_controlled_helper_names
    def test_several_controls_share_one_gate(
        self,
        helper: str,
    ) -> None:
        """A control list produces a single multi-controlled gate."""
        circuit = QuditQuantumCircuit(3, dim=3)

        result = getattr(circuit, helper)([0, 1], 2)

        assert len(result) == 1
        assert result[0].qudits == (
            circuit.qudits[0],
            circuit.qudits[1],
            circuit.qudits[2],
        )
        assert result[0].operation.control_dims == (3, 3)
        assert result[0].operation.num_control_qudits == 2

    @parametrize_controlled_helper_names
    def test_gate_is_built_with_each_targets_own_dimension(
        self,
        helper: str,
    ) -> None:
        """Mixed-dimension circuits get one gate per target dim."""
        circuit = mixed_circuit()

        result = getattr(circuit, helper)(0, [1, 2])

        assert [entry.operation.target_dim for entry in result] == [3, 5]
        assert all(entry.operation.control_dims == (2,) for entry in result)
        assert [entry.dims for entry in result] == [(2, 3), (2, 5)]

    @parametrize_controlled_helper_names
    def test_helper_forwards_the_label(
        self,
        helper: str,
    ) -> None:
        """``label`` reaches every gate the helper creates."""
        circuit = QuditQuantumCircuit(3, dim=3)

        result = getattr(circuit, helper)(0, [1, 2], label="tag")

        assert [entry.operation.label for entry in result] == ["tag", "tag"]

    @parametrize_controlled_helper_names
    def test_duplicate_controls_are_rejected(
        self,
        helper: str,
    ) -> None:
        """The same control qudit cannot appear twice."""
        circuit = QuditQuantumCircuit(2, dim=3)

        with pytest.raises(QuditCircuitError, match="duplicate control"):
            getattr(circuit, helper)([0, 0], 1)

        assert len(circuit.data) == 0

    @parametrize_controlled_helper_names
    def test_a_control_cannot_also_be_the_target(
        self,
        helper: str,
    ) -> None:
        """Control and target must be distinct wires."""
        circuit = QuditQuantumCircuit(2, dim=3)

        with pytest.raises(QuditCircuitError, match="duplicate qudit"):
            getattr(circuit, helper)(0, 0)

    @parametrize_controlled_helper_names
    def test_at_least_one_control_is_required(
        self,
        helper: str,
    ) -> None:
        """An empty control specifier is rejected."""
        circuit = QuditQuantumCircuit(2, dim=3)

        with pytest.raises(QuditCircuitError, match="at least one control"):
            getattr(circuit, helper)((), 0)

    @parametrize_controlled_helper_names
    def test_at_least_one_target_is_required(
        self,
        helper: str,
    ) -> None:
        """An empty target specifier is rejected."""
        circuit = QuditQuantumCircuit(2, dim=3)

        with pytest.raises(QuditCircuitError, match="at least one target"):
            getattr(circuit, helper)(0, ())


class TestSumpHelper:
    """The one-angle controlled helper ``sump``."""

    def test_sump_takes_the_angle_first(self) -> None:
        """The signature mirrors ``QuantumCircuit.cp``."""
        circuit = QuditQuantumCircuit(2, dim=3)

        result = circuit.sump(0.5, 0, 1)

        assert isinstance(result[0].operation, QuditSUMPGate)
        assert result[0].operation.params == [0.5]
        assert result[0].operation.theta == pytest.approx(0.5)
        assert result[0].qudits == (circuit.qudits[0], circuit.qudits[1])
        assert circuit.circuit.data[0].operation.name == "SUMP"

    def test_sump_broadcasts_and_forwards_the_label(self) -> None:
        """``sump`` broadcasts exactly like the other helpers."""
        circuit = QuditQuantumCircuit(3, dim=3)

        result = circuit.sump(0.25, 0, [1, 2], label="phase")

        assert len(result) == 2
        assert [entry.operation.label for entry in result] == [
            "phase",
            "phase",
        ]
        assert all(
            entry.operation.theta == pytest.approx(0.25) for entry in result
        )

    def test_sump_uses_each_targets_own_dimension(self) -> None:
        """The angle is shared, the target dimension is not."""
        circuit = mixed_circuit()

        result = circuit.sump(0.25, 0, [1, 2])

        assert [entry.operation.target_dim for entry in result] == [3, 5]
        assert all(entry.operation.control_dims == (2,) for entry in result)

    @pytest.mark.parametrize(
        "theta",
        [float("inf"), float("nan")],
        ids=["inf", "nan"],
    )
    def test_sump_rejects_a_non_finite_angle(self, theta: float) -> None:
        """A non-finite angle never reaches the circuit."""
        circuit = QuditQuantumCircuit(2, dim=3)

        # NOTE: the angle is validated by the gate, so this is a plain
        # `ValueError` rather than a `QuditCircuitError`.
        with pytest.raises(ValueError, match="must be finite"):
            circuit.sump(theta, 0, 1)

        assert len(circuit.data) == 0


class TestSwapHelper:
    """The pairwise ``swap`` helper."""

    def test_swap_records_one_gate_per_pair(self) -> None:
        """A single pair appends a single SWAP gate."""
        circuit = QuditQuantumCircuit(2, dim=3)

        result = circuit.swap(0, 1)

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0].operation, QuditSWAPGate)
        assert result[0].operation.dims == (3, 3)
        assert result[0].qudits == (circuit.qudits[0], circuit.qudits[1])
        assert circuit.circuit.data[0].operation.name == "SWAP"
        assert list(circuit.circuit.data[0].qubits) == [
            *circuit.qudits[0].qubits,
            *circuit.qudits[1].qubits,
        ]

    def test_swap_pairs_operands_one_to_one(self) -> None:
        """Operand ``i`` of each list is paired with operand ``i``."""
        circuit = QuditQuantumCircuit(4, dim=3)

        result = circuit.swap([0, 1], [2, 3])

        assert [entry.qudits for entry in result] == [
            (circuit.qudits[0], circuit.qudits[2]),
            (circuit.qudits[1], circuit.qudits[3]),
        ]

    def test_swap_forwards_the_label(self) -> None:
        """``label`` reaches every SWAP the helper creates."""
        circuit = QuditQuantumCircuit(4, dim=3)

        result = circuit.swap([0, 1], [2, 3], label="tag")

        assert [entry.operation.label for entry in result] == ["tag", "tag"]

    def test_swap_rejects_mismatched_operand_counts(self) -> None:
        """Two firsts cannot share a single second."""
        circuit = QuditQuantumCircuit(3, dim=3)

        with pytest.raises(QuditCircuitError, match="one qudit per qudit"):
            circuit.swap([0, 1], 2)

    def test_swap_rejects_a_dimension_mismatch(self) -> None:
        """The two qudits of a pair must share a dimension."""
        circuit = mixed_circuit()

        with pytest.raises(QuditCircuitError, match="equal dimension"):
            circuit.swap(0, 1)

        assert len(circuit.data) == 0

    def test_swap_rejects_the_same_qudit_twice(self) -> None:
        """A qudit cannot be swapped with itself."""
        circuit = QuditQuantumCircuit(2, dim=3)

        with pytest.raises(QuditCircuitError, match="duplicate qudit"):
            circuit.swap(0, 0)


class TestQftHelper:
    """The register-wide ``qft`` helper."""

    def test_qft_defaults_to_every_qudit(self) -> None:
        """Omitting ``qudits`` transforms the whole circuit."""
        circuit = QuditQuantumCircuit(3, dim=3)

        instruction = circuit.qft()

        assert isinstance(instruction, QuditCircuitInstruction)
        assert isinstance(instruction.operation, QuditQFTGate)
        assert instruction.operation.num_qudits == 3
        assert instruction.operation.dims == (3, 3, 3)
        assert instruction.qudits == circuit.qudits
        assert circuit.circuit.data[0].operation.name == "QFT"
        assert list(circuit.circuit.data[0].qubits) == circuit.qubits

    def test_qft_honours_an_explicit_target_order(self) -> None:
        """Targets are used in the order the specifier implies."""
        circuit = QuditQuantumCircuit(3, dim=3)

        instruction = circuit.qft([2, 0])

        assert instruction.operation.num_qudits == 2
        assert instruction.qudits == (circuit.qudits[2], circuit.qudits[0])
        assert list(circuit.circuit.data[0].qubits) == [
            *circuit.qudits[2].qubits,
            *circuit.qudits[0].qubits,
        ]

    def test_qft_forwards_the_label(self) -> None:
        """``label`` is stored on the gate."""
        circuit = QuditQuantumCircuit(2, dim=3)

        instruction = circuit.qft(label="tag")

        assert instruction.operation.label == "tag"

    @pytest.mark.parametrize(
        "qudits",
        [[0], ()],
        ids=["one-target", "no-targets"],
    )
    def test_qft_needs_at_least_two_qudits(self, qudits: object) -> None:
        """A one-qudit Fourier transform is just a Hadamard."""
        circuit = QuditQuantumCircuit(3, dim=3)

        with pytest.raises(QuditCircuitError, match="at least 2 qudits"):
            circuit.qft(qudits)

    def test_qft_rejects_duplicate_targets(self) -> None:
        """The same qudit cannot appear twice in the register."""
        circuit = QuditQuantumCircuit(3, dim=3)

        with pytest.raises(QuditCircuitError, match="duplicate qudit"):
            circuit.qft([0, 0])

    def test_qft_rejects_mixed_dimensions(self) -> None:
        """The transform is only defined on a homogeneous register."""
        circuit = mixed_circuit()

        with pytest.raises(QuditCircuitError, match="equal dimension"):
            circuit.qft()

        assert len(circuit.data) == 0


class TestBarrier:
    """The ``barrier`` directive."""

    def test_barrier_without_arguments_covers_every_qudit(self) -> None:
        """An argument-free barrier spans the whole circuit."""
        circuit = QuditQuantumCircuit(3, dim=3)

        instruction = circuit.barrier()

        assert isinstance(instruction, QuditCircuitInstruction)
        assert isinstance(instruction.operation, QuditBarrier)
        assert instruction.qudits == circuit.qudits
        assert len(circuit.circuit.data) == 1
        assert list(circuit.circuit.data[0].qubits) == circuit.qubits

    def test_barrier_deduplicates_while_preserving_order(self) -> None:
        """Repeated targets are collapsed, first occurrence wins."""
        circuit = QuditQuantumCircuit(3, dim=3)

        instruction = circuit.barrier(2, 0, 2)

        assert instruction.qudits == (circuit.qudits[2], circuit.qudits[0])

    def test_barrier_forwards_the_label(self) -> None:
        """``label`` is stored on the directive."""
        circuit = QuditQuantumCircuit(2, dim=3)

        instruction = circuit.barrier(label="sync")

        assert instruction.operation.label == "sync"

    def test_barrier_does_not_count_towards_size_or_depth(self) -> None:
        """Barriers are directives and are filtered out by default."""
        circuit = QuditQuantumCircuit(1, dim=3)
        circuit.x(0)
        circuit.barrier()
        circuit.x(0)

        assert len(circuit.data) == 3
        assert circuit.size() == 2
        assert circuit.depth() == 2


class TestReset:
    """The ``reset`` directive."""

    def test_reset_creates_one_instruction_per_target(self) -> None:
        """``reset`` broadcasts like the gate helpers."""
        circuit = QuditQuantumCircuit(3, dim=3)

        result = circuit.reset([0, 2])

        assert len(result) == 2
        assert all(isinstance(entry.operation, QuditReset) for entry in result)
        assert [entry.qudits[0] for entry in result] == [
            circuit.qudits[0],
            circuit.qudits[2],
        ]

    def test_reset_expands_to_one_reset_per_encoding_qubit(self) -> None:
        """Every encoding qubit is reset, in little-endian order."""
        circuit = QuditQuantumCircuit(1, dim=3)

        circuit.reset(0)

        encoded = circuit.circuit.data
        assert [entry.operation.name for entry in encoded] == [
            "reset",
            "reset",
        ]
        assert [entry.qubits[0] for entry in encoded] == list(
            circuit.qudits[0].qubits,
        )


class TestMeasure:
    """The ``measure`` directive."""

    def test_measure_pairs_qudits_and_clbytes_one_to_one(self) -> None:
        """Operand ``i`` of each list is paired with operand ``i``."""
        circuit = QuditQuantumCircuit(2, 2, dim=3)

        result = circuit.measure([0, 1], [1, 0])

        assert all(
            isinstance(entry.operation, QuditMeasure) for entry in result
        )
        assert [entry.qudits[0] for entry in result] == [
            circuit.qudits[0],
            circuit.qudits[1],
        ]
        assert [entry.clbytes[0] for entry in result] == [
            circuit.clbytes[1],
            circuit.clbytes[0],
        ]

    def test_measure_rejects_mismatched_operand_counts(self) -> None:
        """Two qudits cannot share a single clbyte."""
        circuit = QuditQuantumCircuit(2, 2, dim=3)

        with pytest.raises(QuditCircuitError, match="one clbyte per qudit"):
            circuit.measure([0, 1], 0)

    def test_encoded_measurements_are_little_endian(self) -> None:
        """Qubit ``j`` of a qudit lands on clbit ``j`` of its byte."""
        circuit = QuditQuantumCircuit(1, 1, dim=3)

        circuit.measure(0, 0)

        encoded = circuit.circuit.data
        assert len(encoded) == 2
        for index, entry in enumerate(encoded):
            assert entry.operation.name == "measure"
            # Qiskit rebuilds bit handles when reading back from the
            # Rust-backed `CircuitData`, so compare by value, not by
            # identity.
            assert entry.qubits[0] == circuit.qudits[0].qubits[index]
            assert entry.clbits[0] == circuit.clbytes[0].clbits[index]


class TestMeasureAll:
    """The ``measure_all`` convenience wrapper."""

    def test_measure_all_adds_a_register_sized_from_the_dims(self) -> None:
        """The new register mirrors the circuit's dimensions."""
        circuit = mixed_circuit()

        assert circuit.measure_all() is None

        register = circuit.cbregs[-1]
        assert register.name == "meas"
        assert register.dims == MIXED_DIMS
        assert circuit.clbyte_widths == MIXED_WIDTHS
        assert circuit.num_clbits == sum(MIXED_WIDTHS)

    def test_measure_all_inserts_a_barrier_before_measuring(self) -> None:
        """A barrier separates the circuit from its readout."""
        circuit = QuditQuantumCircuit(2, dim=3)

        circuit.measure_all()

        assert [entry.name for entry in circuit.data] == [
            "barrier",
            "measure",
            "measure",
        ]
        assert circuit.data[0].qudits == circuit.qudits

    def test_measure_all_avoids_an_existing_register_name(self) -> None:
        """A taken ``'meas'`` name is suffixed."""
        circuit = QuditQuantumCircuit(
            QuditRegister(1, 3, "q"),
            ClByteRegister(1, 3, "meas"),
        )

        circuit.measure_all()

        assert [register.name for register in circuit.cbregs] == [
            "meas",
            "meas0",
        ]

    def test_measure_all_can_reuse_the_existing_clbytes(self) -> None:
        """``add_bytes=False`` measures into clbyte ``i``."""
        circuit = QuditQuantumCircuit(2, 3, dim=3)

        circuit.measure_all(add_bytes=False)

        assert len(circuit.cbregs) == 1
        assert circuit.num_clbytes == 3
        assert [entry.clbytes[0] for entry in circuit.data[1:]] == list(
            circuit.clbytes[:2],
        )

    def test_measure_all_requires_enough_clbytes(self) -> None:
        """Reusing clbytes needs at least one byte per qudit."""
        circuit = QuditQuantumCircuit(2, 1, dim=3)

        with pytest.raises(QuditCircuitError, match="at least the number"):
            circuit.measure_all(add_bytes=False)

    def test_measure_all_out_of_place_leaves_the_original(self) -> None:
        """``inplace=False`` returns a new, measured circuit."""
        circuit = QuditQuantumCircuit(2, dim=3)

        measured = circuit.measure_all(inplace=False)

        assert measured is not None
        assert measured is not circuit
        assert len(circuit.data) == 0
        assert circuit.num_clbytes == 0
        assert len(measured.data) == 3
        assert measured.num_clbytes == 2


class TestInitializeLevels:
    """The basis-state initialisation entry point."""

    def test_a_bare_integer_targets_a_single_qudit(self) -> None:
        """An ``int`` is the level of the one target qudit."""
        circuit = QuditQuantumCircuit(2, dim=5)

        instruction = circuit.initialize_levels(3, 1)

        assert isinstance(instruction.operation, QuditInitializeLevels)
        assert instruction.operation.values == (3,)
        assert instruction.qudits == (circuit.qudits[1],)
        assert circuit.circuit.data[0].operation.name == "initialize"

    def test_a_bare_integer_needs_exactly_one_target(self) -> None:
        """With several targets the level would be ambiguous."""
        circuit = QuditQuantumCircuit(2, dim=5)

        with pytest.raises(QuditCircuitError, match="bare integer level"):
            circuit.initialize_levels(3)

    def test_the_string_form_uses_qiskit_ordering(self) -> None:
        """The rightmost token belongs to the first target."""
        circuit = QuditQuantumCircuit(3, dim=16)

        instruction = circuit.initialize_levels("11 3 0")

        assert instruction.operation.values == (0, 3, 11)

    def test_the_sequence_form_is_in_target_order(self) -> None:
        """A ``Sequence[int]`` is *not* reversed."""
        circuit = QuditQuantumCircuit(3, dim=16)

        instruction = circuit.initialize_levels([0, 3, 11])

        assert instruction.operation.values == (0, 3, 11)

    def test_levels_default_to_every_qudit(self) -> None:
        """Omitting ``qudits`` targets the whole circuit."""
        circuit = QuditQuantumCircuit(3, dim=3)

        instruction = circuit.initialize_levels("0 1 2")

        assert instruction.qudits == circuit.qudits
        assert instruction.operation.values == (2, 1, 0)

    def test_initialize_levels_needs_a_qudit(self) -> None:
        """An empty circuit has nothing to initialise."""
        circuit = QuditQuantumCircuit()

        with pytest.raises(QuditCircuitError, match="at least one qudit"):
            circuit.initialize_levels(0)

    def test_a_non_integer_level_is_rejected(self) -> None:
        """Sequence elements must be integral."""
        circuit = QuditQuantumCircuit(2, dim=3)

        with pytest.raises(QuditCircuitError, match="non-integer level"):
            circuit.initialize_levels([0, 1.5])

    def test_an_out_of_range_level_is_rejected(self) -> None:
        """A level must fit into the qudit's dimension."""
        circuit = QuditQuantumCircuit(1, dim=3)

        # NOTE: the docstring advertises `QuditCircuitError`, but the
        # check lives in `validate_basis_states`, which raises the
        # stdlib `ValueError`.
        with pytest.raises(ValueError, match=r"must be in \[0, 2\]"):
            circuit.initialize_levels(5)

        assert len(circuit.data) == 0

    def test_a_wrong_length_sequence_raises_value_error(self) -> None:
        """One level per target is required."""
        circuit = QuditQuantumCircuit(3, dim=3)

        # NOTE: same as above - `validate_basis_states` raises a plain
        # `ValueError`, not the documented `QuditCircuitError`.
        with pytest.raises(ValueError, match=r"2 state.* for 3 qudit"):
            circuit.initialize_levels([0, 1])

    def test_a_wrong_token_count_raises_value_error(self) -> None:
        """Concatenated digits are ambiguous and rejected."""
        circuit = QuditQuantumCircuit(3, dim=3)

        # NOTE: `parse_level_tokens` also raises a plain `ValueError`.
        with pytest.raises(ValueError, match="expected 3 whitespace"):
            circuit.initialize_levels("012")


class TestInitialize:
    """The general initialisation entry point."""

    def test_an_integer_is_forwarded_to_initialize_levels(self) -> None:
        """Integers are labels, exactly like in Qiskit."""
        circuit = QuditQuantumCircuit(1, dim=3)

        instruction = circuit.initialize(2)

        assert isinstance(instruction.operation, QuditInitializeLevels)
        assert instruction.operation.values == (2,)

    def test_a_string_is_forwarded_to_initialize_levels(self) -> None:
        """Strings are labels too."""
        circuit = QuditQuantumCircuit(2, dim=3)

        instruction = circuit.initialize("2 0")

        assert isinstance(instruction.operation, QuditInitializeLevels)
        assert instruction.operation.values == (0, 2)

    def test_a_logical_vector_is_kept_as_is(self) -> None:
        """``prod(dims)`` amplitudes are the logical state."""
        circuit = QuditQuantumCircuit(1, dim=3)
        amplitudes = np.ones(3) / np.sqrt(3)

        instruction = circuit.initialize(amplitudes)

        assert isinstance(instruction.operation, QuditStatePreparation)
        assert instruction.operation.amplitudes.size == 3
        assert np.allclose(instruction.operation.amplitudes, amplitudes)
        assert circuit.circuit.data[0].operation.name == "initialize"

    def test_an_encoded_vector_is_projected_back(self) -> None:
        """``2**N`` amplitudes are read as an encoded state."""
        circuit = QuditQuantumCircuit(1, dim=3)

        instruction = circuit.initialize(np.array([0.0, 1.0, 0.0, 0.0]))

        assert instruction.operation.amplitudes.size == 3
        assert np.allclose(
            instruction.operation.amplitudes,
            [0.0, 1.0, 0.0],
        )

    def test_an_encoded_vector_may_not_leak(self) -> None:
        """Amplitude on an invalid basis state is an error."""
        circuit = QuditQuantumCircuit(1, dim=3)

        with pytest.raises(ValueError, match="outside the qudit subspace"):
            circuit.initialize(np.array([0.0, 0.0, 0.0, 1.0]))

        assert len(circuit.data) == 0

    def test_a_wrong_length_vector_is_rejected(self) -> None:
        """Neither the logical nor the encoded length matches."""
        circuit = QuditQuantumCircuit(1, dim=3)

        with pytest.raises(
            QuditCircuitError,
            match=r"expected 3 \(logical\) or 4 \(encoded\)",
        ):
            circuit.initialize(np.zeros(5))

    def test_a_non_normalised_vector_is_rejected(self) -> None:
        """The amplitudes must describe a unit vector."""
        circuit = QuditQuantumCircuit(1, dim=3)

        with pytest.raises(ValueError, match="not normalised"):
            circuit.initialize(np.array([1.0, 1.0, 0.0]))

    def test_a_non_vector_is_rejected(self) -> None:
        """Only 1-D data can be a state-vector."""
        circuit = QuditQuantumCircuit(1, dim=3)

        with pytest.raises(
            QuditCircuitError,
            match="invalid state specification",
        ):
            circuit.initialize(np.eye(2))

    def test_a_plain_sequence_is_read_as_amplitudes(self) -> None:
        """Sequences are amplitudes; only ``str``/``int`` are labels."""
        circuit = QuditQuantumCircuit(1, dim=3)

        instruction = circuit.initialize([0, 1, 0])

        assert isinstance(instruction.operation, QuditStatePreparation)

    def test_initialize_needs_a_qudit(self) -> None:
        """An empty circuit has nothing to initialise."""
        circuit = QuditQuantumCircuit()

        with pytest.raises(
            QuditCircuitError,
            match="initialize needs at least one qudit",
        ):
            circuit.initialize(np.array([1.0, 0.0]))
