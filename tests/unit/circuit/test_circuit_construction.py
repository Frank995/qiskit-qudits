"""Construction and wire bookkeeping of ``QuditQuantumCircuit``.

Covers the constructor forms, the ``name`` / ``metadata`` /
``global_phase`` properties, register and loose-object registration,
the lookup helpers and every read-only property of the class.

Nothing here appends a gate helper: operations are exercised in
``test_circuit_operations.py``. When an instruction is needed as a
fixture it is appended explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import (
    Clbit,
    Parameter,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
)
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import RZGate

from qiskit_qudits.circuit.clbyte import ClByte, ClByteRegister
from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.circuit.instruction import QuditCircuitInstruction
from qiskit_qudits.circuit.quantumcircuit import QuditQuantumCircuit
from qiskit_qudits.circuit.qudit import Qudit, QuditRegister
from qiskit_qudits.gates import QuditXGate

# ------------------------------------------------------------------ #
# Factories
# ------------------------------------------------------------------ #
MIXED_DIMS = (2, 3, 5)

#: Encoding width of each dimension in :data:`MIXED_DIMS`.
MIXED_WIDTHS = (1, 2, 3)


def mixed_circuit() -> QuditQuantumCircuit:
    """Return a heterogeneous circuit with matching clbytes."""
    return QuditQuantumCircuit(
        QuditRegister.from_dims(MIXED_DIMS, "mix"),
        ClByteRegister.from_dims(MIXED_DIMS, "out"),
    )


class TestConstructorForms:
    """The ``*regs`` overloads accepted by ``__init__``."""

    def test_integer_form_creates_a_qudit_register(self) -> None:
        """``(3, dim=4)`` creates one auto-named qudit register."""
        circuit = QuditQuantumCircuit(3, dim=4)

        assert circuit.num_qudits == 3
        assert circuit.dims == (4, 4, 4)
        assert [reg.name for reg in circuit.qdregs] == ["qd"]
        assert circuit.num_clbytes == 0
        assert circuit.num_qubits == 6

    def test_integer_form_creates_both_registers(self) -> None:
        """``(3, 3, dim=4)`` also creates the clbyte register."""
        circuit = QuditQuantumCircuit(3, 3, dim=4)

        assert [reg.name for reg in circuit.qdregs] == ["qd"]
        assert [reg.name for reg in circuit.cbregs] == ["cb"]
        assert circuit.num_clbytes == 3
        assert circuit.num_clbits == 6

    def test_auto_created_registers_back_the_encoded_circuit(self) -> None:
        """The ``'qd'``/``'cb'`` names reach the encoded circuit."""
        circuit = QuditQuantumCircuit(2, 2, dim=3)

        assert [reg.name for reg in circuit.circuit.qregs] == ["qd"]
        assert [reg.name for reg in circuit.circuit.cregs] == ["cb"]

    def test_register_form_keeps_the_given_objects(self) -> None:
        """Registers are stored by identity, in argument order."""
        qdreg = QuditRegister(2, 3, "alice")
        cbreg = ClByteRegister(2, 3, "bob")

        circuit = QuditQuantumCircuit(qdreg, cbreg)

        assert circuit.qdregs == (qdreg,)
        assert circuit.cbregs == (cbreg,)
        assert circuit.qudits == qdreg.qudits
        assert circuit.clbytes == cbreg.clbytes

    def test_no_argument_form_is_empty(self) -> None:
        """``QuditQuantumCircuit()`` has no wires and no data."""
        circuit = QuditQuantumCircuit()

        assert circuit.num_qudits == 0
        assert circuit.num_clbytes == 0
        assert circuit.qdregs == ()
        assert circuit.cbregs == ()
        assert len(circuit) == 0

    def test_integer_form_requires_dim(self) -> None:
        """The integer form without ``dim`` is rejected."""
        with pytest.raises(QuditCircuitError, match="requires"):
            QuditQuantumCircuit(3)

    def test_more_than_two_integers_are_rejected(self) -> None:
        """At most ``(qudits, clbytes)`` may be given as integers."""
        with pytest.raises(QuditCircuitError, match="at most 2 integer"):
            QuditQuantumCircuit(1, 1, 1, dim=2)

    def test_mixing_integers_and_registers_is_rejected(self) -> None:
        """Registers and integers cannot be combined."""
        register = QuditRegister(1, 2, "solo")

        for args in ((2, register), (register, 2)):
            with pytest.raises(
                QuditCircuitError,
                match="registers or integers",
            ):
                QuditQuantumCircuit(*args, dim=2)

    @pytest.mark.parametrize(
        ("sizes", "attribute"),
        [((0,), "qdregs"), ((2, 0), "cbregs")],
        ids=["no-qudits", "no-clbytes"],
    )
    def test_a_zero_size_creates_no_register(
        self,
        sizes: tuple[int, ...],
        attribute: str,
    ) -> None:
        """A size of zero is silently skipped."""
        circuit = QuditQuantumCircuit(*sizes, dim=4)

        assert getattr(circuit, attribute) == ()


class TestName:
    """The ``name`` property and its validation."""

    def test_name_is_auto_generated_from_a_counter(self) -> None:
        """Unnamed circuits get ``quditcircuit-<n>`` names."""
        first = QuditQuantumCircuit()
        second = QuditQuantumCircuit()

        assert first.name == "quditcircuit-0"
        assert second.name == "quditcircuit-1"

    def test_name_is_shared_with_the_encoded_circuit(self) -> None:
        """The encoded circuit carries the same name."""
        circuit = QuditQuantumCircuit(name="demo")

        assert circuit.name == "demo"
        assert circuit.circuit.name == "demo"

    def test_setter_renames_both_circuits(self) -> None:
        """Assigning ``name`` renames the encoded circuit too."""
        circuit = QuditQuantumCircuit(1, dim=2)

        circuit.name = "renamed"

        assert circuit.name == "renamed"
        assert circuit.circuit.name == "renamed"

    def test_empty_name_is_rejected_at_construction(self) -> None:
        """An empty name is not a valid circuit name."""
        with pytest.raises(QuditCircuitError, match="non-empty string"):
            QuditQuantumCircuit(name="")

    def test_empty_name_is_rejected_by_the_setter(self) -> None:
        """The setter applies the same validation."""
        circuit = QuditQuantumCircuit()

        with pytest.raises(QuditCircuitError, match="non-empty string"):
            circuit.name = ""


class TestMetadata:
    """The ``metadata`` property, stored on the encoded circuit."""

    def test_metadata_defaults_to_an_empty_dict(self) -> None:
        """No metadata means an empty mapping, never ``None``."""
        circuit = QuditQuantumCircuit()

        assert circuit.metadata == {}

    def test_metadata_is_stored_on_the_encoded_circuit(self) -> None:
        """The very same mapping is handed to the encoded circuit."""
        metadata: dict[str, object] = {"author": "alice"}

        circuit = QuditQuantumCircuit(metadata=metadata)

        assert circuit.metadata is metadata
        assert circuit.circuit.metadata is metadata

    def test_metadata_setter_replaces_the_mapping(self) -> None:
        """Assigning ``metadata`` swaps the stored mapping."""
        circuit = QuditQuantumCircuit()

        circuit.metadata = {"run": 3}

        assert circuit.metadata == {"run": 3}
        assert circuit.circuit.metadata == {"run": 3}


class TestGlobalPhase:
    """The ``global_phase`` property and its validation."""

    def test_global_phase_defaults_to_zero(self) -> None:
        """A fresh circuit has no global phase."""
        assert QuditQuantumCircuit().global_phase == 0.0

    def test_global_phase_mirrors_the_encoded_circuit(self) -> None:
        """The value is stored on (and read from) the encoding."""
        circuit = QuditQuantumCircuit(1, dim=2, global_phase=0.25)

        assert circuit.global_phase == 0.25
        assert circuit.circuit.global_phase == 0.25

    def test_global_phase_setter_updates_the_encoding(self) -> None:
        """Assigning the phase forwards it to the encoded circuit."""
        circuit = QuditQuantumCircuit(1, dim=2)

        circuit.global_phase = 0.5

        assert circuit.circuit.global_phase == 0.5

    @pytest.mark.parametrize(
        "value",
        [float("inf"), float("-inf"), float("nan")],
        ids=["inf", "-inf", "nan"],
    )
    def test_non_finite_global_phase_is_rejected(self, value: float) -> None:
        """A non-finite angle raises a plain ``ValueError``."""
        # NOTE: `validate_float_finite` deliberately raises the stdlib
        # exceptions, so this is *not* a `QuditCircuitError`.
        with pytest.raises(ValueError, match="must be finite"):
            QuditQuantumCircuit(global_phase=value)

        circuit = QuditQuantumCircuit()
        with pytest.raises(ValueError, match="must be finite"):
            circuit.global_phase = value

    def test_non_real_global_phase_is_rejected(self) -> None:
        """A non-numeric angle raises ``TypeError``."""
        with pytest.raises(TypeError, match="must be a float"):
            QuditQuantumCircuit(global_phase="zero")


class TestAddRegister:
    """``add_register`` and its bookkeeping."""

    def test_registers_extend_both_views(self) -> None:
        """The backing registers land on the encoded circuit."""
        circuit = QuditQuantumCircuit()
        qdreg = QuditRegister(2, 3, "alice")
        cbreg = ClByteRegister(2, 3, "bob")

        circuit.add_register(qdreg, cbreg)

        assert circuit.qdregs == (qdreg,)
        assert circuit.cbregs == (cbreg,)
        assert circuit.circuit.qregs == [qdreg.qreg]
        assert circuit.circuit.cregs == [cbreg.creg]
        assert circuit.num_qubits == 4
        assert circuit.num_clbits == 4

    def test_adding_the_same_register_twice_is_rejected(self) -> None:
        """A register object cannot be added twice."""
        register = QuditRegister(2, 3, "twice")
        circuit = QuditQuantumCircuit(register)

        # NOTE: the name check runs first, so the dedicated
        # "is already in this circuit" branch of `add_register` is
        # unreachable and a name clash is reported instead.
        with pytest.raises(QuditCircuitError, match="already exists"):
            circuit.add_register(register)

    def test_a_name_clash_across_kinds_is_rejected(self) -> None:
        """Qudit and clbyte register names share one namespace."""
        circuit = QuditQuantumCircuit(QuditRegister(1, 3, "shared"))

        with pytest.raises(QuditCircuitError, match="already exists"):
            circuit.add_register(ClByteRegister(1, 3, "shared"))

    def test_an_unsupported_register_type_is_rejected(self) -> None:
        """Plain Qiskit registers are not qudit registers."""
        circuit = QuditQuantumCircuit()

        with pytest.raises(
            QuditCircuitError,
            match="expected a QuditRegister",
        ):
            circuit.add_register(
                QuantumRegister(2, "plain"),
            )


class TestLooseObjects:
    """``add_qudits`` and ``add_clbytes``."""

    def test_loose_qudits_add_their_qubits_to_the_encoding(self) -> None:
        """Encoding qubits become loose qubits of the encoding."""
        circuit = QuditQuantumCircuit()
        qudit = Qudit(3)

        circuit.add_qudits([qudit])

        assert circuit.qudits == (qudit,)
        assert circuit.qdregs == ()
        assert circuit.qubits == list(qudit.qubits)
        assert circuit.circuit.qregs == []

    def test_loose_clbytes_add_their_clbits_to_the_encoding(self) -> None:
        """Clbits become loose clbits of the encoded circuit."""
        circuit = QuditQuantumCircuit()
        clbyte = ClByte(3)

        circuit.add_clbytes([clbyte])

        assert circuit.clbytes == (clbyte,)
        assert circuit.cbregs == ()
        assert circuit.clbits == list(clbyte.clbits)

    def test_add_qudits_rejects_a_foreign_type(self) -> None:
        """Only :class:`Qudit` objects are accepted."""
        circuit = QuditQuantumCircuit()

        with pytest.raises(QuditCircuitError, match="expected a Qudit"):
            circuit.add_qudits([Qubit()])

    def test_add_clbytes_rejects_a_foreign_type(self) -> None:
        """Only :class:`ClByte` objects are accepted."""
        circuit = QuditQuantumCircuit()

        with pytest.raises(QuditCircuitError, match="expected a ClByte"):
            circuit.add_clbytes([Clbit()])

    def test_adding_the_same_qudit_twice_is_rejected(self) -> None:
        """A qudit cannot be registered twice."""
        qudit = Qudit(3)
        circuit = QuditQuantumCircuit()
        circuit.add_qudits([qudit])

        # NOTE: Qiskit's `add_bits` rejects the duplicate encoding
        # qubits first, so this is a plain `CircuitError` and *not*
        # the `QuditCircuitError` of `_register_qudits`.
        with pytest.raises(CircuitError, match="already in circuit"):
            circuit.add_qudits([qudit])

    def test_adding_the_same_clbyte_twice_is_rejected(self) -> None:
        """A clbyte cannot be registered twice."""
        clbyte = ClByte(3)
        circuit = QuditQuantumCircuit()
        circuit.add_clbytes([clbyte])

        with pytest.raises(CircuitError, match="already in circuit"):
            circuit.add_clbytes([clbyte])


class TestLookup:
    """``has_register``, ``find_qudit`` and ``find_clbyte``."""

    def test_has_register_compares_by_identity(self) -> None:
        """Only the very same register object is recognised."""
        qdreg = QuditRegister(1, 3, "alice")
        cbreg = ClByteRegister(1, 3, "bob")
        circuit = QuditQuantumCircuit(qdreg, cbreg)

        assert circuit.has_register(qdreg)
        assert circuit.has_register(cbreg)
        assert not circuit.has_register(QuditRegister(1, 3, "alice2"))
        assert not circuit.has_register(ClByteRegister(1, 3, "bob2"))

    def test_find_qudit_returns_the_circuit_wide_index(self) -> None:
        """Indices follow the order in which qudits were added."""
        circuit = mixed_circuit()

        indices = [circuit.find_qudit(qudit) for qudit in circuit.qudits]

        assert indices == [0, 1, 2]

    def test_find_clbyte_returns_the_circuit_wide_index(self) -> None:
        """Clbyte indices follow the clbit order."""
        circuit = mixed_circuit()

        indices = [circuit.find_clbyte(byte) for byte in circuit.clbytes]

        assert indices == [0, 1, 2]

    def test_find_qudit_rejects_a_foreign_qudit(self) -> None:
        """A qudit from another circuit has no index here."""
        circuit = QuditQuantumCircuit(1, dim=3)

        with pytest.raises(QuditCircuitError, match="not in this circuit"):
            circuit.find_qudit(Qudit(3))

    def test_find_clbyte_rejects_a_foreign_clbyte(self) -> None:
        """A clbyte from another circuit has no index here."""
        circuit = QuditQuantumCircuit(1, 1, dim=3)

        with pytest.raises(QuditCircuitError, match="not in this circuit"):
            circuit.find_clbyte(ClByte(3))


class TestProperties:
    """The read-only view of the circuit's wires and data."""

    def test_qudit_and_clbyte_collections_are_tuples(self) -> None:
        """The public collections are immutable snapshots."""
        circuit = QuditQuantumCircuit(2, 2, dim=3)

        assert isinstance(circuit.qudits, tuple)
        assert isinstance(circuit.clbytes, tuple)
        assert isinstance(circuit.qdregs, tuple)
        assert isinstance(circuit.cbregs, tuple)
        assert isinstance(circuit.data, tuple)

    def test_wire_counts(self) -> None:
        """Qudit/clbyte counts and their encoded counterparts."""
        circuit = QuditQuantumCircuit(2, 3, dim=3)

        assert circuit.num_qudits == 2
        assert circuit.num_clbytes == 3
        assert circuit.num_qubits == 4
        assert circuit.num_clbits == 6

    def test_bit_properties_delegate_to_the_encoded_circuit(self) -> None:
        """``qubits``, ``clbits`` and ``qregs`` are proxies."""
        circuit = QuditQuantumCircuit(2, 2, dim=3)

        assert circuit.qubits == circuit.circuit.qubits
        assert circuit.clbits == circuit.circuit.clbits
        assert circuit.qregs == circuit.circuit.qregs
        assert circuit.qubits == [
            qubit for qudit in circuit.qudits for qubit in qudit.qubits
        ]
        assert circuit.clbits == [
            clbit for clbyte in circuit.clbytes for clbit in clbyte.clbits
        ]

    def test_dims_lists_every_qudit_dimension(self) -> None:
        """``dims`` is in qudit order."""
        assert QuditQuantumCircuit(3, dim=5).dims == (5, 5, 5)

    def test_dim_returns_the_common_dimension(self) -> None:
        """A homogeneous circuit exposes a single ``dim``."""
        assert QuditQuantumCircuit(3, dim=5).dim == 5

    def test_dim_is_zero_for_an_empty_circuit(self) -> None:
        """Without qudits there is nothing to disagree about."""
        assert QuditQuantumCircuit().dim == 0

    def test_dim_rejects_a_heterogeneous_circuit(self) -> None:
        """Mixed dimensions must be read through ``dims``."""
        circuit = mixed_circuit()

        with pytest.raises(QuditCircuitError, match="heterogeneous"):
            _ = circuit.dim

    def test_clbyte_layout_properties(self) -> None:
        """Clbyte widths and dimensions are in clbit order."""
        circuit = mixed_circuit()

        assert circuit.clbyte_dims == MIXED_DIMS
        assert circuit.clbyte_widths == MIXED_WIDTHS

    def test_len_and_getitem_expose_the_instruction_log(self) -> None:
        """``len`` and indexing address ``data`` directly."""
        circuit = QuditQuantumCircuit(2, dim=3)
        first = circuit.append(QuditXGate(3), 0)
        second = circuit.append(QuditXGate(3), 1)

        assert len(circuit) == 2
        assert circuit[0] is first
        assert circuit[1] is second
        assert circuit[-1] is second
        assert circuit.data == (first, second)
        assert isinstance(circuit[0], QuditCircuitInstruction)

    def test_getitem_accepts_a_numpy_integer(self) -> None:
        """Index-like NumPy scalars are coerced with ``int``."""
        circuit = QuditQuantumCircuit(1, dim=3)
        instruction = circuit.append(QuditXGate(3), 0)

        assert circuit[np.int64(0)] is instruction

    def test_parameter_properties_delegate_to_the_encoding(self) -> None:
        """Compile-time parameters come from the encoded circuit."""
        theta = Parameter("theta")
        circuit = QuditQuantumCircuit(1, dim=2)

        assert circuit.num_parameters == 0

        circuit.append(RZGate(theta), 0, copy=False)

        assert circuit.num_parameters == 1
        assert list(circuit.parameters) == [theta]


class TestMixedDimensions:
    """Circuits built from ``QuditRegister.from_dims``."""

    def test_mixed_register_reports_per_qudit_dimensions(self) -> None:
        """Each qudit keeps its own dimension and width."""
        circuit = mixed_circuit()

        assert circuit.dims == MIXED_DIMS
        assert [qudit.num_qubits for qudit in circuit.qudits] == list(
            MIXED_WIDTHS,
        )
        assert circuit.num_qubits == sum(MIXED_WIDTHS)
        assert circuit.num_clbits == sum(MIXED_WIDTHS)

    def test_mixed_circuit_supports_the_lookup_helpers(self) -> None:
        """Bookkeeping is dimension agnostic."""
        circuit = mixed_circuit()

        assert circuit.find_qudit(circuit.qudits[2]) == 2
        assert circuit.find_clbyte(circuit.clbytes[2]) == 2
        assert circuit.qudits[2].qubits == tuple(circuit.qubits[3:])

    def test_mixed_circuit_appends_per_dimension_gates(self) -> None:
        """A gate must match the dimension of its target qudit."""
        circuit = mixed_circuit()

        instruction = circuit.append(QuditXGate(5), 2)

        assert instruction.dims == (5,)
        assert list(circuit.circuit.data[0].qubits) == list(
            circuit.qudits[2].qubits,
        )

    def test_encoded_registers_are_sized_from_the_widths(self) -> None:
        """A heterogeneous register widens the encoded registers."""
        circuit = mixed_circuit()

        assert isinstance(circuit.circuit, QuantumCircuit)
        assert circuit.circuit.qregs[0].size == sum(MIXED_WIDTHS)
        assert circuit.circuit.cregs[0].size == sum(MIXED_WIDTHS)
        assert circuit.circuit.qregs[0].name == "mix"
