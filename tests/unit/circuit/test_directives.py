"""Unit tests for :mod:`qiskit_qudits.circuit.directives`.

The directives are the only place where the qudit data model touches a
real :class:`~qiskit.circuit.QuantumCircuit`, so every ``apply`` test
builds a plain Qiskit circuit, wraps :class:`Qudit`/:class:`ClByte`
handles around its bits, applies the directive and then inspects
``circuit.data``.

Assertions are always on the *order* of the emitted operations: the
little-endian pairing between a qudit's qubits and a clbyte's clbits is
the whole point of the encoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from qiskit_qudits.circuit.clbyte import ClByte
from qiskit_qudits.circuit.directives import (
    QuditBarrier,
    QuditDirective,
    QuditInitializeLevels,
    QuditMeasure,
    QuditReset,
    QuditStatePreparation,
)
from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.circuit.qudit import Qudit
from tests.helpers import assert_allclose

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit_qudits.utils.consts import IntLike

#: ``ceil(log2 d)`` for every dimension used below.
WIDTHS = {2: 1, 3: 2, 4: 2, 5: 3, 7: 3, 8: 3}


class DummyDirective(QuditDirective):
    """Smallest possible concrete directive.

    It exists purely to reach the otherwise-abstract behaviour of
    :class:`QuditDirective`; ``apply`` validates its targets and then
    records the call instead of emitting anything.
    """

    def __init__(
        self,
        dims: Sequence[IntLike],
        *,
        num_clbytes: int = 0,
        num_clbits: int = 0,
        params: Sequence[object] = (),
        label: str | None = None,
    ) -> None:
        """Create the dummy directive.

        Args:
            dims: Dimension of each target qudit.
            num_clbytes: Number of target clbytes.
            num_clbits: Total number of target clbits.
            params: Values forwarded to :attr:`params`.
            label: Optional display label.
        """
        super().__init__(
            "dummy",
            dims,
            num_clbytes=num_clbytes,
            num_clbits=num_clbits,
            params=params,
            label=label,
        )
        self.calls: list[
            tuple[QuantumCircuit, tuple[Qudit, ...], tuple[ClByte, ...]]
        ] = []

    def apply(
        self,
        circuit: QuantumCircuit,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Validate the resolved targets and record the call.

        Args:
            circuit: The encoded circuit (left untouched).
            qudits: Resolved target qudits.
            clbytes: Resolved target clbytes.
        """
        self._check_targets(qudits, clbytes)
        self.calls.append((circuit, tuple(qudits), tuple(clbytes)))


def build_targets(
    dims: Sequence[int],
    clbyte_dims: Sequence[int] = (),
) -> tuple[QuantumCircuit, list[Qudit], list[ClByte]]:
    """Build a plain circuit plus qudits/clbytes over its own bits.

    The bits are handed out in order, so qudit ``0`` owns the lowest
    qubit indices and clbyte ``0`` the lowest clbit indices.

    Args:
        dims: Dimension of each qudit, in target order.
        clbyte_dims: Dimension of each clbyte, in target order.

    Returns:
        The circuit, its qudits and its clbytes.
    """
    widths = [WIDTHS[dim] for dim in dims]
    clbyte_widths = [WIDTHS[dim] for dim in clbyte_dims]
    circuit = (
        QuantumCircuit(sum(widths), sum(clbyte_widths))
        if clbyte_widths
        else QuantumCircuit(sum(widths))
    )

    qudits: list[Qudit] = []
    offset = 0
    for dim, width in zip(dims, widths, strict=True):
        qudits.append(Qudit(dim, circuit.qubits[offset : offset + width]))
        offset += width

    clbytes: list[ClByte] = []
    offset = 0
    for dim, width in zip(clbyte_dims, clbyte_widths, strict=True):
        clbytes.append(ClByte(dim, circuit.clbits[offset : offset + width]))
        offset += width

    return circuit, qudits, clbytes


def operation_names(circuit: QuantumCircuit) -> list[str]:
    """Return the name of every operation in ``circuit``, in order.

    Args:
        circuit: The circuit to inspect.

    Returns:
        One name per entry of ``circuit.data``.
    """
    return [entry.operation.name for entry in circuit.data]


class TestQuditDirectiveBase:
    """Behaviour inherited by every concrete directive."""

    def test_the_base_class_cannot_be_instantiated(self) -> None:
        """:class:`QuditDirective` is abstract."""
        with pytest.raises(TypeError, match="abstract"):
            QuditDirective("dummy", [3])

    @pytest.mark.parametrize(
        ("dims", "widths"),
        [
            ((2,), (1,)),
            ((3,), (2,)),
            ((2, 3), (1, 2)),
            ((5, 8, 4), (3, 3, 2)),
            ((7, 7), (3, 3)),
        ],
    )
    def test_dims_widths_and_qubit_count(
        self,
        dims: tuple[int, ...],
        widths: tuple[int, ...],
    ) -> None:
        """The Qiskit qubit count is the sum of the encoding widths."""
        directive = DummyDirective(dims)
        assert directive.dims == dims
        assert directive.widths == widths
        assert directive.num_qudits == len(dims)
        assert directive.num_qubits == sum(widths)

    def test_dimensions_are_coerced_to_plain_ints(self) -> None:
        """Numpy integers survive as ordinary ``int`` dimensions."""
        directive = DummyDirective([np.int64(3), np.int32(4)])
        assert directive.dims == (3, 4)
        assert all(type(dim) is int for dim in directive.dims)

    @pytest.mark.parametrize(
        ("dim", "error", "message"),
        [
            (1, ValueError, r"dim must be >= 2"),
            (0, ValueError, r"dim must be >= 2"),
            (True, TypeError, "dim must be an integer"),
            (2.5, TypeError, "dim must be an integer"),
        ],
    )
    def test_invalid_dimensions_are_rejected(
        self,
        dim: Any,
        error: type[Exception],
        message: str,
    ) -> None:
        """A single bad entry in ``dims`` rejects the directive."""
        with pytest.raises(error, match=message):
            DummyDirective([3, dim])

    def test_classical_widths_default_to_zero(self) -> None:
        """A purely quantum directive has no classical operands."""
        directive = DummyDirective([3])
        assert directive.num_clbytes == 0
        assert directive.num_clbits == 0

    def test_classical_widths_are_stored(self) -> None:
        """Clbyte and clbit counts are kept independently."""
        directive = DummyDirective([3, 2], num_clbytes=2, num_clbits=3)
        assert directive.num_clbytes == 2
        assert directive.num_clbits == 3

    def test_params_and_label_reach_the_qiskit_instruction(self) -> None:
        """Directives are inspectable like any Qiskit instruction."""
        directive = DummyDirective([3], params=(1, "a"), label="tag")
        assert directive.name == "dummy"
        assert list(directive.params) == [1, "a"]
        assert directive.label == "tag"


class TestCheckTargets:
    """``_check_targets`` is the shared validation entry point."""

    def test_matching_targets_are_accepted(self) -> None:
        """Correct arity and dimensions let ``apply`` run."""
        circuit, qudits, clbytes = build_targets([3, 2], [3, 2])
        directive = DummyDirective([3, 2], num_clbytes=2, num_clbits=3)
        directive.apply(circuit, qudits, clbytes)
        assert len(directive.calls) == 1
        assert directive.calls[0][0] is circuit
        assert directive.calls[0][1] == tuple(qudits)
        assert directive.calls[0][2] == tuple(clbytes)

    def test_too_few_qudits_are_rejected(self) -> None:
        """The qudit arity must match exactly."""
        circuit, qudits, _ = build_targets([3, 3])
        directive = DummyDirective([3, 3])
        with pytest.raises(
            QuditCircuitError,
            match=r"'dummy' acts on 2 qudit\(s\), got 1",
        ):
            directive.apply(circuit, qudits[:1], [])

    def test_too_many_qudits_are_rejected(self) -> None:
        """Extra qudits are an error too."""
        circuit, qudits, _ = build_targets([3, 3])
        directive = DummyDirective([3])
        with pytest.raises(
            QuditCircuitError,
            match=r"'dummy' acts on 1 qudit\(s\), got 2",
        ):
            directive.apply(circuit, qudits, [])

    def test_clbyte_count_mismatch_is_rejected(self) -> None:
        """The clbyte arity must match exactly."""
        circuit, qudits, _ = build_targets([3])
        directive = DummyDirective([3], num_clbytes=1, num_clbits=2)
        with pytest.raises(
            QuditCircuitError,
            match=r"'dummy' acts on 1 clbyte\(s\), got 0",
        ):
            directive.apply(circuit, qudits, [])

    def test_unexpected_clbytes_are_rejected(self) -> None:
        """A directive without classical operands refuses clbytes."""
        circuit, qudits, clbytes = build_targets([3], [3])
        directive = DummyDirective([3])
        with pytest.raises(
            QuditCircuitError,
            match=r"'dummy' acts on 0 clbyte\(s\), got 1",
        ):
            directive.apply(circuit, qudits, clbytes)

    def test_a_dimension_mismatch_names_the_position(self) -> None:
        """The failing operand position is reported."""
        circuit, qudits, _ = build_targets([3, 8])
        directive = DummyDirective([3, 4])
        with pytest.raises(
            QuditCircuitError,
            match="expects a 4-level qudit at position 1, got 8 levels",
        ):
            directive.apply(circuit, qudits, [])


class TestQuditBarrier:
    """:class:`QuditBarrier`."""

    def test_name_arity_and_directive_flag(self) -> None:
        """A barrier is a Qiskit directive, so metrics skip it."""
        barrier = QuditBarrier([3, 2])
        assert barrier.name == "barrier"
        assert barrier.num_qudits == 2
        assert barrier.num_qubits == 3
        assert barrier.num_clbytes == 0
        assert barrier.label is None
        assert barrier._directive is True

    def test_the_label_is_stored(self) -> None:
        """An explicit label is kept on the directive."""
        assert QuditBarrier([3], label="sync").label == "sync"

    @pytest.mark.parametrize(
        ("dims", "num_qubits"),
        [((2,), 1), ((3,), 2), ((3, 2), 3), ((5, 8), 6)],
    )
    def test_apply_emits_one_barrier_over_every_encoding_qubit(
        self,
        dims: tuple[int, ...],
        num_qubits: int,
    ) -> None:
        """Exactly one Qiskit barrier spans all the target qubits."""
        circuit, qudits, _ = build_targets(dims)
        QuditBarrier(dims).apply(circuit, qudits, [])
        assert operation_names(circuit) == ["barrier"]
        entry = circuit.data[0]
        assert entry.qubits == tuple(circuit.qubits)
        assert len(entry.qubits) == num_qubits
        assert entry.clbits == ()

    def test_apply_forwards_the_label_to_qiskit(self) -> None:
        """The emitted Qiskit barrier carries the directive label."""
        circuit, qudits, _ = build_targets([3])
        QuditBarrier([3], label="sync").apply(circuit, qudits, [])
        assert circuit.data[0].operation.label == "sync"

    def test_apply_without_targets_emits_nothing(self) -> None:
        """An empty barrier must not fall back to every qubit."""
        # NOTE: `QuantumCircuit.barrier()` with no argument barriers
        # the whole circuit; the guard in `apply` prevents that.
        circuit = QuantumCircuit(2)
        QuditBarrier([]).apply(circuit, [], [])
        assert len(circuit.data) == 0


class TestQuditReset:
    """:class:`QuditReset`."""

    def test_name_and_arity(self) -> None:
        """A reset is quantum-only."""
        reset = QuditReset([3, 2])
        assert reset.name == "reset"
        assert reset.num_qudits == 2
        assert reset.num_qubits == 3
        assert reset.num_clbytes == 0

    @pytest.mark.parametrize(
        "dims",
        [(2,), (3,), (5,), (3, 2), (8, 4)],
    )
    def test_apply_resets_every_encoding_qubit_in_order(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """One primitive ``reset`` per encoding qubit, LSB first."""
        circuit, qudits, _ = build_targets(dims)
        QuditReset(dims).apply(circuit, qudits, [])
        expected_qubits = [qubit for qudit in qudits for qubit in qudit.qubits]
        assert operation_names(circuit) == ["reset"] * len(expected_qubits)
        assert [entry.qubits[0] for entry in circuit.data] == expected_qubits
        assert expected_qubits == list(circuit.qubits)


class TestQuditMeasure:
    """:class:`QuditMeasure`."""

    @pytest.mark.parametrize(
        ("dims", "num_bits"),
        [((2,), 1), ((3,), 2), ((3, 2), 3), ((5, 8), 6), ((7, 7), 6)],
    )
    def test_name_and_classical_width(
        self,
        dims: tuple[int, ...],
        num_bits: int,
    ) -> None:
        """One clbyte per qudit and one clbit per encoding qubit."""
        measure = QuditMeasure(dims)
        assert measure.name == "measure"
        assert measure.num_qudits == len(dims)
        assert measure.num_clbytes == len(dims)
        assert measure.num_qubits == num_bits
        assert measure.num_clbits == num_bits

    def test_apply_pairs_qubits_and_clbits_in_matching_order(self) -> None:
        """Qubit ``j`` of a qudit goes to clbit ``j`` of its clbyte."""
        dims = (3, 2)
        circuit, qudits, clbytes = build_targets(dims, dims)
        QuditMeasure(dims).apply(circuit, qudits, clbytes)
        assert operation_names(circuit) == ["measure"] * 3
        pairs = [(entry.qubits[0], entry.clbits[0]) for entry in circuit.data]
        assert pairs == [
            (qudits[0].qubits[0], clbytes[0].clbits[0]),
            (qudits[0].qubits[1], clbytes[0].clbits[1]),
            (qudits[1].qubits[0], clbytes[1].clbits[0]),
        ]

    def test_a_clbyte_that_is_too_narrow_is_rejected(self) -> None:
        """A qutrit cannot be measured into a one-bit clbyte."""
        circuit, qudits, clbytes = build_targets([3], [2])
        measure = QuditMeasure([3])
        with pytest.raises(
            QuditCircuitError,
            match=r"cannot measure a 3-level qudit \(2 qubit\(s\)\) into",
        ):
            measure.apply(circuit, qudits, clbytes)

    def test_a_rejected_measurement_leaves_the_circuit_untouched(
        self,
    ) -> None:
        """Validation runs fully before anything is emitted."""
        circuit, qudits, clbytes = build_targets([3, 3], [3, 2])
        measure = QuditMeasure([3, 3])
        with pytest.raises(QuditCircuitError, match="cannot measure"):
            measure.apply(circuit, qudits, clbytes)
        assert len(circuit.data) == 0


class TestQuditInitializeLevels:
    """:class:`QuditInitializeLevels`."""

    def test_name_values_and_params(self) -> None:
        """The prepared levels are exposed twice, as Qiskit params."""
        directive = QuditInitializeLevels([3, 2], [2, 1])
        assert directive.name == "initialize"
        assert directive.values == (2, 1)
        assert list(directive.params) == [2, 1]
        assert directive.num_qudits == 2
        assert directive.num_qubits == 3
        assert directive.num_clbytes == 0

    @pytest.mark.parametrize(
        ("dim", "level"),
        [(3, -1), (2, 2), (5, 5), (8, 8)],
    )
    def test_out_of_range_levels_are_rejected(
        self,
        dim: int,
        level: int,
    ) -> None:
        """A level must lie in ``[0, d - 1]``."""
        with pytest.raises(
            ValueError,
            match="state for qudit 0 must be in",
        ):
            QuditInitializeLevels([dim], [level])

    def test_the_highest_encodable_level_is_still_rejected(self) -> None:
        """Padding levels are outside the qudit subspace."""
        # NOTE: d=3 encodes into 2 qubits, so level 3 is representable
        # but invalid; the check is against `dim`, not `2**width`.
        with pytest.raises(ValueError, match=r"must be in \[0, 2\]"):
            QuditInitializeLevels([3], [3])

    def test_one_level_per_qudit_is_required(self) -> None:
        """The two sequences must have the same length."""
        with pytest.raises(
            ValueError,
            match=r"got 1 state\(s\) for 2 qudit\(s\)",
        ):
            QuditInitializeLevels([3, 2], [1])

    @pytest.mark.parametrize(
        ("dims", "levels", "labels"),
        [
            ((3,), (0,), ["00"]),
            ((3,), (2,), ["10"]),
            ((2,), (1,), ["1"]),
            ((5,), (4,), ["100"]),
            ((3, 2), (2, 1), ["10", "1"]),
            ((8, 4), (5, 2), ["101", "10"]),
        ],
    )
    def test_apply_initialises_each_qudit_from_an_msb_first_label(
        self,
        dims: tuple[int, ...],
        levels: tuple[int, ...],
        labels: list[str],
    ) -> None:
        """One Qiskit ``initialize`` per qudit, with its bit-string."""
        circuit, qudits, _ = build_targets(dims)
        QuditInitializeLevels(dims, levels).apply(circuit, qudits, [])
        assert operation_names(circuit) == ["initialize"] * len(dims)
        for entry, qudit, label in zip(
            circuit.data,
            qudits,
            labels,
            strict=True,
        ):
            assert "".join(entry.operation.params) == label
            assert entry.qubits == tuple(qudit.qubits)
            assert entry.clbits == ()


class TestQuditStatePreparation:
    """:class:`QuditStatePreparation`."""

    def test_name_and_logical_amplitudes(self) -> None:
        """The *logical* vector is stored, not the embedded one."""
        directive = QuditStatePreparation([3], [0.6, 0.0, 0.8])
        assert directive.name == "initialize"
        assert directive.num_qudits == 1
        assert directive.num_qubits == 2
        assert directive.num_clbytes == 0
        assert directive.amplitudes.shape == (3,)
        assert_allclose(
            directive.amplitudes,
            np.array([0.6, 0.0, 0.8], dtype=np.complex128),
            message="amplitudes must stay logical",
        )

    def test_params_hold_the_logical_vector(self) -> None:
        """Introspection through Qiskit sees the same vector."""
        directive = QuditStatePreparation([3], [0.6, 0.0, 0.8])
        assert len(directive.params) == 1
        assert_allclose(
            np.asarray(directive.params[0], dtype=np.complex128),
            np.array([0.6, 0.0, 0.8], dtype=np.complex128),
        )

    def test_a_wrong_length_vector_is_rejected(self) -> None:
        """``prod(dims)`` amplitudes are required."""
        with pytest.raises(
            ValueError,
            match=r"expected 6 amplitude\(s\)",
        ):
            QuditStatePreparation([3, 2], [1.0, 0.0, 0.0])

    def test_a_non_normalised_vector_is_rejected(self) -> None:
        """The state must be a unit vector."""
        with pytest.raises(ValueError, match="not normalised"):
            QuditStatePreparation([3], [1.0, 1.0, 0.0])

    def test_apply_emits_a_single_embedded_initialize(self) -> None:
        """One ``initialize`` covers every target qubit at once."""
        dims = (3, 2)
        amplitudes = np.zeros(6, dtype=np.complex128)
        amplitudes[3] = 1.0
        circuit, qudits, _ = build_targets(dims)
        QuditStatePreparation(dims, amplitudes).apply(circuit, qudits, [])

        assert operation_names(circuit) == ["initialize"]
        entry = circuit.data[0]
        assert entry.qubits == tuple(circuit.qubits)
        assert entry.clbits == ()

        # Logical index 3 of a (3, 2) system is qutrit level 0 and
        # qubit level 1, i.e. encoded index 0 + 1 * 4 = 4.
        expected = np.zeros(8, dtype=np.complex128)
        expected[4] = 1.0
        assert_allclose(
            np.asarray(entry.operation.params, dtype=np.complex128),
            expected,
            message="the embedded vector is wrong",
        )

    def test_invalid_basis_states_get_zero_amplitude(self) -> None:
        """The padding entries of the encoded vector are exactly 0."""
        amplitudes = np.full(3, 1 / np.sqrt(3), dtype=np.complex128)
        circuit, qudits, _ = build_targets([3])
        QuditStatePreparation([3], amplitudes).apply(circuit, qudits, [])
        params = np.asarray(
            circuit.data[0].operation.params,
            dtype=np.complex128,
        )
        assert params.shape == (4,)
        assert_allclose(params[:3], amplitudes)
        assert params[3] == 0
