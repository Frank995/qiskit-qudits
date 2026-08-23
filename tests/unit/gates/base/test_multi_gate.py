"""Software-contract tests for :class:`QuditMultiGate`.

``QuditMultiGate`` is abstract in practice: :meth:`_build_unitary` is
still inherited unimplemented from ``QuditGateMixin``, so everything
here is exercised through the minimal concrete subclass defined below.
The bookkeeping the shared ``QuditMultiGateMixin`` provides (dims
validation, qubit ranges, strides, leakage counting) is covered
through the same dummy.

Only the contract is covered: register layout, constructor
validation, Qiskit interoperability and the not-implemented inverse.
The physics lives in ``tests/quantum``.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from qiskit.circuit import Gate, QuantumCircuit

from qiskit_qudits.gates.base.mixins import QuditMultiGateMixin
from qiskit_qudits.gates.base.multigate import QuditMultiGate

#: ``(dims, total qubit count)`` pairs, worked out by hand so the
#: tests never validate the library against itself.
DIMS_TO_NUM_QUBITS: list[tuple[tuple[int, ...], int]] = [
    ((2, 2), 2),
    ((3, 2), 3),
    ((2, 3), 3),
    ((4, 4), 4),
    ((3, 5), 5),
    ((2, 3, 5), 6),
]

#: ``(dims, fills_hilbert_space)`` pairs.
DIMS_TO_FILLING: list[tuple[tuple[int, ...], bool]] = [
    ((2, 2), True),
    ((2, 4), True),
    ((8, 2, 4), True),
    ((2, 3), False),
    ((3, 3), False),
    ((5, 8), False),
]

#: ``(dims, 2**n - prod(dims))`` pairs.
DIMS_TO_INVALID_STATES: list[tuple[tuple[int, ...], int]] = [
    ((2, 2), 0),
    ((2, 3), 2),
    ((3, 3), 7),
    ((3, 5), 17),
    ((4, 4), 0),
]

#: ``(dims, per-qudit strides)`` pairs (qudit 0 first).
DIMS_TO_STRIDES: list[tuple[tuple[int, ...], list[int]]] = [
    ((2, 2), [1, 2]),
    ((3, 3), [1, 4]),
    ((5, 2), [1, 8]),
    ((2, 3, 5), [1, 2, 8]),
]


class _DummyMultiGate(QuditMultiGate):
    """Minimal concrete :class:`QuditMultiGate` used by these tests."""

    def __init__(
        self,
        dims: Any,
        *,
        name: str = "dummy",
        params: list[Any] | None = None,
        label: str | None = None,
    ) -> None:
        """Create the dummy gate.

        Args:
            dims: Candidate dimensions, validated by the base class.
            name: Qiskit gate name.
            params: Gate parameters forwarded to Qiskit.
            label: Optional display label.
        """
        super().__init__(
            name,
            dims,
            [] if params is None else params,
            label=label,
        )

    def _define(self) -> None:
        """Assign a trivial (empty) qubit-level circuit."""
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        """Return the identity on the full Hilbert space."""
        return np.eye(self.hilbert_dim, dtype=np.complex128)


class TestMixinDefaults:
    """Class-level knobs of :class:`QuditMultiGateMixin`."""

    def test_default_qudit_bounds(self) -> None:
        """The mixin defaults span the library-wide register range."""
        assert QuditMultiGateMixin.MIN_QUDITS == 2
        assert QuditMultiGateMixin.MAX_QUDITS == 8

    def test_validate_dims_is_a_classmethod(self) -> None:
        """It must be callable without an instance."""
        raw = inspect.getattr_static(QuditMultiGateMixin, "_validate_dims")
        assert isinstance(raw, classmethod)

    def test_validate_dims_returns_plain_ints(self) -> None:
        """NumPy dimensions are coerced on the way out."""
        result = QuditMultiGateMixin._validate_dims(np.array([2, 3]))
        assert result == (2, 3)
        assert all(type(dim) is int for dim in result)


class TestMultiDimensionBookkeeping:
    """Everything the base class derives from the qudit dimensions."""

    def test_dims_echo_the_constructor_argument(self) -> None:
        """``dims`` is the validated register, in order."""
        gate = _DummyMultiGate([3, 5])
        assert gate.dims == (3, 5)
        assert gate.num_qudits == 2
        assert all(type(dim) is int for dim in gate.dims)

    @pytest.mark.parametrize(("dims", "expected"), DIMS_TO_NUM_QUBITS)
    def test_num_qubits_is_the_sum_of_the_encodings(
        self,
        dims: tuple[int, ...],
        expected: int,
    ) -> None:
        """Each qudit contributes ``ceil(log2 d_i)`` qubits."""
        assert _DummyMultiGate(dims).num_qubits == expected

    @pytest.mark.parametrize(("dims", "expected"), DIMS_TO_NUM_QUBITS)
    def test_hilbert_dim_is_two_to_the_num_qubits(
        self,
        dims: tuple[int, ...],
        expected: int,
    ) -> None:
        """``hilbert_dim`` and ``num_qubits`` stay consistent."""
        assert _DummyMultiGate(dims).hilbert_dim == 2**expected

    @pytest.mark.parametrize(("dims", "expected"), DIMS_TO_FILLING)
    def test_fills_hilbert_space_needs_every_dim_to_be_a_power_of_two(
        self,
        dims: tuple[int, ...],
        expected: bool,
    ) -> None:
        """A single leaky qudit spoils the whole register."""
        assert _DummyMultiGate(dims).fills_hilbert_space is expected

    @pytest.mark.parametrize(("dims", "expected"), DIMS_TO_INVALID_STATES)
    def test_num_invalid_states_counts_the_leftover_basis_states(
        self,
        dims: tuple[int, ...],
        expected: int,
    ) -> None:
        """``num_invalid_states`` is ``2**n - prod(dims)``."""
        assert _DummyMultiGate(dims).num_invalid_states == expected

    def test_qudit_ranges_partition_the_qubits(self) -> None:
        """Qudit ``i`` owns a contiguous little-endian qubit slice."""
        gate = _DummyMultiGate([2, 3, 5])
        assert gate._qudit_range(0) == range(1)
        assert gate._qudit_range(1) == range(1, 3)
        assert gate._qudit_range(2) == range(3, 6)

    def test_qudit_range_accepts_negative_indices(self) -> None:
        """Negative indices address the register from the end."""
        gate = _DummyMultiGate([2, 3, 5])
        assert gate._qudit_range(-1) == range(3, 6)
        assert gate._qudit_range(-2) == range(1, 3)

    @pytest.mark.parametrize(("dims", "expected"), DIMS_TO_STRIDES)
    def test_strides_follow_the_little_endian_layout(
        self,
        dims: tuple[int, ...],
        expected: list[int],
    ) -> None:
        """Each stride is the Hilbert size of the preceding qudits."""
        assert _DummyMultiGate(dims)._compute_strides() == expected

    def test_num_qubits_is_not_a_constructor_parameter(self) -> None:
        """The caller cannot choose the encoding width."""
        parameters = list(
            inspect.signature(QuditMultiGate.__init__).parameters,
        )
        assert "num_qubits" not in parameters
        assert parameters == ["self", "name", "dims", "params", "label"]


class TestMultiDimsValidation:
    """Constructor-level validation of the qudit dimensions."""

    def test_fewer_than_two_qudits_are_rejected(self) -> None:
        """A multi-qudit gate needs at least two operands."""
        with pytest.raises(ValueError, match="at least 2 qudits"):
            _DummyMultiGate([3])

    def test_empty_dims_are_rejected(self) -> None:
        """An empty register fails before the arity check."""
        with pytest.raises(ValueError, match="dims must not be empty"):
            _DummyMultiGate([])

    def test_more_than_max_qudits_are_rejected(self) -> None:
        """The register may hold at most ``MAX_QUDITS`` qudits."""
        with pytest.raises(ValueError, match="at most 8"):
            _DummyMultiGate([2] * 9)

    @pytest.mark.parametrize("dim", [1, 17])
    def test_out_of_range_dims_are_rejected(self, dim: int) -> None:
        """A single bad entry rejects the whole register."""
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            _DummyMultiGate([3, dim])

    @pytest.mark.parametrize(
        "dims",
        [[3, 2.5], [3.0, 2]],
        ids=["float-element", "float-first"],
    )
    def test_non_integer_elements_are_rejected(self, dims: list[Any]) -> None:
        """Non-integer dimensions raise ``TypeError``."""
        with pytest.raises(TypeError, match="dim must be an integer"):
            _DummyMultiGate(dims)

    @pytest.mark.parametrize(
        "dims",
        [5, None, np.eye(2)],
        ids=["scalar", "none", "matrix"],
    )
    def test_non_vector_dims_are_rejected(self, dims: Any) -> None:
        """``dims`` must be a one-dimensional sequence."""
        with pytest.raises(TypeError, match="dims must be a 1-D vector"):
            _DummyMultiGate(dims)


class TestMultiQiskitInterface:
    """What Qiskit itself sees when handed a multi-qudit gate."""

    def test_is_a_qiskit_gate_subclass(self) -> None:
        """Transpiler compatibility rests on this inheritance."""
        assert issubclass(QuditMultiGate, Gate)
        assert isinstance(_DummyMultiGate([3, 3]), Gate)

    def test_gates_carry_no_classical_bits(self) -> None:
        """A gate is purely quantum."""
        assert _DummyMultiGate([3, 3]).num_clbits == 0

    def test_name_params_and_label_are_forwarded(self) -> None:
        """The Qiskit metadata reaches the base class untouched."""
        gate = _DummyMultiGate(
            [3, 3],
            name="my-gate",
            params=[0.25],
            label="tag",
        )
        assert gate.name == "my-gate"
        assert gate.params == [0.25]
        assert gate.label == "tag"


class TestAbstractness:
    """The base class is abstract, but only partially so."""

    def test_qudit_multi_gate_cannot_be_instantiated(self) -> None:
        """``_build_unitary`` keeps the base class abstract."""
        with pytest.raises(TypeError, match="abstract"):
            QuditMultiGate("g", [3, 3], [])

    def test_only_build_unitary_is_left_abstract(self) -> None:
        """``_define`` is satisfied by Qiskit's no-op implementation."""
        # NOTE: ``QuditGateMixin`` declares both ``_define`` and
        # ``_build_unitary`` abstract, but ``Instruction`` comes first
        # in the MRO and already provides a no-op ``_define``, so ABC
        # never sees it as missing.
        assert QuditMultiGate.__abstractmethods__ == frozenset(
            {"_build_unitary"},
        )


class TestInverse:
    """The base implementation refuses to guess an inverse."""

    def test_inverse_raises_not_implemented(self) -> None:
        """The error names the offending subclass."""
        with pytest.raises(
            NotImplementedError,
            match="_DummyMultiGate must implement inverse",
        ):
            _DummyMultiGate([3, 3]).inverse()

    def test_inverse_raises_not_implemented_when_annotated(self) -> None:
        """``annotated=True`` does not unlock a fallback."""
        with pytest.raises(
            NotImplementedError,
            match="_DummyMultiGate must implement inverse",
        ):
            _DummyMultiGate([3, 3]).inverse(annotated=True)
