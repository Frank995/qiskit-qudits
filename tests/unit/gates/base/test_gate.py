"""Software-contract tests for :class:`QuditGate`.

``QuditGate`` is abstract in practice: :meth:`_build_unitary` is still
inherited unimplemented from ``QuditGateMixin``, so everything here is
exercised through the minimal concrete subclass defined below.

Only the contract is covered: dimension bookkeeping, constructor
validation, parameter/label passthrough, Qiskit interoperability and
the not-implemented inverse. The physics lives in ``tests/quantum``.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from qiskit.circuit import Gate, QuantumCircuit

from qiskit_qudits.gates.base.gate import QuditGate
from tests.helpers import (
    NON_POWER_OF_TWO_DIMS,
    POWER_OF_TWO_DIMS,
    parametrize_dims,
)

#: ``(dimension, ceil(log2 d))`` pairs, worked out by hand so the test
#: does not re-implement ``qubits_per_qudit``.
DIM_TO_NUM_QUBITS: list[tuple[int, int]] = [
    (2, 1),
    (3, 2),
    (4, 2),
    (5, 3),
    (7, 3),
    (8, 3),
    (9, 4),
    (15, 4),
    (16, 4),
]

#: ``(dimension, 2**n)`` pairs for the encoded Hilbert space.
DIM_TO_HILBERT_DIM: list[tuple[int, int]] = [
    (2, 2),
    (3, 4),
    (5, 8),
    (7, 8),
    (11, 16),
    (16, 16),
]

#: ``(dimension, 2**n - d)`` pairs.
DIM_TO_INVALID_STATES: list[tuple[int, int]] = [
    (2, 0),
    (3, 1),
    (4, 0),
    (5, 3),
    (7, 1),
    (8, 0),
    (11, 5),
    (16, 0),
]

#: ``(dimension, fills_hilbert_space)`` pairs.
DIM_TO_FILLING: list[tuple[int, bool]] = [
    *[(dim, True) for dim in POWER_OF_TWO_DIMS],
    *[(dim, False) for dim in NON_POWER_OF_TWO_DIMS],
]


class _DummyQuditGate(QuditGate):
    """Minimal concrete :class:`QuditGate` used by these tests."""

    def __init__(
        self,
        dim: Any,
        *,
        name: str = "dummy",
        params: list[Any] | None = None,
        label: str | None = None,
    ) -> None:
        """Create the dummy gate.

        Args:
            dim: Candidate dimension, validated by the base class.
            name: Qiskit gate name.
            params: Gate parameters forwarded to Qiskit.
            label: Optional display label.
        """
        super().__init__(
            name,
            dim,
            [] if params is None else params,
            label=label,
        )

    def _define(self) -> None:
        """Assign a trivial (empty) qubit-level circuit."""
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        """Return the identity on the full Hilbert space."""
        return np.eye(self.hilbert_dim, dtype=np.complex128)


class _NoDefineQuditGate(QuditGate):
    """Concrete gate that deliberately never overrides ``_define``."""

    def __init__(self, dim: Any) -> None:
        """Create the gate with a fixed name and no parameters.

        Args:
            dim: Candidate dimension, validated by the base class.
        """
        super().__init__("no-define", dim, [])

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        """Return the identity on the full Hilbert space."""
        return np.eye(self.hilbert_dim, dtype=np.complex128)


class TestDimensionBookkeeping:
    """Everything the base class derives from the qudit dimension."""

    @parametrize_dims()
    def test_dim_returns_the_requested_dimension(self, dim: int) -> None:
        """``dim`` echoes the constructor argument."""
        assert _DummyQuditGate(dim).dim == dim

    @pytest.mark.parametrize(("dim", "expected"), DIM_TO_NUM_QUBITS)
    def test_num_qubits_is_ceil_log2_of_the_dimension(
        self,
        dim: int,
        expected: int,
    ) -> None:
        """The qubit count is derived, never supplied."""
        assert _DummyQuditGate(dim).num_qubits == expected

    @pytest.mark.parametrize(("dim", "expected"), DIM_TO_HILBERT_DIM)
    def test_hilbert_dim_is_the_encoded_space_size(
        self,
        dim: int,
        expected: int,
    ) -> None:
        """``hilbert_dim`` is the size of the qubit register space."""
        assert _DummyQuditGate(dim).hilbert_dim == expected

    @parametrize_dims()
    def test_hilbert_dim_is_two_to_the_num_qubits(self, dim: int) -> None:
        """``hilbert_dim`` and ``num_qubits`` stay consistent."""
        gate = _DummyQuditGate(dim)
        assert gate.hilbert_dim == 2**gate.num_qubits

    @pytest.mark.parametrize(("dim", "expected"), DIM_TO_FILLING)
    def test_fills_hilbert_space_flags_powers_of_two(
        self,
        dim: int,
        expected: bool,
    ) -> None:
        """Only power-of-two dimensions fill the Hilbert space."""
        assert _DummyQuditGate(dim).fills_hilbert_space is expected

    @pytest.mark.parametrize(("dim", "expected"), DIM_TO_INVALID_STATES)
    def test_num_invalid_states_counts_the_leftover_basis_states(
        self,
        dim: int,
        expected: int,
    ) -> None:
        """``num_invalid_states`` is ``2**n - d``."""
        assert _DummyQuditGate(dim).num_invalid_states == expected

    def test_num_qubits_is_not_a_constructor_parameter(self) -> None:
        """The caller cannot choose the encoding width."""
        parameters = list(inspect.signature(QuditGate.__init__).parameters)
        assert "num_qubits" not in parameters
        assert parameters == ["self", "name", "dim", "params", "label"]

    def test_passing_num_qubits_is_rejected(self) -> None:
        """A ``num_qubits`` keyword is not accepted anywhere."""
        with pytest.raises(TypeError, match="num_qubits"):
            _DummyQuditGate(3, num_qubits=1)


class TestDimValidation:
    """Constructor-level validation of the qudit dimension."""

    @pytest.mark.parametrize(
        "dim",
        [True, False, 3.0, 2.5, "3", None, 3j, [3], np.float64(4.0)],
        ids=[
            "true",
            "false",
            "float-integral",
            "float",
            "str",
            "none",
            "complex",
            "list",
            "numpy-float",
        ],
    )
    def test_non_integer_dim_raises_type_error(self, dim: Any) -> None:
        """Anything that is not integer-like is a ``TypeError``."""
        with pytest.raises(TypeError, match="dim must be an integer"):
            _DummyQuditGate(dim)

    @pytest.mark.parametrize("dim", [-5, 0, 1, 17, 100])
    def test_out_of_range_dim_raises_value_error(self, dim: int) -> None:
        """``d`` must lie in ``[MIN_DIM, MAX_DIM]``."""
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            _DummyQuditGate(dim)

    @pytest.mark.parametrize("dim", [2, 16])
    def test_range_bounds_are_inclusive(self, dim: int) -> None:
        """The smallest and largest dimensions are accepted."""
        assert _DummyQuditGate(dim).dim == dim

    @pytest.mark.parametrize(
        "dim",
        [np.int8(3), np.int32(5), np.int64(8), np.uint8(16)],
        ids=["int8", "int32", "int64", "uint8"],
    )
    def test_numpy_integers_are_accepted(self, dim: Any) -> None:
        """NumPy integers are valid dimensions."""
        assert _DummyQuditGate(dim).dim == int(dim)

    def test_dim_is_coerced_to_a_plain_int(self) -> None:
        """The stored dimension is never a NumPy scalar."""
        assert type(_DummyQuditGate(np.int64(5)).dim) is int


class TestQiskitInterface:
    """What Qiskit itself sees when handed a qudit gate."""

    def test_is_a_qiskit_gate_subclass(self) -> None:
        """Transpiler compatibility rests on this inheritance."""
        assert issubclass(QuditGate, Gate)

    def test_instances_are_qiskit_gates(self) -> None:
        """Instances pass an ``isinstance`` check against ``Gate``."""
        assert isinstance(_DummyQuditGate(3), Gate)

    def test_gates_carry_no_classical_bits(self) -> None:
        """A gate is purely quantum."""
        assert _DummyQuditGate(3).num_clbits == 0

    def test_name_is_forwarded(self) -> None:
        """The name reaches Qiskit's ``name`` property untouched."""
        assert _DummyQuditGate(3, name="my-gate").name == "my-gate"

    def test_params_default_to_an_empty_list(self) -> None:
        """A gate without parameters exposes ``[]``."""
        assert _DummyQuditGate(3).params == []

    def test_params_are_forwarded(self) -> None:
        """Parameters reach Qiskit's ``params`` property in order."""
        assert _DummyQuditGate(3, params=[0.25, -1.0]).params == [0.25, -1.0]

    @pytest.mark.parametrize("label", [None, "custom-label"])
    def test_label_is_forwarded(self, label: str | None) -> None:
        """The label is stored verbatim, ``None`` included."""
        assert _DummyQuditGate(3, label=label).label == label


class TestAbstractness:
    """The base class is abstract, but only partially so."""

    def test_qudit_gate_cannot_be_instantiated(self) -> None:
        """``_build_unitary`` keeps the base class abstract."""
        with pytest.raises(TypeError, match="abstract"):
            QuditGate("g", 3, [])

    def test_only_build_unitary_is_left_abstract(self) -> None:
        """``_define`` is satisfied by Qiskit's no-op implementation."""
        # NOTE: ``QuditGateMixin`` declares both ``_define`` and
        # ``_build_unitary`` abstract, but ``Instruction`` comes first
        # in the MRO and already provides a no-op ``_define``, so ABC
        # never sees it as missing.
        assert QuditGate.__abstractmethods__ == frozenset({"_build_unitary"})

    def test_subclass_without_define_is_instantiable(self) -> None:
        """Forgetting ``_define`` is not caught at construction."""
        assert _NoDefineQuditGate(3).num_qubits == 2

    def test_subclass_without_define_has_no_definition(self) -> None:
        """The silently inherited no-op leaves ``definition`` unset."""
        # NOTE: this is the visible consequence of the previous test:
        # the mixin's raising ``_define`` is shadowed, so the failure
        # mode is a ``None`` definition rather than an exception.
        assert _NoDefineQuditGate(3).definition is None


class TestInverse:
    """The base implementation refuses to guess an inverse."""

    def test_inverse_raises_not_implemented(self) -> None:
        """The error names the offending subclass."""
        with pytest.raises(
            NotImplementedError,
            match="_DummyQuditGate must implement inverse",
        ):
            _DummyQuditGate(3).inverse()

    def test_inverse_raises_not_implemented_when_annotated(self) -> None:
        """``annotated=True`` does not unlock a fallback."""
        with pytest.raises(
            NotImplementedError,
            match="_DummyQuditGate must implement inverse",
        ):
            _DummyQuditGate(3).inverse(annotated=True)
