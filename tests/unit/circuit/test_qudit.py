"""Unit tests for :mod:`qiskit_qudits.circuit.qudit`.

Covers the :class:`~qiskit_qudits.circuit.qudit.Qudit` wire handle and
the :class:`~qiskit_qudits.circuit.qudit.QuditRegister` container:
construction, validation, identity semantics, the qubit layout and the
container protocol. Nothing here runs a simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit.circuit import QuantumRegister, Qubit

from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.circuit.qudit import Qudit, QuditRegister
from tests.helpers import ALL_DIMS, parametrize_dims

if TYPE_CHECKING:
    from collections.abc import Callable

#: ``ceil(log2 d)`` for every dimension the suite exercises. Hard-coded
#: so the tests never validate the library against itself.
WIDTHS = {2: 1, 3: 2, 4: 2, 5: 3, 7: 3, 8: 3}

assert set(WIDTHS) == set(ALL_DIMS), "WIDTHS is out of sync with ALL_DIMS"

#: Dimensions whose encoding exactly fills the qubit Hilbert space.
POW2 = [2, 4, 8]

#: Dimensions that leave invalid basis states behind.
NON_POW2 = [3, 5, 7]

#: Dimensions wide enough (>= 2 qubits) to have duplicate qubits.
WIDE = [3, 4, 5, 7, 8]


def fresh_qubits(count: int) -> list[Qubit]:
    """Return ``count`` distinct registerless qubits.

    Args:
        count: How many qubits to create.

    Returns:
        A list of freshly created, pairwise distinct qubits.
    """
    return [Qubit() for _ in range(count)]


class TestQuditDimensionValidation:
    """Dimension checks performed by :class:`Qudit`."""

    @pytest.mark.parametrize("dim", [-3, 0, 1])
    def test_dimension_below_two_is_rejected(self, dim: int) -> None:
        """A qudit needs at least two levels."""
        with pytest.raises(ValueError, match=r"dim must be >= 2"):
            Qudit(dim)

    @pytest.mark.parametrize("dim", [True, False, 3.0, 2.5, "3", None])
    def test_non_integer_dimension_is_rejected(self, dim: Any) -> None:
        """Booleans, floats, strings and ``None`` are not dimensions."""
        with pytest.raises(TypeError, match="dim must be an integer"):
            Qudit(dim)

    def test_numpy_integers_are_accepted_and_coerced(self) -> None:
        """A numpy integer is a valid dimension and becomes an int."""
        qudit = Qudit(np.int64(5))
        assert qudit.dim == 5
        assert type(qudit.dim) is int


class TestQuditQubitAllocation:
    """How a qudit gets hold of its encoding qubits."""

    @parametrize_dims()
    def test_fresh_qubits_are_created_when_none_given(
        self,
        dim: int,
    ) -> None:
        """A loose qudit allocates its own encoding qubits."""
        qudit = Qudit(dim)
        assert qudit.num_qubits == WIDTHS[dim]
        assert len(qudit.qubits) == WIDTHS[dim]
        assert all(isinstance(qubit, Qubit) for qubit in qudit.qubits)
        assert len(set(qudit.qubits)) == WIDTHS[dim]

    def test_two_loose_qudits_do_not_share_qubits(self) -> None:
        """Auto-created qubits are unique per qudit."""
        first = Qudit(3)
        second = Qudit(3)
        assert set(first.qubits).isdisjoint(second.qubits)

    @parametrize_dims()
    def test_supplied_qubits_are_kept_in_order(self, dim: int) -> None:
        """The given qubits become :attr:`Qudit.qubits` verbatim."""
        register = QuantumRegister(WIDTHS[dim], "r")
        qudit = Qudit(dim, list(register))
        assert isinstance(qudit.qubits, tuple)
        assert qudit.qubits == tuple(register)

    @pytest.mark.parametrize(
        ("dim", "count"),
        [(2, 0), (2, 2), (3, 1), (3, 3), (5, 2), (5, 4), (8, 0)],
    )
    def test_wrong_number_of_qubits_is_rejected(
        self,
        dim: int,
        count: int,
    ) -> None:
        """Exactly ``ceil(log2 d)`` qubits must be supplied."""
        with pytest.raises(
            QuditCircuitError,
            match=r"needs exactly \d+ qubit\(s\)",
        ):
            Qudit(dim, fresh_qubits(count))

    @parametrize_dims(WIDE)
    def test_duplicate_qubits_are_rejected(self, dim: int) -> None:
        """The same qubit cannot appear twice in one qudit."""
        qubit = Qubit()
        with pytest.raises(QuditCircuitError, match="duplicate qubits"):
            Qudit(dim, [qubit] * WIDTHS[dim])


class TestQuditProperties:
    """Read-only attributes of a qudit."""

    @parametrize_dims()
    def test_dim_and_num_qubits(self, dim: int) -> None:
        """Dimension and encoding width agree with the table."""
        qudit = Qudit(dim)
        assert qudit.dim == dim
        assert qudit.num_qubits == WIDTHS[dim]

    def test_a_loose_qudit_has_no_register_or_index(self) -> None:
        """Both back-pointers are ``None`` outside a register."""
        qudit = Qudit(3)
        assert qudit.register is None
        assert qudit.index is None

    @parametrize_dims(POW2)
    def test_power_of_two_dimensions_fill_the_hilbert_space(
        self,
        dim: int,
    ) -> None:
        """No invalid basis state is left when ``d`` is a power of 2."""
        assert Qudit(dim).fills_hilbert_space

    @parametrize_dims(NON_POW2)
    def test_other_dimensions_leave_invalid_states(self, dim: int) -> None:
        """A non-power-of-two qudit leaves unused basis states."""
        assert not Qudit(dim).fills_hilbert_space


class TestQuditRepr:
    """The two shapes of :meth:`Qudit.__repr__`."""

    def test_loose_qudit_repr(self) -> None:
        """A registerless qudit only shows its dimension."""
        assert repr(Qudit(5)) == "Qudit(d=5)"

    def test_register_owned_qudit_repr(
        self,
        qutrit_register: QuditRegister,
    ) -> None:
        """A register member shows ``name[index]`` as well."""
        assert repr(qutrit_register[1]) == "Qudit(qt[1], d=3)"

    def test_repr_needs_both_register_and_index(
        self,
        qutrit_register: QuditRegister,
    ) -> None:
        """A register without an index falls back to the loose form."""
        # NOTE: `register=` alone is accepted; `__repr__` only uses the
        # qualified form when `index` is set too.
        qudit = Qudit(3, register=qutrit_register)
        assert qudit.register is qutrit_register
        assert qudit.index is None
        assert repr(qudit) == "Qudit(d=3)"


class TestQuditIdentity:
    """Qudits behave like Qiskit bits: identity, not structure."""

    def test_qudits_over_the_same_qubits_are_not_equal(self) -> None:
        """Structural twins are still two different wires."""
        qubits = fresh_qubits(2)
        first = Qudit(3, qubits)
        second = Qudit(3, qubits)
        assert first.qubits == second.qubits
        assert first != second

    def test_a_qudit_equals_itself(self) -> None:
        """Default identity equality is reflexive."""
        qudit = Qudit(3)
        alias = qudit
        assert qudit == alias

    def test_qudits_hash_by_identity(self) -> None:
        """Twins occupy two distinct slots in a set."""
        qubits = fresh_qubits(2)
        first = Qudit(3, qubits)
        second = Qudit(3, qubits)
        alias = first
        assert hash(first) == hash(alias)
        assert len({first, second}) == 2
        assert first in {first, second}

    def test_the_class_is_final(self) -> None:
        """:class:`Qudit` is not meant to be subclassed."""
        assert getattr(Qudit, "__final__", False) is True

    def test_the_class_uses_slots(self) -> None:
        """No ``__dict__``, so attributes cannot be injected."""
        qudit = Qudit(3)
        assert not hasattr(qudit, "__dict__")
        with pytest.raises(AttributeError, match="attribute"):
            qudit.extra = 1


class TestQuditRegisterConstruction:
    """Building homogeneous and heterogeneous registers."""

    @parametrize_dims()
    def test_homogeneous_constructor(self, dim: int) -> None:
        """The single ``dim`` is repeated ``size`` times."""
        register = QuditRegister(2, dim, "r")
        assert register.dims == (dim, dim)
        assert register.dim == dim
        assert register.widths == (WIDTHS[dim],) * 2
        assert register.num_qubits == 2 * WIDTHS[dim]
        assert register.size == 2
        assert all(qudit.dim == dim for qudit in register)

    def test_from_dims_builds_a_heterogeneous_register(self) -> None:
        """Each qudit may have its own dimension."""
        register = QuditRegister.from_dims([2, 3, 8], "mix")
        assert register.dims == (2, 3, 8)
        assert register.widths == (1, 2, 3)
        assert register.num_qubits == 6
        assert register.size == 3
        assert [qudit.dim for qudit in register] == [2, 3, 8]

    def test_from_dims_validates_every_dimension(self) -> None:
        """A single bad entry rejects the whole register."""
        with pytest.raises(ValueError, match=r"dim must be >= 2"):
            QuditRegister.from_dims([2, 1, 8])

    def test_auto_generated_names_use_the_q_prefix(self) -> None:
        """Anonymous registers are named ``Q0``, ``Q1``, ..."""
        assert QuditRegister.prefix == "Q"
        assert QuditRegister(1, 2).name == "Q0"
        assert QuditRegister(2, 3).name == "Q1"
        assert QuditRegister.from_dims([2, 3]).name == "Q2"

    def test_explicit_none_still_auto_generates(self) -> None:
        """``name=None`` is the documented way to ask for a name."""
        assert QuditRegister(1, 2, None).name == "Q0"

    @pytest.mark.parametrize(
        "build",
        [
            lambda: QuditRegister(2, 3, ""),
            lambda: QuditRegister.from_dims([2, 3], ""),
        ],
        ids=["init", "from_dims"],
    )
    def test_empty_name_is_rejected(
        self,
        build: Callable[[], QuditRegister],
    ) -> None:
        """An empty string is not a usable register name."""
        with pytest.raises(QuditCircuitError, match="non-empty string"):
            build()

    def test_the_backing_register_is_a_real_quantum_register(self) -> None:
        """``qreg`` holds ``sum(widths)`` qubits and shares the name."""
        register = QuditRegister.from_dims([2, 3, 8], "mix")
        assert isinstance(register.qreg, QuantumRegister)
        assert register.qreg.name == "mix"
        assert register.qreg.size == 6
        assert register.num_qubits == register.qreg.size

    def test_each_qudit_gets_a_contiguous_qubit_slice(self) -> None:
        """Slices follow the register order with no gaps."""
        register = QuditRegister.from_dims([2, 3, 8], "mix")
        bounds = [(0, 1), (1, 3), (3, 6)]
        for index, (start, stop) in enumerate(bounds):
            qudit = register[index]
            assert qudit.qubits == tuple(register.qreg[start:stop])
            assert qudit.register is register
            assert qudit.index == index

    def test_a_size_zero_register_is_degenerate_but_valid(self) -> None:
        """An empty register has no qudits and no qubits."""
        register = QuditRegister(0, 3, "empty")
        assert register.dims == ()
        assert register.widths == ()
        assert register.size == 0
        assert len(register) == 0
        assert register.num_qubits == 0
        assert register.dim == 0
        assert list(register) == []
        assert repr(register) == "QuditRegister(0, d=0, 'empty')"


class TestQuditRegisterProperties:
    """Derived attributes of a register."""

    def test_dim_returns_the_common_dimension(self) -> None:
        """A homogeneous register answers ``dim`` directly."""
        assert QuditRegister.from_dims([4, 4, 4], "r").dim == 4

    def test_dim_rejects_a_heterogeneous_register(self) -> None:
        """Mixed dimensions have no single ``dim``."""
        register = QuditRegister.from_dims([2, 3], "mix")
        with pytest.raises(QuditCircuitError, match="heterogeneous"):
            _ = register.dim

    def test_qudits_matches_iteration(
        self,
        qutrit_register: QuditRegister,
    ) -> None:
        """``qudits`` is the tuple the register iterates over."""
        assert isinstance(qutrit_register.qudits, tuple)
        assert qutrit_register.qudits == tuple(qutrit_register)


class TestQuditRegisterContainer:
    """``len``/``iter``/``in``/``[]`` on a register."""

    def test_len_counts_the_qudits(
        self,
        qutrit_register: QuditRegister,
    ) -> None:
        """``len`` agrees with :attr:`QuditRegister.size`."""
        assert len(qutrit_register) == 3
        assert len(qutrit_register) == qutrit_register.size

    def test_iteration_is_in_register_order(
        self,
        qutrit_register: QuditRegister,
    ) -> None:
        """Iteration yields the very same qudit objects, in order."""
        assert all(
            member is qudit
            for member, qudit in zip(
                qutrit_register,
                qutrit_register.qudits,
                strict=True,
            )
        )

    def test_members_are_contained(
        self,
        qutrit_register: QuditRegister,
    ) -> None:
        """Every qudit of the register is ``in`` it."""
        assert all(qudit in qutrit_register for qudit in qutrit_register)

    def test_containment_is_identity_based(
        self,
        qutrit_register: QuditRegister,
    ) -> None:
        """A structural twin is not a member."""
        twin = Qudit(3, qutrit_register[0].qubits)
        assert twin.qubits == qutrit_register[0].qubits
        assert twin not in qutrit_register

    @pytest.mark.parametrize("other", [0, "qt", None])
    def test_unrelated_objects_are_not_contained(
        self,
        qutrit_register: QuditRegister,
        other: Any,
    ) -> None:
        """``__contains__`` never raises on foreign objects."""
        assert other not in qutrit_register

    @pytest.mark.parametrize(
        "key",
        [0, 1, 2, -1, -3, np.int64(1), np.int32(2)],
    )
    def test_getitem_with_integers(
        self,
        qutrit_register: QuditRegister,
        key: Any,
    ) -> None:
        """Plain, negative and numpy integers all index the register."""
        assert qutrit_register[key] is qutrit_register.qudits[int(key)]

    @pytest.mark.parametrize("key", [slice(0, 2), slice(None), slice(1, 1)])
    def test_getitem_with_a_slice_returns_a_list(
        self,
        qutrit_register: QuditRegister,
        key: slice,
    ) -> None:
        """Slicing yields a ``list``, not a tuple or a register."""
        selected = qutrit_register[key]
        assert isinstance(selected, list)
        assert selected == list(qutrit_register.qudits[key])

    @pytest.mark.parametrize("key", [3, -4, 99])
    def test_getitem_out_of_range(
        self,
        qutrit_register: QuditRegister,
        key: int,
    ) -> None:
        """An out-of-range index raises ``IndexError``."""
        with pytest.raises(IndexError, match="out of range"):
            _ = qutrit_register[key]


class TestQuditRegisterRepr:
    """Homogeneous and heterogeneous representations differ."""

    def test_homogeneous_repr_shows_a_single_dimension(self) -> None:
        """``d=`` is used when every qudit has the same dimension."""
        assert repr(QuditRegister(3, 3, "qt")) == "QuditRegister(3, d=3, 'qt')"

    def test_heterogeneous_repr_shows_every_dimension(self) -> None:
        """``dims=`` is used as soon as the dimensions differ."""
        assert (
            repr(QuditRegister.from_dims([2, 3], "mix"))
            == "QuditRegister(2, dims=(2, 3), 'mix')"
        )
