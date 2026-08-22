"""Unit tests for :mod:`qiskit_qudits.circuit.cldigit`.

:class:`~qiskit_qudits.circuit.cldigit.ClDigit` and
:class:`~qiskit_qudits.circuit.cldigit.ClDigitRegister` mirror the qudit
classes but are *not* subclasses of them, so their behaviour is pinned
down here as well. The overlap with ``test_qudit.py`` is kept to the
bare minimum: only the classical side is asserted, and the places where
the two APIs genuinely diverge get their own test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit.circuit import ClassicalRegister, Clbit

from qiskit_qudits.circuit.cldigit import ClDigit, ClDigitRegister
from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.circuit.qudit import QuditRegister
from tests.helpers import ALL_DIMS, parametrize_dims

if TYPE_CHECKING:
    from collections.abc import Callable

#: ``ceil(log2 d)`` for every dimension the suite exercises.
WIDTHS = {2: 1, 3: 2, 4: 2, 5: 3, 7: 3, 8: 3}

assert set(WIDTHS) == set(ALL_DIMS), "WIDTHS is out of sync with ALL_DIMS"

#: Dimensions wide enough (>= 2 clbits) to have duplicate clbits.
WIDE = [3, 4, 5, 7, 8]


@pytest.fixture
def out_register() -> ClDigitRegister:
    """A three-digit register sized for qutrit outcomes."""
    return ClDigitRegister(3, 3, "out")


class TestClDigitConstruction:
    """Creating a cldigit and validating its width."""

    @parametrize_dims()
    def test_fresh_clbits_are_created_when_none_given(
        self,
        dim: int,
    ) -> None:
        """A loose cldigit allocates ``ceil(log2 d)`` clbits."""
        cldigit = ClDigit(dim)
        assert cldigit.dim == dim
        assert cldigit.num_clbits == WIDTHS[dim]
        assert isinstance(cldigit.clbits, tuple)
        assert all(isinstance(clbit, Clbit) for clbit in cldigit.clbits)
        assert len(set(cldigit.clbits)) == WIDTHS[dim]

    @parametrize_dims()
    def test_supplied_clbits_are_kept_in_order(self, dim: int) -> None:
        """The given clbits become :attr:`ClDigit.clbits` verbatim."""
        register = ClassicalRegister(WIDTHS[dim], "c")
        cldigit = ClDigit(dim, list(register))
        assert cldigit.clbits == tuple(register)

    @pytest.mark.parametrize(
        ("dim", "count"),
        [(2, 0), (2, 2), (3, 1), (3, 3), (5, 2), (5, 4)],
    )
    def test_wrong_number_of_clbits_is_rejected(
        self,
        dim: int,
        count: int,
    ) -> None:
        """Exactly ``ceil(log2 d)`` clbits must be supplied."""
        with pytest.raises(
            QuditCircuitError,
            match=r"needs exactly \d+ clbit\(s\)",
        ):
            ClDigit(dim, [Clbit() for _ in range(count)])

    @parametrize_dims(WIDE)
    def test_duplicate_clbits_are_rejected(self, dim: int) -> None:
        """The same clbit cannot appear twice in one digit."""
        clbit = Clbit()
        with pytest.raises(QuditCircuitError, match="duplicate clbits"):
            ClDigit(dim, [clbit] * WIDTHS[dim])

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
        """A cldigit is sized from a *valid* qudit dimension."""
        with pytest.raises(error, match=message):
            ClDigit(dim)


class TestClDigitProperties:
    """Attributes and representation of a cldigit."""

    def test_a_loose_cldigit_has_no_register_or_index(self) -> None:
        """Both back-pointers are ``None`` outside a register."""
        cldigit = ClDigit(3)
        assert cldigit.register is None
        assert cldigit.index is None

    def test_register_members_carry_their_position(
        self,
        out_register: ClDigitRegister,
    ) -> None:
        """A member points back at its register and index."""
        for index, cldigit in enumerate(out_register):
            assert cldigit.register is out_register
            assert cldigit.index == index

    def test_loose_repr(self) -> None:
        """A registerless digit only shows its dimension."""
        assert repr(ClDigit(5)) == "ClDigit(d=5)"

    def test_register_owned_repr(
        self,
        out_register: ClDigitRegister,
    ) -> None:
        """A member shows ``name[index]`` as well."""
        assert repr(out_register[2]) == "ClDigit(out[2], d=3)"

    def test_cldigits_compare_by_identity(self) -> None:
        """Structural twins are two different classical wires."""
        clbits = [Clbit(), Clbit()]
        first = ClDigit(3, clbits)
        second = ClDigit(3, clbits)
        assert first.clbits == second.clbits
        assert first != second
        assert len({first, second}) == 2

    def test_the_class_is_final_and_slotted(self) -> None:
        """:class:`ClDigit` is closed for subclassing and injection."""
        cldigit = ClDigit(3)
        assert getattr(ClDigit, "__final__", False) is True
        assert not hasattr(cldigit, "__dict__")
        with pytest.raises(AttributeError, match="attribute"):
            cldigit.extra = 1


class TestClDigitRegisterConstruction:
    """Building homogeneous and heterogeneous digit registers."""

    @parametrize_dims()
    def test_homogeneous_constructor(self, dim: int) -> None:
        """The single ``dim`` is repeated ``size`` times."""
        register = ClDigitRegister(2, dim, "c")
        assert register.dims == (dim, dim)
        assert register.widths == (WIDTHS[dim],) * 2
        assert register.num_clbits == 2 * WIDTHS[dim]
        assert register.size == 2
        assert all(cldigit.dim == dim for cldigit in register)

    def test_from_dims_builds_a_heterogeneous_register(self) -> None:
        """Each digit may be sized for its own dimension."""
        register = ClDigitRegister.from_dims([2, 3, 8], "mix")
        assert register.dims == (2, 3, 8)
        assert register.widths == (1, 2, 3)
        assert register.num_clbits == 6
        assert register.size == 3
        assert isinstance(register.cldigits, tuple)
        assert [cldigit.dim for cldigit in register] == [2, 3, 8]

    def test_auto_generated_names_use_the_c_prefix(self) -> None:
        """Anonymous registers are named ``C0``, ``C1``, ..."""
        assert ClDigitRegister.prefix == "C"
        assert ClDigitRegister(1, 2).name == "C0"
        assert ClDigitRegister(2, 3).name == "C1"
        assert ClDigitRegister.from_dims([2, 3]).name == "C2"

    @pytest.mark.parametrize(
        "build",
        [
            lambda: ClDigitRegister(2, 3, ""),
            lambda: ClDigitRegister.from_dims([2, 3], ""),
        ],
        ids=["init", "from_dims"],
    )
    def test_empty_name_is_rejected(
        self,
        build: Callable[[], ClDigitRegister],
    ) -> None:
        """An empty string is not a usable register name."""
        with pytest.raises(QuditCircuitError, match="non-empty string"):
            build()

    def test_the_backing_register_is_a_real_classical_register(
        self,
    ) -> None:
        """``creg`` holds ``sum(widths)`` clbits and shares the name."""
        register = ClDigitRegister.from_dims([2, 3, 8], "mix")
        assert isinstance(register.creg, ClassicalRegister)
        assert register.creg.name == "mix"
        assert register.creg.size == 6
        assert register.num_clbits == register.creg.size

    def test_each_digit_gets_a_contiguous_clbit_slice(self) -> None:
        """Slices follow the register order with no gaps."""
        register = ClDigitRegister.from_dims([2, 3, 8], "mix")
        bounds = [(0, 1), (1, 3), (3, 6)]
        for index, (start, stop) in enumerate(bounds):
            cldigit = register[index]
            assert cldigit.clbits == tuple(register.creg[start:stop])
            assert cldigit.index == index


class TestClDigitRegisterDivergence:
    """Where :class:`ClDigitRegister` differs from ``QuditRegister``."""

    def test_there_is_no_dim_property(
        self,
        out_register: ClDigitRegister,
    ) -> None:
        """Only the qudit register exposes a scalar ``dim``."""
        # NOTE: `QuditRegister.dim` exists (and raises on heterogeneous
        # registers); `ClDigitRegister` deliberately has no counterpart.
        assert hasattr(QuditRegister, "dim")
        assert not hasattr(ClDigitRegister, "dim")
        assert not hasattr(out_register, "dim")
        with pytest.raises(AttributeError, match="dim"):
            _ = out_register.dim

    def test_repr_always_lists_every_dimension(self) -> None:
        """Unlike the qudit register, ``dims=`` is always used."""
        assert (
            repr(ClDigitRegister(2, 3, "out"))
            == "ClDigitRegister(2, dims=(3, 3), 'out')"
        )
        assert (
            repr(ClDigitRegister.from_dims([2, 3], "mix"))
            == "ClDigitRegister(2, dims=(2, 3), 'mix')"
        )


class TestClDigitRegisterContainer:
    """``len``/``iter``/``in``/``[]`` on a digit register."""

    def test_len_and_iteration(
        self,
        out_register: ClDigitRegister,
    ) -> None:
        """``len`` and iteration agree with :attr:`cldigits`."""
        assert len(out_register) == out_register.size == 3
        assert all(
            member is cldigit
            for member, cldigit in zip(
                out_register,
                out_register.cldigits,
                strict=True,
            )
        )

    def test_containment_is_identity_based(
        self,
        out_register: ClDigitRegister,
    ) -> None:
        """Members are contained, structural twins are not."""
        twin = ClDigit(3, out_register[0].clbits)
        assert all(cldigit in out_register for cldigit in out_register)
        assert twin not in out_register
        assert ClDigit(3) not in out_register

    @pytest.mark.parametrize(
        "key",
        [0, 1, 2, -1, -3, np.int64(1), np.int32(2)],
    )
    def test_getitem_with_integers(
        self,
        out_register: ClDigitRegister,
        key: Any,
    ) -> None:
        """Plain, negative and numpy integers all index the register."""
        assert out_register[key] is out_register.cldigits[int(key)]

    @pytest.mark.parametrize("key", [slice(0, 2), slice(None)])
    def test_getitem_with_a_slice_returns_a_list(
        self,
        out_register: ClDigitRegister,
        key: slice,
    ) -> None:
        """Slicing yields a ``list``."""
        selected = out_register[key]
        assert isinstance(selected, list)
        assert selected == list(out_register.cldigits[key])

    def test_getitem_out_of_range(
        self,
        out_register: ClDigitRegister,
    ) -> None:
        """An out-of-range index raises ``IndexError``."""
        with pytest.raises(IndexError, match="out of range"):
            _ = out_register[3]
