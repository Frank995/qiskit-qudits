"""Unit tests for :mod:`qiskit_qudits.circuit.clbyte`.

:class:`~qiskit_qudits.circuit.clbyte.ClByte` and
:class:`~qiskit_qudits.circuit.clbyte.ClByteRegister` mirror the qudit
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

from qiskit_qudits.circuit.clbyte import ClByte, ClByteRegister
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
def out_register() -> ClByteRegister:
    """A three-byte register sized for qutrit outcomes."""
    return ClByteRegister(3, 3, "out")


class TestClByteConstruction:
    """Creating a clbyte and validating its width."""

    @parametrize_dims()
    def test_fresh_clbits_are_created_when_none_given(
        self,
        dim: int,
    ) -> None:
        """A loose clbyte allocates ``ceil(log2 d)`` clbits."""
        clbyte = ClByte(dim)
        assert clbyte.dim == dim
        assert clbyte.num_clbits == WIDTHS[dim]
        assert isinstance(clbyte.clbits, tuple)
        assert all(isinstance(clbit, Clbit) for clbit in clbyte.clbits)
        assert len(set(clbyte.clbits)) == WIDTHS[dim]

    @parametrize_dims()
    def test_supplied_clbits_are_kept_in_order(self, dim: int) -> None:
        """The given clbits become :attr:`ClByte.clbits` verbatim."""
        register = ClassicalRegister(WIDTHS[dim], "c")
        clbyte = ClByte(dim, list(register))
        assert clbyte.clbits == tuple(register)

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
            ClByte(dim, [Clbit() for _ in range(count)])

    @parametrize_dims(WIDE)
    def test_duplicate_clbits_are_rejected(self, dim: int) -> None:
        """The same clbit cannot appear twice in one byte."""
        clbit = Clbit()
        with pytest.raises(QuditCircuitError, match="duplicate clbits"):
            ClByte(dim, [clbit] * WIDTHS[dim])

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
        """A clbyte is sized from a *valid* qudit dimension."""
        with pytest.raises(error, match=message):
            ClByte(dim)


class TestClByteProperties:
    """Attributes and representation of a clbyte."""

    def test_a_loose_clbyte_has_no_register_or_index(self) -> None:
        """Both back-pointers are ``None`` outside a register."""
        clbyte = ClByte(3)
        assert clbyte.register is None
        assert clbyte.index is None

    def test_register_members_carry_their_position(
        self,
        out_register: ClByteRegister,
    ) -> None:
        """A member points back at its register and index."""
        for index, clbyte in enumerate(out_register):
            assert clbyte.register is out_register
            assert clbyte.index == index

    def test_loose_repr(self) -> None:
        """A registerless byte only shows its dimension."""
        assert repr(ClByte(5)) == "ClByte(d=5)"

    def test_register_owned_repr(
        self,
        out_register: ClByteRegister,
    ) -> None:
        """A member shows ``name[index]`` as well."""
        assert repr(out_register[2]) == "ClByte(out[2], d=3)"

    def test_clbytes_compare_by_identity(self) -> None:
        """Structural twins are two different classical wires."""
        clbits = [Clbit(), Clbit()]
        first = ClByte(3, clbits)
        second = ClByte(3, clbits)
        assert first.clbits == second.clbits
        assert first != second
        assert len({first, second}) == 2

    def test_the_class_is_final_and_slotted(self) -> None:
        """:class:`ClByte` is closed for subclassing and injection."""
        clbyte = ClByte(3)
        assert getattr(ClByte, "__final__", False) is True
        assert not hasattr(clbyte, "__dict__")
        with pytest.raises(AttributeError, match="attribute"):
            clbyte.extra = 1


class TestClByteRegisterConstruction:
    """Building homogeneous and heterogeneous byte registers."""

    @parametrize_dims()
    def test_homogeneous_constructor(self, dim: int) -> None:
        """The single ``dim`` is repeated ``size`` times."""
        register = ClByteRegister(2, dim, "c")
        assert register.dims == (dim, dim)
        assert register.widths == (WIDTHS[dim],) * 2
        assert register.num_clbits == 2 * WIDTHS[dim]
        assert register.size == 2
        assert all(clbyte.dim == dim for clbyte in register)

    def test_from_dims_builds_a_heterogeneous_register(self) -> None:
        """Each byte may be sized for its own dimension."""
        register = ClByteRegister.from_dims([2, 3, 8], "mix")
        assert register.dims == (2, 3, 8)
        assert register.widths == (1, 2, 3)
        assert register.num_clbits == 6
        assert register.size == 3
        assert isinstance(register.clbytes, tuple)
        assert [clbyte.dim for clbyte in register] == [2, 3, 8]

    def test_auto_generated_names_use_the_c_prefix(self) -> None:
        """Anonymous registers are named ``C0``, ``C1``, ..."""
        assert ClByteRegister.prefix == "C"
        assert ClByteRegister(1, 2).name == "C0"
        assert ClByteRegister(2, 3).name == "C1"
        assert ClByteRegister.from_dims([2, 3]).name == "C2"

    @pytest.mark.parametrize(
        "build",
        [
            lambda: ClByteRegister(2, 3, ""),
            lambda: ClByteRegister.from_dims([2, 3], ""),
        ],
        ids=["init", "from_dims"],
    )
    def test_empty_name_is_rejected(
        self,
        build: Callable[[], ClByteRegister],
    ) -> None:
        """An empty string is not a usable register name."""
        with pytest.raises(QuditCircuitError, match="non-empty string"):
            build()

    def test_the_backing_register_is_a_real_classical_register(
        self,
    ) -> None:
        """``creg`` holds ``sum(widths)`` clbits and shares the name."""
        register = ClByteRegister.from_dims([2, 3, 8], "mix")
        assert isinstance(register.creg, ClassicalRegister)
        assert register.creg.name == "mix"
        assert register.creg.size == 6
        assert register.num_clbits == register.creg.size

    def test_each_byte_gets_a_contiguous_clbit_slice(self) -> None:
        """Slices follow the register order with no gaps."""
        register = ClByteRegister.from_dims([2, 3, 8], "mix")
        bounds = [(0, 1), (1, 3), (3, 6)]
        for index, (start, stop) in enumerate(bounds):
            clbyte = register[index]
            assert clbyte.clbits == tuple(register.creg[start:stop])
            assert clbyte.index == index


class TestClByteRegisterDivergence:
    """Where :class:`ClByteRegister` differs from ``QuditRegister``."""

    def test_there_is_no_dim_property(
        self,
        out_register: ClByteRegister,
    ) -> None:
        """Only the qudit register exposes a scalar ``dim``."""
        # NOTE: `QuditRegister.dim` exists (and raises on heterogeneous
        # registers); `ClByteRegister` deliberately has no counterpart.
        assert hasattr(QuditRegister, "dim")
        assert not hasattr(ClByteRegister, "dim")
        assert not hasattr(out_register, "dim")
        with pytest.raises(AttributeError, match="dim"):
            _ = out_register.dim

    def test_repr_always_lists_every_dimension(self) -> None:
        """Unlike the qudit register, ``dims=`` is always used."""
        assert (
            repr(ClByteRegister(2, 3, "out"))
            == "ClByteRegister(2, dims=(3, 3), 'out')"
        )
        assert (
            repr(ClByteRegister.from_dims([2, 3], "mix"))
            == "ClByteRegister(2, dims=(2, 3), 'mix')"
        )


class TestClByteRegisterContainer:
    """``len``/``iter``/``in``/``[]`` on a byte register."""

    def test_len_and_iteration(
        self,
        out_register: ClByteRegister,
    ) -> None:
        """``len`` and iteration agree with :attr:`clbytes`."""
        assert len(out_register) == out_register.size == 3
        assert all(
            member is clbyte
            for member, clbyte in zip(
                out_register,
                out_register.clbytes,
                strict=True,
            )
        )

    def test_containment_is_identity_based(
        self,
        out_register: ClByteRegister,
    ) -> None:
        """Members are contained, structural twins are not."""
        twin = ClByte(3, out_register[0].clbits)
        assert all(clbyte in out_register for clbyte in out_register)
        assert twin not in out_register
        assert ClByte(3) not in out_register

    @pytest.mark.parametrize(
        "key",
        [0, 1, 2, -1, -3, np.int64(1), np.int32(2)],
    )
    def test_getitem_with_integers(
        self,
        out_register: ClByteRegister,
        key: Any,
    ) -> None:
        """Plain, negative and numpy integers all index the register."""
        assert out_register[key] is out_register.clbytes[int(key)]

    @pytest.mark.parametrize("key", [slice(0, 2), slice(None)])
    def test_getitem_with_a_slice_returns_a_list(
        self,
        out_register: ClByteRegister,
        key: slice,
    ) -> None:
        """Slicing yields a ``list``."""
        selected = out_register[key]
        assert isinstance(selected, list)
        assert selected == list(out_register.clbytes[key])

    def test_getitem_out_of_range(
        self,
        out_register: ClByteRegister,
    ) -> None:
        """An out-of-range index raises ``IndexError``."""
        with pytest.raises(IndexError, match="out of range"):
            _ = out_register[3]
