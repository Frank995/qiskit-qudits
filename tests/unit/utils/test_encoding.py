"""Unit tests for :mod:`qiskit_qudits.utils.encoding`.

This module carries the library's endianness conventions, so the
tests below pin them down explicitly:

* inside a qudit a level is stored MSB-first, the rightmost character
  living on qubit 0;
* inside a counts key the rightmost characters belong to the lowest
  clbit indices, and register separators are pure decoration;
* a logical index is mixed-radix with qudit 0 as the least
  significant factor.
"""

from __future__ import annotations

import re
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from qiskit_qudits.utils.dims import qubits_per_qudit
from qiskit_qudits.utils.encoding import (
    _bitstring_to_level,
    _encoded_index,
    _split_bitstring,
    decode_bitstring,
    decode_counts,
    embed_state,
    format_levels,
    level_to_bitstring,
    parse_level_tokens,
    project_state,
    validate_basis_states,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit_qudits.utils.encoding import InvalidPolicy

#: Qudit layouts exercised by the state-vector round-trips.
LAYOUTS: list[tuple[int, ...]] = [
    (2,),
    (3,),
    (5,),
    (8,),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (3, 5),
    (4, 4),
    (2, 3, 5),
]

#: Layouts whose encoding leaves no invalid basis state behind.
DENSE_LAYOUTS: list[tuple[int, ...]] = [(2,), (8,), (2, 2), (4, 4)]


def state_norm(vector: np.typing.NDArray[np.complex128]) -> float:
    """Return the Euclidean norm of a complex vector."""
    return float(
        np.linalg.norm(vector),
    )


def normalised_state(
    size: int,
    seed: int = 1234,
) -> np.typing.NDArray[np.complex128]:
    """Return a deterministic, normalised complex vector.

    Args:
        size: Number of amplitudes.
        seed: Generator seed, pinned so the suite never flakes.

    Returns:
        A unit-norm ``complex128`` vector with no zero entry.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    vector = np.asarray(raw, dtype=np.complex128)
    return vector / state_norm(vector)


def logical_index(dims: Sequence[int], levels: Sequence[int]) -> int:
    """Return the mixed-radix index of ``levels`` (qudit 0 lowest)."""
    index = 0
    stride = 1
    for dim, level in zip(dims, levels, strict=True):
        index += level * stride
        stride *= dim
    return index


class TestLevelToBitstring:
    """Behaviour of :func:`level_to_bitstring`."""

    @pytest.mark.parametrize(
        ("level", "width", "expected"),
        [
            (0, 1, "0"),
            (1, 1, "1"),
            (0, 2, "00"),
            (2, 2, "10"),
            (3, 2, "11"),
            (0, 3, "000"),
            (5, 3, "101"),
            (7, 3, "111"),
            (1, 4, "0001"),
            (8, 4, "1000"),
            (11, 4, "1011"),
            (5, 8, "00000101"),
            (255, 8, "11111111"),
        ],
    )
    def test_encodes_msb_first_with_zero_padding(
        self,
        level: int,
        width: int,
        expected: str,
    ) -> None:
        """The level is rendered MSB-first, padded to ``width``."""
        assert level_to_bitstring(level, width) == expected

    @pytest.mark.parametrize("width", [1, 2, 3, 5])
    def test_result_always_has_the_requested_width(
        self,
        width: int,
    ) -> None:
        """Every representable level yields exactly ``width`` chars."""
        for level in range(1 << width):
            assert len(level_to_bitstring(level, width)) == width

    @pytest.mark.parametrize(("level", "width"), [(2, 3), (5, 4), (9, 4)])
    def test_rightmost_character_is_qubit_zero(
        self,
        level: int,
        width: int,
    ) -> None:
        """Character ``width - 1 - j`` carries the bit of qubit j."""
        bits = level_to_bitstring(level, width)
        for qubit in range(width):
            assert int(bits[width - 1 - qubit]) == (level >> qubit) & 1

    @pytest.mark.parametrize("width", [1, 2, 3, 4])
    def test_round_trips_through_the_decoder(self, width: int) -> None:
        """Decoding an encoded level returns the level."""
        for level in range(1 << width):
            bits = level_to_bitstring(level, width)
            assert _bitstring_to_level(bits) == level

    @pytest.mark.parametrize(
        ("level", "width"),
        [(2, 1), (4, 2), (8, 3), (16, 4), (256, 8), (-1, 3), (-5, 1)],
    )
    def test_rejects_levels_that_do_not_fit(
        self,
        level: int,
        width: int,
    ) -> None:
        """Out-of-range and negative levels raise ``ValueError``."""
        with pytest.raises(ValueError, match="does not fit into"):
            level_to_bitstring(level, width)

    def test_zero_width_still_returns_one_character(self) -> None:
        """A degenerate ``width=0`` yields ``'0'``, not ``''``."""
        # NOTE: the docstring promises exactly ``width`` characters,
        # but ``format(0, '00b')`` has a one-character minimum.
        assert level_to_bitstring(0, 0) == "0"


class TestBitstringToLevel:
    """Behaviour of the private :func:`_bitstring_to_level`."""

    @pytest.mark.parametrize(
        ("bits", "expected"),
        [
            ("0", 0),
            ("1", 1),
            ("00", 0),
            ("10", 2),
            ("101", 5),
            ("0000", 0),
            ("1111", 15),
            ("00101", 5),
        ],
    )
    def test_decodes_msb_first(self, bits: str, expected: int) -> None:
        """The leftmost character is the most significant bit."""
        assert _bitstring_to_level(bits) == expected

    def test_rejects_the_empty_string(self) -> None:
        """An empty chunk carries no level."""
        with pytest.raises(ValueError, match="is not a binary string"):
            _bitstring_to_level("")

    @pytest.mark.parametrize(
        "bits",
        ["2", "01a", " 1", "1 ", "0b1", "-1", "1.0", "01_1"],
    )
    def test_rejects_non_binary_characters(self, bits: str) -> None:
        """Anything outside ``'0'``/``'1'`` is refused."""
        with pytest.raises(ValueError, match="is not a binary string"):
            _bitstring_to_level(bits)


class TestSplitBitstring:
    """Behaviour of the private :func:`_split_bitstring`."""

    @pytest.mark.parametrize(
        ("bitstring", "widths", "expected"),
        [
            ("10 01", [2, 2], ("01", "10")),
            ("1001", [2, 2], ("01", "10")),
            ("  10   01  ", [2, 2], ("01", "10")),
            ("10\t01", [2, 2], ("01", "10")),
            ("10110", [2, 3], ("10", "101")),
            ("10110", [3, 2], ("110", "10")),
            ("1011", [4], ("1011",)),
            ("1011", [1, 1, 1, 1], ("1", "1", "0", "1")),
            ("11 10 01", [2, 2, 2], ("01", "10", "11")),
            ("", [], ()),
        ],
    )
    def test_lowest_clbits_come_from_the_right(
        self,
        bitstring: str,
        widths: list[int],
        expected: tuple[str, ...],
    ) -> None:
        """Chunk 0 is the rightmost slice; whitespace is ignored."""
        assert _split_bitstring(bitstring, widths) == expected

    @pytest.mark.parametrize(
        ("bitstring", "widths"),
        [("101", [2, 2]), ("10101", [2, 2]), ("", [1]), ("1", [])],
    )
    def test_rejects_a_width_mismatch(
        self,
        bitstring: str,
        widths: list[int],
    ) -> None:
        """The total width must match the stripped string length."""
        with pytest.raises(ValueError, match=r"bit\(s\) but the layout"):
            _split_bitstring(bitstring, widths)

    def test_chunks_reassemble_into_the_stripped_string(self) -> None:
        """Concatenating the chunks back-to-front rebuilds the key."""
        chunks = _split_bitstring("110 01 1", [1, 2, 3])
        assert chunks == ("1", "01", "110")
        assert "".join(reversed(chunks)) == "110011"


class TestDecodeBitstring:
    """Behaviour of :func:`decode_bitstring`."""

    @pytest.mark.parametrize(
        ("bitstring", "widths", "expected"),
        [
            ("10 01", [2, 2], (1, 2)),
            ("11 10 01", [2, 2, 2], (1, 2, 3)),
            ("101", [3], (5,)),
            ("10110", [2, 3], (2, 5)),
            ("", [], ()),
        ],
    )
    def test_decodes_without_dimensions(
        self,
        bitstring: str,
        widths: list[int],
        expected: tuple[int, ...],
    ) -> None:
        """Each chunk becomes one level, digit 0 first."""
        assert decode_bitstring(bitstring, widths) == expected

    def test_keeps_in_subspace_levels_when_dims_are_given(self) -> None:
        """Valid levels are returned untouched."""
        assert decode_bitstring("10 01", [2, 2], (3, 3)) == (1, 2)

    def test_keep_policy_returns_the_leaked_level(self) -> None:
        """``'keep'`` reports the out-of-subspace level as measured."""
        decoded = decode_bitstring("11 00", [2, 2], (3, 3), on_invalid="keep")
        assert decoded == (0, 3)

    def test_keep_is_the_default_policy(self) -> None:
        """Leakage is kept unless another policy is requested."""
        assert decode_bitstring("11 00", [2, 2], (3, 3)) == (0, 3)

    def test_drop_policy_returns_none(self) -> None:
        """``'drop'`` discards the whole shot."""
        decoded = decode_bitstring("11 00", [2, 2], (3, 3), on_invalid="drop")
        assert decoded is None

    def test_raise_policy_reports_the_offending_digit(self) -> None:
        """``'raise'`` names the digit, its level and the subspace."""
        with pytest.raises(
            ValueError,
            match="digit 1 decoded to level 3, which is outside the 3-level",
        ):
            decode_bitstring("11 00", [2, 2], (3, 3), on_invalid="raise")

    @pytest.mark.parametrize("policy", ["keep", "drop", "raise"])
    def test_policies_agree_when_there_is_no_leakage(
        self,
        policy: InvalidPolicy,
    ) -> None:
        """The policy only matters for out-of-subspace levels."""
        decoded = decode_bitstring("10 01", [2, 2], (3, 3), on_invalid=policy)
        assert decoded == (1, 2)

    @pytest.mark.parametrize("dims", [(3,), (3, 3, 3), ()])
    def test_rejects_a_dims_length_mismatch(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """``dims`` must have one entry per digit."""
        with pytest.raises(ValueError, match=r"dimension\(s\) for 2 digit"):
            decode_bitstring("10 01", [2, 2], dims)

    def test_propagates_a_width_mismatch(self) -> None:
        """A malformed key fails before any level is decoded."""
        with pytest.raises(ValueError, match="the layout requires 4"):
            decode_bitstring("101", [2, 2])

    def test_a_dense_subspace_never_leaks(self) -> None:
        """With power-of-two dims every bit pattern is valid."""
        assert decode_bitstring("11 11", [2, 2], (4, 4)) == (3, 3)


class TestDecodeCounts:
    """Behaviour of :func:`decode_counts`."""

    def test_decodes_a_simple_histogram(self) -> None:
        """Each key becomes a level tuple, values are preserved."""
        counts = {"01": 5, "10": 3}
        assert decode_counts(counts, [2]) == {(1,): 5, (2,): 3}

    def test_aggregates_keys_that_decode_alike(self) -> None:
        """Spacing differences collapse onto the same level tuple."""
        counts = {"10 01": 3, "1001": 4}
        assert decode_counts(counts, [2, 2]) == {(1, 2): 7}

    def test_drop_policy_removes_leaked_shots(self) -> None:
        """Dropped shots disappear from the histogram entirely."""
        counts = {"11 00": 2, "10 01": 3}
        decoded = decode_counts(counts, [2, 2], (3, 3), on_invalid="drop")
        assert decoded == {(1, 2): 3}

    def test_keep_policy_retains_leaked_shots(self) -> None:
        """With ``'keep'`` the shot total is conserved."""
        counts = {"11 00": 2, "10 01": 3}
        decoded = decode_counts(counts, [2, 2], (3, 3))
        assert decoded == {(0, 3): 2, (1, 2): 3}
        assert sum(decoded.values()) == sum(counts.values())

    def test_raise_policy_propagates(self) -> None:
        """A single leaked key aborts the whole decoding."""
        with pytest.raises(ValueError, match="outside the 3-level"):
            decode_counts({"11 00": 2}, [2, 2], (3, 3), on_invalid="raise")

    def test_empty_counts_give_an_empty_mapping(self) -> None:
        """Nothing in, nothing out."""
        assert decode_counts({}, [2, 2]) == {}

    def test_dropping_everything_gives_an_empty_mapping(self) -> None:
        """All-leakage input decodes to no levels at all."""
        counts = {"11 00": 2, "00 11": 1}
        assert decode_counts(counts, [2, 2], (3, 3), on_invalid="drop") == {}

    def test_shot_values_are_coerced_to_int(self) -> None:
        """Numpy shot counts become builtin integers."""
        counts: dict[str, Any] = {"01": np.int64(4)}
        decoded = decode_counts(counts, [2])
        assert decoded == {(1,): 4}
        assert type(decoded[(1,)]) is int


class TestFormatLevels:
    """Behaviour of :func:`format_levels`."""

    @pytest.mark.parametrize(
        ("levels", "expected"),
        [
            ((0, 3, 11), "11 3 0"),
            ((1, 2, 3), "3 2 1"),
            ((5,), "5"),
            ((), ""),
            ((0, 0), "0 0"),
            ((10, 2), "2 10"),
        ],
    )
    def test_renders_the_last_digit_leftmost(
        self,
        levels: tuple[int, ...],
        expected: str,
    ) -> None:
        """Rendering reverses digit order, like Qiskit bit-strings."""
        assert format_levels(levels) == expected

    @pytest.mark.parametrize(
        ("separator", "expected"),
        [(",", "11,3,0"), ("|", "11|3|0"), ("", "1130"), ("  ", "11  3  0")],
    )
    def test_honours_a_custom_separator(
        self,
        separator: str,
        expected: str,
    ) -> None:
        """Any separator is inserted verbatim between tokens."""
        assert format_levels((0, 3, 11), separator=separator) == expected

    def test_accepts_any_iterable(self) -> None:
        """A one-shot iterator is materialised before reversing."""
        assert format_levels(iter((0, 3, 11))) == "11 3 0"

    def test_matches_the_decoded_bitstring_order(self) -> None:
        """Formatting a decoded key mirrors the key's own order."""
        levels = decode_bitstring("11 10 01", [2, 2, 2])
        assert levels is not None
        assert format_levels(levels) == "3 2 1"


class TestEncodedIndex:
    """Behaviour of the private :func:`_encoded_index`."""

    @pytest.mark.parametrize(
        ("dims", "expected"),
        [
            ((3,), [0, 1, 2]),
            ((5,), [0, 1, 2, 3, 4]),
            ((2, 3), [0, 1, 2, 3, 4, 5]),
            ((3, 2), [0, 1, 2, 4, 5, 6]),
            ((3, 3), [0, 1, 2, 4, 5, 6, 8, 9, 10]),
            (
                (3, 5),
                [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18],
            ),
        ],
    )
    def test_maps_the_full_mixed_radix_table(
        self,
        dims: tuple[int, ...],
        expected: list[int],
    ) -> None:
        """Every logical index lands on its hand-computed slot."""
        actual = [
            _encoded_index(dims, index) for index in range(len(expected))
        ]
        assert actual == expected

    @pytest.mark.parametrize("dims", DENSE_LAYOUTS)
    def test_is_the_identity_for_power_of_two_layouts(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """A dense encoding wastes no index."""
        size = prod(dims)
        actual = [_encoded_index(dims, index) for index in range(size)]
        assert actual == list(range(size))

    @pytest.mark.parametrize("dims", LAYOUTS)
    def test_is_injective_and_strictly_increasing(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """Distinct logical states occupy distinct encoded slots."""
        size = prod(dims)
        indices = [_encoded_index(dims, index) for index in range(size)]
        assert len(set(indices)) == size
        assert indices == sorted(indices)

    @pytest.mark.parametrize("dims", LAYOUTS)
    def test_stays_inside_the_encoded_space(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """No index exceeds the ``2**N`` encoded dimension."""
        num_qubits = sum(qubits_per_qudit(dim) for dim in dims)
        assert all(
            0 <= _encoded_index(dims, index) < (1 << num_qubits)
            for index in range(prod(dims))
        )

    @pytest.mark.parametrize(
        ("dims", "levels"),
        [
            ((3,), (2,)),
            ((3, 2), (2, 1)),
            ((3, 5), (2, 3)),
            ((2, 3, 5), (1, 2, 4)),
        ],
    )
    def test_agrees_with_concatenated_level_bitstrings(
        self,
        dims: tuple[int, ...],
        levels: tuple[int, ...],
    ) -> None:
        """The encoded index is the concatenation of the chunks."""
        chunks = [
            level_to_bitstring(level, qubits_per_qudit(dim))
            for dim, level in zip(dims, levels, strict=True)
        ]
        expected = int("".join(reversed(chunks)), 2)
        assert _encoded_index(dims, logical_index(dims, levels)) == expected

    def test_an_empty_layout_maps_to_zero(self) -> None:
        """No qudits means a single, trivial encoded state."""
        assert _encoded_index((), 0) == 0


class TestEmbedState:
    """Behaviour of :func:`embed_state`."""

    @pytest.mark.parametrize("dims", LAYOUTS)
    def test_result_spans_the_full_encoded_space(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """The output has ``2**N`` complex amplitudes."""
        encoded = embed_state(dims, normalised_state(prod(dims)))
        num_qubits = sum(qubits_per_qudit(dim) for dim in dims)
        assert encoded.shape == (1 << num_qubits,)
        assert encoded.dtype == np.complex128

    @pytest.mark.parametrize(
        ("index", "expected"),
        [(0, 0), (1, 1), (2, 2), (3, 4), (4, 5), (5, 6)],
    )
    def test_places_a_basis_state_on_its_encoded_slot(
        self,
        index: int,
        expected: int,
    ) -> None:
        """Amplitude ``k`` ends up at ``_encoded_index(dims, k)``."""
        amplitudes = np.zeros(6, dtype=np.complex128)
        amplitudes[index] = 1.0
        encoded = embed_state((3, 2), amplitudes)
        assert encoded[expected] == 1.0
        assert np.count_nonzero(encoded) == 1

    def test_invalid_slots_are_exactly_zero(self) -> None:
        """Leakage slots are hard zeros, not just small numbers."""
        encoded = embed_state((3, 2), normalised_state(6))
        assert encoded[3] == 0
        assert encoded[7] == 0
        assert np.count_nonzero(encoded) == 6

    def test_preserves_the_amplitudes(self) -> None:
        """Values are copied verbatim onto their encoded slots."""
        amplitudes = normalised_state(3)
        encoded = embed_state((3,), amplitudes)
        assert encoded[0] == amplitudes[0]
        assert encoded[1] == amplitudes[1]
        assert encoded[2] == amplitudes[2]
        assert encoded[3] == 0

    def test_preserves_the_norm(self) -> None:
        """Embedding is an isometry."""
        encoded = embed_state((3, 5), normalised_state(15))
        assert state_norm(encoded) == pytest.approx(1.0)

    @pytest.mark.parametrize("size", [1, 2, 4, 9])
    def test_rejects_a_size_mismatch(self, size: int) -> None:
        """The input must have exactly ``prod(dims)`` amplitudes."""
        amplitudes = normalised_state(size)
        with pytest.raises(ValueError, match=r"expected 3 amplitude\(s\)"):
            embed_state((3,), amplitudes)

    @pytest.mark.parametrize(
        "amplitudes",
        [
            np.zeros(3, dtype=np.complex128),
            np.ones(3, dtype=np.complex128),
            np.array([2.0, 0.0, 0.0], dtype=np.complex128),
        ],
    )
    def test_rejects_a_non_normalised_state(
        self,
        amplitudes: np.typing.NDArray[np.complex128],
    ) -> None:
        """The input must be a unit vector."""
        with pytest.raises(ValueError, match="not normalised"):
            embed_state((3,), amplitudes)

    def test_atol_controls_the_normalisation_check(self) -> None:
        """A slightly off-norm state passes only with a loose atol."""
        amplitudes = np.array([1.0 + 1e-6, 0.0, 0.0], dtype=np.complex128)
        with pytest.raises(ValueError, match="not normalised"):
            embed_state((3,), amplitudes)
        assert embed_state((3,), amplitudes, atol=1e-3).shape == (4,)


class TestProjectState:
    """Behaviour of :func:`project_state`."""

    @pytest.mark.parametrize("dims", LAYOUTS)
    def test_round_trips_with_embed_state(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """Projecting an embedded state returns the original."""
        amplitudes = normalised_state(prod(dims))
        recovered = project_state(dims, embed_state(dims, amplitudes))
        assert np.array_equal(recovered, amplitudes)

    @pytest.mark.parametrize("dims", LAYOUTS)
    def test_preserves_the_norm(self, dims: tuple[int, ...]) -> None:
        """No amplitude is lost on the way back."""
        encoded = embed_state(dims, normalised_state(prod(dims)))
        assert state_norm(project_state(dims, encoded)) == pytest.approx(1.0)

    @pytest.mark.parametrize("dims", LAYOUTS)
    def test_embed_undoes_project(self, dims: tuple[int, ...]) -> None:
        """Embedding a projected state rebuilds the encoded vector."""
        encoded = embed_state(dims, normalised_state(prod(dims)))
        rebuilt = embed_state(dims, project_state(dims, encoded))
        assert np.array_equal(rebuilt, encoded)

    def test_reads_the_encoded_slots(self) -> None:
        """A single encoded basis state maps to its logical index."""
        encoded = np.zeros(8, dtype=np.complex128)
        encoded[4] = 1.0
        logical = project_state((3, 2), encoded)
        assert logical.shape == (6,)
        assert logical[3] == 1.0
        assert np.count_nonzero(logical) == 1

    def test_flattens_its_input(self) -> None:
        """A non-flat encoded array is ravelled first."""
        encoded = np.zeros((2, 2), dtype=np.complex128)
        encoded[0, 1] = 1.0
        logical = project_state((3,), encoded)
        assert logical[1] == 1.0

    @pytest.mark.parametrize("size", [3, 5, 8])
    def test_rejects_a_size_mismatch(self, size: int) -> None:
        """The input must have exactly ``2**N`` amplitudes."""
        encoded = np.zeros(size, dtype=np.complex128)
        with pytest.raises(ValueError, match=r"expected 4 amplitude\(s\)"):
            project_state((3,), encoded)

    def test_rejects_amplitude_outside_the_subspace(self) -> None:
        """Population on a leakage slot is an error, not a silent 0."""
        encoded = np.zeros(4, dtype=np.complex128)
        encoded[3] = 1.0
        with pytest.raises(ValueError, match="outside the qudit subspace"):
            project_state((3,), encoded)

    def test_atol_controls_the_leakage_check(self) -> None:
        """Negligible leakage is tolerated, tiny tolerances are not."""
        encoded = np.zeros(4, dtype=np.complex128)
        encoded[0] = 1.0
        encoded[3] = 1e-9
        assert project_state((3,), encoded)[0] == 1.0
        with pytest.raises(ValueError, match="outside the qudit subspace"):
            project_state((3,), encoded, atol=1e-12)

    @pytest.mark.parametrize("dims", DENSE_LAYOUTS)
    def test_a_dense_layout_can_never_leak(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """With power-of-two dims every slot is a valid state."""
        size = prod(dims)
        encoded = normalised_state(size)
        assert project_state(dims, encoded).shape == (size,)


class TestParseLevelTokens:
    """Behaviour of :func:`parse_level_tokens`."""

    @pytest.mark.parametrize(
        ("text", "num_qudits", "expected"),
        [
            ("11 3 0", 3, (0, 3, 11)),
            ("0", 1, (0,)),
            ("1 2", 2, (2, 1)),
            ("  11 \t 3   0  ", 3, (0, 3, 11)),
            ("", 0, ()),
            ("2 2 2", 3, (2, 2, 2)),
        ],
    )
    def test_rightmost_token_is_the_first_target(
        self,
        text: str,
        num_qudits: int,
        expected: tuple[int, ...],
    ) -> None:
        """Tokens are read in Qiskit order and reversed."""
        assert parse_level_tokens(text, num_qudits) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("0b1011 0x3", (3, 11)),
            ("0o17 10", (10, 15)),
            ("0xb 0B1", (1, 11)),
            ("1_0 2", (2, 10)),
        ],
    )
    def test_accepts_every_python_integer_literal(
        self,
        text: str,
        expected: tuple[int, ...],
    ) -> None:
        """Tokens are parsed with ``int(token, 0)``."""
        assert parse_level_tokens(text, 2) == expected

    def test_parses_negative_tokens(self) -> None:
        """No range check happens here, only parsing."""
        assert parse_level_tokens("-1 2", 2) == (2, -1)

    def test_rejects_leading_zeros(self) -> None:
        """``int('011', 0)`` is a syntax error in Python."""
        with pytest.raises(ValueError, match="not an integer level"):
            parse_level_tokens("011", 1)

    def test_accepts_repeated_zeros(self) -> None:
        """All-zero tokens are the one leading-zero form allowed."""
        assert parse_level_tokens("00 0", 2) == (0, 0)

    @pytest.mark.parametrize(
        ("text", "num_qudits"),
        [("1 2", 3), ("1 2 3", 2), ("", 1), ("1", 0)],
    )
    def test_rejects_a_token_count_mismatch(
        self,
        text: str,
        num_qudits: int,
    ) -> None:
        """The number of tokens must match the number of qudits."""
        with pytest.raises(
            ValueError,
            match=f"expected {num_qudits} whitespace-separated level",
        ):
            parse_level_tokens(text, num_qudits)

    @pytest.mark.parametrize(
        ("text", "offender"),
        [("a", "a"), ("1 x", "x"), ("1.5", "1.5"), ("+", "+"), ("2j", "2j")],
    )
    def test_rejects_unparsable_tokens(
        self,
        text: str,
        offender: str,
    ) -> None:
        """The offending token is quoted in the error message."""
        message = f"'{re.escape(offender)}' is not an integer level"
        with pytest.raises(ValueError, match=message):
            parse_level_tokens(text, len(text.split()))

    @pytest.mark.parametrize(
        "levels",
        [(0, 3, 11), (1,), (2, 2), (0, 0, 0, 7)],
    )
    def test_round_trips_with_format_levels(
        self,
        levels: tuple[int, ...],
    ) -> None:
        """Rendering then parsing is the identity."""
        rendered = format_levels(levels)
        assert parse_level_tokens(rendered, len(levels)) == levels


class TestValidateBasisStates:
    """Behaviour of :func:`validate_basis_states`."""

    @pytest.mark.parametrize(
        ("states", "dims"),
        [
            ((0, 1, 2), (3, 3, 3)),
            ((0,), (2,)),
            ((1,), (2,)),
            ((2, 4), (3, 5)),
            ((), ()),
        ],
    )
    def test_accepts_in_range_states(
        self,
        states: tuple[int, ...],
        dims: tuple[int, ...],
    ) -> None:
        """In-range indices are returned in target order."""
        assert validate_basis_states(states, dims) == states

    def test_coerces_numpy_integers(self) -> None:
        """The result always holds builtin integers."""
        result = validate_basis_states([np.int64(2), np.int32(0)], (3, 3))
        assert result == (2, 0)
        assert all(type(state) is int for state in result)

    @pytest.mark.parametrize(
        ("states", "dims", "message"),
        [
            ((0, 1), (3, 3, 3), r"got 2 state\(s\) for 3 qudit\(s\)"),
            ((0, 1, 2), (3,), r"got 3 state\(s\) for 1 qudit\(s\)"),
            ((0,), (), r"got 1 state\(s\) for 0 qudit\(s\)"),
        ],
    )
    def test_rejects_a_length_mismatch(
        self,
        states: tuple[int, ...],
        dims: tuple[int, ...],
        message: str,
    ) -> None:
        """One state per qudit is required."""
        with pytest.raises(ValueError, match=message):
            validate_basis_states(states, dims)

    @pytest.mark.parametrize("state", [3, 4, 100, -1, -10])
    def test_rejects_out_of_range_levels(self, state: int) -> None:
        """Levels must lie in ``[0, dim - 1]``."""
        with pytest.raises(
            ValueError,
            match=r"state for qudit 0 must be in \[0, 2\]",
        ):
            validate_basis_states((state,), (3,))

    def test_error_names_the_offending_qudit(self) -> None:
        """The index in the message is the target position."""
        with pytest.raises(ValueError, match="state for qudit 1 must be in"):
            validate_basis_states((0, 5), (3, 3))

    @pytest.mark.parametrize("state", [1.0, "1", None, True])
    def test_rejects_non_integer_states(self, state: Any) -> None:
        """Non-integers fail the delegated integer validation."""
        # NOTE: the docstring only advertises ValueError, but
        # validate_integer_range raises TypeError for a wrong type.
        with pytest.raises(TypeError, match="state for qudit 0 must be an"):
            validate_basis_states([state], (3,))

    @pytest.mark.parametrize("dim", [2, 3, 5, 8])
    def test_accepts_the_highest_level(self, dim: int) -> None:
        """``dim - 1`` is inside the closed range."""
        assert validate_basis_states((dim - 1,), (dim,)) == (dim - 1,)
