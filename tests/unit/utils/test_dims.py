"""Unit tests for :mod:`qiskit_qudits.utils.dims`.

Also covers :data:`qiskit_qudits.utils.consts.MIN_QUDIT_DIM`, whose
only job is to define the lower bound enforced here.
"""

from __future__ import annotations

import itertools

import pytest

from qiskit_qudits.utils.consts import MIN_QUDIT_DIM
from qiskit_qudits.utils.dims import qubits_per_qudit
from tests.helpers import ALL_DIMS

#: Dimensions and the number of qubits their encoding needs.
EXACT_WIDTHS = [
    (2, 1),
    (3, 2),
    (4, 2),
    (5, 3),
    (6, 3),
    (7, 3),
    (8, 3),
    (9, 4),
    (15, 4),
    (16, 4),
    (17, 5),
    (32, 5),
    (33, 6),
    (1024, 10),
    (1025, 11),
]


class TestQubitsPerQudit:
    """Behaviour of :func:`qubits_per_qudit`."""

    @pytest.mark.parametrize(("dim", "expected"), EXACT_WIDTHS)
    def test_returns_the_documented_width(
        self,
        dim: int,
        expected: int,
    ) -> None:
        """The width matches the hand-computed ceil(log2 d) table."""
        assert qubits_per_qudit(dim) == expected

    @pytest.mark.parametrize("exponent", list(range(1, 13)))
    def test_is_exact_on_powers_of_two(self, exponent: int) -> None:
        """A dimension of 2**n needs exactly n qubits."""
        assert qubits_per_qudit(2**exponent) == exponent

    @pytest.mark.parametrize("exponent", list(range(1, 13)))
    def test_one_level_above_a_power_of_two_costs_one_qubit_more(
        self,
        exponent: int,
    ) -> None:
        """Crossing a power of two adds exactly one qubit."""
        assert qubits_per_qudit(2**exponent + 1) == exponent + 1

    def test_is_monotonically_non_decreasing(self) -> None:
        """Wider dimensions never need fewer qubits."""
        widths = [qubits_per_qudit(dim) for dim in range(2, 130)]
        assert all(
            later >= earlier for earlier, later in itertools.pairwise(widths)
        )

    def test_returns_the_smallest_sufficient_width(self) -> None:
        """2**n is large enough for d, and 2**(n - 1) is not."""
        for dim in range(2, 130):
            width = qubits_per_qudit(dim)
            assert (1 << width) >= dim, dim
            assert (1 << (width - 1)) < dim, dim

    def test_accepts_every_dimension_of_the_test_suite(self) -> None:
        """The shared representative dimensions are all encodable."""
        for dim in ALL_DIMS:
            assert (1 << qubits_per_qudit(dim)) >= dim

    @pytest.mark.parametrize("dim", [1, 0, -1, -2, -1024])
    def test_rejects_dimensions_below_two(self, dim: int) -> None:
        """A qudit needs at least two levels."""
        with pytest.raises(ValueError, match="dimension at least 2"):
            qubits_per_qudit(dim)

    def test_rejects_true_because_it_equals_one(self) -> None:
        """``True`` is an ``int`` worth 1, so it is out of range."""
        dim: int = True
        with pytest.raises(ValueError, match="dimension at least 2"):
            qubits_per_qudit(dim)


class TestMinQuditDim:
    """Behaviour of the :data:`MIN_QUDIT_DIM` constant."""

    def test_is_two(self) -> None:
        """The smallest meaningful qudit is a qubit."""
        assert MIN_QUDIT_DIM == 2

    def test_is_the_smallest_accepted_dimension(self) -> None:
        """It is accepted while the value just below it is not."""
        assert qubits_per_qudit(MIN_QUDIT_DIM) == 1
        with pytest.raises(ValueError, match="dimension at least 2"):
            qubits_per_qudit(MIN_QUDIT_DIM - 1)
