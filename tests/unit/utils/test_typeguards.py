"""Unit tests for :mod:`qiskit_qudits.utils.typeguards`.

The guards are deliberately stricter than ``isinstance``: ``bool`` is
an ``int`` for Python but never a valid qudit index or angle here, and
an object-dtype array is never a valid vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qiskit_qudits.utils.typeguards import is_integral, is_real, is_vector

#: Values every guard for numbers must accept as integers.
INTEGRAL_VALUES: list[Any] = [
    0,
    1,
    -5,
    10**30,
    np.int8(1),
    np.int16(-2),
    np.int32(3),
    np.int64(-4),
    np.uint8(5),
    np.uint64(6),
]

#: Values that are real but not integral.
NON_INTEGRAL_REAL_VALUES: list[Any] = [
    0.0,
    -0.0,
    2.5,
    -1e308,
    float("inf"),
    float("-inf"),
    float("nan"),
    np.float16(1.5),
    np.float32(-2.5),
    np.float64(3.5),
]

#: Values neither guard for numbers may accept.
NON_NUMERIC_VALUES: list[Any] = [
    True,
    False,
    np.True_,
    np.False_,
    "1",
    "1.5",
    "",
    None,
    1j,
    complex(1, 0),
    np.complex64(1),
    np.complex128(1),
    [1],
    (1,),
    {"a": 1},
    np.array(3),
    np.array([1, 2]),
    object(),
]


class TestIsIntegral:
    """Behaviour of :func:`is_integral`."""

    @pytest.mark.parametrize("value", INTEGRAL_VALUES)
    def test_accepts_python_and_numpy_integers(self, value: Any) -> None:
        """Plain ``int`` and every ``numpy.integer`` width pass."""
        assert is_integral(value) is True

    @pytest.mark.parametrize(
        "value",
        NON_INTEGRAL_REAL_VALUES + NON_NUMERIC_VALUES,
    )
    def test_rejects_everything_else(self, value: Any) -> None:
        """Floats, booleans, strings and containers are rejected."""
        assert is_integral(value) is False

    @pytest.mark.parametrize("value", [True, False, np.True_])
    def test_rejects_booleans_despite_int_subclassing(
        self,
        value: Any,
    ) -> None:
        """``bool`` is an ``int`` for Python but not for this guard."""
        assert isinstance(value, (bool, np.bool_))
        assert is_integral(value) is False

    def test_rejects_zero_dimensional_integer_arrays(self) -> None:
        """A 0-d ``ndarray`` is not a ``numpy.integer`` scalar."""
        assert is_integral(np.array(7)) is False


class TestIsReal:
    """Behaviour of :func:`is_real`."""

    @pytest.mark.parametrize(
        "value",
        INTEGRAL_VALUES + NON_INTEGRAL_REAL_VALUES,
    )
    def test_accepts_integers_and_floats(self, value: Any) -> None:
        """Every integral or floating value passes."""
        assert is_real(value) is True

    @pytest.mark.parametrize("value", NON_NUMERIC_VALUES)
    def test_rejects_non_real_values(self, value: Any) -> None:
        """Booleans, strings, complex numbers and arrays fail."""
        assert is_real(value) is False

    @pytest.mark.parametrize("value", INTEGRAL_VALUES)
    def test_every_integral_value_is_also_real(self, value: Any) -> None:
        """The integral guard is strictly narrower than the real one."""
        assert is_integral(value) is True
        assert is_real(value) is True

    @pytest.mark.parametrize(
        "value",
        [float("inf"), float("-inf"), float("nan"), np.float64("inf")],
    )
    def test_accepts_non_finite_floats(self, value: Any) -> None:
        """Finiteness is checked by the validators, not by the guard."""
        assert is_real(value) is True


class TestIsVector:
    """Behaviour of :func:`is_vector`."""

    @pytest.mark.parametrize(
        "value",
        [
            [1, 2, 3],
            [1.5, -2.5],
            [1 + 2j],
            (1, 2),
            (),
            [],
            range(4),
            np.array([1, 2, 3]),
            np.zeros(3),
            np.zeros(0),
            np.arange(6, dtype=np.complex128),
        ],
    )
    def test_accepts_one_dimensional_numeric_sequences(
        self,
        value: Any,
    ) -> None:
        """Lists, tuples, ranges and 1-D arrays are vectors."""
        assert is_vector(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            5,
            5.0,
            1j,
            True,
            None,
            "abc",
            b"ab",
            np.array(5),
            np.float64(1.0),
            {"a": 1},
            {1, 2},
        ],
    )
    def test_rejects_scalars_and_non_sequences(self, value: Any) -> None:
        """Anything numpy sees as 0-dimensional is not a vector."""
        assert is_vector(value) is False

    @pytest.mark.parametrize(
        "value",
        [
            [[1, 2], [3, 4]],
            [[1, 2]],
            np.zeros((2, 2)),
            np.zeros((1, 3)),
            np.zeros((3, 1)),
            np.zeros((2, 2, 2)),
        ],
    )
    def test_rejects_higher_dimensional_arrays(self, value: Any) -> None:
        """Only ``ndim == 1`` qualifies; a column vector does not."""
        assert is_vector(value) is False

    @pytest.mark.parametrize(
        "value",
        [
            np.array([1, 2], dtype=object),
            [None, 1],
            [{}, 1],
            [object(), object()],
        ],
    )
    def test_rejects_object_dtype(self, value: Any) -> None:
        """Object arrays carry no usable numeric payload."""
        assert is_vector(value) is False

    def test_rejects_ragged_input(self) -> None:
        """A ragged nested list cannot become a numeric array."""
        assert is_vector([[1, 2], [3]]) is False

    def test_accepts_a_list_of_strings(self) -> None:
        """A ``<U`` dtype is 1-D and non-object, so the guard passes."""
        # NOTE: the guard only rejects object dtype, so string arrays
        # slip through; ``validate_vector`` catches them at cast time.
        assert is_vector(["a", "b"]) is True
