"""Unit tests for :mod:`qiskit_qudits.utils.validation`.

The validators are the library's front door for user input: they must
raise ``TypeError`` for the wrong *kind* of value and ``ValueError``
for the wrong *magnitude*, and they always return plain Python or
numpy objects rather than whatever exotic scalar came in.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from qiskit_qudits.utils.consts import MIN_QUDIT_DIM
from qiskit_qudits.utils.validation import (
    validate_dim,
    validate_float_finite,
    validate_integer_range,
    validate_vector,
)
from tests.helpers import ALL_DIMS

#: Values rejected with ``TypeError`` by the integer validator.
NON_INTEGERS: list[Any] = [
    1.0,
    2.5,
    float("nan"),
    True,
    False,
    np.True_,
    np.float64(3.0),
    "3",
    None,
    1j,
    [3],
    (3,),
]

#: Values rejected with ``TypeError`` by the float validator.
NON_FLOATS: list[Any] = [
    True,
    False,
    np.False_,
    "1.5",
    "",
    None,
    1j,
    np.complex128(1),
    [1.5],
    np.array([1.5]),
]


class TestValidateIntegerRange:
    """Behaviour of :func:`validate_integer_range`."""

    @pytest.mark.parametrize("value", [0, 1, -7, 10**18])
    def test_unbounded_accepts_any_integer(self, value: int) -> None:
        """Without bounds every integer is returned unchanged."""
        assert validate_integer_range(value) == value

    @pytest.mark.parametrize("value", [1, 2, 3, 4, 5])
    def test_two_sided_bounds_are_inclusive(self, value: int) -> None:
        """Both endpoints of ``[minimum, maximum]`` are allowed."""
        assert validate_integer_range(value, minimum=1, maximum=5) == value

    @pytest.mark.parametrize("value", [0, -1, 6, 100])
    def test_two_sided_bounds_reject_outside_values(
        self,
        value: int,
    ) -> None:
        """Values outside the closed interval raise ``ValueError``."""
        with pytest.raises(ValueError, match=r"must be in \[1, 5\]"):
            validate_integer_range(value, minimum=1, maximum=5)

    @pytest.mark.parametrize("value", [2, 3, 10**6])
    def test_lower_bound_only_accepts_from_the_minimum_up(
        self,
        value: int,
    ) -> None:
        """A lone ``minimum`` leaves the upper side unbounded."""
        assert validate_integer_range(value, minimum=2) == value

    @pytest.mark.parametrize("value", [1, 0, -10])
    def test_lower_bound_only_rejects_smaller_values(
        self,
        value: int,
    ) -> None:
        """Below ``minimum`` the ">=" message is used."""
        with pytest.raises(ValueError, match="must be >= 2"):
            validate_integer_range(value, minimum=2)

    @pytest.mark.parametrize("value", [10, 0, -(10**6)])
    def test_upper_bound_only_accepts_up_to_the_maximum(
        self,
        value: int,
    ) -> None:
        """A lone ``maximum`` leaves the lower side unbounded."""
        assert validate_integer_range(value, maximum=10) == value

    @pytest.mark.parametrize("value", [11, 12, 10**6])
    def test_upper_bound_only_rejects_larger_values(
        self,
        value: int,
    ) -> None:
        """Above ``maximum`` the "<=" message is used."""
        with pytest.raises(ValueError, match="must be <= 10"):
            validate_integer_range(value, maximum=10)

    def test_degenerate_range_accepts_only_one_value(self) -> None:
        """``minimum == maximum`` pins the value down."""
        assert validate_integer_range(3, minimum=3, maximum=3) == 3
        with pytest.raises(ValueError, match=r"must be in \[3, 3\]"):
            validate_integer_range(4, minimum=3, maximum=3)

    @pytest.mark.parametrize(
        "value",
        [np.int8(5), np.int32(5), np.int64(5), np.uint16(5)],
    )
    def test_coerces_numpy_integers_to_plain_int(self, value: Any) -> None:
        """The return value is always a builtin ``int``."""
        result = validate_integer_range(value, minimum=0, maximum=10)
        assert result == 5
        assert type(result) is int

    @pytest.mark.parametrize("value", NON_INTEGERS)
    def test_rejects_non_integers_with_type_error(
        self,
        value: Any,
    ) -> None:
        """Wrong kinds raise ``TypeError``, never ``ValueError``."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_integer_range(value)

    def test_type_error_names_the_offending_type(self) -> None:
        """The message reports the rejected type."""
        with pytest.raises(TypeError, match="got float"):
            validate_integer_range(1.5)

    def test_default_name_is_value(self) -> None:
        """Messages fall back to the generic ``value`` label."""
        with pytest.raises(ValueError, match=r"^value must be >= 0"):
            validate_integer_range(-1, minimum=0)

    @pytest.mark.parametrize(
        ("minimum", "maximum"),
        [(0, 3), (0, None), (None, 3)],
    )
    def test_name_is_used_in_range_messages(
        self,
        minimum: int | None,
        maximum: int | None,
    ) -> None:
        """The ``name`` argument prefixes every range message."""
        with pytest.raises(ValueError, match=r"^num_qudits "):
            validate_integer_range(
                -5 if minimum is not None else 99,
                minimum=minimum,
                maximum=maximum,
                name="num_qudits",
            )

    def test_name_is_used_in_type_messages(self) -> None:
        """The ``name`` argument prefixes the type message too."""
        with pytest.raises(TypeError, match=r"^angle must be an integer"):
            validate_integer_range("x", name="angle")


class TestValidateFloatFinite:
    """Behaviour of :func:`validate_float_finite`."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, 0.0),
            (3, 3.0),
            (-7, -7.0),
            (0.5, 0.5),
            (-2.25, -2.25),
            (np.int64(4), 4.0),
            (np.float32(0.5), 0.5),
            (np.float64(-1.5), -1.5),
        ],
    )
    def test_accepts_and_coerces_real_numbers(
        self,
        value: Any,
        expected: float,
    ) -> None:
        """Ints and numpy reals come back as builtin ``float``."""
        result = validate_float_finite(value)
        assert result == expected
        assert type(result) is float

    @pytest.mark.parametrize("value", NON_FLOATS)
    def test_rejects_non_reals_with_type_error(self, value: Any) -> None:
        """Booleans, strings, ``None`` and complex numbers fail."""
        with pytest.raises(TypeError, match="must be a float"):
            validate_float_finite(value)

    @pytest.mark.parametrize(
        "value",
        [
            float("nan"),
            float("inf"),
            float("-inf"),
            np.float64("nan"),
            np.float32("inf"),
            -np.inf,
        ],
    )
    def test_rejects_non_finite_values(self, value: Any) -> None:
        """NaN and both infinities raise ``ValueError``."""
        with pytest.raises(ValueError, match="must be finite"):
            validate_float_finite(value)

    def test_name_is_used_in_messages(self) -> None:
        """The ``name`` argument prefixes both error kinds."""
        with pytest.raises(TypeError, match=r"^theta must be a float"):
            validate_float_finite(None, name="theta")
        with pytest.raises(ValueError, match=r"^theta must be finite"):
            validate_float_finite(math.inf, name="theta")


class TestValidateVector:
    """Behaviour of :func:`validate_vector`."""

    @pytest.mark.parametrize(
        "value",
        [[1, 2, 3], (1, 2, 3), np.array([1, 2, 3]), np.arange(1, 4)],
    )
    def test_accepts_any_one_dimensional_sequence(
        self,
        value: Any,
    ) -> None:
        """Lists, tuples and arrays all become a 1-D array."""
        result = validate_vector(value)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (3,)
        assert np.array_equal(result, np.array([1, 2, 3], dtype=complex))

    def test_defaults_to_complex128(self) -> None:
        """Without an explicit dtype the result is complex."""
        assert validate_vector([1.0, 2.0]).dtype == np.complex128

    @pytest.mark.parametrize(
        "dtype",
        [np.float64, np.int64, np.complex64, np.complex128],
    )
    def test_casts_to_the_requested_dtype(self, dtype: Any) -> None:
        """The requested dtype is honoured exactly."""
        result = validate_vector([1, 2, 3], dtype=dtype)
        assert result.dtype == np.dtype(dtype)

    @pytest.mark.parametrize(
        "value",
        [[1, 2, 3], [1.5, 2.5], np.array([1 + 2j]), ["a", "b"]],
    )
    def test_dtype_none_keeps_the_natural_dtype(self, value: Any) -> None:
        """``dtype=None`` performs no conversion at all."""
        result = validate_vector(value, dtype=None)
        assert result.dtype == np.asarray(value).dtype

    def test_accepts_an_empty_vector(self) -> None:
        """An empty sequence is a legal 0-length vector."""
        result = validate_vector([])
        assert result.shape == (0,)

    def test_returns_a_new_array_for_sequence_input(self) -> None:
        """Values are copied out of the input sequence."""
        source = [1, 2, 3]
        result = validate_vector(source, dtype=None)
        result[0] = 99
        assert source[0] == 1

    @pytest.mark.parametrize(
        "value",
        [
            np.zeros((2, 2)),
            [[1, 2], [3, 4]],
            [[1, 2]],
            np.zeros((3, 1)),
            5,
            None,
            "abc",
            np.array([1, 2], dtype=object),
        ],
    )
    def test_rejects_non_vectors_with_type_error(
        self,
        value: Any,
    ) -> None:
        """Scalars, matrices and object arrays raise ``TypeError``."""
        with pytest.raises(TypeError, match="must be a 1-D vector"):
            validate_vector(value)

    def test_type_error_names_the_offending_type(self) -> None:
        """The message reports the rejected type."""
        with pytest.raises(TypeError, match="got ndarray"):
            validate_vector(np.zeros((2, 2)))

    @pytest.mark.parametrize("dtype", [np.complex128, np.float64, np.int64])
    def test_rejects_values_not_castable_to_the_dtype(
        self,
        dtype: Any,
    ) -> None:
        """A 1-D string array is a vector but not a numeric one."""
        # NOTE: numpy raises ValueError here; the validator re-raises it
        # as TypeError, so the "wrong kind of data" contract holds.
        with pytest.raises(TypeError, match="could not be cast to dtype"):
            validate_vector(["a", "b"], dtype=dtype)

    def test_cast_failure_chains_the_numpy_error(self) -> None:
        """The original numpy error is kept as ``__cause__``."""
        with pytest.raises(TypeError) as excinfo:
            validate_vector(["a", "b"])
        assert isinstance(excinfo.value.__cause__, (TypeError, ValueError))

    def test_name_is_used_in_messages(self) -> None:
        """The ``name`` argument prefixes both failure modes."""
        with pytest.raises(TypeError, match=r"^amplitudes must be a 1-D"):
            validate_vector(5, name="amplitudes")
        with pytest.raises(TypeError, match=r"^amplitudes could not be cast"):
            validate_vector(["a"], name="amplitudes")


class TestValidateDim:
    """Behaviour of :func:`validate_dim`."""

    @pytest.mark.parametrize("value", list(ALL_DIMS))
    def test_accepts_every_representative_dimension(
        self,
        value: int,
    ) -> None:
        """All dimensions used by the suite validate unchanged."""
        assert validate_dim(value) == value

    def test_coerces_to_plain_int(self) -> None:
        """A numpy integer comes back as a builtin ``int``."""
        result = validate_dim(np.int32(5))
        assert result == 5
        assert type(result) is int

    @pytest.mark.parametrize("value", [1, 0, -1, -100])
    def test_rejects_dimensions_below_the_minimum(
        self,
        value: int,
    ) -> None:
        """The lower bound is :data:`MIN_QUDIT_DIM`."""
        with pytest.raises(ValueError, match=r"^dim must be >= 2"):
            validate_dim(value)

    def test_accepts_the_minimum_dimension(self) -> None:
        """``MIN_QUDIT_DIM`` itself is valid."""
        assert validate_dim(MIN_QUDIT_DIM) == 2

    @pytest.mark.parametrize("value", [3.0, "3", None, True])
    def test_rejects_non_integers(self, value: Any) -> None:
        """Non-integers raise ``TypeError`` under the ``dim`` label."""
        with pytest.raises(TypeError, match=r"^dim must be an integer"):
            validate_dim(value)

    def test_has_no_upper_bound(self) -> None:
        """Arbitrarily large dimensions are accepted."""
        assert validate_dim(10**9) == 10**9

    @pytest.mark.parametrize("value", list(ALL_DIMS))
    def test_matches_the_delegated_call(self, value: int) -> None:
        """It is exactly ``validate_integer_range`` with minimum 2."""
        expected = validate_integer_range(
            value,
            minimum=MIN_QUDIT_DIM,
            name="dim",
        )
        assert validate_dim(value) == expected
