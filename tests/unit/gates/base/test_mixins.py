"""Software-contract tests for the qudit gate mixins.

:class:`QuditGateMixin` and :class:`QuditPhaseGateMixin` are ABCs, so
every test goes through a minimal concrete implementation defined in
this module rather than through a real gate. That keeps the mixin
contract (validation hooks, ``__array__`` semantics, abstractness)
separated from the gates that happen to use it.
"""

from __future__ import annotations

import inspect
from typing import Any, ClassVar

import numpy as np
import pytest

from qiskit_qudits.gates.base.mixins import (
    QuditGateMixin,
    QuditPhaseGateMixin,
)
from qiskit_qudits.utils.consts import MIN_QUDIT_DIM


class _MinimalGate(QuditGateMixin):
    """Smallest possible concrete :class:`QuditGateMixin`."""

    def __init__(self, num_qubits: int = 2) -> None:
        """Store the qubit count and pre-build the unitary.

        Args:
            num_qubits: Width of the fake gate.
        """
        self._num_qubits = num_qubits
        self._matrix = np.eye(1 << num_qubits, dtype=np.complex128)
        self.defined = False

    @property
    def num_qubits(self) -> int:
        """Number of qubits the fake gate acts on."""
        return self._num_qubits

    def _define(self) -> None:
        """Record that a decomposition was requested."""
        self.defined = True

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        """Return the *same* array on every call.

        Handing out a cached array is what makes the ``copy``
        semantics of :meth:`QuditGateMixin.__array__` observable.
        """
        return self._matrix


class _NarrowGate(_MinimalGate):
    """Implementation that only accepts dimensions in ``[4, 6]``."""

    MIN_DIM: ClassVar[int] = 4
    MAX_DIM: ClassVar[int] = 6


class _WideGate(_MinimalGate):
    """Implementation that accepts dimensions up to 32."""

    MAX_DIM: ClassVar[int] = 32


class _NoUnitaryGate(QuditGateMixin):
    """Concrete except for :meth:`_build_unitary`."""

    @property
    def num_qubits(self) -> int:
        """Number of qubits the fake gate acts on."""
        return 1

    def _define(self) -> None:
        """Do nothing."""


class _NoDefineGate(QuditGateMixin):
    """Concrete except for :meth:`_define`."""

    @property
    def num_qubits(self) -> int:
        """Number of qubits the fake gate acts on."""
        return 1

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        """Return a one-qubit identity."""
        return np.eye(2, dtype=np.complex128)


class _NoNumQubitsGate(QuditGateMixin):
    """Concrete except for the ``num_qubits`` property."""

    def _define(self) -> None:
        """Do nothing."""

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        """Return a one-qubit identity."""
        return np.eye(2, dtype=np.complex128)


class _MinimalPhaseGate(QuditPhaseGateMixin):
    """Smallest possible concrete :class:`QuditPhaseGateMixin`."""

    def __init__(self, theta: Any) -> None:
        """Validate and store the angle as ``params[0]``.

        Args:
            theta: Candidate rotation angle.
        """
        self._params: list[Any] = [self._validate_theta(theta)]

    @property
    def params(self) -> list[Any]:
        """Gate parameters, the angle first."""
        return self._params


class _NoParamsPhaseGate(QuditPhaseGateMixin):
    """Concrete except for the ``params`` property."""


class TestDimValidationHook:
    """``_validate_dim`` and the class-level bounds it reads."""

    def test_validate_dim_is_a_classmethod(self) -> None:
        """It must be callable without an instance."""
        raw = inspect.getattr_static(QuditGateMixin, "_validate_dim")
        assert isinstance(raw, classmethod)

    def test_validate_dim_binds_to_the_calling_class(self) -> None:
        """Subclasses see themselves as ``cls``."""
        assert _NarrowGate._validate_dim.__self__ is _NarrowGate

    def test_default_bounds(self) -> None:
        """The mixin defaults span the library-wide qudit range."""
        assert QuditGateMixin.MIN_DIM == MIN_QUDIT_DIM
        assert QuditGateMixin.MAX_DIM == 16

    def test_validate_dim_returns_a_plain_int(self) -> None:
        """NumPy integers are coerced on the way out."""
        result = QuditGateMixin._validate_dim(np.int64(5))
        assert type(result) is int
        assert result == 5

    @pytest.mark.parametrize("dim", [4, 5, 6])
    def test_narrowed_bounds_accept_their_own_range(self, dim: int) -> None:
        """A subclass may shrink the accepted range."""
        assert _NarrowGate._validate_dim(dim) == dim

    @pytest.mark.parametrize("dim", [2, 3, 7, 16])
    def test_narrowed_bounds_reject_everything_else(self, dim: int) -> None:
        """Dimensions the base class allows can be refused."""
        with pytest.raises(ValueError, match=r"dim must be in \[4, 6\]"):
            _NarrowGate._validate_dim(dim)

    @pytest.mark.parametrize("dim", [2, 3, 7, 16])
    def test_the_base_class_still_accepts_them(self, dim: int) -> None:
        """Narrowing is local to the subclass."""
        assert QuditGateMixin._validate_dim(dim) == dim

    def test_widened_bounds_accept_larger_dimensions(self) -> None:
        """A subclass may also grow the accepted range."""
        assert _WideGate._validate_dim(32) == 32

    def test_widening_does_not_leak_into_the_base_class(self) -> None:
        """The default maximum is unaffected by the subclass."""
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            QuditGateMixin._validate_dim(32)


class TestHilbertDim:
    """``hilbert_dim`` is derived from ``num_qubits`` alone."""

    @pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
    def test_hilbert_dim_is_two_to_the_num_qubits(
        self,
        num_qubits: int,
    ) -> None:
        """The mixin never looks at a dimension to compute it."""
        assert _MinimalGate(num_qubits).hilbert_dim == 2**num_qubits


class TestArrayProtocol:
    """``__array__`` and the NumPy conversions that call it."""

    def test_np_array_returns_the_unitary(self) -> None:
        """``np.array(gate)`` goes through ``__array__``."""
        gate = _MinimalGate(2)
        assert np.array_equal(np.array(gate), gate._matrix)

    def test_np_asarray_returns_the_unitary(self) -> None:
        """``np.asarray(gate)`` goes through ``__array__`` too."""
        gate = _MinimalGate(2)
        assert np.array_equal(np.asarray(gate), gate._matrix)

    def test_default_dtype_is_the_builders(self) -> None:
        """Without a ``dtype`` the built matrix is handed back as-is."""
        assert np.asarray(_MinimalGate(1)).dtype == np.complex128

    @pytest.mark.parametrize(
        "dtype",
        [np.complex64, np.complex128, complex],
        ids=["complex64", "complex128", "builtin-complex"],
    )
    def test_dtype_conversion_is_honoured(self, dtype: Any) -> None:
        """An explicit ``dtype`` is applied to the result."""
        result = np.asarray(_MinimalGate(1), dtype=dtype)
        assert result.dtype == np.dtype(dtype)

    def test_dtype_conversion_via_np_array(self) -> None:
        """The ``dtype`` also flows through ``np.array``."""
        result = np.array(_MinimalGate(1), dtype=np.complex64)
        assert result.dtype == np.complex64

    def test_copy_true_returns_a_new_array(self) -> None:
        """``copy=True`` never aliases the built matrix."""
        gate = _MinimalGate(1)
        assert gate.__array__(copy=True) is not gate._matrix

    def test_copy_true_result_is_independent(self) -> None:
        """Mutating the copy cannot affect a later conversion."""
        gate = _MinimalGate(1)
        copied = gate.__array__(copy=True)
        copied[0, 0] = 42.0
        assert np.asarray(gate)[0, 0] == 1.0

    def test_copy_none_avoids_copying_when_it_can(self) -> None:
        """The default path only copies when it has to."""
        gate = _MinimalGate(1)
        assert gate.__array__() is gate._matrix

    def test_copy_none_copies_when_the_dtype_changes(self) -> None:
        """A conversion forces a fresh allocation."""
        gate = _MinimalGate(1)
        result = gate.__array__(np.complex64)
        assert result is not gate._matrix
        assert result.dtype == np.complex64

    def test_copy_false_raises(self) -> None:
        """The unitary is built on the fly, so it cannot be shared."""
        with pytest.raises(ValueError, match="copy=False is not supported"):
            _MinimalGate(1).__array__(copy=False)

    def test_np_array_with_copy_false_raises(self) -> None:
        """NumPy's own no-copy request is refused as well."""
        with pytest.raises(ValueError, match="copy"):
            np.array(_MinimalGate(1), copy=False)


class TestGateMixinAbstractness:
    """A half-implemented gate must fail loudly at construction."""

    @pytest.mark.parametrize(
        ("gate_cls", "missing"),
        [
            (_NoUnitaryGate, "_build_unitary"),
            (_NoDefineGate, "_define"),
            (_NoNumQubitsGate, "num_qubits"),
        ],
        ids=["no-build-unitary", "no-define", "no-num-qubits"],
    )
    def test_incomplete_subclasses_cannot_be_instantiated(
        self,
        gate_cls: type[QuditGateMixin],
        missing: str,
    ) -> None:
        """Each missing member keeps the subclass abstract."""
        assert gate_cls.__abstractmethods__ == frozenset({missing})
        with pytest.raises(TypeError, match="abstract"):
            gate_cls()

    def test_a_full_implementation_is_instantiable(self) -> None:
        """The reference implementation has nothing left abstract."""
        assert _MinimalGate.__abstractmethods__ == frozenset()


class TestPhaseMixinValidation:
    """``_validate_theta``, the only logic the phase mixin adds."""

    @pytest.mark.parametrize(
        "theta",
        [0, 1, -3, 0.5, -2.25, np.float64(0.25), np.int32(3), np.float32(1)],
        ids=[
            "zero",
            "int",
            "negative-int",
            "float",
            "negative-float",
            "numpy-float64",
            "numpy-int32",
            "numpy-float32",
        ],
    )
    def test_valid_angles_are_returned_as_floats(self, theta: Any) -> None:
        """Ints and NumPy reals are coerced to plain floats."""
        result = QuditPhaseGateMixin._validate_theta(theta)
        assert type(result) is float
        assert result == pytest.approx(float(theta))

    @pytest.mark.parametrize(
        "theta",
        [True, False, "0.5", None, 1j, [0.5]],
        ids=["true", "false", "str", "none", "complex", "list"],
    )
    def test_non_real_angles_raise_type_error(self, theta: Any) -> None:
        """Booleans and non-numbers are not angles."""
        with pytest.raises(TypeError, match="theta must be a float"):
            QuditPhaseGateMixin._validate_theta(theta)

    @pytest.mark.parametrize(
        "theta",
        [float("nan"), float("inf"), float("-inf"), np.float64("nan")],
        ids=["nan", "inf", "-inf", "numpy-nan"],
    )
    def test_non_finite_angles_raise_value_error(self, theta: Any) -> None:
        """``nan`` and the infinities are rejected."""
        with pytest.raises(ValueError, match="theta must be finite"):
            QuditPhaseGateMixin._validate_theta(theta)


class TestPhaseMixinTheta:
    """The ``theta`` property and its ``params[0]`` contract."""

    @pytest.mark.parametrize("theta", [0.0, 0.75, -1.5])
    def test_theta_reads_the_first_parameter(self, theta: float) -> None:
        """``theta`` is exactly ``params[0]``."""
        gate = _MinimalPhaseGate(theta)
        assert gate.theta == pytest.approx(theta)
        assert gate.theta is gate.params[0]

    def test_theta_follows_later_parameter_changes(self) -> None:
        """Nothing is cached: the property re-reads ``params``."""
        gate = _MinimalPhaseGate(0.5)
        gate.params[0] = -0.5
        assert gate.theta == pytest.approx(-0.5)

    def test_params_is_left_abstract(self) -> None:
        """The mixin cannot supply the parameter storage itself."""
        assert _NoParamsPhaseGate.__abstractmethods__ == frozenset({"params"})
        with pytest.raises(TypeError, match="abstract"):
            _NoParamsPhaseGate()
