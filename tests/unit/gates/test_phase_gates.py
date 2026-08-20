"""Software-contract tests for the :class:`QuditPGate` family.

``Z``, ``Zdg``, ``S``, ``Sdg``, ``T`` and ``Tdg`` are nothing but a
:class:`QuditPGate` with a hard-coded angle, so the only thing tested
here is that wiring: the angle each subclass forwards, the ``theta``
plumbing of ``QuditPGate`` itself and its parameter validation.

Behaviour inherited unchanged from ``QuditGate`` (names, labels,
dimension bookkeeping, inverse *types*) is covered by
``test_gate_base.py`` and ``test_single_gates.py``; the phases
themselves are checked in ``tests/quantum``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qiskit_qudits.gates import (
    QuditPGate,
    QuditSdgGate,
    QuditSGate,
    QuditTdgGate,
    QuditTGate,
    QuditZdgGate,
    QuditZGate,
)
from qiskit_qudits.gates.base.gate import QuditGate
from qiskit_qudits.gates.base.mixins import QuditPhaseGateMixin

#: ``(class, angle it must forward to QuditPGate)``. The classes are
#: typed loosely because their ``__init__`` drops the ``theta``
#: argument that :class:`QuditPGate` requires.
FIXED_ANGLE_GATES: list[tuple[type[Any], float]] = [
    (QuditZGate, np.pi),
    (QuditZdgGate, -np.pi),
    (QuditSGate, np.pi / 2),
    (QuditSdgGate, -np.pi / 2),
    (QuditTGate, np.pi / 4),
    (QuditTdgGate, -np.pi / 4),
]

#: Readable ids for the table above.
ANGLE_IDS = [gate_cls.__name__ for gate_cls, _ in FIXED_ANGLE_GATES]

#: Just the classes.
FIXED_ANGLE_CLASSES = [gate_cls for gate_cls, _ in FIXED_ANGLE_GATES]


class TestFixedAngleSubclasses:
    """Each subclass is a ``QuditPGate`` at one specific angle."""

    @pytest.mark.parametrize(
        ("gate_cls", "expected"),
        FIXED_ANGLE_GATES,
        ids=ANGLE_IDS,
    )
    def test_theta_is_the_hard_coded_angle(
        self,
        gate_cls: type[Any],
        expected: float,
    ) -> None:
        """``theta`` exposes the angle the subclass forwards."""
        assert gate_cls(3).theta == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("gate_cls", "expected"),
        FIXED_ANGLE_GATES,
        ids=ANGLE_IDS,
    )
    def test_angle_reaches_the_qiskit_parameters(
        self,
        gate_cls: type[Any],
        expected: float,
    ) -> None:
        """The angle is visible to Qiskit as ``params[0]``."""
        assert gate_cls(3).params[0] == pytest.approx(expected)

    @pytest.mark.parametrize(
        "gate_cls",
        FIXED_ANGLE_CLASSES,
        ids=ANGLE_IDS,
    )
    def test_the_angle_is_the_only_parameter(
        self,
        gate_cls: type[Any],
    ) -> None:
        """No subclass smuggles extra parameters in."""
        assert len(gate_cls(3).params) == 1

    @pytest.mark.parametrize(
        "gate_cls",
        FIXED_ANGLE_CLASSES,
        ids=ANGLE_IDS,
    )
    def test_the_angle_is_not_a_constructor_argument(
        self,
        gate_cls: type[Any],
    ) -> None:
        """The whole point of the subclass is to fix the angle."""
        with pytest.raises(TypeError, match="argument"):
            gate_cls(3, 0.5)

    @pytest.mark.parametrize(
        "gate_cls",
        FIXED_ANGLE_CLASSES,
        ids=ANGLE_IDS,
    )
    def test_the_paired_dagger_carries_the_opposite_angle(
        self,
        gate_cls: type[Any],
    ) -> None:
        """Inverting a fixed-angle gate negates its angle."""
        gate = gate_cls(3)
        assert gate.inverse().theta == pytest.approx(-gate.theta)

    @pytest.mark.parametrize(
        "gate_cls",
        FIXED_ANGLE_CLASSES,
        ids=ANGLE_IDS,
    )
    def test_subclasses_are_phase_gates(
        self,
        gate_cls: type[Any],
    ) -> None:
        """Every fixed-angle gate is usable as a ``QuditPGate``."""
        gate = gate_cls(3)
        assert isinstance(gate, QuditPGate)
        assert isinstance(gate, QuditPhaseGateMixin)

    @pytest.mark.parametrize(
        "gate_cls",
        FIXED_ANGLE_CLASSES,
        ids=ANGLE_IDS,
    )
    def test_subclasses_are_qudit_gates(
        self,
        gate_cls: type[Any],
    ) -> None:
        """They also remain plain qudit gates."""
        assert isinstance(gate_cls(3), QuditGate)


class TestQuditPGateAngle:
    """``QuditPGate`` stores and exposes a caller-supplied angle."""

    @pytest.mark.parametrize("theta", [0.0, 0.75, -1.5, np.pi])
    def test_theta_is_stored_as_the_first_parameter(
        self,
        theta: float,
    ) -> None:
        """The subclassing contract of the mixin is honoured."""
        assert QuditPGate(3, theta).params == [pytest.approx(theta)]

    @pytest.mark.parametrize("theta", [0.0, 0.75, -1.5, np.pi])
    def test_theta_property_reads_that_parameter(
        self,
        theta: float,
    ) -> None:
        """``theta`` and ``params[0]`` are the same object."""
        gate = QuditPGate(3, theta)
        assert gate.theta is gate.params[0]

    @pytest.mark.parametrize(
        "theta",
        [1, -2, np.int32(3), np.float64(0.5)],
        ids=["int", "negative-int", "numpy-int32", "numpy-float64"],
    )
    def test_angles_are_coerced_to_plain_floats(self, theta: Any) -> None:
        """Integers and NumPy reals are accepted and normalised."""
        gate = QuditPGate(3, theta)
        assert type(gate.theta) is float
        assert gate.theta == pytest.approx(float(theta))

    @pytest.mark.parametrize(
        "theta",
        [True, "0.5", None, 1j],
        ids=["bool", "str", "none", "complex"],
    )
    def test_non_real_angles_are_rejected(self, theta: Any) -> None:
        """Validation happens at construction, not at first use."""
        with pytest.raises(TypeError, match="theta must be a float"):
            QuditPGate(3, theta)

    @pytest.mark.parametrize(
        "theta",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_non_finite_angles_are_rejected(self, theta: float) -> None:
        """A non-finite phase would produce a non-unitary matrix."""
        with pytest.raises(ValueError, match="theta must be finite"):
            QuditPGate(3, theta)

    def test_the_angle_is_required(self) -> None:
        """Unlike its subclasses, ``QuditPGate`` needs an angle."""
        with pytest.raises(TypeError, match="theta"):
            QuditPGate(3)


class TestQuditPGateInverse:
    """``QuditPGate`` inverts itself by negating the angle."""

    @pytest.mark.parametrize("theta", [0.0, 0.75, -1.5, np.pi])
    def test_inverse_negates_the_angle(self, theta: float) -> None:
        """``P(-theta)`` is the announced inverse."""
        assert QuditPGate(3, theta).inverse().theta == pytest.approx(-theta)

    @pytest.mark.parametrize("dim", [2, 3, 5, 16])
    def test_inverse_keeps_the_dimension_and_the_class(
        self,
        dim: int,
    ) -> None:
        """Nothing but the angle changes."""
        inverse = QuditPGate(dim, 0.4).inverse()
        assert type(inverse) is QuditPGate
        assert inverse.dim == dim
