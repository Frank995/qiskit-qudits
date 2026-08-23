"""Software-contract tests for :class:`QuditControlledGate`.

``QuditControlledGate`` is abstract in practice: :meth:`_build_unitary`
is still inherited unimplemented from ``QuditGateMixin``, so everything
here is exercised through the minimal concrete subclass defined below.
The dims-validation, qubit-range and stride bookkeeping the shared
``QuditMultiGateMixin`` provides is already covered, through
``QuditMultiGate``, in ``test_multi_gate_base.py``; this file focuses
on the control/target split layered on top.

Only the contract is covered: control/target layout, constructor
validation, Qiskit interoperability and the not-implemented inverse.
The physics lives in ``tests/quantum``.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from qiskit.circuit import ControlledGate, Gate, QuantumCircuit

from qiskit_qudits.gates.base.controlledgate import QuditControlledGate
from qiskit_qudits.gates.base.gate import QuditGate


class _DummyBaseGate(QuditGate):
    """Minimal single-qudit gate used as a controlled base gate."""

    def __init__(self, dim: Any) -> None:
        """Create the base gate.

        Args:
            dim: Candidate dimension, validated by the base class.
        """
        super().__init__("base-dummy", dim, [])

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        """Return the identity on the full Hilbert space."""
        return np.eye(self.hilbert_dim, dtype=np.complex128)


class _DummyControlledGate(QuditControlledGate):
    """Minimal concrete :class:`QuditControlledGate`."""

    def __init__(
        self,
        target_dim: Any,
        control_dims: Any,
        *,
        name: str = "cdummy",
        params: list[Any] | None = None,
        label: str | None = None,
    ) -> None:
        """Create the dummy controlled gate.

        Args:
            target_dim: Candidate target dimension.
            control_dims: Candidate control dimensions.
            name: Qiskit gate name.
            params: Gate parameters forwarded to Qiskit.
            label: Optional display label.
        """
        # NOTE: a base gate is always supplied because Qiskit's
        # ControlledGate delegates ``params`` to it; its dimension is
        # deliberately fixed so the dims validation below is what
        # raises in the validation tests.
        super().__init__(
            name,
            target_dim,
            control_dims,
            [] if params is None else params,
            base_gate=_DummyBaseGate(2),
            label=label,
        )

    def _define(self) -> None:
        """Assign a trivial (empty) qubit-level circuit."""
        self.definition = QuantumCircuit(self.num_qubits, name=self.name)

    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        """Return the identity on the full Hilbert space."""
        return np.eye(self.hilbert_dim, dtype=np.complex128)


class TestControlledLayout:
    """The control/target split of :class:`QuditControlledGate`."""

    def test_dims_put_the_controls_first(self) -> None:
        """Register order is ``(c_0, ..., c_{m-1}, t)``."""
        gate = _DummyControlledGate(5, [2, 3])
        assert gate.dims == (2, 3, 5)
        assert gate.target_dim == 5
        assert gate.control_dims == (2, 3)
        assert gate.num_control_qudits == 2
        assert gate.num_qudits == 3

    def test_qubit_level_bookkeeping(self) -> None:
        """Widths, control count and the default control state."""
        gate = _DummyControlledGate(5, [2, 3])
        assert gate.num_qubits == 6
        assert gate.num_ctrl_qubits == 3
        assert gate.ctrl_state == 0b111

    @pytest.mark.parametrize(
        ("target_dim", "expected"),
        [(3, 4), (4, 4), (5, 8)],
    )
    def test_target_hilbert_dim_pads_to_a_power_of_two(
        self,
        target_dim: int,
        expected: int,
    ) -> None:
        """``target_hilbert_dim`` mirrors the single-qudit case."""
        gate = _DummyControlledGate(target_dim, [2])
        assert gate.target_hilbert_dim == expected

    def test_control_and_target_ranges_partition_the_qubits(self) -> None:
        """Controls sit on the low qubits, the target on the high."""
        gate = _DummyControlledGate(5, [2, 3])
        assert gate._control_qudit_range(0) == range(1)
        assert gate._control_qudit_range(1) == range(1, 3)
        assert gate._target_qudit_range() == range(3, 6)

    def test_control_and_target_strides_split_the_layout(self) -> None:
        """The target stride follows all the control strides."""
        gate = _DummyControlledGate(5, [2, 3])
        control_strides, target_stride = gate._compute_control_target_strides()
        assert control_strides == [1, 2]
        assert target_stride == 8

    def test_leakage_bookkeeping_is_inherited(self) -> None:
        """The mixin properties see the joint register."""
        gate = _DummyControlledGate(3, [2])
        assert gate.fills_hilbert_space is False
        assert gate.num_invalid_states == 2
        assert _DummyControlledGate(4, [2]).fills_hilbert_space is True


class TestControlledValidation:
    """Constructor-level validation of the controlled layout."""

    def test_at_least_one_control_is_required(self) -> None:
        """Empty ``control_dims`` is rejected explicitly."""
        with pytest.raises(ValueError, match="at least one control"):
            _DummyControlledGate(3, [])

    @pytest.mark.parametrize("dim", [1, 17])
    def test_out_of_range_dims_are_rejected(self, dim: int) -> None:
        """Both operand kinds share the ``[2, 16]`` bound."""
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            _DummyControlledGate(dim, [2])
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            _DummyControlledGate(2, [dim])

    def test_non_integer_dims_are_rejected(self) -> None:
        """Non-integer dimensions raise ``TypeError``."""
        with pytest.raises(TypeError, match="dim must be an integer"):
            _DummyControlledGate(3.0, [2])

    def test_too_many_qudits_are_rejected(self) -> None:
        """Controls plus target may not exceed ``MAX_QUDITS``."""
        with pytest.raises(ValueError, match="at most 8"):
            _DummyControlledGate(2, [2] * 8)

    def test_num_qubits_is_not_a_constructor_parameter(self) -> None:
        """The caller cannot choose the encoding width."""
        parameters = list(
            inspect.signature(QuditControlledGate.__init__).parameters,
        )
        assert "num_qubits" not in parameters
        assert parameters == [
            "self",
            "name",
            "target_dim",
            "control_dims",
            "params",
            "base_gate",
            "label",
        ]


class TestControlledQiskitInterface:
    """What Qiskit itself sees when handed a controlled gate."""

    def test_is_a_qiskit_controlled_gate_subclass(self) -> None:
        """Transpiler compatibility rests on this inheritance."""
        assert issubclass(QuditControlledGate, ControlledGate)
        gate = _DummyControlledGate(3, [2])
        assert isinstance(gate, ControlledGate)
        assert isinstance(gate, Gate)
        assert gate.num_clbits == 0

    def test_the_base_gate_is_stored(self) -> None:
        """The uncontrolled partner is kept for Qiskit."""
        gate = _DummyControlledGate(3, [2])
        assert isinstance(gate.base_gate, _DummyBaseGate)

    def test_params_delegate_to_the_base_gate(self) -> None:
        """Qiskit's ControlledGate reads params off the base gate."""
        gate = _DummyControlledGate(3, [2], params=[0.25])
        assert gate.params == [0.25]
        assert gate.base_gate is not None
        assert gate.base_gate.params == [0.25]

    @pytest.mark.parametrize("label", [None, "custom-label"])
    def test_label_is_forwarded(self, label: str | None) -> None:
        """The label is stored verbatim, ``None`` included."""
        assert _DummyControlledGate(3, [2], label=label).label == label


class TestAbstractness:
    """The base class is abstract, but only partially so."""

    def test_qudit_controlled_gate_cannot_be_instantiated(self) -> None:
        """``_build_unitary`` keeps the base class abstract."""
        with pytest.raises(TypeError, match="abstract"):
            QuditControlledGate("g", 3, [3], [])

    def test_only_build_unitary_is_left_abstract(self) -> None:
        """``_define`` is satisfied by Qiskit's no-op implementation."""
        # NOTE: as for ``QuditGate``, ``Instruction`` comes first in
        # the MRO and already provides a no-op ``_define``, so ABC
        # never sees it as missing.
        assert QuditControlledGate.__abstractmethods__ == frozenset(
            {"_build_unitary"},
        )


class TestInverse:
    """The base implementation refuses to guess an inverse."""

    def test_inverse_raises_not_implemented(self) -> None:
        """The error names the offending subclass."""
        with pytest.raises(
            NotImplementedError,
            match="_DummyControlledGate must implement inverse",
        ):
            _DummyControlledGate(3, [2]).inverse()

    def test_inverse_raises_not_implemented_when_annotated(self) -> None:
        """``annotated=True`` does not unlock a fallback."""
        with pytest.raises(
            NotImplementedError,
            match="_DummyControlledGate must implement inverse",
        ):
            _DummyControlledGate(3, [2]).inverse(annotated=True)
