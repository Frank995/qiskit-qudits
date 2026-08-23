"""Software-contract tests for the concrete controlled qudit gates.

``SUMX``, ``SUMXdg`` and ``SUMP`` are the controlled entries of the
gate catalogue. What is pinned down here is the *controlled*
bookkeeping: the register order (controls first, target last), the
qubit-level attributes derived from the qudit dimensions
(``num_ctrl_qubits``, ``ctrl_state``), the stored ``base_gate``, the
inverse wiring and the package exports.

Matrices, gate algebra and the equivalence of ``definition`` with
``_build_unitary`` belong to ``tests/quantum``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qiskit.circuit import ControlledGate

from qiskit_qudits import gates
from qiskit_qudits.gates import (
    QuditPGate,
    QuditSUMPGate,
    QuditSUMXdgGate,
    QuditSUMXGate,
    QuditXdgGate,
    QuditXGate,
    controlled,
)
from qiskit_qudits.gates.base.controlledgate import QuditControlledGate
from qiskit_qudits.gates.base.mixins import QuditPhaseGateMixin

#: Angle used whenever the parametrised SUMP gate has to be built.
THETA = 0.7

#: ``(class, expected name, expected inverse class)`` for every
#: controlled gate.
GATES: list[
    tuple[type[QuditControlledGate], str, type[QuditControlledGate]]
] = [
    (QuditSUMXGate, "SUMX", QuditSUMXdgGate),
    (QuditSUMXdgGate, "SUMXdg", QuditSUMXGate),
    (QuditSUMPGate, "SUMP", QuditSUMPGate),
]

#: Readable ids for the table above.
GATE_IDS = [gate_cls.__name__ for gate_cls, _, _ in GATES]

#: The table reduced to ``(class, expected name)``.
NAMED_GATES = [(gate_cls, name) for gate_cls, name, _ in GATES]

#: The table reduced to ``(class, expected inverse class)``.
INVERSE_GATES = [(gate_cls, inverse) for gate_cls, _, inverse in GATES]

#: Just the classes.
GATE_CLASSES: list[type[QuditControlledGate]] = [
    gate_cls for gate_cls, _, _ in GATES
]

#: ``(class, expected base gate class)``.
BASE_GATES = [
    (QuditSUMXGate, QuditXGate),
    (QuditSUMXdgGate, QuditXdgGate),
    (QuditSUMPGate, QuditPGate),
]

#: ``((target, controls), total qubit count)`` worked out by hand so
#: the test never validates the library against itself.
QUBIT_TABLE = [
    ((2, (2,)), 2),
    ((3, (2,)), 3),
    ((2, (3,)), 3),
    ((4, (4,)), 4),
    ((5, (2, 3)), 6),
]

#: ``((target, controls), control qubit count)`` worked out by hand.
CTRL_QUBIT_TABLE = [
    ((2, (2,)), 1),
    ((3, (3,)), 2),
    ((5, (2, 3)), 3),
    ((2, (4, 4)), 4),
]

#: ``(target dim, target Hilbert-space dimension)`` pairs.
TARGET_HILBERT_TABLE = [(2, 2), (3, 4), (5, 8), (8, 8)]

#: ``((target, controls), fills_hilbert_space)`` pairs.
FILLING_TABLE = [
    ((4, (2,)), True),
    ((2, (2, 4)), True),
    ((3, (2,)), False),
    ((2, (3,)), False),
    ((4, (4, 3)), False),
]

#: ``((target, controls), 2**n - prod(dims))`` pairs.
INVALID_STATES_TABLE = [
    ((4, (4,)), 0),
    ((3, (2,)), 2),
    ((5, (3,)), 17),
    ((3, (2, 3)), 14),
]

#: One configuration per decomposition branch: ``(3, (2,))`` leaves
#: invalid states behind, ``(4, (2,))`` fills its qubits exactly.
BRANCH_CONFIGS = [(3, (2,)), (4, (2,))]

#: Every controlled-gate test iterates over this parametrization.
parametrize_gate_classes = pytest.mark.parametrize(
    "gate_cls",
    GATE_CLASSES,
    ids=GATE_IDS,
)


def make_gate(
    gate_cls: type[QuditControlledGate],
    target_dim: Any,
    control_dims: Any,
    *,
    label: str | None = None,
) -> QuditControlledGate:
    """Build ``gate_cls``, supplying ``theta`` when it is required.

    Args:
        gate_cls: The gate class to instantiate.
        target_dim: Target qudit dimension.
        control_dims: Control qudit dimensions, in register order.
        label: Optional display label.

    Returns:
        A freshly built gate.
    """
    if gate_cls is QuditSUMPGate:
        return QuditSUMPGate(target_dim, control_dims, THETA, label=label)
    return gate_cls(target_dim, control_dims, label=label)


class TestNaming:
    """Every gate advertises a stable, hard-coded name."""

    @pytest.mark.parametrize(
        ("gate_cls", "expected"),
        NAMED_GATES,
        ids=GATE_IDS,
    )
    def test_gate_name_class_var(
        self,
        gate_cls: type[QuditControlledGate],
        expected: str,
    ) -> None:
        """``gate_name`` is readable without an instance."""
        name: str = gate_cls.gate_name
        assert name == expected

    @pytest.mark.parametrize(
        ("gate_cls", "expected"),
        NAMED_GATES,
        ids=GATE_IDS,
    )
    def test_instance_name_matches_the_class_var(
        self,
        gate_cls: type[QuditControlledGate],
        expected: str,
    ) -> None:
        """The Qiskit ``name`` is taken from ``gate_name``."""
        assert make_gate(gate_cls, 3, (2,)).name == expected

    def test_names_are_unique(self) -> None:
        """No two gates share a Qiskit name."""
        names = [name for _, name, _ in GATES]
        assert len(set(names)) == len(names)


class TestConstruction:
    """Dimension handling shared by the controlled catalogue."""

    @pytest.mark.parametrize(("config", "expected"), QUBIT_TABLE)
    @parametrize_gate_classes
    def test_num_qubits_follows_the_dimensions(
        self,
        gate_cls: type[QuditControlledGate],
        config: tuple[int, tuple[int, ...]],
        expected: int,
    ) -> None:
        """Each gate is as wide as the joint encoding requires."""
        target_dim, control_dims = config
        gate = make_gate(gate_cls, target_dim, control_dims)
        assert gate.num_qubits == expected

    @parametrize_gate_classes
    def test_dims_list_the_controls_first(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """Register order is ``(c_0, ..., c_{m-1}, t)``."""
        gate = make_gate(gate_cls, 5, (2, 3))
        assert gate.dims == (2, 3, 5)
        assert gate.target_dim == 5
        assert gate.control_dims == (2, 3)
        assert gate.num_control_qudits == 2
        assert gate.num_qudits == 3

    @parametrize_gate_classes
    def test_dimensions_are_coerced_to_plain_ints(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """NumPy dimensions survive as ordinary ``int`` values."""
        gate = make_gate(gate_cls, np.int64(3), np.array([2, 3]))
        assert gate.target_dim == 3
        assert type(gate.target_dim) is int
        assert gate.control_dims == (2, 3)
        assert all(type(dim) is int for dim in gate.control_dims)

    @pytest.mark.parametrize(
        ("target_dim", "expected"),
        TARGET_HILBERT_TABLE,
    )
    @parametrize_gate_classes
    def test_target_hilbert_dim_pads_to_a_power_of_two(
        self,
        gate_cls: type[QuditControlledGate],
        target_dim: int,
        expected: int,
    ) -> None:
        """``target_hilbert_dim`` is ``2**ceil(log2 d_t)``."""
        gate = make_gate(gate_cls, target_dim, (2,))
        assert gate.target_hilbert_dim == expected

    @pytest.mark.parametrize(("config", "expected"), FILLING_TABLE)
    @parametrize_gate_classes
    def test_fills_hilbert_space_flags_powers_of_two(
        self,
        gate_cls: type[QuditControlledGate],
        config: tuple[int, tuple[int, ...]],
        expected: bool,
    ) -> None:
        """The flag needs *every* dimension to be a power of two."""
        target_dim, control_dims = config
        gate = make_gate(gate_cls, target_dim, control_dims)
        assert gate.fills_hilbert_space is expected

    @pytest.mark.parametrize(("config", "expected"), INVALID_STATES_TABLE)
    @parametrize_gate_classes
    def test_num_invalid_states_counts_the_leftover_basis_states(
        self,
        gate_cls: type[QuditControlledGate],
        config: tuple[int, tuple[int, ...]],
        expected: int,
    ) -> None:
        """``num_invalid_states`` is ``2**n - prod(dims)``."""
        target_dim, control_dims = config
        gate = make_gate(gate_cls, target_dim, control_dims)
        assert gate.num_invalid_states == expected

    @pytest.mark.parametrize("label", [None, "custom-label"])
    @parametrize_gate_classes
    def test_label_is_stored(
        self,
        gate_cls: type[QuditControlledGate],
        label: str | None,
    ) -> None:
        """The label is kept verbatim, ``None`` included."""
        gate = make_gate(gate_cls, 3, (2,), label=label)
        assert gate.label == label


class TestControlAttributes:
    """The qubit-level control attributes Qiskit sees."""

    @pytest.mark.parametrize(("config", "expected"), CTRL_QUBIT_TABLE)
    @parametrize_gate_classes
    def test_num_ctrl_qubits_is_the_encoded_control_width(
        self,
        gate_cls: type[QuditControlledGate],
        config: tuple[int, tuple[int, ...]],
        expected: int,
    ) -> None:
        """Every control qudit contributes its own encoding width."""
        target_dim, control_dims = config
        gate = make_gate(gate_cls, target_dim, control_dims)
        assert gate.num_ctrl_qubits == expected

    @pytest.mark.parametrize("config", [(3, (3,)), (5, (2, 3))])
    @parametrize_gate_classes
    def test_ctrl_state_defaults_to_all_control_qubits_high(
        self,
        gate_cls: type[QuditControlledGate],
        config: tuple[int, tuple[int, ...]],
    ) -> None:
        """``ctrl_state`` customisation is not supported yet."""
        target_dim, control_dims = config
        gate = make_gate(gate_cls, target_dim, control_dims)
        assert gate.ctrl_state == (1 << gate.num_ctrl_qubits) - 1

    @parametrize_gate_classes
    def test_instances_are_qiskit_controlled_gates(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """Transpiler compatibility rests on this inheritance."""
        assert isinstance(make_gate(gate_cls, 3, (2,)), ControlledGate)

    @parametrize_gate_classes
    def test_gates_carry_no_classical_bits(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """A controlled gate is purely quantum."""
        assert make_gate(gate_cls, 3, (2,)).num_clbits == 0


class TestBaseGate:
    """Every controlled gate advertises its uncontrolled partner."""

    @pytest.mark.parametrize(
        ("gate_cls", "base_cls"),
        BASE_GATES,
        ids=GATE_IDS,
    )
    def test_base_gate_is_the_uncontrolled_gate(
        self,
        gate_cls: type[QuditControlledGate],
        base_cls: type,
    ) -> None:
        """The base gate class and dimension match the target."""
        # NOTE: Qiskit's ControlledGate copies the base gate, so the
        # check is by type and dimension, never by identity.
        gate = make_gate(gate_cls, 5, (2, 3))
        assert isinstance(gate.base_gate, base_cls)
        assert gate.base_gate.dim == 5

    def test_sump_base_gate_carries_the_angle(self) -> None:
        """The base ``P`` gate is built at the same angle."""
        gate = QuditSUMPGate(3, (2,), THETA)
        assert gate.base_gate is not None
        assert gate.base_gate.theta == pytest.approx(THETA)


class TestValidation:
    """Constructor-level validation of the qudit dimensions."""

    @pytest.mark.parametrize("dim", [1, 17])
    @parametrize_gate_classes
    def test_out_of_range_target_dim_is_rejected(
        self,
        gate_cls: type[QuditControlledGate],
        dim: int,
    ) -> None:
        """The target dimension must lie in ``[2, 16]``."""
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            make_gate(gate_cls, dim, (2,))

    @pytest.mark.parametrize("dim", [1, 17])
    @parametrize_gate_classes
    def test_out_of_range_control_dim_is_rejected(
        self,
        gate_cls: type[QuditControlledGate],
        dim: int,
    ) -> None:
        """Control dimensions are validated one by one."""
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            make_gate(gate_cls, 2, (dim,))

    @parametrize_gate_classes
    def test_non_integer_target_dim_is_rejected(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """A float target dimension is a ``TypeError``."""
        with pytest.raises(TypeError, match="dim must be an integer"):
            make_gate(gate_cls, 3.0, (2,))

    @parametrize_gate_classes
    def test_non_integer_control_dim_is_rejected(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """A float control dimension is a ``TypeError``."""
        with pytest.raises(TypeError, match="dim must be an integer"):
            make_gate(gate_cls, 3, (2.5,))

    @parametrize_gate_classes
    def test_at_least_one_control_is_required(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """A ``ControlledGate`` needs at least one control qudit."""
        with pytest.raises(ValueError, match="at least one control"):
            make_gate(gate_cls, 3, ())

    @parametrize_gate_classes
    def test_too_many_qudits_are_rejected(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """Controls plus target may not exceed ``MAX_QUDITS``."""
        with pytest.raises(ValueError, match="at most 8"):
            make_gate(gate_cls, 2, (2,) * 8)


class TestDefinition:
    """The lazily built qubit-level decomposition."""

    def test_definition_is_built_lazily(self) -> None:
        """Nothing is decomposed until ``definition`` is read."""
        gate = QuditSUMXGate(3, (2,))
        assert gate._definition is None
        assert gate.definition is not None

    def test_definition_is_cached(self) -> None:
        """The circuit is built once and then reused."""
        gate = QuditSUMXGate(3, (2,))
        assert gate.definition is gate.definition

    @pytest.mark.parametrize("config", BRANCH_CONFIGS)
    @parametrize_gate_classes
    def test_definition_is_named_after_the_label(
        self,
        gate_cls: type[QuditControlledGate],
        config: tuple[int, tuple[int, ...]],
    ) -> None:
        """A label renames the decomposition circuit."""
        target_dim, control_dims = config
        gate = make_gate(gate_cls, target_dim, control_dims, label="my-label")
        definition = gate.definition
        assert definition is not None
        assert definition.name == "my-label"

    @pytest.mark.parametrize("config", BRANCH_CONFIGS)
    @parametrize_gate_classes
    def test_definition_falls_back_to_the_gate_name(
        self,
        gate_cls: type[QuditControlledGate],
        config: tuple[int, tuple[int, ...]],
    ) -> None:
        """Without a label the gate name is used instead."""
        target_dim, control_dims = config
        gate = make_gate(gate_cls, target_dim, control_dims)
        definition = gate.definition
        assert definition is not None
        assert definition.name == gate.name

    @pytest.mark.parametrize("config", BRANCH_CONFIGS)
    @parametrize_gate_classes
    def test_definition_is_as_wide_as_the_gate(
        self,
        gate_cls: type[QuditControlledGate],
        config: tuple[int, tuple[int, ...]],
    ) -> None:
        """The decomposition acts on the encoded qubits only."""
        target_dim, control_dims = config
        gate = make_gate(gate_cls, target_dim, control_dims)
        definition = gate.definition
        assert definition is not None
        assert definition.num_qubits == gate.num_qubits

    @pytest.mark.parametrize("config", BRANCH_CONFIGS)
    @parametrize_gate_classes
    def test_definition_has_no_classical_bits(
        self,
        gate_cls: type[QuditControlledGate],
        config: tuple[int, tuple[int, ...]],
    ) -> None:
        """The decomposition circuit is purely quantum."""
        target_dim, control_dims = config
        gate = make_gate(gate_cls, target_dim, control_dims)
        definition = gate.definition
        assert definition is not None
        assert definition.num_clbits == 0


class TestInverse:
    """Which class each gate hands back as its inverse."""

    @pytest.mark.parametrize(
        ("gate_cls", "expected"),
        INVERSE_GATES,
        ids=GATE_IDS,
    )
    def test_inverse_type(
        self,
        gate_cls: type[QuditControlledGate],
        expected: type[QuditControlledGate],
    ) -> None:
        """The dagger pairs (and SUMP's self-wiring) are set up."""
        assert type(make_gate(gate_cls, 3, (2,)).inverse()) is expected

    @parametrize_gate_classes
    def test_inverse_preserves_the_dimensions(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """Inverting never changes the qudits the gate acts on."""
        gate = make_gate(gate_cls, 5, (2, 3))
        inverse = gate.inverse()
        assert inverse.target_dim == gate.target_dim
        assert inverse.control_dims == gate.control_dims
        assert inverse.num_qubits == gate.num_qubits

    @parametrize_gate_classes
    def test_inverse_returns_a_new_object(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """Self-inverse wiring still returns a fresh instance."""
        gate = make_gate(gate_cls, 3, (2,))
        assert gate.inverse() is not gate

    @parametrize_gate_classes
    def test_annotated_inverse_is_not_supported(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """``annotated=True`` is rejected, naming the gate class."""
        gate = make_gate(gate_cls, 3, (2,))
        expected = (
            "annotated inverse is not yet supported for "
            f"{gate_cls.__name__}"
        )
        with pytest.raises(NotImplementedError, match=expected):
            gate.inverse(annotated=True)

    def test_sump_inverse_negates_the_angle(self) -> None:
        """``SUMP(-theta)`` is the announced inverse."""
        inverse = QuditSUMPGate(3, (2,), THETA).inverse()
        assert type(inverse) is QuditSUMPGate
        assert inverse.theta == pytest.approx(-THETA)


class TestSumpAngle:
    """``SUMP`` stores and exposes a caller-supplied angle."""

    @pytest.mark.parametrize("theta", [0.0, 0.75, -1.5, np.pi])
    def test_theta_is_stored_as_the_first_parameter(
        self,
        theta: float,
    ) -> None:
        """The subclassing contract of the phase mixin holds."""
        # NOTE: Qiskit's ControlledGate delegates ``params`` to the
        # base gate, so this also pins that delegation down.
        gate = QuditSUMPGate(3, (2,), theta)
        assert gate.params == [pytest.approx(theta)]
        assert gate.theta == pytest.approx(theta)

    @pytest.mark.parametrize(
        "theta",
        [1, -2, np.int32(3), np.float64(0.5)],
        ids=["int", "negative-int", "numpy-int32", "numpy-float64"],
    )
    def test_angles_are_coerced_to_plain_floats(self, theta: Any) -> None:
        """Integers and NumPy reals are accepted and normalised."""
        gate = QuditSUMPGate(3, (2,), theta)
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
            QuditSUMPGate(3, (2,), theta)

    @pytest.mark.parametrize(
        "theta",
        [float("nan"), float("inf"), float("-inf")],
        ids=["nan", "inf", "-inf"],
    )
    def test_non_finite_angles_are_rejected(self, theta: float) -> None:
        """A non-finite phase would produce a non-unitary matrix."""
        with pytest.raises(ValueError, match="theta must be finite"):
            QuditSUMPGate(3, (2,), theta)

    def test_the_angle_is_required(self) -> None:
        """Unlike ``SUMX``, ``SUMP`` needs an angle."""
        with pytest.raises(TypeError, match="theta"):
            QuditSUMPGate(3, (2,))

    def test_sump_is_a_phase_gate(self) -> None:
        """``SUMP`` is usable wherever the phase mixin is expected."""
        assert isinstance(QuditSUMPGate(3, (2,), THETA), QuditPhaseGateMixin)


class TestExports:
    """``__all__`` is the public catalogue and must stay accurate."""

    def test_controlled_all_lists_exactly_the_known_gates(self) -> None:
        """No gate is missing from, or stale in, ``controlled.__all__``."""
        assert set(controlled.__all__) == {
            cls.__name__ for cls in GATE_CLASSES
        }

    def test_controlled_all_has_no_duplicates(self) -> None:
        """Each name appears exactly once."""
        assert len(controlled.__all__) == len(set(controlled.__all__))

    def test_controlled_all_is_sorted(self) -> None:
        """The list is kept alphabetically sorted."""
        assert controlled.__all__ == sorted(controlled.__all__)

    def test_controlled_all_is_a_subset_of_gates_all(self) -> None:
        """Everything the subpackage exports is re-exported by the facade."""
        assert set(controlled.__all__) <= set(gates.__all__)

    @pytest.mark.parametrize("name", sorted(controlled.__all__))
    def test_every_exported_name_is_importable(self, name: str) -> None:
        """Both modules really expose the advertised attribute."""
        assert getattr(gates, name) is getattr(controlled, name)

    @pytest.mark.parametrize("name", sorted(controlled.__all__))
    def test_every_exported_name_is_a_qudit_gate(self, name: str) -> None:
        """Only :class:`QuditControlledGate` subclasses are exported."""
        exported = getattr(controlled, name)
        assert isinstance(exported, type)
        assert issubclass(exported, QuditControlledGate)

    @parametrize_gate_classes
    def test_table_classes_are_the_exported_ones(
        self,
        gate_cls: type[QuditControlledGate],
    ) -> None:
        """The table above is not shadowing a different class."""
        assert getattr(gates, gate_cls.__name__) is gate_cls
