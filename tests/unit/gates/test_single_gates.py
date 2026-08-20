"""Software-contract tests for the concrete single-qudit gates.

The whole gate catalogue is driven from one table so that adding a
gate to :mod:`qiskit_qudits.gates.single` without registering it here
makes the export tests fail.

What is checked: names, arity, labels, the *type* of the inverse and
the package exports. Matrices, gate algebra and the equivalence of
``definition`` with ``_build_unitary`` belong to ``tests/quantum``.
"""

from __future__ import annotations

from typing import Any

import pytest

from qiskit_qudits import gates
from qiskit_qudits.gates import (
    QuditHdgGate,
    QuditHGate,
    QuditIGate,
    QuditKGate,
    QuditNOTGate,
    QuditPGate,
    QuditSdgGate,
    QuditSGate,
    QuditTdgGate,
    QuditTGate,
    QuditXdgGate,
    QuditXGate,
    QuditZdgGate,
    QuditZGate,
    single,
)
from qiskit_qudits.gates.base.gate import QuditGate

#: Angle used whenever a parametrised gate has to be built.
THETA = 0.7

#: ``(class, expected name, expected inverse class)`` for every gate
#: exported by :mod:`qiskit_qudits.gates`.
GATES: list[tuple[type[QuditGate], str, type[QuditGate]]] = [
    (QuditIGate, "I", QuditIGate),
    (QuditHGate, "H", QuditHdgGate),
    (QuditHdgGate, "Hdg", QuditHGate),
    (QuditXGate, "X", QuditXdgGate),
    (QuditXdgGate, "Xdg", QuditXGate),
    (QuditZGate, "Z", QuditZdgGate),
    (QuditZdgGate, "Zdg", QuditZGate),
    (QuditSGate, "S", QuditSdgGate),
    (QuditSdgGate, "Sdg", QuditSGate),
    (QuditTGate, "T", QuditTdgGate),
    (QuditTdgGate, "Tdg", QuditTGate),
    (QuditPGate, "P", QuditPGate),
    (QuditKGate, "K", QuditKGate),
    (QuditNOTGate, "NOT", QuditNOTGate),
]

#: Readable ids for the table above.
GATE_IDS = [gate_cls.__name__ for gate_cls, _, _ in GATES]

#: The table reduced to ``(class, expected name)``.
NAMED_GATES = [(gate_cls, name) for gate_cls, name, _ in GATES]

#: The table reduced to ``(class, expected inverse class)``.
INVERSE_GATES = [(gate_cls, inverse) for gate_cls, _, inverse in GATES]

#: Just the classes.
GATE_CLASSES: list[type[QuditGate]] = [gate_cls for gate_cls, _, _ in GATES]

#: One dimension per decomposition branch: 3 leaves invalid states
#: behind, 4 fills the two-qubit Hilbert space exactly.
BRANCH_DIMS = [3, 4]


def make_gate(
    gate_cls: type[QuditGate],
    dim: Any,
    *,
    label: str | None = None,
) -> QuditGate:
    """Build ``gate_cls``, supplying ``theta`` when it is required.

    Args:
        gate_cls: The gate class to instantiate.
        dim: Qudit dimension.
        label: Optional display label.

    Returns:
        A freshly built gate.
    """
    if gate_cls is QuditPGate:
        return QuditPGate(dim, THETA, label=label)
    return gate_cls(dim, label=label)


class TestNaming:
    """Every gate advertises a stable, hard-coded name."""

    @pytest.mark.parametrize(
        ("gate_cls", "expected"),
        NAMED_GATES,
        ids=GATE_IDS,
    )
    def test_gate_name_class_var(
        self,
        gate_cls: type[QuditGate],
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
        gate_cls: type[QuditGate],
        expected: str,
    ) -> None:
        """The Qiskit ``name`` is taken from ``gate_name``."""
        assert make_gate(gate_cls, 3).name == expected

    def test_names_are_unique(self) -> None:
        """No two gates share a Qiskit name."""
        names = [name for _, name, _ in GATES]
        assert len(set(names)) == len(names)


class TestConstruction:
    """Dimension handling shared by the whole catalogue."""

    @pytest.mark.parametrize(("dim", "expected"), [(2, 1), (3, 2), (5, 3)])
    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_num_qubits_follows_the_dimension(
        self,
        gate_cls: type[QuditGate],
        dim: int,
        expected: int,
    ) -> None:
        """Each gate is as wide as the encoding requires."""
        assert make_gate(gate_cls, dim).num_qubits == expected

    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_invalid_dimension_is_rejected(
        self,
        gate_cls: type[QuditGate],
    ) -> None:
        """Validation is inherited by every subclass' ``__init__``."""
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            make_gate(gate_cls, 1)

    def test_non_integer_dimension_is_rejected(self) -> None:
        """Spot-check that the type guard survives the subclassing."""
        with pytest.raises(TypeError, match="dim must be an integer"):
            QuditXGate(3.0)

    @pytest.mark.parametrize("label", [None, "custom-label"])
    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_label_is_stored(
        self,
        gate_cls: type[QuditGate],
        label: str | None,
    ) -> None:
        """The label is kept verbatim, ``None`` included."""
        assert make_gate(gate_cls, 3, label=label).label == label


class TestDefinition:
    """The lazily built qubit-level decomposition."""

    def test_definition_is_built_lazily(self) -> None:
        """Nothing is decomposed until ``definition`` is read."""
        gate = QuditXGate(3)
        assert gate._definition is None
        assert gate.definition is not None

    def test_definition_is_cached(self) -> None:
        """The circuit is built once and then reused."""
        gate = QuditXGate(3)
        assert gate.definition is gate.definition

    @pytest.mark.parametrize("dim", BRANCH_DIMS)
    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_definition_is_named_after_the_label(
        self,
        gate_cls: type[QuditGate],
        dim: int,
    ) -> None:
        """A label renames the decomposition circuit."""
        definition = make_gate(gate_cls, dim, label="my-label").definition
        assert definition is not None
        assert definition.name == "my-label"

    @pytest.mark.parametrize("dim", BRANCH_DIMS)
    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_definition_falls_back_to_the_gate_name(
        self,
        gate_cls: type[QuditGate],
        dim: int,
    ) -> None:
        """Without a label the gate name is used instead."""
        gate = make_gate(gate_cls, dim)
        definition = gate.definition
        assert definition is not None
        assert definition.name == gate.name

    @pytest.mark.parametrize("dim", BRANCH_DIMS)
    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_definition_is_as_wide_as_the_gate(
        self,
        gate_cls: type[QuditGate],
        dim: int,
    ) -> None:
        """The decomposition acts on the encoded qubits only."""
        gate = make_gate(gate_cls, dim)
        definition = gate.definition
        assert definition is not None
        assert definition.num_qubits == gate.num_qubits


class TestInverse:
    """Which class each gate hands back as its inverse."""

    @pytest.mark.parametrize(
        ("gate_cls", "expected"),
        INVERSE_GATES,
        ids=GATE_IDS,
    )
    def test_inverse_type(
        self,
        gate_cls: type[QuditGate],
        expected: type[QuditGate],
    ) -> None:
        """The dagger pairs (and self-inverse gates) are wired up."""
        assert type(make_gate(gate_cls, 3).inverse()) is expected

    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_inverse_preserves_the_dimension(
        self,
        gate_cls: type[QuditGate],
    ) -> None:
        """Inverting never changes the qudit the gate acts on."""
        gate = make_gate(gate_cls, 5)
        inverse = gate.inverse()
        assert inverse.dim == gate.dim
        assert inverse.num_qubits == gate.num_qubits

    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_inverse_returns_a_new_object(
        self,
        gate_cls: type[QuditGate],
    ) -> None:
        """Self-inverse gates still return a fresh instance."""
        gate = make_gate(gate_cls, 3)
        assert gate.inverse() is not gate

    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_annotated_inverse_is_not_supported(
        self,
        gate_cls: type[QuditGate],
    ) -> None:
        """``annotated=True`` is rejected, naming the gate class."""
        gate = make_gate(gate_cls, 3)
        expected = (
            "annotated inverse is not yet supported for "
            f"{gate_cls.__name__}"
        )
        with pytest.raises(NotImplementedError, match=expected):
            gate.inverse(annotated=True)


class TestExports:
    """``__all__`` is the public catalogue and must stay accurate."""

    def test_single_all_lists_exactly_the_known_gates(self) -> None:
        """No gate is missing from, or stale in, ``single.__all__``."""
        assert set(single.__all__) == {cls.__name__ for cls in GATE_CLASSES}

    def test_single_all_has_no_duplicates(self) -> None:
        """Each name appears exactly once."""
        assert len(single.__all__) == len(set(single.__all__))

    def test_single_all_is_sorted(self) -> None:
        """The list is kept alphabetically sorted."""
        assert single.__all__ == sorted(single.__all__)

    def test_single_all_is_a_subset_of_gates_all(self) -> None:
        """Everything the subpackage exports is re-exported by the facade."""
        assert set(single.__all__) <= set(gates.__all__)

    @pytest.mark.parametrize("name", sorted(single.__all__))
    def test_every_exported_name_is_importable(self, name: str) -> None:
        """Both modules really expose the advertised attribute."""
        assert getattr(gates, name) is getattr(single, name)

    @pytest.mark.parametrize("name", sorted(single.__all__))
    def test_every_exported_name_is_a_qudit_gate(self, name: str) -> None:
        """Only :class:`QuditGate` subclasses are exported."""
        exported = getattr(single, name)
        assert isinstance(exported, type)
        assert issubclass(exported, QuditGate)

    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_table_classes_are_the_exported_ones(
        self,
        gate_cls: type[QuditGate],
    ) -> None:
        """The table above is not shadowing a different class."""
        assert getattr(gates, gate_cls.__name__) is gate_cls
