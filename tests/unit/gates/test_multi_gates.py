"""Software-contract tests for the concrete multi-qudit gates.

``SWAP`` and ``QFT`` are the (uncontrolled) multi-qudit entries of
the catalogue. What is checked here: names, register layout,
constructor validation, labels, the lazily built definition, the
*type* of the inverse and the package exports. Matrices and the
equivalence of ``definition`` with ``_build_unitary`` belong to
``tests/quantum``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qiskit.circuit import Gate

from qiskit_qudits import gates
from qiskit_qudits.gates import QuditQFTGate, QuditSWAPGate, multi
from qiskit_qudits.gates.base.multigate import QuditMultiGate

#: ``(class, expected name)`` for every multi-qudit gate exported by
#: :mod:`qiskit_qudits.gates`.
GATES: list[tuple[type[QuditMultiGate], str]] = [
    (QuditSWAPGate, "SWAP"),
    (QuditQFTGate, "QFT"),
]

#: Readable ids for the table above.
GATE_IDS = [gate_cls.__name__ for gate_cls, _ in GATES]

#: Just the classes.
GATE_CLASSES: list[type[QuditMultiGate]] = [gate_cls for gate_cls, _ in GATES]

#: ``(dimension, 2 * ceil(log2 d))`` pairs, worked out by hand.
SWAP_QUBIT_TABLE = [(2, 2), (3, 4), (5, 6), (8, 6)]

#: ``((num_qudits, dim), total qubit count)`` pairs for the QFT.
QFT_QUBIT_TABLE = [
    ((2, 2), 2),
    ((3, 2), 3),
    ((2, 3), 4),
    ((3, 4), 6),
    ((2, 5), 6),
]

#: One ``(dim,)`` / ``(num_qudits, dim)`` constructor configuration per
#: decomposition branch, keyed by gate class.
BRANCH_CONFIGS: dict[type[QuditMultiGate], list[tuple[int, ...]]] = {
    QuditSWAPGate: [(3,), (4,)],
    QuditQFTGate: [(2, 3), (2, 2)],
}

#: ``BRANCH_CONFIGS`` flattened for parametrisation.
BRANCH_CASES: list[tuple[type[QuditMultiGate], tuple[int, ...]]] = [
    (gate_cls, args)
    for gate_cls, configs in BRANCH_CONFIGS.items()
    for args in configs
]

#: Readable ids for ``BRANCH_CASES``.
BRANCH_IDS = [f"{gate_cls.__name__}{args}" for gate_cls, args in BRANCH_CASES]


def make_gate(
    gate_cls: type[QuditMultiGate],
    *,
    label: str | None = None,
) -> QuditMultiGate:
    """Build ``gate_cls`` with a representative, valid configuration.

    Args:
        gate_cls: The gate class to instantiate.
        label: Optional display label.

    Returns:
        A freshly built gate.
    """
    if gate_cls is QuditQFTGate:
        return QuditQFTGate(2, 3, label=label)
    return QuditSWAPGate(3, label=label)


class TestNaming:
    """Every gate advertises a stable, hard-coded name."""

    @pytest.mark.parametrize(("gate_cls", "expected"), GATES, ids=GATE_IDS)
    def test_gate_name_class_var(
        self,
        gate_cls: type[QuditMultiGate],
        expected: str,
    ) -> None:
        """``gate_name`` is readable without an instance."""
        name: str = gate_cls.gate_name
        assert name == expected

    @pytest.mark.parametrize(("gate_cls", "expected"), GATES, ids=GATE_IDS)
    def test_instance_name_matches_the_class_var(
        self,
        gate_cls: type[QuditMultiGate],
        expected: str,
    ) -> None:
        """The Qiskit ``name`` is taken from ``gate_name``."""
        assert make_gate(gate_cls).name == expected

    def test_names_are_unique(self) -> None:
        """No two gates share a Qiskit name."""
        names = [name for _, name in GATES]
        assert len(set(names)) == len(names)


class TestConstruction:
    """Behaviour shared by every multi-qudit gate, regardless of signature."""

    @pytest.mark.parametrize("label", [None, "custom-label"])
    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_label_is_stored(
        self,
        gate_cls: type[QuditMultiGate],
        label: str | None,
    ) -> None:
        """The label is kept verbatim, ``None`` included."""
        assert make_gate(gate_cls, label=label).label == label

    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_instances_are_multi_qudit_gates(
        self,
        gate_cls: type[QuditMultiGate],
    ) -> None:
        """Every gate goes through the multi-qudit base class."""
        gate = make_gate(gate_cls)
        assert isinstance(gate, QuditMultiGate)
        assert isinstance(gate, Gate)
        assert gate.num_clbits == 0


class TestSwapConstruction:
    """Register layout and validation specific to :class:`QuditSWAPGate`."""

    @pytest.mark.parametrize("dim", [2, 3, 5, 8])
    def test_swap_acts_on_two_equal_qudits(self, dim: int) -> None:
        """The single ``dim`` is repeated for both operands."""
        gate = QuditSWAPGate(dim)
        assert gate.dims == (dim, dim)
        assert gate.num_qudits == 2

    @pytest.mark.parametrize(("dim", "expected"), SWAP_QUBIT_TABLE)
    def test_num_qubits_follows_the_dimension(
        self,
        dim: int,
        expected: int,
    ) -> None:
        """The gate is twice as wide as one encoded qudit."""
        assert QuditSWAPGate(dim).num_qubits == expected

    def test_numpy_dimensions_are_coerced(self) -> None:
        """NumPy integers become plain ``int`` dimensions."""
        gate = QuditSWAPGate(np.int64(3))
        assert gate.dims == (3, 3)
        assert all(type(dim) is int for dim in gate.dims)

    @pytest.mark.parametrize("dim", [1, 17])
    def test_out_of_range_dimensions_are_rejected(self, dim: int) -> None:
        """The dimension must lie in ``[2, 16]``."""
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            QuditSWAPGate(dim)

    def test_non_integer_dimensions_are_rejected(self) -> None:
        """A float dimension is a ``TypeError``."""
        with pytest.raises(TypeError, match="dim must be an integer"):
            QuditSWAPGate(3.0)


class TestQftConstruction:
    """Register layout and validation specific to :class:`QuditQFTGate`."""

    @pytest.mark.parametrize(("num_qudits", "dim"), [(2, 3), (3, 4), (4, 2)])
    def test_qft_acts_on_a_homogeneous_register(
        self,
        num_qudits: int,
        dim: int,
    ) -> None:
        """The single ``dim`` is repeated ``num_qudits`` times."""
        gate = QuditQFTGate(num_qudits, dim)
        assert gate.dims == (dim,) * num_qudits
        assert gate.num_qudits == num_qudits

    @pytest.mark.parametrize(("config", "expected"), QFT_QUBIT_TABLE)
    def test_num_qubits_follows_the_register(
        self,
        config: tuple[int, int],
        expected: int,
    ) -> None:
        """The gate is ``n`` encoded qudits wide."""
        num_qudits, dim = config
        assert QuditQFTGate(num_qudits, dim).num_qubits == expected

    def test_numpy_num_qudits_is_accepted(self) -> None:
        """A NumPy integer is a valid qudit count."""
        assert QuditQFTGate(np.int64(3), 2).num_qudits == 3

    @pytest.mark.parametrize(
        "num_qudits",
        [2.0, "2", None, True],
        ids=["float", "str", "none", "bool"],
    )
    def test_non_integer_num_qudits_is_rejected(
        self,
        num_qudits: Any,
    ) -> None:
        """The qudit count must be integer-like."""
        with pytest.raises(TypeError, match="num_qudits must be an integer"):
            QuditQFTGate(num_qudits, 3)

    def test_a_single_qudit_is_rejected(self) -> None:
        """The multi-qudit base class needs at least two qudits."""
        with pytest.raises(ValueError, match="at least 2 qudits"):
            QuditQFTGate(1, 3)

    @pytest.mark.parametrize("num_qudits", [0, -1])
    def test_an_empty_register_is_rejected(self, num_qudits: int) -> None:
        """Zero or negative qudit counts produce empty ``dims``."""
        with pytest.raises(ValueError, match="dims must not be empty"):
            QuditQFTGate(num_qudits, 3)

    def test_too_many_qudits_are_rejected(self) -> None:
        """The register may hold at most ``MAX_QUDITS`` qudits."""
        with pytest.raises(ValueError, match="at most 8"):
            QuditQFTGate(9, 2)

    @pytest.mark.parametrize("dim", [1, 17])
    def test_out_of_range_dimensions_are_rejected(self, dim: int) -> None:
        """The dimension must lie in ``[2, 16]``."""
        with pytest.raises(ValueError, match=r"dim must be in \[2, 16\]"):
            QuditQFTGate(2, dim)

    def test_non_integer_dimensions_are_rejected(self) -> None:
        """A float dimension is a ``TypeError``."""
        with pytest.raises(TypeError, match="dim must be an integer"):
            QuditQFTGate(2, 3.0)


class TestDefinition:
    """The lazily built qubit-level decomposition."""

    def test_definition_is_built_lazily(self) -> None:
        """Nothing is decomposed until ``definition`` is read."""
        gate = QuditSWAPGate(3)
        assert gate._definition is None
        assert gate.definition is not None

    def test_definition_is_cached(self) -> None:
        """The circuit is built once and then reused."""
        gate = QuditSWAPGate(3)
        assert gate.definition is gate.definition

    @pytest.mark.parametrize(
        ("gate_cls", "args"),
        BRANCH_CASES,
        ids=BRANCH_IDS,
    )
    def test_definition_is_named_after_the_label(
        self,
        gate_cls: type[QuditMultiGate],
        args: tuple[int, ...],
    ) -> None:
        """A label renames the decomposition circuit."""
        definition = gate_cls(*args, label="my-label").definition
        assert definition is not None
        assert definition.name == "my-label"

    @pytest.mark.parametrize(
        ("gate_cls", "args"),
        BRANCH_CASES,
        ids=BRANCH_IDS,
    )
    def test_definition_falls_back_to_the_gate_name(
        self,
        gate_cls: type[QuditMultiGate],
        args: tuple[int, ...],
    ) -> None:
        """Without a label the gate name is used instead."""
        gate = gate_cls(*args)
        definition = gate.definition
        assert definition is not None
        assert definition.name == gate.name

    @pytest.mark.parametrize(
        ("gate_cls", "args"),
        BRANCH_CASES,
        ids=BRANCH_IDS,
    )
    def test_definition_is_as_wide_as_the_gate(
        self,
        gate_cls: type[QuditMultiGate],
        args: tuple[int, ...],
    ) -> None:
        """The decomposition acts on the encoded qubits only."""
        gate = gate_cls(*args)
        definition = gate.definition
        assert definition is not None
        assert definition.num_qubits == gate.num_qubits
        assert definition.num_clbits == 0


class TestInverse:
    """Which type each gate hands back as its inverse."""

    def test_swap_inverse_type(self) -> None:
        """SWAP is self-inverse."""
        assert type(QuditSWAPGate(3).inverse()) is QuditSWAPGate

    def test_swap_inverse_preserves_the_dimension(self) -> None:
        """Inverting never changes the qudits SWAP acts on."""
        gate = QuditSWAPGate(5)
        inverse = gate.inverse()
        assert inverse.dims == gate.dims
        assert inverse.num_qubits == gate.num_qubits

    def test_qft_inverse_is_a_plain_qiskit_gate(self) -> None:
        """QFT's inverse is synthesised, not a qudit gate class."""
        inverse = QuditQFTGate(2, 3).inverse()
        assert isinstance(inverse, Gate)
        assert not isinstance(inverse, QuditMultiGate)

    def test_qft_inverse_preserves_the_qubit_width(self) -> None:
        """The inverse spans exactly the encoded qubits."""
        gate = QuditQFTGate(2, 3)
        assert gate.inverse().num_qubits == gate.num_qubits

    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_inverse_returns_a_new_object(
        self,
        gate_cls: type[QuditMultiGate],
    ) -> None:
        """Every inverse call returns a fresh instance."""
        gate = make_gate(gate_cls)
        assert gate.inverse() is not gate

    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_annotated_inverse_is_not_supported(
        self,
        gate_cls: type[QuditMultiGate],
    ) -> None:
        """``annotated=True`` is rejected, naming the gate class."""
        gate = make_gate(gate_cls)
        expected = (
            "annotated inverse is not yet supported for "
            f"{gate_cls.__name__}"
        )
        with pytest.raises(NotImplementedError, match=expected):
            gate.inverse(annotated=True)


class TestExports:
    """``__all__`` is the public catalogue and must stay accurate."""

    def test_multi_all_lists_exactly_the_known_gates(self) -> None:
        """No gate is missing from, or stale in, ``multi.__all__``."""
        assert set(multi.__all__) == {cls.__name__ for cls in GATE_CLASSES}

    def test_multi_all_has_no_duplicates(self) -> None:
        """Each name appears exactly once."""
        assert len(multi.__all__) == len(set(multi.__all__))

    def test_multi_all_is_sorted(self) -> None:
        """The list is kept alphabetically sorted."""
        assert multi.__all__ == sorted(multi.__all__)

    def test_multi_all_is_a_subset_of_gates_all(self) -> None:
        """Everything the subpackage exports is re-exported by the facade."""
        assert set(multi.__all__) <= set(gates.__all__)

    @pytest.mark.parametrize("name", sorted(multi.__all__))
    def test_every_exported_name_is_importable(self, name: str) -> None:
        """Both modules really expose the advertised attribute."""
        assert getattr(gates, name) is getattr(multi, name)

    @pytest.mark.parametrize("name", sorted(multi.__all__))
    def test_every_exported_name_is_a_qudit_gate(self, name: str) -> None:
        """Only :class:`QuditMultiGate` subclasses are exported."""
        exported = getattr(multi, name)
        assert isinstance(exported, type)
        assert issubclass(exported, QuditMultiGate)

    @pytest.mark.parametrize("gate_cls", GATE_CLASSES, ids=GATE_IDS)
    def test_table_classes_are_the_exported_ones(
        self,
        gate_cls: type[QuditMultiGate],
    ) -> None:
        """The table above is not shadowing a different class."""
        assert getattr(gates, gate_cls.__name__) is gate_cls
