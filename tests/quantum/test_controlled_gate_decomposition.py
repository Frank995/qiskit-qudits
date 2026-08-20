"""Physical equivalence between a gate and its qubit decomposition.

Exactly as for the single-qudit suite, every controlled qudit gate
carries a lazily built ``definition``: a plain
:class:`~qiskit.circuit.QuantumCircuit` over the qubits that encode
every control and the target that a backend actually executes. The
decomposition is only correct if it realises *exactly* the same
unitary as the gate matrix, including the phases of the leakage
levels; anything else would make the encoded circuit and the
mathematical qudit model disagree.

Two synthesis branches exist and both are checked:

* when the target and every control dimension is a power of two, an
  mcx ladder handles ``SUMX``/``SUMXdg`` and a ``cp`` cascade handles
  ``SUMP``;
* the remaining dimensions fall back to generic unitary synthesis.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from qiskit_qudits.gates import QuditSUMPGate, QuditSUMXdgGate, QuditSUMXGate
from tests.helpers import assert_allclose, definition_matrix, gate_matrix

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit.circuit import Gate

    from qiskit_qudits.gates.base.controlledgate import QuditControlledGate

#: Angle of the parametric controlled-phase gate.
THETA = 0.7

#: Power-of-two configurations: the structured branch.
DENSE_CONTROLLED_CONFIGS = (
    (2, (2,)),
    (4, (2,)),
    (2, (4,)),
    (4, (4,)),
    (2, (2, 2)),
)

#: Leaky configurations: the unitary-synthesis branch.
LEAKY_CONTROLLED_CONFIGS = (
    (3, (2,)),
    (2, (3,)),
    (3, (3,)),
    (5, (3,)),
    (3, (2, 3)),
)

#: Every configuration needed, both branches mixed.
CONTROLLED_CONFIGS = (*DENSE_CONTROLLED_CONFIGS, *LEAKY_CONTROLLED_CONFIGS)

CONTROLLED_FACTORIES: tuple[
    tuple[str, Callable[[int, tuple[int, ...]], QuditControlledGate]],
    ...,
] = (
    ("SUMX", QuditSUMXGate),
    ("SUMXdg", QuditSUMXdgGate),
    (
        "SUMP(0.7)",
        lambda target, controls: QuditSUMPGate(target, controls, THETA),
    ),
)


def _config_id(target_dim: int, control_dims: tuple[int, ...]) -> str:
    """Return a readable id for a controlled configuration."""
    controls = "x".join(str(dim) for dim in control_dims)
    return f"c{controls}-t{target_dim}"


def _parametrize_configs(
    configs: tuple[tuple[int, tuple[int, ...]], ...],
) -> pytest.MarkDecorator:
    """Return a controlled-configuration parametrisation."""
    return pytest.mark.parametrize(
        ("target_dim", "control_dims"),
        [
            pytest.param(target, controls, id=_config_id(target, controls))
            for target, controls in configs
        ],
    )


parametrize_controlled_gates = pytest.mark.parametrize(
    "make_gate",
    [pytest.param(factory, id=name) for name, factory in CONTROLLED_FACTORIES],
)


def _qubit_width(dim: int) -> int:
    """Return the qubits needed by one ``dim``-level qudit."""
    return math.ceil(math.log2(dim))


def _operation_names(gate: Gate) -> list[str]:
    """Return the instruction names of the gate's definition."""
    definition = gate.definition
    assert definition is not None
    return [instruction.operation.name for instruction in definition.data]


@parametrize_controlled_gates
@_parametrize_configs(CONTROLLED_CONFIGS)
def test_definition_reproduces_the_gate_matrix(
    make_gate: Callable[[int, tuple[int, ...]], QuditControlledGate],
    target_dim: int,
    control_dims: tuple[int, ...],
) -> None:
    """Check the qubit circuit realises exactly the gate unitary."""
    gate = make_gate(target_dim, control_dims)

    assert_allclose(
        definition_matrix(gate),
        gate_matrix(gate),
        message="the decomposition does not implement the gate matrix",
    )


@parametrize_controlled_gates
@_parametrize_configs(CONTROLLED_CONFIGS)
def test_definition_acts_on_the_encoded_qubits_only(
    make_gate: Callable[[int, tuple[int, ...]], QuditControlledGate],
    target_dim: int,
    control_dims: tuple[int, ...],
) -> None:
    """Check the definition spans exactly the encoding qubits."""
    gate = make_gate(target_dim, control_dims)
    definition = gate.definition
    assert definition is not None

    expected = sum(_qubit_width(dim) for dim in gate.dims)
    assert definition.num_qubits == expected == gate.num_qubits
    assert definition.num_clbits == 0


@parametrize_controlled_gates
@_parametrize_configs(DENSE_CONTROLLED_CONFIGS)
def test_power_of_two_definitions_are_structured(
    make_gate: Callable[[int, tuple[int, ...]], QuditControlledGate],
    target_dim: int,
    control_dims: tuple[int, ...],
) -> None:
    """Check a full Hilbert space never triggers unitary synthesis."""
    names = _operation_names(make_gate(target_dim, control_dims))

    assert names, "the decomposition is empty"
    assert "unitary" not in names, (
        "a dense register fills its qubits, so the decomposition should "
        f"be structured, but it contains a synthesised block: {names}"
    )


@_parametrize_configs(DENSE_CONTROLLED_CONFIGS)
def test_power_of_two_sump_definition_is_a_cp_cascade(
    target_dim: int,
    control_dims: tuple[int, ...],
) -> None:
    """Check SUMP emits one ``cp`` per control-target qubit pair."""
    gate = QuditSUMPGate(target_dim, control_dims, THETA)
    names = _operation_names(gate)

    expected = sum(_qubit_width(dim) for dim in control_dims) * _qubit_width(
        target_dim,
    )
    assert names == ["cp"] * expected


@parametrize_controlled_gates
@_parametrize_configs(LEAKY_CONTROLLED_CONFIGS)
def test_non_power_of_two_definitions_fall_back_to_synthesis(
    make_gate: Callable[[int, tuple[int, ...]], QuditControlledGate],
    target_dim: int,
    control_dims: tuple[int, ...],
) -> None:
    """Check leaky registers use a single synthesised unitary."""
    gate = make_gate(target_dim, control_dims)
    definition = gate.definition
    assert definition is not None

    assert (
        len(definition.data) == 1
    ), f"expected one synthesised block, got {_operation_names(gate)}"
    instruction = definition.data[0]
    assert instruction.operation.name == "unitary"
    assert len(instruction.qubits) == gate.num_qubits
