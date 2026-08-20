"""Physical equivalence between a gate and its qubit decomposition.

Every qudit gate carries a lazily built ``definition``: a plain
:class:`~qiskit.circuit.QuantumCircuit` over ``n = ceil(log2 d)``
qubits that a backend actually executes. The decomposition is only
correct if it realises *exactly* the same unitary as the gate matrix,
including the phases of the leakage levels; anything else would make
the encoded circuit and the mathematical qudit model disagree.

Two synthesis branches exist and both are checked:

* power-of-two dimensions use hand-written, transpiler-friendly
  constructions (an increment ladder of multi-controlled X for
  ``X``/``Xdg``, ``synth_qft_full`` for ``H``/``Hdg``, one phase gate
  per qubit for ``P``, an X on every qubit for ``NOT``, and
  ``NOT`` followed by ``X`` for ``K``);
* the remaining dimensions fall back to generic unitary synthesis.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

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
)
from tests.helpers import (
    NON_POWER_OF_TWO_DIMS,
    assert_allclose,
    definition_matrix,
    gate_matrix,
    parametrize_dims,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit_qudits.gates.base.gate import QuditGate

#: Generic angle for the parametric phase gate.
THETA = 0.7

#: Largest supported dimension; a four-qubit synthesis, hence slow.
MAX_DIM = 16

#: Power-of-two dimensions that stay cheap to synthesise.
SMALL_POWER_OF_TWO_DIMS = (2, 4, 8)

#: Every dimension needing at most four qubits, both branches mixed.
DECOMPOSITION_DIMS = (2, 3, 4, 5, 6, 7, 8, 11)

#: Gates that switch between a structured and a synthesised
#: decomposition depending on the dimension. ``I`` is excluded because
#: it is always an explicit ladder of ``id`` instructions.
STRUCTURED_FACTORIES: tuple[tuple[str, Callable[[int], QuditGate]], ...] = (
    ("X", QuditXGate),
    ("Xdg", QuditXdgGate),
    ("Z", QuditZGate),
    ("Zdg", QuditZdgGate),
    ("S", QuditSGate),
    ("Sdg", QuditSdgGate),
    ("T", QuditTGate),
    ("Tdg", QuditTdgGate),
    ("H", QuditHGate),
    ("Hdg", QuditHdgGate),
    ("NOT", QuditNOTGate),
    ("K", QuditKGate),
    ("P(0.7)", lambda dim: QuditPGate(dim, THETA)),
)

GATE_FACTORIES: tuple[tuple[str, Callable[[int], QuditGate]], ...] = (
    ("I", QuditIGate),
    *STRUCTURED_FACTORIES,
)

parametrize_gates = pytest.mark.parametrize(
    "make_gate",
    [pytest.param(factory, id=name) for name, factory in GATE_FACTORIES],
)

parametrize_structured_gates = pytest.mark.parametrize(
    "make_gate",
    [pytest.param(factory, id=name) for name, factory in STRUCTURED_FACTORIES],
)


def _operation_names(gate: QuditGate) -> list[str]:
    """Return the instruction names of the gate's definition."""
    definition = gate.definition
    assert definition is not None
    return [instruction.operation.name for instruction in definition.data]


@parametrize_gates
@parametrize_dims(DECOMPOSITION_DIMS)
def test_definition_reproduces_the_gate_matrix(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check the qubit circuit realises exactly the gate unitary."""
    gate = make_gate(dim)

    assert_allclose(
        definition_matrix(gate),
        gate_matrix(gate),
        message="the decomposition does not implement the gate matrix",
    )


@pytest.mark.slow
@parametrize_gates
def test_definition_reproduces_the_gate_matrix_at_dim_16(
    make_gate: Callable[[int], QuditGate],
) -> None:
    """Check the four-qubit ``d=16`` decompositions are exact too."""
    gate = make_gate(MAX_DIM)

    assert_allclose(
        definition_matrix(gate),
        gate_matrix(gate),
        message="the d=16 decomposition does not implement the gate matrix",
    )


@parametrize_gates
@parametrize_dims(DECOMPOSITION_DIMS)
def test_definition_acts_on_the_encoded_qubits_only(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check the definition spans exactly ``ceil(log2 d)`` qubits."""
    gate = make_gate(dim)
    definition = gate.definition
    assert definition is not None

    assert definition.num_qubits == math.ceil(math.log2(dim))
    assert definition.num_qubits == gate.num_qubits
    assert definition.num_clbits == 0


@parametrize_structured_gates
@parametrize_dims(SMALL_POWER_OF_TWO_DIMS)
def test_power_of_two_definitions_are_structured(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check a full Hilbert space never triggers unitary synthesis."""
    names = _operation_names(make_gate(dim))

    assert names, "the decomposition is empty"
    assert "unitary" not in names, (
        f"a d={dim} qudit fills its qubits, so the decomposition should "
        f"be structured, but it contains a synthesised block: {names}"
    )


@parametrize_structured_gates
@parametrize_dims(NON_POWER_OF_TWO_DIMS)
def test_non_power_of_two_definitions_fall_back_to_synthesis(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check leakage dimensions use a single synthesised unitary."""
    gate = make_gate(dim)
    definition = gate.definition
    assert definition is not None

    assert (
        len(definition.data) == 1
    ), f"expected one synthesised block, got {_operation_names(gate)}"
    instruction = definition.data[0]
    assert instruction.operation.name == "unitary"
    assert len(instruction.qubits) == gate.num_qubits


@parametrize_dims(DECOMPOSITION_DIMS)
def test_identity_definition_never_needs_synthesis(dim: int) -> None:
    """Check ``I`` decomposes into one ``id`` per encoded qubit."""
    gate = QuditIGate(dim)
    names = _operation_names(gate)

    assert names == ["id"] * gate.num_qubits
