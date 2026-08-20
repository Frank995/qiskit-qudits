"""Physical equivalence between a gate and its qubit decomposition.

Exactly as for the single-qudit and controlled-qudit suites, a
decomposition is only correct if it realises *the same unitary* as
the gate matrix, including the leakage sector. Two synthesis branches
exist and both are checked:

* when every register dimension is a power of two, hand-written
  constructions are used (the SUMXdg-SUMX-SUMXdg-K sequence for
  ``SWAP`` and the H/SUMP/SWAP cascade for ``QFT``);
* the remaining dimensions fall back to generic unitary synthesis.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from qiskit_qudits.gates import QuditQFTGate, QuditSWAPGate
from tests.helpers import (
    assert_allclose,
    definition_matrix,
    gate_matrix,
    parametrize_dims,
)

if TYPE_CHECKING:
    from qiskit.circuit import Gate

#: Every SWAP dimension needed, both branches mixed.
SWAP_DIMS = (2, 3, 4, 5, 8)

#: Power-of-two SWAP dimensions: the structured branch.
DENSE_SWAP_DIMS = (2, 4, 8)

#: Leaky SWAP dimensions: the unitary-synthesis branch.
LEAKY_SWAP_DIMS = (3, 5)

#: Power-of-two QFT configurations: the structured branch.
DENSE_QFT_CONFIGS = ((2, 2), (3, 2), (4, 2), (2, 4))

#: Leaky QFT configurations: the unitary-synthesis branch.
LEAKY_QFT_CONFIGS = ((2, 3), (3, 3), (2, 5))

#: Every QFT configuration needed, both branches mixed.
QFT_CONFIGS = (*DENSE_QFT_CONFIGS, *LEAKY_QFT_CONFIGS)


def _parametrize_qft(
    configs: tuple[tuple[int, int], ...],
) -> pytest.MarkDecorator:
    """Return a QFT-configuration parametrisation."""
    return pytest.mark.parametrize(
        ("num_qudits", "dim"),
        [pytest.param(n, d, id=f"n{n}-d{d}") for n, d in configs],
    )


def _qubit_width(dim: int) -> int:
    """Return the qubits needed by one ``dim``-level qudit."""
    return math.ceil(math.log2(dim))


def _operation_names(gate: Gate) -> list[str]:
    """Return the instruction names of the gate's definition."""
    definition = gate.definition
    assert definition is not None
    return [instruction.operation.name for instruction in definition.data]


def _assert_synthesised(gate: Gate) -> None:
    """Assert the definition is a single synthesised unitary block."""
    definition = gate.definition
    assert definition is not None
    assert (
        len(definition.data) == 1
    ), f"expected one synthesised block, got {_operation_names(gate)}"
    instruction = definition.data[0]
    assert instruction.operation.name == "unitary"
    assert len(instruction.qubits) == gate.num_qubits


@parametrize_dims(SWAP_DIMS)
def test_swap_definition_reproduces_the_gate_matrix(dim: int) -> None:
    """Check the SUMX/K sequence realises exactly the SWAP matrix."""
    gate = QuditSWAPGate(dim)

    assert_allclose(
        definition_matrix(gate),
        gate_matrix(gate),
        message="the decomposition does not implement the gate matrix",
    )


@_parametrize_qft(QFT_CONFIGS)
def test_qft_definition_reproduces_the_gate_matrix(
    num_qudits: int,
    dim: int,
) -> None:
    """Check the H/SUMP/SWAP cascade realises the QFT matrix."""
    # NOTE: the structured branch composes the library's H, which is
    # the *inverse* DFT (``H|k> ~ w^{-jk}``, see ``test_gate_algebra``),
    # with positive SUMP angles. A failure here for d in {4, 8, 16}
    # means the cascade produces conjugated cross-qudit phases and
    # disagrees with ``_build_unitary``; for d = 2 the two conventions
    # coincide, which is why qubit-only registers pass.
    gate = QuditQFTGate(num_qudits, dim)

    assert_allclose(
        definition_matrix(gate),
        gate_matrix(gate),
        message="the decomposition does not implement the gate matrix",
    )


@parametrize_dims(SWAP_DIMS)
def test_swap_definition_acts_on_the_encoded_qubits_only(dim: int) -> None:
    """Check the SWAP definition spans exactly its qubits."""
    gate = QuditSWAPGate(dim)
    definition = gate.definition
    assert definition is not None

    assert definition.num_qubits == 2 * _qubit_width(dim) == gate.num_qubits
    assert definition.num_clbits == 0


@_parametrize_qft(QFT_CONFIGS)
def test_qft_definition_acts_on_the_encoded_qubits_only(
    num_qudits: int,
    dim: int,
) -> None:
    """Check the QFT definition spans exactly its qubits."""
    gate = QuditQFTGate(num_qudits, dim)
    definition = gate.definition
    assert definition is not None

    expected = num_qudits * _qubit_width(dim)
    assert definition.num_qubits == expected == gate.num_qubits
    assert definition.num_clbits == 0


@parametrize_dims(DENSE_SWAP_DIMS)
def test_power_of_two_swap_definition_is_the_documented_cascade(
    dim: int,
) -> None:
    """Check SWAP is the documented SUMXdg-SUMX-SUMXdg-K sequence."""
    names = _operation_names(QuditSWAPGate(dim))

    assert names == ["SUMXdg", "SUMX", "SUMXdg", "K"]


@_parametrize_qft(DENSE_QFT_CONFIGS)
def test_power_of_two_qft_definitions_use_the_qudit_gate_cascade(
    num_qudits: int,
    dim: int,
) -> None:
    """Check the QFT cascade has the textbook gate counts."""
    counter = Counter(_operation_names(QuditQFTGate(num_qudits, dim)))

    assert counter == Counter(
        {
            "Hdg": num_qudits,
            "SUMP": num_qudits * (num_qudits - 1) // 2,
            "SWAP": num_qudits // 2,
        },
    )


@parametrize_dims(LEAKY_SWAP_DIMS)
def test_non_power_of_two_swap_definitions_fall_back_to_synthesis(
    dim: int,
) -> None:
    """Check leaky SWAP registers use a single synthesised unitary."""
    _assert_synthesised(QuditSWAPGate(dim))


@_parametrize_qft(LEAKY_QFT_CONFIGS)
def test_non_power_of_two_qft_definitions_fall_back_to_synthesis(
    num_qudits: int,
    dim: int,
) -> None:
    """Check leaky QFT registers use a single synthesised unitary."""
    _assert_synthesised(QuditQFTGate(num_qudits, dim))
