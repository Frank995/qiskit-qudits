"""Fixtures shared by the whole :mod:`qiskit_qudits` test-suite.

The reusable *logic* lives in :mod:`tests.helpers`; this module only
exposes it as fixtures and parametrisations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.helpers import (
    ALL_DIMS,
    NON_POWER_OF_TWO_DIMS,
    POWER_OF_TWO_DIMS,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from qiskit_qudits.circuit.qudit import QuditRegister

#: Shot count used by the statistical tests. Large enough for a 5%
#: tolerance on a d=8 uniform distribution, small enough to stay fast
#: on the pure-Python reference simulator.
DEFAULT_SHOTS = 4096

#: Seed pinned everywhere so the suite is reproducible.
DEFAULT_SEED = 1234


@pytest.fixture(params=ALL_DIMS, ids=[f"d{d}" for d in ALL_DIMS])
def dim(request: pytest.FixtureRequest) -> int:
    """A representative qudit dimension (both kinds of encoding)."""
    return int(request.param)


@pytest.fixture(
    params=POWER_OF_TWO_DIMS,
    ids=[f"d{d}" for d in POWER_OF_TWO_DIMS],
)
def pow2_dim(request: pytest.FixtureRequest) -> int:
    """A dimension whose encoding fills the qubit Hilbert space."""
    return int(request.param)


@pytest.fixture(
    params=NON_POWER_OF_TWO_DIMS,
    ids=[f"d{d}" for d in NON_POWER_OF_TWO_DIMS],
)
def non_pow2_dim(request: pytest.FixtureRequest) -> int:
    """A dimension that leaves invalid basis states behind."""
    return int(request.param)


@pytest.fixture
def shots() -> int:
    """Number of shots used by the sampling tests."""
    return DEFAULT_SHOTS


@pytest.fixture
def seed() -> int:
    """Seed used for every simulator run."""
    return DEFAULT_SEED


@pytest.fixture(autouse=True)
def _reset_register_counters() -> Iterator[None]:
    """Keep auto-generated register names stable across tests.

    :class:`~qiskit_qudits.circuit.qudit.QuditRegister` and friends
    name themselves from a module-level counter, so a test that
    asserts on ``'Q0'`` would otherwise depend on execution order.
    """
    import itertools

    from qiskit_qudits.circuit.cldigit import ClDigitRegister
    from qiskit_qudits.circuit.quantumcircuit import QuditQuantumCircuit
    from qiskit_qudits.circuit.qudit import QuditRegister as _QuditRegister

    saved: list[tuple[type, object]] = [
        (_QuditRegister, _QuditRegister._instances_counter),
        (ClDigitRegister, ClDigitRegister._instances_counter),
        (
            QuditQuantumCircuit,
            QuditQuantumCircuit._instances_counter,
        ),
    ]
    for owner, _ in saved:
        owner._instances_counter = itertools.count()
    yield
    for owner, counter in saved:
        owner._instances_counter = counter


@pytest.fixture
def qutrit_register() -> QuditRegister:
    """A three-qutrit register, the workhorse of the circuit tests."""
    from qiskit_qudits.circuit.qudit import QuditRegister

    return QuditRegister(3, 3, "qt")
