"""Shared helpers for the :mod:`qiskit_qudits` test-suite.

Everything that more than one test module needs lives here: matrix
extraction, unitarity assertions, simulation shortcuts and the
statistical comparison used by the user-level tests.

Conventions mirror the library (see
:mod:`qiskit_qudits.utils.encoding`): everything is little-endian, so
qudit ``0`` occupies the lowest-index qubits and is the least
significant factor of a logical state-vector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from qiskit import transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Operator, Statevector

from qiskit_qudits.utils.encoding import project_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from qiskit.circuit import QuantumCircuit

    from qiskit_qudits.circuit.quantumcircuit import QuditQuantumCircuit
    from qiskit_qudits.gates.base.gate import QuditGate

#: Default absolute tolerance for floating-point matrix comparisons.
ATOL = 1e-10

#: Dimensions whose encoding exactly fills the qubit Hilbert space.
POWER_OF_TWO_DIMS = (2, 4, 8, 16)

#: Dimensions that leave invalid ("leakage") basis states behind.
NON_POWER_OF_TWO_DIMS = (3, 5, 6, 7, 11)

#: A representative mix, small enough to keep the suite fast.
ALL_DIMS = (2, 3, 4, 5, 7, 8)


# ------------------------------------------------------------------ #
# Matrices
# ------------------------------------------------------------------ #
def gate_matrix(gate: QuditGate) -> np.typing.NDArray[np.complex128]:
    """Return the full ``2**n x 2**n`` unitary of ``gate``."""
    return np.asarray(gate, dtype=np.complex128)


def subspace_block(gate: QuditGate) -> np.typing.NDArray[np.complex128]:
    """Return the ``d x d`` block of ``gate`` acting on valid levels."""
    dim = gate.dim
    return gate_matrix(gate)[:dim, :dim]


def definition_matrix(gate: QuditGate) -> np.typing.NDArray[np.complex128]:
    """Return the unitary realised by the gate's qubit decomposition.

    Touching ``gate.definition`` is what triggers the lazy ``_define``
    call, so this doubles as a check that the decomposition builds.
    """
    definition = gate.definition
    assert definition is not None, f"{type(gate).__name__} has no definition"
    return np.asarray(Operator(definition).data, dtype=np.complex128)


def omega(dim: int) -> complex:
    r"""Return the primitive root of unity :math:`e^{2 \pi i / d}`."""
    return complex(np.exp(2j * np.pi / dim))


def assert_unitary(
    matrix: np.typing.NDArray[np.complex128],
    *,
    atol: float = ATOL,
) -> None:
    """Assert that ``matrix`` is square and unitary."""
    rows, columns = matrix.shape
    assert rows == columns, f"matrix is not square: {matrix.shape}"
    identity = np.eye(rows, dtype=np.complex128)
    assert np.allclose(
        matrix @ matrix.conj().T,
        identity,
        atol=atol,
    ), "matrix is not unitary"


def assert_allclose(
    actual: np.typing.NDArray[Any],
    expected: np.typing.NDArray[Any],
    *,
    atol: float = ATOL,
    message: str = "",
) -> None:
    """Assert two arrays agree, with a readable failure message."""
    assert np.allclose(actual, expected, atol=atol), (
        f"{message}\nactual:\n{np.round(actual, 6)}\n"
        f"expected:\n{np.round(expected, 6)}"
    )


def assert_leakage_states_are_fixed(
    gate: QuditGate,
    *,
    atol: float = ATOL,
) -> None:
    """Assert basis states outside the qudit subspace are untouched.

    Every gate in the library promises that the ``2**n - d`` invalid
    basis states map to themselves, which is what keeps the padded
    matrix unitary.
    """
    matrix = gate_matrix(gate)
    dim = gate.dim
    hilbert_dim = gate.hilbert_dim
    if dim == hilbert_dim:
        return
    invalid = matrix[dim:, dim:]
    assert_allclose(
        invalid,
        np.eye(hilbert_dim - dim, dtype=np.complex128),
        atol=atol,
        message="invalid basis states are not fixed points",
    )
    assert_allclose(
        matrix[:dim, dim:],
        np.zeros((dim, hilbert_dim - dim), dtype=np.complex128),
        atol=atol,
        message="the qudit subspace is not decoupled from leakage states",
    )
    assert_allclose(
        matrix[dim:, :dim],
        np.zeros((hilbert_dim - dim, dim), dtype=np.complex128),
        atol=atol,
        message="the qudit subspace is not decoupled from leakage states",
    )


# ------------------------------------------------------------------ #
# Simulation
# ------------------------------------------------------------------ #
def encoded_statevector(circuit: QuditQuantumCircuit) -> Statevector:
    """Return the encoded (``2**N`` amplitudes) state of a circuit."""
    return Statevector.from_instruction(circuit.circuit)


def logical_statevector(
    circuit: QuditQuantumCircuit,
) -> np.typing.NDArray[np.complex128]:
    r"""Return the :math:`\prod_i d_i` amplitudes of a qudit circuit.

    Qudit ``0`` is the least significant factor, exactly like Qiskit
    orders qubits.
    """
    encoded = np.asarray(
        encoded_statevector(circuit).data,
        dtype=np.complex128,
    )
    return project_state(circuit.dims, encoded)


def basis_state(
    dims: Sequence[int],
    levels: Sequence[int],
) -> np.typing.NDArray[np.complex128]:
    r"""Build the logical basis vector :math:`\lvert levels \rangle`.

    Args:
        dims: Dimension of each qudit, least significant first.
        levels: Level of each qudit, in the same order.

    Returns:
        A unit vector over the ``prod(dims)`` dimensional qudit space.
    """
    index = 0
    stride = 1
    for dim, level in zip(dims, levels, strict=True):
        index += level * stride
        stride *= dim
    vector = np.zeros(stride, dtype=np.complex128)
    vector[index] = 1.0
    return vector


def run_counts(
    circuit: QuditQuantumCircuit | QuantumCircuit,
    *,
    shots: int = 4096,
    seed: int = 1234,
) -> dict[str, int]:
    """Sample a circuit on Qiskit's built-in reference simulator.

    Aer is not a dependency of this project, so the pure-Python
    :class:`~qiskit.providers.basic_provider.BasicSimulator` is used;
    it is deterministic given ``seed``.

    Args:
        circuit: A qudit circuit (its encoded circuit is used) or a
            plain Qiskit circuit.
        shots: Number of shots.
        seed: Simulator seed, fixed so the suite never flakes.

    Returns:
        The raw bit-string counts.
    """
    encoded = getattr(circuit, "circuit", circuit)
    simulator = BasicSimulator()
    transpiled = transpile(encoded, simulator, seed_transpiler=seed)
    result = simulator.run(
        transpiled,
        shots=shots,
        seed_simulator=seed,
    ).result()
    return dict(result.get_counts())


def sample_levels(
    circuit: QuditQuantumCircuit,
    *,
    shots: int = 4096,
    seed: int = 1234,
    on_invalid: str = "keep",
) -> dict[tuple[int, ...], int]:
    """Sample a qudit circuit and decode the outcomes into levels.

    Args:
        circuit: The qudit circuit; it must contain measurements.
        shots: Number of shots.
        seed: Simulator seed.
        on_invalid: Leakage policy forwarded to ``decode_counts``.

    Returns:
        Mapping from a tuple of levels (clbyte order) to shot counts.
    """
    counts = run_counts(circuit, shots=shots, seed=seed)
    return circuit.decode_counts(counts, on_invalid=on_invalid)


# ------------------------------------------------------------------ #
# Statistics
# ------------------------------------------------------------------ #
def assert_distribution_close(
    observed: Mapping[Any, int],
    expected_probabilities: Mapping[Any, float],
    *,
    shots: int,
    atol: float = 0.05,
) -> None:
    """Assert sampled frequencies match a theoretical distribution.

    Uses a plain per-outcome tolerance on the frequency rather than a
    p-value: with a fixed simulator seed the comparison is
    deterministic, and a flat tolerance states the physics
    expectation directly.

    Args:
        observed: Sampled counts.
        expected_probabilities: Theoretical probability of each
            outcome. Outcomes absent from it must not be observed.
        shots: Total number of shots drawn.
        atol: Allowed absolute deviation of each frequency.
    """
    assert (
        sum(observed.values()) == shots
    ), f"expected {shots} shots, got {sum(observed.values())}"
    unexpected = {
        outcome: count
        for outcome, count in observed.items()
        if expected_probabilities.get(outcome, 0.0) == 0.0 and count
    }
    assert not unexpected, f"impossible outcomes were sampled: {unexpected}"

    for outcome, probability in expected_probabilities.items():
        frequency = observed.get(outcome, 0) / shots
        assert abs(frequency - probability) <= atol, (
            f"outcome {outcome!r}: sampled {frequency:.4f}, "
            f"expected {probability:.4f} (tolerance {atol})"
        )


def assert_deterministic_outcome(
    observed: Mapping[Any, int],
    outcome: Any,
    *,
    shots: int,
) -> None:
    """Assert every shot produced ``outcome``."""
    assert observed == {
        outcome: shots,
    }, f"expected all {shots} shots on {outcome!r}, got {dict(observed)}"


def parametrize_dims(dims: Sequence[int] = ALL_DIMS) -> pytest.MarkDecorator:
    """Return a ``dim`` parametrisation with readable test ids."""
    return pytest.mark.parametrize("dim", dims, ids=[f"d{d}" for d in dims])
