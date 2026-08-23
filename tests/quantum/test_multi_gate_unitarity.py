"""Unitarity and encoding safety of the multi-target qudit gate matrices.

``SWAP`` and the qudit ``QFT`` act jointly on several qudits sharing a
common dimension. As for the controlled gates, the register is
encoded as the tensor product of one padded ``2**n`` block per qudit,
so the valid subspace is not the leading contiguous block: a basis
state is valid only when *every* qudit level is below the shared
dimension. The checks here therefore work on explicit index sets:

* the full ``2**N x 2**N`` matrix is unitary;
* the ``d**N x d**N`` block spanned by the valid states is already
  unitary on its own, so probability never flows out of the
  computational subspace;
* every invalid basis state is a fixed point, decoupled from the
  valid subspace in both directions;
* the padding bookkeeping (``num_qubits``, ``hilbert_dim``,
  ``num_invalid_states``, ``fills_hilbert_space``) follows from the
  dimensions alone.

As for the single-qudit suite, these checks are complemented by unit
determinant, unimodular spectrum, and preservation of norms and inner
products of physical register states.
"""

from __future__ import annotations

import math
from math import prod
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qiskit_qudits.gates import QuditQFTGate, QuditSWAPGate
from tests.helpers import ATOL, assert_allclose, assert_unitary, gate_matrix

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from qiskit.circuit import Gate

#: Largest dimension the library accepts (``QuditGateMixin.MAX_DIM``).
MAX_DIM = 16

#: Seed base of the random physical states used below.
SEED = 20_240_611

#: Dimensions exercised by the SWAP tests.
SWAP_DIMS = (2, 3, 4, 5, 8)

#: ``(num_qudits, dim)`` register layouts for the QFT tests.
QFT_CONFIGS = ((2, 2), (3, 2), (4, 2), (2, 3), (2, 4), (3, 3), (2, 5))


def _has_leakage(dims: Sequence[int]) -> bool:
    """Return whether any dimension is not a power of two."""
    return any(dim & (dim - 1) for dim in dims)


def _gate_cases(*, leaky_only: bool) -> list[object]:
    """Build the parametrisation for every gate under test.

    Args:
        leaky_only: Restrict to registers with invalid basis states.

    Returns:
        ``pytest.param`` entries holding zero-argument factories.
    """
    cases: list[object] = []
    for dim in SWAP_DIMS:
        if leaky_only and not _has_leakage((dim, dim)):
            continue
        cases.append(
            pytest.param(
                lambda dim=dim: QuditSWAPGate(dim),
                id=f"SWAP-d{dim}",
            ),
        )
    for num_qudits, dim in QFT_CONFIGS:
        if leaky_only and not _has_leakage((dim,) * num_qudits):
            continue
        cases.append(
            pytest.param(
                lambda n=num_qudits, d=dim: QuditQFTGate(n, d),
                id=f"QFT-n{num_qudits}-d{dim}",
            ),
        )
    return cases


parametrize_gates = pytest.mark.parametrize(
    "make_gate",
    _gate_cases(leaky_only=False),
)

parametrize_leaky_gates = pytest.mark.parametrize(
    "make_gate",
    _gate_cases(leaky_only=True),
)


def _qubit_width(dim: int) -> int:
    """Return the qubits needed by one ``dim``-level qudit."""
    return math.ceil(math.log2(dim))


def _decoded_levels(dims: Sequence[int], index: int) -> tuple[int, ...]:
    """Split an encoded index into one raw level per qudit."""
    levels: list[int] = []
    remainder = index
    for dim in dims:
        width = _qubit_width(dim)
        levels.append(remainder & ((1 << width) - 1))
        remainder >>= width
    return tuple(levels)


def _valid_indices(dims: Sequence[int], hilbert_dim: int) -> list[int]:
    """Return every encoded index whose levels are all valid."""
    return [
        index
        for index in range(hilbert_dim)
        if all(
            level < dim
            for level, dim in zip(
                _decoded_levels(dims, index),
                dims,
                strict=True,
            )
        )
    ]


def _random_physical_state(
    valid: Sequence[int],
    hilbert_dim: int,
    rng: np.random.Generator,
) -> np.typing.NDArray[np.complex128]:
    """Return a normalised state supported on the valid levels only.

    Args:
        valid: Encoded indices spanned by the physical register.
        hilbert_dim: Size of the encoded Hilbert space.
        rng: Seeded generator, so the suite never flakes.

    Returns:
        A ``hilbert_dim`` dimensional unit vector that vanishes on
        every invalid (leakage) index.
    """
    amplitudes = rng.normal(size=len(valid)) + 1j * rng.normal(size=len(valid))
    amplitudes /= np.linalg.norm(amplitudes)
    state = np.zeros(hilbert_dim, dtype=np.complex128)
    state[valid] = amplitudes
    return state


@parametrize_gates
def test_full_matrix_is_unitary(make_gate: Callable[[], Gate]) -> None:
    """Check the padded ``2**N x 2**N`` matrix is unitary."""
    matrix = gate_matrix(make_gate())

    assert_unitary(matrix)
    assert_allclose(
        matrix.conj().T @ matrix,
        np.eye(matrix.shape[0], dtype=np.complex128),
        message="U^dagger U is not the identity",
    )


@parametrize_gates
def test_the_valid_subspace_is_closed_and_unitary(
    make_gate: Callable[[], Gate],
) -> None:
    """Check the block over the valid states is unitary on its own."""
    gate = make_gate()
    matrix = gate_matrix(gate)
    valid = _valid_indices(gate.dims, gate.hilbert_dim)
    block = matrix[np.ix_(valid, valid)]

    assert block.shape == (prod(gate.dims), prod(gate.dims))
    assert_unitary(block)


@parametrize_leaky_gates
def test_leakage_states_are_fixed_and_decoupled(
    make_gate: Callable[[], Gate],
) -> None:
    """Check invalid states are frozen and never mix with the rest."""
    gate = make_gate()
    matrix = gate_matrix(gate)
    valid = _valid_indices(gate.dims, gate.hilbert_dim)
    valid_set = set(valid)
    invalid = [
        index for index in range(gate.hilbert_dim) if index not in valid_set
    ]

    assert invalid, "expected at least one leakage state"
    assert len(invalid) == gate.num_invalid_states
    assert_allclose(
        matrix[np.ix_(invalid, invalid)],
        np.eye(len(invalid), dtype=np.complex128),
        message="invalid basis states are not fixed points",
    )
    assert_allclose(
        matrix[np.ix_(valid, invalid)],
        np.zeros((len(valid), len(invalid)), dtype=np.complex128),
        message="the qudit subspace is not decoupled from leakage states",
    )
    assert_allclose(
        matrix[np.ix_(invalid, valid)],
        np.zeros((len(invalid), len(valid)), dtype=np.complex128),
        message="the qudit subspace is not decoupled from leakage states",
    )


@parametrize_gates
def test_padding_size_matches_the_encoding(
    make_gate: Callable[[], Gate],
) -> None:
    """Check the matrix width and leakage count follow the dims."""
    gate = make_gate()
    num_qubits = sum(_qubit_width(dim) for dim in gate.dims)
    hilbert_dim = 2**num_qubits

    assert gate.num_qubits == num_qubits
    assert gate.hilbert_dim == hilbert_dim
    assert gate.num_invalid_states == hilbert_dim - prod(gate.dims)
    assert gate.fills_hilbert_space is not _has_leakage(gate.dims)
    assert gate_matrix(gate).shape == (hilbert_dim, hilbert_dim)


@parametrize_gates
def test_determinant_has_unit_modulus(
    make_gate: Callable[[], Gate],
) -> None:
    """Check ``|det U| = 1``, as required of a norm-preserving map."""
    determinant = np.linalg.det(gate_matrix(make_gate()))

    assert abs(determinant) == pytest.approx(1.0, abs=ATOL)


@parametrize_gates
def test_eigenvalues_lie_on_the_unit_circle(
    make_gate: Callable[[], Gate],
) -> None:
    """Check every eigenvalue is a pure phase, as for any unitary."""
    eigenvalues = np.linalg.eigvals(gate_matrix(make_gate()))

    assert_allclose(
        np.abs(eigenvalues),
        np.ones(eigenvalues.size),
        message="the gate has an eigenvalue off the unit circle",
    )


@parametrize_gates
def test_preserves_the_norm_of_a_physical_state(
    make_gate: Callable[[], Gate],
) -> None:
    """Check a random register state keeps unit norm under the gate."""
    gate = make_gate()
    valid = _valid_indices(gate.dims, gate.hilbert_dim)
    rng = np.random.default_rng(SEED + gate.hilbert_dim)
    state = _random_physical_state(valid, gate.hilbert_dim, rng)

    evolved = gate_matrix(gate) @ state

    assert float(np.linalg.norm(evolved)) == pytest.approx(1.0, abs=ATOL)


@parametrize_gates
def test_preserves_inner_products(make_gate: Callable[[], Gate]) -> None:
    """Check ``<Ua|Ub> = <a|b>``: the gate is an isometry of states."""
    gate = make_gate()
    valid = _valid_indices(gate.dims, gate.hilbert_dim)
    rng = np.random.default_rng(SEED + gate.hilbert_dim)
    first = _random_physical_state(valid, gate.hilbert_dim, rng)
    second = _random_physical_state(valid, gate.hilbert_dim, rng)
    matrix = gate_matrix(gate)

    before = complex(np.vdot(first, second))
    after = complex(np.vdot(matrix @ first, matrix @ second))

    assert after == pytest.approx(before, abs=ATOL)


@pytest.mark.slow
def test_swap_matrix_is_unitary_at_the_maximum_dimension() -> None:
    """Check SWAP stays unitary at the largest supported qudit dimension."""
    gate = QuditSWAPGate(MAX_DIM)
    matrix = gate_matrix(gate)

    assert matrix.shape == (gate.hilbert_dim, gate.hilbert_dim)
    assert_unitary(matrix)


@pytest.mark.slow
def test_qft_matrix_is_unitary_at_the_maximum_dimension() -> None:
    """Check QFT stays unitary at the largest supported qudit dimension."""
    gate = QuditQFTGate(2, MAX_DIM)
    matrix = gate_matrix(gate)

    assert matrix.shape == (gate.hilbert_dim, gate.hilbert_dim)
    assert_unitary(matrix)
