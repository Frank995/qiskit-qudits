"""Unitarity and encoding safety of the single-qudit gate matrices.

A :math:`d`-level gate is stored as a ``2**n x 2**n`` matrix with
``n = ceil(log2 d)``. For that padded matrix to describe a physical
qudit operation three statements must hold simultaneously:

* the full matrix is unitary, otherwise it cannot be run on qubit
  hardware at all;
* the ``d x d`` block spanned by the physical levels is *already*
  unitary, i.e. the qudit evolution is closed and probability never
  flows out of the computational subspace;
* the ``2**n - d`` leakage states are fixed points, which is the only
  padding compatible with the first two requirements.

These tests check those statements together with their immediate
consequences: unit determinant, unimodular spectrum, and preservation
of norms and inner products of physical states.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
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
    ALL_DIMS,
    ATOL,
    NON_POWER_OF_TWO_DIMS,
    assert_allclose,
    assert_leakage_states_are_fixed,
    assert_unitary,
    gate_matrix,
    parametrize_dims,
    subspace_block,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit_qudits.gates.base.gate import QuditGate

#: Generic angles for the parametric phase gate; neither is a special
#: value such as ``pi`` or ``pi/2``.
POSITIVE_THETA = 0.7
NEGATIVE_THETA = -2.5

#: Largest dimension the library accepts (``QuditGateMixin.MAX_DIM``).
MAX_DIM = 16

#: Seed base of the random physical states used below.
SEED = 20_240_611

GATE_FACTORIES: tuple[tuple[str, Callable[[int], QuditGate]], ...] = (
    ("I", QuditIGate),
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
    ("P(0.7)", lambda dim: QuditPGate(dim, POSITIVE_THETA)),
    ("P(-2.5)", lambda dim: QuditPGate(dim, NEGATIVE_THETA)),
)

parametrize_gates = pytest.mark.parametrize(
    "make_gate",
    [pytest.param(factory, id=name) for name, factory in GATE_FACTORIES],
)


def _random_physical_state(
    dim: int,
    hilbert_dim: int,
    rng: np.random.Generator,
) -> np.typing.NDArray[np.complex128]:
    """Return a normalised state supported on the qudit levels only.

    Args:
        dim: Number of physical levels of the qudit.
        hilbert_dim: Size of the encoded qubit Hilbert space.
        rng: Seeded generator, so the suite never flakes.

    Returns:
        A ``hilbert_dim`` dimensional unit vector that vanishes on the
        leakage indices.
    """
    amplitudes = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    amplitudes /= np.linalg.norm(amplitudes)
    state = np.zeros(hilbert_dim, dtype=np.complex128)
    state[:dim] = amplitudes
    return state


@parametrize_gates
@parametrize_dims()
def test_full_matrix_is_unitary(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check the padded ``2**n x 2**n`` matrix is unitary."""
    matrix = gate_matrix(make_gate(dim))

    assert_unitary(matrix)
    assert_allclose(
        matrix.conj().T @ matrix,
        np.eye(matrix.shape[0], dtype=np.complex128),
        message="U^dagger U is not the identity",
    )


@parametrize_gates
@parametrize_dims()
def test_subspace_block_is_unitary(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check the physical ``d x d`` block is unitary on its own."""
    block = subspace_block(make_gate(dim))

    assert block.shape == (dim, dim)
    assert_unitary(block)


@parametrize_gates
@parametrize_dims(NON_POWER_OF_TWO_DIMS)
def test_leakage_states_are_fixed_points(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check leakage levels are frozen and decoupled from the qudit."""
    assert_leakage_states_are_fixed(make_gate(dim))


@parametrize_gates
@parametrize_dims((*ALL_DIMS, 6, 11))
def test_padding_size_matches_the_encoding(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check the matrix width and the leakage count follow from d."""
    gate = make_gate(dim)
    num_qubits = math.ceil(math.log2(dim))
    hilbert_dim = 2**num_qubits

    assert gate.num_qubits == num_qubits
    assert gate.hilbert_dim == hilbert_dim
    assert gate.num_invalid_states == hilbert_dim - dim
    assert gate_matrix(gate).shape == (hilbert_dim, hilbert_dim)


@parametrize_gates
@parametrize_dims()
def test_determinant_has_unit_modulus(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check ``|det U| = 1``, as required of a norm-preserving map."""
    determinant = np.linalg.det(gate_matrix(make_gate(dim)))

    assert abs(determinant) == pytest.approx(1.0, abs=ATOL)


@parametrize_gates
@parametrize_dims()
def test_eigenvalues_lie_on_the_unit_circle(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check every eigenvalue is a pure phase, as for any unitary."""
    eigenvalues = np.linalg.eigvals(gate_matrix(make_gate(dim)))

    assert_allclose(
        np.abs(eigenvalues),
        np.ones(eigenvalues.size),
        message="the gate has an eigenvalue off the unit circle",
    )


@parametrize_gates
@parametrize_dims()
def test_preserves_the_norm_of_a_physical_state(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check a random qudit state keeps unit norm under the gate."""
    gate = make_gate(dim)
    rng = np.random.default_rng(SEED + dim)
    state = _random_physical_state(dim, gate.hilbert_dim, rng)

    evolved = gate_matrix(gate) @ state

    assert float(np.linalg.norm(evolved)) == pytest.approx(1.0, abs=ATOL)


@parametrize_gates
@parametrize_dims()
def test_preserves_inner_products(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check ``<Ua|Ub> = <a|b>``: the gate is an isometry of states."""
    gate = make_gate(dim)
    rng = np.random.default_rng(SEED + dim)
    first = _random_physical_state(dim, gate.hilbert_dim, rng)
    second = _random_physical_state(dim, gate.hilbert_dim, rng)
    matrix = gate_matrix(gate)

    before = complex(np.vdot(first, second))
    after = complex(np.vdot(matrix @ first, matrix @ second))

    assert after == pytest.approx(before, abs=ATOL)


@pytest.mark.slow
@parametrize_gates
def test_matrix_is_unitary_at_the_maximum_dimension(
    make_gate: Callable[[int], QuditGate],
) -> None:
    """Check the largest supported qudit (``d=16``) stays unitary."""
    matrix = gate_matrix(make_gate(MAX_DIM))

    assert matrix.shape == (MAX_DIM, MAX_DIM)
    assert_unitary(matrix)
