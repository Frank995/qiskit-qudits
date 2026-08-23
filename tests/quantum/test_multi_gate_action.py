r"""Level-by-level action of the multi-qudit gates.

Each gate is applied to every valid computational basis state of its
register and compared with the textbook formula, written in terms of
:math:`\omega_D = e^{2 i \pi / D}`:

.. math::

    SWAP \lvert j \rangle \lvert k \rangle
        &= \lvert k \rangle \lvert j \rangle \\
    QFT \lvert x \rangle
        &= \frac{1}{\sqrt{d^n}} \sum_{y=0}^{d^n - 1}
           \omega_{d^n}^{xy} \lvert y \rangle

The reference vectors are assembled here from explicit powers of
:math:`\omega` and a hand-rolled little-endian index map (qudit 0 on
the lowest qubits, each qudit padded to ``ceil(log2 d)`` qubits on
its own), never from the strides or index helpers the implementation
uses, so the tests are an independent statement of the physics.

Leakage handling (basis states outside the valid subspace must be
fixed points) is covered by ``test_multi_gate_unitarity.py``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qiskit_qudits.gates import QuditQFTGate, QuditSWAPGate
from tests.helpers import assert_allclose, gate_matrix, omega, parametrize_dims

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Dimensions exercised by the SWAP tests.
SWAP_DIMS = (2, 3, 4, 5, 8)

#: ``(num_qudits, dim)`` register layouts for the QFT tests.
QFT_CONFIGS = ((2, 2), (3, 2), (4, 2), (2, 3), (2, 4), (3, 3), (2, 5))

parametrize_qft_configs = pytest.mark.parametrize(
    ("num_qudits", "dim"),
    [pytest.param(n, d, id=f"n{n}-d{d}") for n, d in QFT_CONFIGS],
)


def _qubit_width(dim: int) -> int:
    """Return the qubits needed by one ``dim``-level qudit."""
    return math.ceil(math.log2(dim))


def _encoded_index(dims: Sequence[int], levels: Sequence[int]) -> int:
    """Return the encoded qubit-space index of a level tuple.

    Reference implementation written from the documented convention:
    qudit 0 occupies the lowest-index qubits and each qudit is padded
    to its own ``ceil(log2 d)`` qubits.

    Args:
        dims: Dimension of each qudit, in register order.
        levels: Level of each qudit, in the same order.

    Returns:
        The index into the ``2**N`` dimensional encoded space.
    """
    index = 0
    shift = 0
    for dim, level in zip(dims, levels, strict=True):
        index += level << shift
        shift += _qubit_width(dim)
    return index


def _basis_state(
    hilbert_dim: int,
    index: int,
) -> np.typing.NDArray[np.complex128]:
    """Return the encoded basis vector with a one at ``index``."""
    state = np.zeros(hilbert_dim, dtype=np.complex128)
    state[index] = 1.0
    return state


def _digits(dim: int, num_qudits: int, value: int) -> tuple[int, ...]:
    """Split ``value`` into base-``dim`` digits, qudit 0 first."""
    return tuple(
        (value // dim**position) % dim for position in range(num_qudits)
    )


@parametrize_dims(SWAP_DIMS)
def test_swap_exchanges_the_two_levels(dim: int) -> None:
    """Check ``SWAP|j,k> = |k,j>``, equal levels included."""
    gate = QuditSWAPGate(dim)
    matrix = gate_matrix(gate)
    dims = gate.dims

    for first in range(dim):
        for second in range(dim):
            assert_allclose(
                matrix[:, _encoded_index(dims, (first, second))],
                _basis_state(
                    gate.hilbert_dim,
                    _encoded_index(dims, (second, first)),
                ),
                message=f"SWAP (d={dim}) acts wrongly on |{first}, {second}>",
            )


@parametrize_qft_configs
def test_qft_maps_each_basis_state_to_its_fourier_column(
    num_qudits: int,
    dim: int,
) -> None:
    """Check ``QFT|x> = sum_y w^(xy) |y> / sqrt(d^n)``."""
    gate = QuditQFTGate(num_qudits, dim)
    matrix = gate_matrix(gate)
    dims = gate.dims
    total_dim = dim**num_qudits
    root = omega(total_dim)

    for value in range(total_dim):
        expected = np.zeros(gate.hilbert_dim, dtype=np.complex128)
        for image in range(total_dim):
            expected[_encoded_index(dims, _digits(dim, num_qudits, image))] = (
                root ** (value * image % total_dim) / np.sqrt(total_dim)
            )

        assert_allclose(
            matrix[:, _encoded_index(dims, _digits(dim, num_qudits, value))],
            expected,
            message=(
                f"QFT (n={num_qudits}, d={dim}) acts wrongly on |{value}>"
            ),
        )


@parametrize_qft_configs
def test_qft_maps_zero_onto_the_uniform_superposition(
    num_qudits: int,
    dim: int,
) -> None:
    """Check ``QFT|0>`` is flat over the valid subspace only."""
    gate = QuditQFTGate(num_qudits, dim)
    matrix = gate_matrix(gate)
    dims = gate.dims
    total_dim = dim**num_qudits

    expected = np.zeros(gate.hilbert_dim, dtype=np.complex128)
    for image in range(total_dim):
        expected[_encoded_index(dims, _digits(dim, num_qudits, image))] = (
            1.0 / np.sqrt(total_dim)
        )

    assert_allclose(
        matrix[:, _encoded_index(dims, (0,) * num_qudits)],
        expected,
        message=(
            "QFT|0> is not the uniform superposition over the valid subspace"
        ),
    )
