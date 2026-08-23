r"""Level-by-level action of every controlled qudit gate.

Each gate is applied to every valid computational basis state of its
register and compared with the textbook formula, written in terms of
:math:`\omega_D = e^{2 i \pi / D}`:

.. math::

    SUMX \lvert j_0 \cdots j_{m-1} \rangle \lvert k \rangle
        &= \lvert j_0 \cdots j_{m-1} \rangle
           \lvert (k + {\textstyle\sum_i j_i}) \bmod d_t \rangle \\
    SUMX^\dagger \lvert j_0 \cdots j_{m-1} \rangle \lvert k \rangle
        &= \lvert j_0 \cdots j_{m-1} \rangle
           \lvert (k - {\textstyle\sum_i j_i}) \bmod d_t \rangle \\
    SUMP(\theta) \lvert j_0 \cdots j_{m-1} \rangle \lvert k \rangle
        &= \omega_{d_t}^{({\sum_i j_i}) k \theta / \pi}
           \lvert j_0 \cdots j_{m-1} \rangle \lvert k \rangle

The reference vectors are assembled here from explicit powers of
:math:`\omega` and a hand-rolled little-endian index map (qudit 0 on
the lowest qubits, each qudit padded to ``ceil(log2 d)`` qubits on
its own), never from the strides or index helpers the implementation
uses, so the tests are an independent statement of the physics.

Leakage handling (basis states outside the valid subspace must be
fixed points) is covered by ``test_multi_gate_unitarity.py``.
"""

from __future__ import annotations

import itertools as it
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qiskit_qudits.gates import QuditSUMPGate, QuditSUMXdgGate, QuditSUMXGate
from tests.helpers import assert_allclose, gate_matrix, omega

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from qiskit_qudits.gates.base.controlledgate import QuditControlledGate

#: Angle of the parametric controlled-phase gate.
THETA = 0.7

#: ``(target dim, control dims)`` register layouts, mixing dense and
#: leaky encodings, single and multiple controls.
CONTROLLED_CONFIGS = (
    (2, (2,)),
    (4, (2,)),
    (2, (4,)),
    (4, (4,)),
    (2, (2, 2)),
    (3, (2,)),
    (2, (3,)),
    (3, (3,)),
    (5, (3,)),
    (3, (2, 3)),
)

#: Every controlled gate, built from a shared configuration.
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


parametrize_controlled_configs = pytest.mark.parametrize(
    ("target_dim", "control_dims"),
    [
        pytest.param(target, controls, id=_config_id(target, controls))
        for target, controls in CONTROLLED_CONFIGS
    ],
)

parametrize_controlled_gates = pytest.mark.parametrize(
    "make_gate",
    [pytest.param(factory, id=name) for name, factory in CONTROLLED_FACTORIES],
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


@parametrize_controlled_configs
def test_sumx_shifts_the_target_by_the_control_sum(
    target_dim: int,
    control_dims: tuple[int, ...],
) -> None:
    """Check ``SUMX|j...,k> = |j...,(k + sum j) mod d_t>``."""
    gate = QuditSUMXGate(target_dim, control_dims)
    matrix = gate_matrix(gate)
    dims = gate.dims

    for levels in it.product(*(range(dim) for dim in dims)):
        *controls, level = levels
        shifted = (level + sum(controls)) % target_dim
        assert_allclose(
            matrix[:, _encoded_index(dims, levels)],
            _basis_state(
                gate.hilbert_dim,
                _encoded_index(dims, (*controls, shifted)),
            ),
            message=(
                f"SUMX (d_t={target_dim}) acts wrongly on controls "
                f"{tuple(controls)} and target |{level}>"
            ),
        )


@parametrize_controlled_configs
def test_sumxdg_shifts_the_target_down_by_the_control_sum(
    target_dim: int,
    control_dims: tuple[int, ...],
) -> None:
    """Check ``SUMXdg|j...,k> = |j...,(k - sum j) mod d_t>``."""
    gate = QuditSUMXdgGate(target_dim, control_dims)
    matrix = gate_matrix(gate)
    dims = gate.dims

    for levels in it.product(*(range(dim) for dim in dims)):
        *controls, level = levels
        shifted = (level - sum(controls)) % target_dim
        assert_allclose(
            matrix[:, _encoded_index(dims, levels)],
            _basis_state(
                gate.hilbert_dim,
                _encoded_index(dims, (*controls, shifted)),
            ),
            message=(
                f"SUMXdg (d_t={target_dim}) acts wrongly on controls "
                f"{tuple(controls)} and target |{level}>"
            ),
        )


@parametrize_controlled_configs
def test_sump_imprints_the_documented_phase(
    target_dim: int,
    control_dims: tuple[int, ...],
) -> None:
    """Check ``SUMP`` multiplies by ``w^((sum j) k theta / pi)``."""
    gate = QuditSUMPGate(target_dim, control_dims, THETA)
    matrix = gate_matrix(gate)
    dims = gate.dims
    root = omega(target_dim)

    for levels in it.product(*(range(dim) for dim in dims)):
        *controls, level = levels
        index = _encoded_index(dims, levels)
        phase = root ** (sum(controls) * level * THETA / np.pi)
        assert_allclose(
            matrix[:, index],
            phase * _basis_state(gate.hilbert_dim, index),
            message=(
                f"SUMP (d_t={target_dim}) applies the wrong phase on "
                f"controls {tuple(controls)} and target |{level}>"
            ),
        )


@parametrize_controlled_gates
@parametrize_controlled_configs
def test_zero_valued_controls_act_trivially(
    make_gate: Callable[[int, tuple[int, ...]], QuditControlledGate],
    target_dim: int,
    control_dims: tuple[int, ...],
) -> None:
    """Check every gate is the identity when all controls are 0."""
    gate = make_gate(target_dim, control_dims)
    matrix = gate_matrix(gate)
    dims = gate.dims
    idle_controls = (0,) * len(control_dims)

    for level in range(target_dim):
        index = _encoded_index(dims, (*idle_controls, level))
        assert_allclose(
            matrix[:, index],
            _basis_state(gate.hilbert_dim, index),
            message=(
                "a controlled gate moved the target although every "
                f"control was |0> (target level |{level}>)"
            ),
        )
