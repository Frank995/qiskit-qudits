r"""Level-by-level action of every qudit gate.

Each gate is applied to every encoded computational basis state and
compared with the textbook formula, written directly in terms of
:math:`\omega = e^{2 i \pi / d}`:

.. math::

    X \lvert k \rangle       &= \lvert k+1 \bmod d \rangle \\
    X^\dagger \lvert k \rangle &= \lvert k-1 \bmod d \rangle \\
    NOT \lvert k \rangle     &= \lvert d-1-k \rangle \\
    K \lvert k \rangle       &= \lvert (d-k) \bmod d \rangle \\
    Z \lvert k \rangle       &= \omega^{k} \lvert k \rangle \\
    S \lvert k \rangle       &= \omega^{k/2} \lvert k \rangle \\
    T \lvert k \rangle       &= \omega^{k/4} \lvert k \rangle \\
    P(\theta) \lvert k \rangle &= e^{2 i \theta k / d}
                                 \lvert k \rangle \\
    H \lvert k \rangle       &= \frac{1}{\sqrt{d}}
        \sum_{j} \omega^{-jk} \lvert j \rangle

The reference vectors are built here from explicit powers of
:math:`\omega`, never from the routines the implementation uses
(``scipy.linalg.dft``, ``numpy.roll``, ``numpy.flipud``), so the test
is an independent statement of the physics.

The same tests also show that no valid level ever acquires an
amplitude on a leakage index, and that the leakage levels themselves
are inert -- the property that makes the qubit encoding safe.
"""

from __future__ import annotations

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
    NON_POWER_OF_TWO_DIMS,
    assert_allclose,
    gate_matrix,
    omega,
    parametrize_dims,
    subspace_block,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit_qudits.gates.base.gate import QuditGate

#: Angle of the parametric phase gate used in the sweep below.
THETA = 0.7

#: ``k -> k'`` for the gates that only permute the levels.
PERMUTATIONS: dict[str, Callable[[int, int], int]] = {
    "I": lambda dim, level: level % dim,
    "X": lambda dim, level: (level + 1) % dim,
    "Xdg": lambda dim, level: (level - 1) % dim,
    "NOT": lambda dim, level: dim - 1 - level,
    "K": lambda dim, level: (dim - level) % dim,
}

#: Exponent ``e`` of the diagonal gates, whose action is ``w^(e k)``.
PHASE_EXPONENTS: dict[str, float] = {
    "Z": 1.0,
    "Zdg": -1.0,
    "S": 0.5,
    "Sdg": -0.5,
    "T": 0.25,
    "Tdg": -0.25,
}

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
    ("P", lambda dim: QuditPGate(dim, THETA)),
)

parametrize_named_gates = pytest.mark.parametrize(
    ("name", "make_gate"),
    [pytest.param(name, factory, id=name) for name, factory in GATE_FACTORIES],
)

parametrize_gates = pytest.mark.parametrize(
    "make_gate",
    [pytest.param(factory, id=name) for name, factory in GATE_FACTORIES],
)


def _expected_image(
    name: str,
    dim: int,
    level: int,
) -> np.typing.NDArray[np.complex128]:
    r"""Return the analytic image of :math:`\lvert level \rangle`.

    Args:
        name: Short gate name, as registered in ``GATE_FACTORIES``.
        dim: Number of physical levels.
        level: Index of the computational basis state, ``< dim``.

    Returns:
        The ``dim`` amplitudes of the transformed state.
    """
    root = omega(dim)
    image = np.zeros(dim, dtype=np.complex128)
    if name in PERMUTATIONS:
        image[PERMUTATIONS[name](dim, level)] = 1.0
    elif name in PHASE_EXPONENTS:
        image[level] = root ** (PHASE_EXPONENTS[name] * level)
    elif name == "P":
        image[level] = root ** (level * THETA / np.pi)
    elif name in {"H", "Hdg"}:
        sign = -1 if name == "H" else 1
        image = np.array(
            [root ** (sign * index * level) for index in range(dim)],
            dtype=np.complex128,
        ) / np.sqrt(dim)
    else:
        pytest.fail(f"no analytic formula is registered for gate '{name}'")
    return image


def _basis_state(
    hilbert_dim: int,
    index: int,
) -> np.typing.NDArray[np.complex128]:
    """Return the encoded basis vector with a one at ``index``."""
    state = np.zeros(hilbert_dim, dtype=np.complex128)
    state[index] = 1.0
    return state


@parametrize_named_gates
@parametrize_dims()
def test_action_on_the_computational_basis(
    name: str,
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check ``U|k>`` against the analytic formula for every level."""
    gate = make_gate(dim)
    matrix = gate_matrix(gate)

    for level in range(dim):
        expected = np.zeros(gate.hilbert_dim, dtype=np.complex128)
        expected[:dim] = _expected_image(name, dim, level)

        assert_allclose(
            matrix @ _basis_state(gate.hilbert_dim, level),
            expected,
            message=f"{name} (d={dim}) acts wrongly on level |{level}>",
        )


@parametrize_gates
@parametrize_dims(NON_POWER_OF_TWO_DIMS)
def test_leakage_levels_are_left_untouched(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check ``U|k> = |k>`` for every leakage level k >= d."""
    gate = make_gate(dim)
    matrix = gate_matrix(gate)

    for level in range(dim, gate.hilbert_dim):
        state = _basis_state(gate.hilbert_dim, level)

        assert_allclose(
            matrix @ state,
            state,
            message=(
                f"the invalid level |{level}> of a d={dim} qudit is not "
                "a fixed point, so the encoding is not safe"
            ),
        )


@pytest.mark.parametrize(
    "theta",
    [0.0, 0.7, -2.5, np.pi, np.pi / 2, np.pi / 4],
)
@parametrize_dims()
def test_phase_gate_action_on_each_level(theta: float, dim: int) -> None:
    """Check ``P(theta)|k> = exp(2 i theta k / d) |k>``."""
    block = subspace_block(QuditPGate(dim, theta))
    expected = np.diag(
        np.array(
            [np.exp(2j * theta * level / dim) for level in range(dim)],
            dtype=np.complex128,
        ),
    )

    assert_allclose(
        block,
        expected,
        message=f"P({theta}) is not the diagonal phase ladder for d={dim}",
    )


@pytest.mark.parametrize(
    "make_gate",
    [
        pytest.param(QuditHGate, id="H"),
        pytest.param(QuditHdgGate, id="Hdg"),
    ],
)
@parametrize_dims()
def test_fourier_gate_produces_an_unbiased_superposition(
    make_gate: Callable[[int], QuditGate],
    dim: int,
) -> None:
    """Check ``|<j|H|k>|^2 = 1/d``: H spreads a level uniformly."""
    block = subspace_block(make_gate(dim))

    assert_allclose(
        np.abs(block) ** 2,
        np.full((dim, dim), 1.0 / dim),
        message=(
            "the Fourier gate does not map a level onto a flat "
            "superposition of all levels"
        ),
    )


@parametrize_dims()
def test_identity_gate_is_the_identity_on_the_whole_space(dim: int) -> None:
    """Check ``I`` is the identity on encoded *and* leakage levels."""
    gate = QuditIGate(dim)

    assert_allclose(
        gate_matrix(gate),
        np.eye(gate.hilbert_dim, dtype=np.complex128),
        message="the identity gate is not the 2**n identity matrix",
    )
