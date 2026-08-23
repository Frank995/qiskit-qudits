r"""Operator algebra of the qudit gate set.

Everything here is checked on the ``d x d`` physical block, because a
relation carrying a phase (such as the Weyl commutation relation) can
only hold on the qudit subspace: the padded leakage block is the
identity for every gate and would pick up the phase as well.

The relations that are verified are the defining ones of the
generalised Pauli (Weyl-Heisenberg) group and of its Clifford
companions, with :math:`\omega = e^{2 i \pi / d}`:

* :math:`X^d = Z^d = 1` and :math:`Z X = \omega X Z`;
* :math:`\{X^a Z^b\}` is an orthogonal operator basis;
* :math:`S^2 = Z`, :math:`T^2 = S`, :math:`T^4 = Z`;
* :math:`H Z H^\dagger = X` and :math:`H X H^\dagger = Z^\dagger`,
  so ``H`` is the Fourier transform of the clock-shift pair;
* :math:`H^2 = K`, :math:`H^4 = 1`, :math:`NOT^2 = K^2 = 1`;
* every ``inverse()`` is the matrix adjoint.

Note that :math:`H X H^\dagger = Z` is *false* for ``d > 2``: with the
inverse-DFT convention used by the library the conjugation produces
:math:`Z^\dagger` instead, and only for ``d = 2`` do the two coincide.
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
    ATOL,
    assert_allclose,
    gate_matrix,
    omega,
    parametrize_dims,
    subspace_block,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from qiskit_qudits.gates.base.gate import QuditGate

#: Generic angle for the parametric phase gate.
THETA = 0.7

#: Dimensions used by the quadratic-cost operator-basis test.
BASIS_DIMS = (2, 3, 4, 5)

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
    ("P(0.7)", lambda dim: QuditPGate(dim, THETA)),
)

#: Gate classes that the library advertises as adjoint partners.
DAGGER_PAIRS = (
    ("X", QuditXGate, QuditXdgGate),
    ("Z", QuditZGate, QuditZdgGate),
    ("S", QuditSGate, QuditSdgGate),
    ("T", QuditTGate, QuditTdgGate),
    ("H", QuditHGate, QuditHdgGate),
)

#: Angle at which each named phase gate must reproduce ``P(theta)``.
NAMED_PHASE_ANGLES = (
    ("Z", QuditZGate, np.pi),
    ("Zdg", QuditZdgGate, -np.pi),
    ("S", QuditSGate, np.pi / 2),
    ("Sdg", QuditSdgGate, -np.pi / 2),
    ("T", QuditTGate, np.pi / 4),
    ("Tdg", QuditTdgGate, -np.pi / 4),
)

parametrize_gates = pytest.mark.parametrize(
    "make_gate",
    [pytest.param(factory, id=name) for name, factory in GATE_FACTORIES],
)


def _identity(dim: int) -> np.typing.NDArray[np.complex128]:
    """Return the ``d x d`` identity."""
    return np.eye(dim, dtype=np.complex128)


class TestWeylPair:
    """The clock ``Z`` and shift ``X`` generators."""

    @parametrize_dims()
    def test_shift_operator_has_order_d(self, dim: int) -> None:
        """Check ``X^d = 1`` and that no smaller power is trivial."""
        shift = subspace_block(QuditXGate(dim))
        identity = _identity(dim)

        assert_allclose(
            np.linalg.matrix_power(shift, dim),
            identity,
            message=f"X^{dim} is not the identity",
        )
        for power in range(1, dim):
            assert not np.allclose(
                np.linalg.matrix_power(shift, power),
                identity,
                atol=ATOL,
            ), f"X^{power} is already the identity, so X has order < {dim}"

    @parametrize_dims()
    def test_clock_operator_has_order_d(self, dim: int) -> None:
        """Check ``Z^d = 1`` and that no smaller power is trivial."""
        clock = subspace_block(QuditZGate(dim))
        identity = _identity(dim)

        assert_allclose(
            np.linalg.matrix_power(clock, dim),
            identity,
            message=f"Z^{dim} is not the identity",
        )
        for power in range(1, dim):
            assert not np.allclose(
                np.linalg.matrix_power(clock, power),
                identity,
                atol=ATOL,
            ), f"Z^{power} is already the identity, so Z has order < {dim}"

    @parametrize_dims()
    def test_weyl_commutation_relation(self, dim: int) -> None:
        """Check ``Z X = w X Z``, the clock-shift algebra itself."""
        shift = subspace_block(QuditXGate(dim))
        clock = subspace_block(QuditZGate(dim))

        assert_allclose(
            clock @ shift,
            omega(dim) * (shift @ clock),
            message="the Weyl commutation relation Z X = w X Z is violated",
        )

    @parametrize_dims()
    def test_generalised_weyl_commutation_relation(self, dim: int) -> None:
        """Check ``Z^a X^b = w^(ab) X^b Z^a`` for all exponents."""
        shift = subspace_block(QuditXGate(dim))
        clock = subspace_block(QuditZGate(dim))
        root = omega(dim)

        for left in range(dim):
            clock_power = np.linalg.matrix_power(clock, left)
            for right in range(dim):
                shift_power = np.linalg.matrix_power(shift, right)
                assert_allclose(
                    clock_power @ shift_power,
                    root ** (left * right) * (shift_power @ clock_power),
                    message=(
                        f"Z^{left} X^{right} does not pick up the "
                        f"expected phase w^{left * right}"
                    ),
                )

    @parametrize_dims(BASIS_DIMS)
    def test_weyl_operators_form_an_orthogonal_operator_basis(
        self,
        dim: int,
    ) -> None:
        """Check the d^2 Weyl operators are pairwise orthogonal."""
        shift = subspace_block(QuditXGate(dim))
        clock = subspace_block(QuditZGate(dim))
        operators = [
            np.linalg.matrix_power(shift, left)
            @ np.linalg.matrix_power(clock, right)
            for left in range(dim)
            for right in range(dim)
        ]

        gram = np.array(
            [
                [np.trace(row.conj().T @ column) for column in operators]
                for row in operators
            ],
            dtype=np.complex128,
        )

        assert_allclose(
            gram,
            dim * np.eye(dim * dim, dtype=np.complex128),
            message=(
                "the d^2 Weyl operators are not Hilbert-Schmidt orthogonal, "
                "so X and Z do not generate a full operator basis"
            ),
        )

    @parametrize_dims()
    def test_xdg_is_the_d_minus_one_th_power_of_x(self, dim: int) -> None:
        """Check ``Xdg = X^(d-1)``, the inverse cyclic shift."""
        shift = subspace_block(QuditXGate(dim))

        assert_allclose(
            subspace_block(QuditXdgGate(dim)),
            np.linalg.matrix_power(shift, dim - 1),
            message="Xdg is not the (d-1)-th power of X",
        )


class TestPhaseFamily:
    """``P``, ``Z``, ``S``, ``T`` and their daggers."""

    @parametrize_dims()
    def test_s_squared_is_z(self, dim: int) -> None:
        """Check ``S S = Z``: S is the square root of the clock."""
        block = subspace_block(QuditSGate(dim))

        assert_allclose(
            block @ block,
            subspace_block(QuditZGate(dim)),
            message="S^2 is not Z",
        )

    @parametrize_dims()
    def test_t_squared_is_s(self, dim: int) -> None:
        """Check ``T T = S``: T is the fourth root of the clock."""
        block = subspace_block(QuditTGate(dim))

        assert_allclose(
            block @ block,
            subspace_block(QuditSGate(dim)),
            message="T^2 is not S",
        )

    @parametrize_dims()
    def test_t_to_the_fourth_is_z(self, dim: int) -> None:
        """Check ``T^4 = Z``, closing the phase-gate hierarchy."""
        block = subspace_block(QuditTGate(dim))

        assert_allclose(
            np.linalg.matrix_power(block, 4),
            subspace_block(QuditZGate(dim)),
            message="T^4 is not Z",
        )

    @parametrize_dims()
    def test_dagger_phase_gates_obey_the_same_hierarchy(
        self,
        dim: int,
    ) -> None:
        """Check Sdg^2 = Zdg, Tdg^2 = Sdg and Tdg^4 = Zdg."""
        sdg = subspace_block(QuditSdgGate(dim))
        tdg = subspace_block(QuditTdgGate(dim))
        zdg = subspace_block(QuditZdgGate(dim))

        assert_allclose(sdg @ sdg, zdg, message="Sdg^2 is not Zdg")
        assert_allclose(tdg @ tdg, sdg, message="Tdg^2 is not Sdg")
        assert_allclose(
            np.linalg.matrix_power(tdg, 4),
            zdg,
            message="Tdg^4 is not Zdg",
        )

    @pytest.mark.parametrize(
        ("gate_class", "theta"),
        [
            pytest.param(gate_class, theta, id=name)
            for name, gate_class, theta in NAMED_PHASE_ANGLES
        ],
    )
    @parametrize_dims()
    def test_named_phase_gates_are_p_at_special_angles(
        self,
        gate_class: Callable[[int], QuditGate],
        theta: float,
        dim: int,
    ) -> None:
        """Check Z, S, T (and daggers) are ``P`` at pi, pi/2, pi/4."""
        assert_allclose(
            gate_matrix(gate_class(dim)),
            gate_matrix(QuditPGate(dim, theta)),
            message=f"the gate does not equal P({theta})",
        )

    @parametrize_dims()
    def test_phase_angles_add(self, dim: int) -> None:
        """Check ``P(a) P(b) = P(a + b)``: phases form a group."""
        first, second = 0.31, -1.27

        assert_allclose(
            gate_matrix(QuditPGate(dim, first))
            @ gate_matrix(QuditPGate(dim, second)),
            gate_matrix(QuditPGate(dim, first + second)),
            message="phase gates do not add their angles",
        )

    @parametrize_dims()
    def test_phase_gate_is_periodic_in_d_pi(self, dim: int) -> None:
        """Check ``P(theta + d pi) = P(theta)``: w^k has period d."""
        assert_allclose(
            gate_matrix(QuditPGate(dim, THETA + dim * np.pi)),
            gate_matrix(QuditPGate(dim, THETA)),
            message="the phase gate is not 2*pi periodic in w",
        )

    @parametrize_dims()
    def test_zero_angle_phase_gate_is_the_identity(self, dim: int) -> None:
        """Check ``P(0)`` is exactly the identity gate."""
        assert_allclose(
            gate_matrix(QuditPGate(dim, 0.0)),
            gate_matrix(QuditIGate(dim)),
            message="P(0) is not the identity",
        )


class TestFourierGate:
    """``H`` as the discrete Fourier transform of the qudit."""

    @parametrize_dims()
    def test_h_conjugates_the_clock_into_the_shift(self, dim: int) -> None:
        """Check ``H Z Hdg = X``: H maps the clock onto the shift."""
        fourier = subspace_block(QuditHGate(dim))
        fourier_dagger = subspace_block(QuditHdgGate(dim))

        assert_allclose(
            fourier @ subspace_block(QuditZGate(dim)) @ fourier_dagger,
            subspace_block(QuditXGate(dim)),
            message="H does not conjugate Z into X",
        )

    @parametrize_dims()
    def test_h_conjugates_the_shift_into_the_inverse_clock(
        self,
        dim: int,
    ) -> None:
        """Check H X Hdg = Zdg, and not Z, for d > 2."""
        fourier = subspace_block(QuditHGate(dim))
        fourier_dagger = subspace_block(QuditHdgGate(dim))

        assert_allclose(
            fourier @ subspace_block(QuditXGate(dim)) @ fourier_dagger,
            subspace_block(QuditZdgGate(dim)),
            message="H does not conjugate X into Z^dagger",
        )

    @parametrize_dims()
    def test_h_squared_is_the_parity_gate(self, dim: int) -> None:
        """Check ``H^2 = K``: two DFTs reverse the levels."""
        fourier = subspace_block(QuditHGate(dim))

        assert_allclose(
            fourier @ fourier,
            subspace_block(QuditKGate(dim)),
            message="H^2 is not the parity (level-reversal) gate K",
        )

    @parametrize_dims()
    def test_h_has_order_four(self, dim: int) -> None:
        """Check ``H^4 = 1``, as for any discrete Fourier transform."""
        fourier = subspace_block(QuditHGate(dim))

        assert_allclose(
            np.linalg.matrix_power(fourier, 4),
            _identity(dim),
            message="H^4 is not the identity",
        )


class TestInvolutions:
    """The self-inverse permutations ``NOT`` and ``K``."""

    @parametrize_dims()
    def test_complement_is_an_involution(self, dim: int) -> None:
        """Check ``NOT^2 = 1``: complementing twice is trivial."""
        complement = subspace_block(QuditNOTGate(dim))

        assert_allclose(
            complement @ complement,
            _identity(dim),
            message="NOT is not self-inverse",
        )

    @parametrize_dims()
    def test_parity_is_an_involution(self, dim: int) -> None:
        """Check ``K^2 = 1``: negating the level twice is trivial."""
        parity = subspace_block(QuditKGate(dim))

        assert_allclose(
            parity @ parity,
            _identity(dim),
            message="K is not self-inverse",
        )

    @parametrize_dims()
    def test_parity_is_the_shift_after_the_complement(
        self,
        dim: int,
    ) -> None:
        """Check ``K = X NOT``: complement first, then shift up."""
        assert_allclose(
            subspace_block(QuditKGate(dim)),
            subspace_block(QuditXGate(dim))
            @ subspace_block(QuditNOTGate(dim)),
            message="K is not X applied after NOT",
        )

    @parametrize_dims()
    def test_shift_is_the_parity_after_the_complement(
        self,
        dim: int,
    ) -> None:
        """Check ``X = K NOT``, the mirror of ``K = X NOT``."""
        assert_allclose(
            subspace_block(QuditXGate(dim)),
            subspace_block(QuditKGate(dim))
            @ subspace_block(QuditNOTGate(dim)),
            message="X is not K applied after NOT",
        )


class TestInverses:
    """``inverse()`` must reproduce the matrix adjoint."""

    @parametrize_gates
    @parametrize_dims()
    def test_inverse_matrix_is_the_conjugate_transpose(
        self,
        make_gate: Callable[[int], QuditGate],
        dim: int,
    ) -> None:
        """Check ``inverse()`` returns exactly ``U^dagger``."""
        gate = make_gate(dim)

        assert_allclose(
            gate_matrix(gate.inverse()),
            gate_matrix(gate).conj().T,
            message="the inverse gate is not the conjugate transpose",
        )

    @parametrize_gates
    @parametrize_dims()
    def test_gate_composed_with_its_inverse_is_the_identity(
        self,
        make_gate: Callable[[int], QuditGate],
        dim: int,
    ) -> None:
        """Check ``U U^-1 = U^-1 U = 1`` on the whole encoded space."""
        gate = make_gate(dim)
        matrix = gate_matrix(gate)
        inverse = gate_matrix(gate.inverse())
        identity = np.eye(gate.hilbert_dim, dtype=np.complex128)

        assert_allclose(
            matrix @ inverse,
            identity,
            message="U U^-1 is not the identity",
        )
        assert_allclose(
            inverse @ matrix,
            identity,
            message="U^-1 U is not the identity",
        )

    @pytest.mark.parametrize(
        ("gate_class", "dagger_class"),
        [
            pytest.param(gate_class, dagger_class, id=name)
            for name, gate_class, dagger_class in DAGGER_PAIRS
        ],
    )
    @parametrize_dims()
    def test_dagger_partner_is_the_matrix_adjoint(
        self,
        gate_class: Callable[[int], QuditGate],
        dagger_class: Callable[[int], QuditGate],
        dim: int,
    ) -> None:
        """Check each ``...dg`` class is the adjoint of its partner."""
        matrix = gate_matrix(gate_class(dim))

        assert_allclose(
            gate_matrix(dagger_class(dim)),
            matrix.conj().T,
            message="the dagger class is not the adjoint of its partner",
        )
