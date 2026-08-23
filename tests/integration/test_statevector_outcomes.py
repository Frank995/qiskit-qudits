"""Exact amplitudes obtained through the public circuit API.

Nothing is sampled here. Every test builds a measurement-free
:class:`~qiskit_qudits.circuit.quantumcircuit.QuditQuantumCircuit`,
reads its logical state-vector and compares it against a reference
assembled by hand from :func:`tests.helpers.basis_state` and
:func:`numpy.kron`.

Qudit ``0`` is the least significant factor of the logical space, so
a multi-qudit reference is built as
``kron(last_qudit, ..., first_qudit)``.
"""

from __future__ import annotations

import numpy as np
import pytest

from qiskit_qudits.circuit.quantumcircuit import QuditQuantumCircuit
from qiskit_qudits.circuit.qudit import QuditRegister
from tests.helpers import (
    ATOL,
    assert_allclose,
    basis_state,
    encoded_statevector,
    logical_statevector,
)

#: Dimensions used by the (cheap) exact-amplitude tests.
STATEVECTOR_DIMS = (2, 3, 4, 5, 7, 8)

#: Ids matching :data:`STATEVECTOR_DIMS`.
STATEVECTOR_IDS = [f"d{dim}" for dim in STATEVECTOR_DIMS]

#: Dimensions that leave invalid encoded basis states behind.
LEAKY_DIMS = (3, 5, 7)

#: An angle that is not a multiple of pi, so the interference
#: pattern it produces is genuinely non-uniform.
GENERIC_THETA = 0.6


def _single_qudit(dim: int) -> QuditQuantumCircuit:
    """Return a fresh, unmeasured one-qudit circuit."""
    return QuditQuantumCircuit(1, dim=dim)


def _interference_amplitudes(
    dim: int,
    theta: float,
) -> np.typing.NDArray[np.complex128]:
    r"""Return the interference amplitudes, derived by hand.

    The prepared state is
    :math:`H^\dagger P(\theta) H \lvert 0 \rangle`, and the
    reference below is built from the three factors alone, so it is
    independent of the library's own matrices:

    * :math:`H[k, 0] = 1 / \sqrt{d}`;
    * :math:`P(\theta)[k, k] = e^{2 i k \theta / d}`;
    * :math:`H^\dagger[j, k] = e^{2 \pi i j k / d} / \sqrt{d}`.

    Args:
        dim: Number of levels of the qudit.
        theta: Phase-gate angle in radians.

    Returns:
        The ``dim`` logical amplitudes.
    """
    levels = np.arange(dim)
    inverse_dft = np.exp(2j * np.pi * np.outer(levels, levels) / dim)
    diagonal = np.exp(2j * levels * theta / dim)
    return (inverse_dft @ diagonal) / dim


class TestInitializeLevels:
    """Where each level specification lands its qudits."""

    def test_string_form_is_read_right_to_left(self) -> None:
        """The rightmost token belongs to the first target qudit."""
        circuit = QuditQuantumCircuit(3, dim=3)
        circuit.initialize_levels("2 0 1")

        assert_allclose(
            logical_statevector(circuit),
            basis_state((3, 3, 3), (1, 0, 2)),
            message="'2 0 1' must mean qudit 0 -> |1>, 2 -> |2>",
        )

    def test_string_form_matches_a_kron_reference(self) -> None:
        """The kron order is last qudit first, qudit 0 last."""
        circuit = QuditQuantumCircuit(3, dim=3)
        circuit.initialize_levels("2 0 1")

        ket = np.eye(3, dtype=np.complex128)
        # kron(qudit 2, qudit 1, qudit 0)
        expected = np.kron(np.kron(ket[2], ket[0]), ket[1])

        assert_allclose(logical_statevector(circuit), expected)

    def test_sequence_form_is_read_in_target_order(self) -> None:
        """A sequence is *not* reversed: entry i is qudit i."""
        circuit = QuditQuantumCircuit(3, dim=3)
        circuit.initialize_levels([1, 0, 2])

        assert_allclose(
            logical_statevector(circuit),
            basis_state((3, 3, 3), (1, 0, 2)),
        )

    def test_string_and_sequence_forms_agree(self) -> None:
        """The two spellings prepare the very same state."""
        from_string = QuditQuantumCircuit(3, dim=3)
        from_string.initialize_levels("2 0 1")
        from_sequence = QuditQuantumCircuit(3, dim=3)
        from_sequence.initialize_levels([1, 0, 2])

        assert_allclose(
            logical_statevector(from_string),
            logical_statevector(from_sequence),
        )

    def test_integer_form_prepares_a_single_qudit(self) -> None:
        """A bare integer level targets one qudit."""
        circuit = _single_qudit(5)
        circuit.initialize_levels(3)

        assert_allclose(
            logical_statevector(circuit),
            basis_state((5,), (3,)),
        )

    def test_integer_form_leaves_the_other_qudits_alone(self) -> None:
        """Only the addressed qudit moves off level zero."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.initialize_levels(2, 1)

        assert_allclose(
            logical_statevector(circuit),
            basis_state((3, 3), (0, 2)),
        )


class TestShiftGate:
    """Cyclic behaviour of the qudit shift gate."""

    @pytest.mark.parametrize(
        "dim",
        STATEVECTOR_DIMS,
        ids=STATEVECTOR_IDS,
    )
    def test_x_shifts_the_level_cyclically(self, dim: int) -> None:
        """After s shifts the qudit sits on level s mod d."""
        for steps in range(dim + 2):
            circuit = _single_qudit(dim)
            for _ in range(steps):
                circuit.x(0)

            assert_allclose(
                logical_statevector(circuit),
                basis_state((dim,), (steps % dim,)),
                message=f"{steps} shift(s) on a {dim}-level qudit",
            )

    @pytest.mark.parametrize(
        "dim",
        STATEVECTOR_DIMS,
        ids=STATEVECTOR_IDS,
    )
    def test_x_applied_d_times_is_the_identity(self, dim: int) -> None:
        """Applying d shifts brings the qudit back to the start."""
        circuit = _single_qudit(dim)
        circuit.x(0)  # start away from |0> so the test can fail
        for _ in range(dim):
            circuit.x(0)

        assert_allclose(
            logical_statevector(circuit),
            basis_state((dim,), (1,)),
        )

    @pytest.mark.parametrize(
        "dim",
        STATEVECTOR_DIMS,
        ids=STATEVECTOR_IDS,
    )
    def test_xdg_undoes_x(self, dim: int) -> None:
        """The inverse shift walks the level back down."""
        circuit = _single_qudit(dim)
        circuit.x(0)
        circuit.x(0)
        circuit.xdg(0)

        assert_allclose(
            logical_statevector(circuit),
            basis_state((dim,), (1,)),
        )


class TestHadamard:
    """Superposition and phases produced by the qudit Hadamard."""

    @pytest.mark.parametrize(
        "dim",
        STATEVECTOR_DIMS,
        ids=STATEVECTOR_IDS,
    )
    def test_h_on_zero_is_the_uniform_superposition(
        self,
        dim: int,
    ) -> None:
        """H|0> puts 1/sqrt(d) on every level, with no phase."""
        circuit = _single_qudit(dim)
        circuit.h(0)

        expected = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        assert_allclose(logical_statevector(circuit), expected)

    @pytest.mark.parametrize(
        "dim",
        STATEVECTOR_DIMS,
        ids=STATEVECTOR_IDS,
    )
    def test_h_on_level_one_carries_the_documented_phases(
        self,
        dim: int,
    ) -> None:
        """H|1> is the uniform state with phases omega**(-j)."""
        circuit = _single_qudit(dim)
        circuit.x(0)  # prepare |1> unitarily
        circuit.h(0)

        levels = np.arange(dim)
        expected = np.exp(-2j * np.pi * levels / dim) / np.sqrt(dim)
        assert_allclose(logical_statevector(circuit), expected)

    @pytest.mark.parametrize(
        "dim",
        STATEVECTOR_DIMS,
        ids=STATEVECTOR_IDS,
    )
    def test_hdg_undoes_h_exactly(self, dim: int) -> None:
        """H then Hdg is the identity on |0>."""
        circuit = _single_qudit(dim)
        circuit.h(0)
        circuit.hdg(0)

        assert_allclose(
            logical_statevector(circuit),
            basis_state((dim,), (0,)),
        )


class TestInterference:
    """Phases inserted between H and Hdg interfere predictably."""

    @pytest.mark.parametrize(
        "dim",
        STATEVECTOR_DIMS,
        ids=STATEVECTOR_IDS,
    )
    def test_a_pi_phase_shifts_the_level_down_by_one(
        self,
        dim: int,
    ) -> None:
        """Hdg P(pi) H |0> is exactly |d - 1>."""
        circuit = _single_qudit(dim)
        circuit.h(0)
        circuit.p(np.pi, 0)
        circuit.hdg(0)

        assert_allclose(
            logical_statevector(circuit),
            basis_state((dim,), (dim - 1,)),
        )

    @pytest.mark.parametrize(
        "dim",
        STATEVECTOR_DIMS,
        ids=STATEVECTOR_IDS,
    )
    def test_z_between_h_and_hdg_behaves_like_p_of_pi(
        self,
        dim: int,
    ) -> None:
        """Z is P(pi), so Hdg Z H |0> also lands on |d - 1>."""
        circuit = _single_qudit(dim)
        circuit.h(0)
        circuit.z(0)
        circuit.hdg(0)

        assert_allclose(
            logical_statevector(circuit),
            basis_state((dim,), (dim - 1,)),
        )

    @pytest.mark.parametrize(
        "dim",
        STATEVECTOR_DIMS,
        ids=STATEVECTOR_IDS,
    )
    def test_a_generic_phase_matches_the_hand_derived_sum(
        self,
        dim: int,
    ) -> None:
        """The amplitudes are the geometric sum computed by hand."""
        circuit = _single_qudit(dim)
        circuit.h(0)
        circuit.p(GENERIC_THETA, 0)
        circuit.hdg(0)

        expected = _interference_amplitudes(dim, GENERIC_THETA)
        assert_allclose(logical_statevector(circuit), expected)

    def test_the_hand_derived_reference_is_normalised(self) -> None:
        """The analytic reference is itself a legal state."""
        for dim in STATEVECTOR_DIMS:
            amplitudes = _interference_amplitudes(dim, GENERIC_THETA)
            norm = float(np.linalg.norm(amplitudes))
            assert abs(norm - 1.0) < ATOL, f"d={dim} norm {norm}"


class TestProductStates:
    """Independent qudits must stay in a product state."""

    def test_two_qutrits_factorise(self) -> None:
        """The joint amplitudes are the kron of the marginals."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.h(0)
        circuit.x(1)
        circuit.h(1)

        levels = np.arange(3)
        qudit0 = np.ones(3, dtype=np.complex128) / np.sqrt(3)
        qudit1 = np.exp(-2j * np.pi * levels / 3) / np.sqrt(3)
        # qudit 0 is least significant, hence rightmost in the kron.
        expected = np.kron(qudit1, qudit0)

        assert_allclose(logical_statevector(circuit), expected)

    def test_mixed_dimensions_factorise(self) -> None:
        """A qubit and a qutrit keep their own local states."""
        register = QuditRegister.from_dims([2, 3], "mix")
        circuit = QuditQuantumCircuit(register)
        circuit.h(0)
        circuit.x(1)

        qudit0 = np.ones(2, dtype=np.complex128) / np.sqrt(2)
        qudit1 = np.array([0.0, 1.0, 0.0], dtype=np.complex128)
        expected = np.kron(qudit1, qudit0)

        state = logical_statevector(circuit)
        assert state.size == 6
        assert_allclose(state, expected)

    def test_a_local_gate_leaves_the_other_qudit_untouched(self) -> None:
        """Acting on qudit 1 does not disturb qudit 0."""
        circuit = QuditQuantumCircuit(2, dim=4)
        circuit.x(0)
        circuit.h(1)
        circuit.z(1)
        circuit.hdg(1)

        # Hdg Z H |0> = |d - 1> = |3> on qudit 1, qudit 0 stays |1>.
        assert_allclose(
            logical_statevector(circuit),
            basis_state((4, 4), (1, 3)),
        )


class TestSubspaceIsNeverLeft:
    """Non-power-of-two encodings must carry no invalid amplitude."""

    @pytest.mark.parametrize(
        "dim",
        LEAKY_DIMS,
        ids=[f"d{dim}" for dim in LEAKY_DIMS],
    )
    def test_one_qudit_keeps_the_padding_amplitudes_at_zero(
        self,
        dim: int,
    ) -> None:
        """Encoded indices d..2**n - 1 hold exactly zero."""
        circuit = _single_qudit(dim)
        circuit.h(0)
        circuit.p(GENERIC_THETA, 0)
        circuit.x(0)

        encoded = np.asarray(
            encoded_statevector(circuit).data,
            dtype=np.complex128,
        )
        padding = encoded[dim:]

        assert padding.size == encoded.size - dim
        assert_allclose(
            padding,
            np.zeros_like(padding),
            message="amplitude leaked outside the qudit subspace",
        )

    def test_two_qutrits_keep_every_invalid_pattern_at_zero(self) -> None:
        """Neither two-qubit block may ever hold the pattern '11'."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.h(0)
        circuit.x(1)
        circuit.h(1)

        encoded = np.asarray(
            encoded_statevector(circuit).data,
            dtype=np.complex128,
        )
        assert encoded.size == 16

        # Qudit 0 uses qubits 0-1, qudit 1 uses qubits 2-3; the
        # value 3 is outside a 3-level qudit.
        invalid = [
            index
            for index in range(16)
            if (index & 0b11) == 3 or ((index >> 2) & 0b11) == 3
        ]
        assert len(invalid) == 7

        assert_allclose(
            encoded[invalid],
            np.zeros(len(invalid), dtype=np.complex128),
            message="the two-qutrit encoding leaked",
        )

    def test_the_valid_subspace_carries_the_whole_norm(self) -> None:
        """Projecting onto the qudit space loses no probability."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.h(0)
        circuit.h(1)

        logical = logical_statevector(circuit)
        norm = float(np.linalg.norm(logical))

        assert logical.size == 9
        assert abs(norm - 1.0) < ATOL
