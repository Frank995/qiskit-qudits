"""Sampled statistics must agree with quantum theory.

Two independent paths through the library are cross-checked here:

* the **state-vector** path, ``logical_statevector`` of a
  measurement-free circuit, which uses each gate's matrix directly;
* the **sampling** path, which transpiles the gates' qubit-level
  decompositions onto a simulator and draws shots.

Whenever a theoretical probability is awkward to spell out, it is
taken from the first path and compared against the second; the rest
of the time the reference is written out by hand.

Simulator note: the reference backend can sample every shot from a
single state-vector pass only when the circuit contains no ``reset``,
so circuits that use ``initialize_levels`` are run with fewer shots.
"""

from __future__ import annotations

import numpy as np
import pytest

from qiskit_qudits.circuit.quantumcircuit import QuditQuantumCircuit
from qiskit_qudits.circuit.qudit import QuditRegister
from tests.helpers import (
    assert_deterministic_outcome,
    assert_distribution_close,
    logical_statevector,
    run_counts,
    sample_levels,
)

#: Dimensions exercised by the keystone comparison. Kept at d <= 5
#: with two qudits so the pure-Python simulator stays quick.
KEYSTONE_DIMS = (2, 3, 4, 5)

#: Ids matching :data:`KEYSTONE_DIMS`.
KEYSTONE_IDS = [f"d{dim}" for dim in KEYSTONE_DIMS]

#: Shots used when the circuit contains an ``initialize`` (hence a
#: reset), which forces one full simulation per shot.
RESET_SHOTS = 512

#: Tolerance on a sampled frequency, deliberately loose.
FREQUENCY_ATOL = 0.05

#: Probabilities below this are numerical dust, not physics.
NEGLIGIBLE = 1e-12

#: An angle that is not a multiple of pi, so the interference it
#: produces is genuinely non-uniform.
GENERIC_THETA = 0.6


def _levels_of(dims: tuple[int, ...], index: int) -> tuple[int, ...]:
    """Split a logical index into one level per qudit.

    Args:
        dims: Dimension of each qudit, qudit 0 first.
        index: Index into the ``prod(dims)`` dimensional space.

    Returns:
        The levels, qudit 0 first (qudit 0 is least significant).
    """
    levels: list[int] = []
    remainder = index
    for dim in dims:
        levels.append(remainder % dim)
        remainder //= dim
    return tuple(levels)


def _theoretical_distribution(
    circuit: QuditQuantumCircuit,
) -> dict[tuple[int, ...], float]:
    """Return |amplitude|**2 for every outcome of a pure circuit.

    Args:
        circuit: A measurement-free qudit circuit.

    Returns:
        Level tuples mapped to their Born probability. Outcomes whose
        probability is numerical dust are omitted, so that
        ``assert_distribution_close`` treats them as impossible.
    """
    amplitudes = logical_statevector(circuit)
    dims = circuit.dims
    distribution: dict[tuple[int, ...], float] = {}
    for index, amplitude in enumerate(amplitudes):
        probability = float(abs(amplitude) ** 2)
        if probability > NEGLIGIBLE:
            distribution[_levels_of(dims, index)] = probability
    return distribution


def _interference_probabilities(
    dim: int,
    theta: float,
) -> np.typing.NDArray[np.float64]:
    r"""Return the Born law of the interference sandwich.

    The state is :math:`H^\dagger P(\theta) H \lvert 0 \rangle`.
    Written out by hand from the three factors
    :math:`H[k, 0] = 1/\sqrt{d}`,
    :math:`P(\theta)[k, k] = e^{2 i k \theta / d}` and
    :math:`H^\dagger[j, k] = e^{2 \pi i j k / d} / \sqrt{d}`, so this
    reference never touches the library.

    Args:
        dim: Number of levels of the qudit.
        theta: Phase-gate angle in radians.

    Returns:
        One probability per level.
    """
    levels = np.arange(dim)
    inverse_dft = np.exp(2j * np.pi * np.outer(levels, levels) / dim)
    diagonal = np.exp(2j * levels * theta / dim)
    amplitudes = (inverse_dft @ diagonal) / dim
    return np.abs(amplitudes) ** 2


def _nontrivial_circuit(dim: int) -> QuditQuantumCircuit:
    """Build an unmeasured two-qudit circuit with a lopsided state.

    Qudit 0 goes through an interference sandwich (non-uniform
    marginal) while qudit 1 ends up uniformly spread, so the joint
    distribution has both large and small probabilities.

    Args:
        dim: Number of levels of both qudits.

    Returns:
        The measurement-free circuit.
    """
    circuit = QuditQuantumCircuit(2, dim=dim)
    circuit.h(0)
    circuit.p(GENERIC_THETA, 0)
    circuit.hdg(0)
    circuit.x(1)
    circuit.h(1)
    return circuit


def _bookkeeping_circuit() -> QuditQuantumCircuit:
    """Build a measured, three-qudit, mixed-dimension circuit."""
    register = QuditRegister.from_dims([2, 3, 4], "mix")
    circuit = QuditQuantumCircuit(register)
    circuit.h(0)
    circuit.x(1)
    circuit.h(2)
    circuit.measure_all()
    return circuit


class TestSampledFrequenciesMatchTheState:
    """The keystone: sampling agrees with the state-vector."""

    @pytest.mark.parametrize("dim", KEYSTONE_DIMS, ids=KEYSTONE_IDS)
    def test_frequencies_match_the_unmeasured_state(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """Sampled outcomes follow |psi|**2 of the same circuit."""
        base = _nontrivial_circuit(dim)
        expected = _theoretical_distribution(base)

        measured = base.copy()
        measured.measure_all()
        observed = sample_levels(measured, shots=shots, seed=seed)

        assert_distribution_close(
            observed,
            expected,
            shots=shots,
            atol=FREQUENCY_ATOL,
        )

    @pytest.mark.parametrize("dim", KEYSTONE_DIMS, ids=KEYSTONE_IDS)
    def test_the_theoretical_distribution_is_normalised(
        self,
        dim: int,
    ) -> None:
        """The reference itself is a legal probability vector."""
        expected = _theoretical_distribution(_nontrivial_circuit(dim))

        assert len(expected) <= dim * dim
        assert abs(sum(expected.values()) - 1.0) < 1e-9

    def test_the_state_is_genuinely_non_uniform(self) -> None:
        """The keystone circuit is not a trivial flat distribution."""
        expected = _theoretical_distribution(_nontrivial_circuit(3))

        assert max(expected.values()) > 2 * min(expected.values())

    @pytest.mark.slow
    @pytest.mark.parametrize("dim", KEYSTONE_DIMS, ids=KEYSTONE_IDS)
    def test_adding_measurements_does_not_change_the_marginals(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """Qudit 0's marginal survives measuring both qudits."""
        base = _nontrivial_circuit(dim)
        joint = _theoretical_distribution(base)
        expected_marginal = [0.0] * dim
        for levels, probability in joint.items():
            expected_marginal[levels[0]] += probability

        measured = base.copy()
        measured.measure_all()
        observed = sample_levels(measured, shots=shots, seed=seed)

        sampled_marginal = [0.0] * dim
        for levels, count in observed.items():
            sampled_marginal[levels[0]] += count / shots

        for level, (sampled, predicted) in enumerate(
            zip(sampled_marginal, expected_marginal, strict=True),
        ):
            assert (
                abs(sampled - predicted) <= FREQUENCY_ATOL
            ), f"level {level}: {sampled:.4f} vs {predicted:.4f}"


class TestUniformSuperposition:
    """A Hadamard spreads a qudit over exactly d levels."""

    @pytest.mark.parametrize("dim", KEYSTONE_DIMS, ids=KEYSTONE_IDS)
    def test_h_yields_a_uniform_distribution(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """Each of the d levels gets a fraction 1/d of the shots."""
        circuit = QuditQuantumCircuit(1, dim=dim)
        circuit.h(0)
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        expected = {(level,): 1 / dim for level in range(dim)}
        assert_distribution_close(
            observed,
            expected,
            shots=shots,
            atol=FREQUENCY_ATOL,
        )

    @pytest.mark.parametrize("dim", [3, 5], ids=["d3", "d5"])
    def test_h_never_produces_an_invalid_level(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """Padding states of a leaky encoding are never sampled."""
        circuit = QuditQuantumCircuit(1, dim=dim)
        circuit.h(0)
        circuit.measure_all()

        observed = sample_levels(
            circuit,
            shots=shots,
            seed=seed,
            on_invalid="raise",
        )

        assert set(observed) == {(level,) for level in range(dim)}


class TestDeterministicCircuits:
    """Circuits whose outcome theory pins to a single value."""

    def test_initialize_levels_puts_every_shot_on_one_outcome(
        self,
        seed: int,
    ) -> None:
        """The string '2 1' means qudit 0 on |1>, qudit 1 on |2>."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.initialize_levels("2 1")
        circuit.measure_all()

        observed = sample_levels(circuit, shots=RESET_SHOTS, seed=seed)

        assert_deterministic_outcome(
            observed,
            (1, 2),
            shots=RESET_SHOTS,
        )

    @pytest.mark.parametrize(
        ("dim", "steps"),
        [(2, 3), (3, 4), (4, 6), (5, 7)],
        ids=["d2x3", "d3x4", "d4x6", "d5x7"],
    )
    def test_a_chain_of_x_gates_lands_on_one_level(
        self,
        dim: int,
        steps: int,
        shots: int,
        seed: int,
    ) -> None:
        """Applying s shifts from |0> measures level s mod d."""
        circuit = QuditQuantumCircuit(1, dim=dim)
        for _ in range(steps):
            circuit.x(0)
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        assert_deterministic_outcome(
            observed,
            (steps % dim,),
            shots=shots,
        )

    @pytest.mark.parametrize("dim", [3, 5], ids=["d3", "d5"])
    def test_not_maps_a_level_to_its_complement(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """NOT sends level 1 to level d - 2, every single shot."""
        circuit = QuditQuantumCircuit(1, dim=dim)
        circuit.x(0)
        circuit.not_(0)
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        assert_deterministic_outcome(observed, (dim - 2,), shots=shots)

    @pytest.mark.parametrize("dim", [3, 5], ids=["d3", "d5"])
    def test_k_reflects_a_level_around_zero(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """K sends level 2 to level (d - 2) mod d, every shot."""
        circuit = QuditQuantumCircuit(1, dim=dim)
        circuit.x(0)
        circuit.x(0)
        circuit.k(0)
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        assert_deterministic_outcome(
            observed,
            ((dim - 2) % dim,),
            shots=shots,
        )

    @pytest.mark.parametrize("dim", [3, 5], ids=["d3", "d5"])
    def test_k_fixes_the_ground_level(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """K leaves level 0 alone, since (d - 0) mod d is 0."""
        circuit = QuditQuantumCircuit(1, dim=dim)
        circuit.k(0)
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        assert_deterministic_outcome(observed, (0,), shots=shots)


class TestProductDistributions:
    """Independent qudits give a product of their marginals."""

    def test_two_independent_qudits_multiply(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """The joint law is the outer product of two marginals."""
        dim = 3
        circuit = QuditQuantumCircuit(2, dim=dim)
        circuit.h(0)
        circuit.h(1)
        circuit.p(GENERIC_THETA, 1)
        circuit.hdg(1)
        circuit.measure_all()

        marginal = _interference_probabilities(dim, GENERIC_THETA)
        expected = {
            (level0, level1): float(marginal[level1]) / dim
            for level0 in range(dim)
            for level1 in range(dim)
        }

        observed = sample_levels(circuit, shots=shots, seed=seed)

        assert_distribution_close(
            observed,
            expected,
            shots=shots,
            atol=FREQUENCY_ATOL,
        )

    def test_the_hand_derived_marginal_is_normalised(self) -> None:
        """The analytic marginal is a probability distribution."""
        for dim in KEYSTONE_DIMS:
            marginal = _interference_probabilities(dim, GENERIC_THETA)
            assert abs(float(marginal.sum()) - 1.0) < 1e-9
            assert float(marginal.min()) >= 0.0

    def test_a_shifted_qudit_does_not_disturb_its_neighbour(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """Qudit 1 stays sharp while qudit 0 stays uniform."""
        circuit = QuditQuantumCircuit(2, dim=4)
        circuit.h(0)
        circuit.x(1)
        circuit.x(1)
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        expected = {(level, 2): 0.25 for level in range(4)}
        assert_distribution_close(
            observed,
            expected,
            shots=shots,
            atol=FREQUENCY_ATOL,
        )


class TestMeasurementIsBasisFaithful:
    """Measuring in the computational basis reads the level back."""

    @pytest.mark.parametrize("dim", KEYSTONE_DIMS, ids=KEYSTONE_IDS)
    def test_h_then_hdg_returns_the_original_level(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """A round trip through the Fourier basis changes nothing."""
        circuit = QuditQuantumCircuit(1, dim=dim)
        circuit.x(0)  # prepare |1> unitarily
        circuit.h(0)
        circuit.hdg(0)
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        assert_deterministic_outcome(observed, (1,), shots=shots)

    @pytest.mark.parametrize("dim", KEYSTONE_DIMS, ids=KEYSTONE_IDS)
    def test_a_diagonal_gate_is_invisible_to_the_measurement(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """Phases cannot change computational-basis statistics."""
        circuit = QuditQuantumCircuit(1, dim=dim)
        circuit.h(0)
        circuit.z(0)
        circuit.p(GENERIC_THETA, 0)
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        expected = {(level,): 1 / dim for level in range(dim)}
        assert_distribution_close(
            observed,
            expected,
            shots=shots,
            atol=FREQUENCY_ATOL,
        )


class TestShotBookkeeping:
    """Sanity on the shot and cldigit accounting."""

    def test_decoded_counts_sum_to_the_requested_shots(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """Decoding aggregates, it never drops or invents shots."""
        circuit = _bookkeeping_circuit()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        assert sum(observed.values()) == shots

    def test_raw_and_decoded_totals_agree(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """The bit-string counts and the level counts hold as much."""
        circuit = _bookkeeping_circuit()

        raw = run_counts(circuit, shots=shots, seed=seed)
        decoded = circuit.decode_counts(raw)

        assert sum(raw.values()) == shots
        assert sum(decoded.values()) == shots
        assert len(decoded) <= len(raw)

    def test_every_decoded_key_has_one_entry_per_cldigit(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """A decoded key is one level per cldigit, in cldigit order."""
        circuit = _bookkeeping_circuit()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        assert circuit.num_cldigits == 3
        assert circuit.cldigit_dims == (2, 3, 4)
        assert circuit.cldigit_widths == (1, 2, 2)
        for key in observed:
            assert len(key) == circuit.num_cldigits
            for level, dim in zip(key, circuit.cldigit_dims, strict=True):
                assert 0 <= level < dim, f"{key} escapes {circuit.dims}"

    def test_the_expected_joint_outcomes_all_show_up(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """Qudits 0 and 2 spread evenly while qudit 1 stays sharp."""
        circuit = _bookkeeping_circuit()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        expected = {
            (level0, 1, level2): 1 / 8
            for level0 in range(2)
            for level2 in range(4)
        }
        assert_distribution_close(
            observed,
            expected,
            shots=shots,
            atol=FREQUENCY_ATOL,
        )
