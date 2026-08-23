"""End-to-end user journeys through :mod:`qiskit_qudits`.

Every test here plays the part of somebody who wants to *simulate
qudits*: build a circuit with the public
:class:`~qiskit_qudits.circuit.quantumcircuit.QuditQuantumCircuit`
API, run it on a simulator, decode the raw bit-strings back into
levels and check the outcome against what quantum theory predicts.

Two practical notes on speed:

* the reference simulator can sample all shots from a single
  state-vector pass only when the circuit contains no ``reset``;
  ``initialize``/``initialize_levels`` emit one, so those circuits
  are run with fewer shots;
* deterministic outcomes need only a handful of shots to be
  convincing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.circuit.quantumcircuit import QuditQuantumCircuit
from qiskit_qudits.circuit.qudit import QuditRegister
from qiskit_qudits.utils.encoding import format_levels
from tests.helpers import (
    assert_allclose,
    assert_deterministic_outcome,
    assert_distribution_close,
    logical_statevector,
    run_counts,
    sample_levels,
)

if TYPE_CHECKING:
    from qiskit_qudits.circuit.quantumcircuit import CircuitView

#: Shots used when the circuit contains an ``initialize`` (hence a
#: reset), which forces one full simulation per shot.
RESET_SHOTS = 2048

#: Shots used for structural checks and for outcomes that theory
#: pins to a single value; a handful is already convincing.
SHORT_RUN_SHOTS = 256

#: Tolerance on a sampled frequency. Deliberately loose: the point
#: of these tests is the physics, not the last decimal.
FREQUENCY_ATOL = 0.05


def _readme_circuit() -> QuditQuantumCircuit:
    """Build the two-qutrit circuit advertised in the docstring."""
    circuit = QuditQuantumCircuit(2, 2, dim=3)
    circuit.initialize_levels("2 0")
    circuit.h(0)
    circuit.x(1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def _mixed_register() -> QuditRegister:
    """Return a qubit, a qutrit and a five-level qudit."""
    return QuditRegister.from_dims([2, 3, 5], "mix")


def _interop_circuit() -> QuditQuantumCircuit:
    """Build a small measured circuit used for the Qiskit hand-off."""
    circuit = QuditQuantumCircuit(2, dim=3)
    circuit.h(0)
    circuit.x(1)
    circuit.measure_all()
    return circuit


class TestReadmeJourney:
    """The advertised build / run / decode flow."""

    def test_decoded_levels_match_theory(self, seed: int) -> None:
        """H spreads qudit 0 evenly while X moves qudit 1 to |0>."""
        circuit = _readme_circuit()

        observed = sample_levels(
            circuit,
            shots=RESET_SHOTS,
            seed=seed,
        )

        # qudit 0: |0> -> uniform over three levels.
        # qudit 1: |2> -> X -> |0>, with certainty.
        expected = {(level, 0): 1 / 3 for level in range(3)}
        assert_distribution_close(
            observed,
            expected,
            shots=RESET_SHOTS,
            atol=FREQUENCY_ATOL,
        )

    def test_format_levels_renders_qiskit_ordered_tokens(
        self,
        seed: int,
    ) -> None:
        """The leftmost rendered token is the last cldigit."""
        circuit = _readme_circuit()

        observed = sample_levels(
            circuit,
            shots=SHORT_RUN_SHOTS,
            seed=seed,
        )
        rendered = {format_levels(levels) for levels in observed}

        assert rendered == {"0 0", "0 1", "0 2"}
        for levels in observed:
            assert format_levels(levels) == f"{levels[1]} {levels[0]}"

    def test_measure_all_journey_decodes_to_predicted_levels(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """A gate-only circuit measured wholesale decodes right."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.x(0)  # qudit 0 -> |1>
        circuit.h(1)  # qudit 1 -> uniform
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        expected = {(1, level): 1 / 3 for level in range(3)}
        assert_distribution_close(
            observed,
            expected,
            shots=shots,
            atol=FREQUENCY_ATOL,
        )
        assert {format_levels(key) for key in observed} == {
            "0 1",
            "1 1",
            "2 1",
        }


class TestMixedDimensionCircuit:
    """A register whose qudits do not share a dimension."""

    def test_per_qudit_gates_use_each_qudits_own_dimension(
        self,
        seed: int,
    ) -> None:
        """One X shifts each qudit modulo *its* number of levels."""
        register = _mixed_register()
        circuit = QuditQuantumCircuit(register)
        circuit.initialize_levels("3 1 0")
        circuit.x(register)
        circuit.measure_all()

        observed = sample_levels(
            circuit,
            shots=SHORT_RUN_SHOTS,
            seed=seed,
        )

        # Each level is shifted once, modulo its own dimension:
        # levels 0, 1 and 3 over dimensions 2, 3 and 5.
        assert_deterministic_outcome(
            observed,
            (1, 2, 4),
            shots=SHORT_RUN_SHOTS,
        )

    def test_cldigit_layout_matches_the_qudit_dimensions(self) -> None:
        """The cldigit layout mirrors the qudit dimensions."""
        circuit = QuditQuantumCircuit(_mixed_register())
        circuit.measure_all()

        assert circuit.dims == (2, 3, 5)
        assert circuit.cldigit_dims == (2, 3, 5)
        assert circuit.cldigit_widths == (1, 2, 3)
        assert circuit.num_clbits == 6
        assert circuit.num_qubits == 6

    def test_every_decoded_level_stays_inside_its_own_range(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """A Hadamard per qudit gives the uniform product state."""
        register = _mixed_register()
        circuit = QuditQuantumCircuit(register)
        circuit.h(register)
        circuit.measure_all()

        observed = sample_levels(circuit, shots=shots, seed=seed)

        dims = (2, 3, 5)
        for key in observed:
            assert len(key) == 3
            for level, dim in zip(key, dims, strict=True):
                assert 0 <= level < dim, f"{key} is outside {dims}"

        expected = {
            (level0, level1, level2): 1 / 30
            for level0 in range(2)
            for level1 in range(3)
            for level2 in range(5)
        }
        assert_distribution_close(
            observed,
            expected,
            shots=shots,
            atol=FREQUENCY_ATOL,
        )


class TestMeasureAll:
    """The three ways of asking for a full measurement."""

    def test_add_digits_creates_one_cldigit_per_qudit(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """A circuit with no cldigits grows a 'meas' register."""
        circuit = QuditQuantumCircuit(2, dim=3)
        assert circuit.num_cldigits == 0

        circuit.measure_all()

        assert circuit.num_cldigits == 2
        assert circuit.cldigit_widths == (2, 2)

        observed = sample_levels(circuit, shots=shots, seed=seed)
        assert_deterministic_outcome(observed, (0, 0), shots=shots)

    def test_add_digits_false_reuses_the_existing_cldigits(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """No register is added; cldigit i receives qudit i."""
        circuit = QuditQuantumCircuit(2, 2, dim=3)
        circuit.x(0)
        circuit.measure_all(add_digits=False)

        assert circuit.num_cldigits == 2
        assert circuit.num_clbits == 4

        observed = sample_levels(circuit, shots=shots, seed=seed)
        assert_deterministic_outcome(observed, (1, 0), shots=shots)

    def test_inplace_false_leaves_the_original_unmeasured(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """The source circuit stays a runnable, pure state."""
        original = QuditQuantumCircuit(1, dim=3)
        original.h(0)

        measured = original.measure_all(inplace=False)

        assert measured is not None
        assert original.num_cldigits == 0
        assert original.num_clbits == 0
        assert measured.num_cldigits == 1

        # Still measurement-free, so it has a state-vector.
        assert_allclose(
            logical_statevector(original),
            np.ones(3, dtype=np.complex128) / np.sqrt(3),
        )

        observed = sample_levels(measured, shots=shots, seed=seed)
        assert_distribution_close(
            observed,
            {(level,): 1 / 3 for level in range(3)},
            shots=shots,
            atol=FREQUENCY_ATOL,
        )


class TestQiskitInteroperability:
    """The encoded circuit is an ordinary Qiskit circuit."""

    def test_encoded_circuit_transpiles_and_runs(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """Handing ``qc.circuit`` to a backend just works."""
        circuit = _interop_circuit()

        counts = run_counts(circuit.circuit, shots=shots, seed=seed)

        assert sum(counts.values()) == shots
        for key in counts:
            assert len(key.replace(" ", "")) == circuit.num_clbits

    def test_raw_counts_decode_through_the_qudit_circuit(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """Counts from the encoded run decode back into levels."""
        circuit = _interop_circuit()

        counts = run_counts(circuit.circuit, shots=shots, seed=seed)
        decoded = circuit.decode_counts(counts)

        assert sum(decoded.values()) == shots
        expected = {(level, 1): 1 / 3 for level in range(3)}
        assert_distribution_close(
            decoded,
            expected,
            shots=shots,
            atol=FREQUENCY_ATOL,
        )

    @pytest.mark.parametrize(
        "view",
        ["ideal", "real", "decomposed"],
    )
    def test_draw_produces_text_for_every_view(
        self,
        view: CircuitView,
    ) -> None:
        """All three renderings return a non-empty drawing."""
        circuit = _interop_circuit()

        drawing = str(circuit.draw(output="text", view=view))

        assert drawing.strip(), f"view '{view}' drew nothing"

    def test_the_ideal_view_has_one_wire_per_qudit(self) -> None:
        """The presentation circuit collapses the encoding qubits."""
        circuit = _interop_circuit()

        ideal = circuit.to_ideal_circuit()

        assert ideal.num_qubits == circuit.num_qudits == 2
        assert ideal.num_clbits == circuit.num_cldigits == 2
        assert circuit.num_qubits == 4

    def test_an_unknown_view_is_rejected(self) -> None:
        """Only the three documented views are accepted."""
        circuit = _interop_circuit()

        with pytest.raises(QuditCircuitError, match="unknown view"):
            circuit.draw(
                output="text",
                view="nonsense",
            )


class TestComposeInverseRoundTrip:
    """Applying a block and its adjoint must do nothing."""

    def test_u_then_u_inverse_restores_the_initial_level(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """Every shot comes back on the level prepared up front."""
        circuit = QuditQuantumCircuit(1, 1, dim=3)
        circuit.x(0)  # prepare |1> unitarily (no reset)

        block = QuditQuantumCircuit(1, dim=3)
        block.h(0)
        block.p(0.7, 0)
        block.k(0)

        assert circuit.compose(block, inplace=True) is None
        assert circuit.compose(block.inverse(), inplace=True) is None
        circuit.measure(0, 0)

        observed = sample_levels(circuit, shots=shots, seed=seed)

        assert_deterministic_outcome(observed, (1,), shots=shots)

    def test_the_round_trip_is_exact_in_the_state_vector(self) -> None:
        """The same round trip is the identity on amplitudes."""
        circuit = QuditQuantumCircuit(1, dim=3)
        circuit.x(0)

        block = QuditQuantumCircuit(1, dim=3)
        block.h(0)
        block.p(0.7, 0)
        block.k(0)

        composed = circuit.compose(block)
        assert composed is not None
        composed = composed.compose(block.inverse())
        assert composed is not None

        assert_allclose(
            logical_statevector(composed),
            logical_statevector(circuit),
        )


class TestArbitraryStatePreparation:
    """Preparing a qudit state from raw amplitudes."""

    def test_sampling_recovers_the_prepared_probabilities(
        self,
        seed: int,
    ) -> None:
        """The measured frequencies are |amplitude|**2."""
        probabilities = np.array([0.5, 0.3, 0.2])
        circuit = QuditQuantumCircuit(1, 1, dim=3)
        circuit.initialize(np.sqrt(probabilities))
        circuit.measure(0, 0)

        observed = sample_levels(circuit, shots=RESET_SHOTS, seed=seed)

        expected = {
            (level,): float(probability)
            for level, probability in enumerate(probabilities)
        }
        assert_distribution_close(
            observed,
            expected,
            shots=RESET_SHOTS,
            atol=FREQUENCY_ATOL,
        )

    def test_the_prepared_state_has_the_requested_amplitudes(
        self,
    ) -> None:
        """The logical state-vector is the requested vector."""
        amplitudes = np.sqrt(np.array([0.5, 0.3, 0.2]))
        circuit = QuditQuantumCircuit(1, dim=3)
        circuit.initialize(amplitudes)

        assert_allclose(
            logical_statevector(circuit),
            amplitudes.astype(np.complex128),
        )


class TestSubspaceIntegrity:
    """Non-power-of-two circuits never leave the qudit subspace."""

    @pytest.mark.parametrize("dim", [3, 5], ids=["d3", "d5"])
    def test_strict_decoding_never_raises(
        self,
        dim: int,
        shots: int,
        seed: int,
    ) -> None:
        """Strict decoding stays quiet: no shot ever leaks."""
        circuit = QuditQuantumCircuit(1, dim=dim)
        circuit.h(0)
        circuit.p(0.7, 0)
        circuit.x(0)
        circuit.measure_all()

        # A leaked outcome would make `decode_counts` raise here.
        observed = sample_levels(
            circuit,
            shots=shots,
            seed=seed,
            on_invalid="raise",
        )

        assert sum(observed.values()) == shots
        assert set(observed) == {(level,) for level in range(dim)}

    def test_strict_decoding_survives_two_leaky_qudits(
        self,
        shots: int,
        seed: int,
    ) -> None:
        """Two qutrits together still stay inside their subspace."""
        circuit = QuditQuantumCircuit(2, dim=3)
        circuit.h(0)
        circuit.x(1)
        circuit.h(1)
        circuit.measure_all()

        observed = sample_levels(
            circuit,
            shots=shots,
            seed=seed,
            on_invalid="raise",
        )

        assert sum(observed.values()) == shots
        assert len(observed) == 9
