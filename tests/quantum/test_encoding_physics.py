r"""The qudit-into-qubit encoding as a physical isometry.

``embed_state`` maps the :math:`\prod_i d_i` amplitudes of a register
of qudits into the :math:`2^N` amplitudes of the qubits that simulate
it, and ``project_state`` maps them back. For the simulation to mean
anything, that pair must be a genuine isometry :math:`E` of Hilbert
spaces:

* :math:`E^\dagger E = 1`, so norms and inner products (hence all
  transition probabilities) are preserved;
* :math:`E E^\dagger` is the orthogonal projector onto the code space,
  and every amplitude outside it is exactly zero;
* the mixed-radix index map is little-endian, qudit ``0`` sitting on
  the lowest-index qubits, and each qudit is padded to
  :math:`\lceil \log_2 d_i \rceil` qubits *independently* -- so the
  isometry factorises as a tensor product;
* the encoding intertwines the two pictures:
  :math:`E^\dagger U_{\text{encoded}} E = U_{\text{qudit}}`, i.e.
  running a gate on the qubits and projecting back is the same as
  applying the ``d x d`` operator directly.
"""

from __future__ import annotations

import math
from math import prod
from typing import TYPE_CHECKING

import numpy as np
import pytest

from qiskit_qudits.gates import (
    QuditHGate,
    QuditKGate,
    QuditNOTGate,
    QuditPGate,
    QuditTGate,
    QuditXdgGate,
    QuditXGate,
    QuditZGate,
)
from qiskit_qudits.utils.encoding import embed_state, project_state
from tests.helpers import (
    ATOL,
    assert_allclose,
    gate_matrix,
    parametrize_dims,
    subspace_block,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from qiskit_qudits.gates.base.gate import QuditGate

#: Seed base of the random logical states.
SEED = 20_240_611

#: Angle of the parametric phase gate.
THETA = 0.7

#: Registers exercised by the isometry tests; both single qudits and
#: mixed-radix registers, with and without leakage.
LAYOUTS = (
    (2,),
    (3,),
    (4,),
    (5,),
    (7,),
    (8,),
    (2, 3),
    (3, 2),
    (3, 5),
    (2, 2, 3),
)

#: Hand-computed ``logical index -> encoded index`` tables. Each qudit
#: is padded on its own, so a non-power-of-two radix leaves holes.
INDEX_TABLES = (
    ((2, 3), (0, 1, 2, 3, 4, 5)),
    ((3, 2), (0, 1, 2, 4, 5, 6)),
    (
        (3, 5),
        (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18),
    ),
)

GATE_FACTORIES: tuple[tuple[str, Callable[[int], QuditGate]], ...] = (
    ("X", QuditXGate),
    ("Xdg", QuditXdgGate),
    ("Z", QuditZGate),
    ("T", QuditTGate),
    ("H", QuditHGate),
    ("NOT", QuditNOTGate),
    ("K", QuditKGate),
    ("P(0.7)", lambda dim: QuditPGate(dim, THETA)),
)

parametrize_layouts = pytest.mark.parametrize(
    "dims",
    [
        pytest.param(layout, id="x".join(str(dim) for dim in layout))
        for layout in LAYOUTS
    ],
)

parametrize_gates = pytest.mark.parametrize(
    "make_gate",
    [pytest.param(factory, id=name) for name, factory in GATE_FACTORIES],
)


def _qubit_width(dim: int) -> int:
    """Return the qubits needed by one ``dim``-level qudit."""
    return math.ceil(math.log2(dim))


def _num_qubits(dims: Sequence[int]) -> int:
    """Return the total width of an encoded register."""
    return sum(_qubit_width(dim) for dim in dims)


def _encoded_index(dims: Sequence[int], logical: int) -> int:
    """Return the encoded index of a mixed-radix logical index.

    Reference implementation written from the documented convention:
    qudit ``0`` is the least significant factor and occupies the
    lowest-index qubits.

    Args:
        dims: Level counts, least significant qudit first.
        logical: Index into the ``prod(dims)`` dimensional space.

    Returns:
        The index into the ``2**N`` dimensional encoded space.
    """
    index = 0
    shift = 0
    remainder = logical
    for dim in dims:
        index += (remainder % dim) << shift
        remainder //= dim
        shift += _qubit_width(dim)
    return index


def _random_state(
    size: int,
    rng: np.random.Generator,
) -> np.typing.NDArray[np.complex128]:
    """Return a random normalised complex vector of length ``size``."""
    amplitudes = rng.normal(size=size) + 1j * rng.normal(size=size)
    return np.asarray(
        amplitudes / np.linalg.norm(amplitudes),
        dtype=np.complex128,
    )


def _isometry(dims: Sequence[int]) -> np.typing.NDArray[np.complex128]:
    """Return the ``2**N x prod(dims)`` matrix of the embedding."""
    logical_dim = prod(dims)
    columns: list[np.typing.NDArray[np.complex128]] = []
    for level in range(logical_dim):
        vector = np.zeros(logical_dim, dtype=np.complex128)
        vector[level] = 1.0
        columns.append(embed_state(dims, vector))
    return np.array(columns, dtype=np.complex128).T


class TestIndexMapping:
    """Where an amplitude of qudit ``i`` ends up among the qubits."""

    @pytest.mark.parametrize(
        ("dims", "table"),
        [
            pytest.param(
                dims,
                table,
                id="x".join(str(dim) for dim in dims),
            )
            for dims, table in INDEX_TABLES
        ],
    )
    def test_embedding_matches_the_hand_computed_index_table(
        self,
        dims: tuple[int, ...],
        table: tuple[int, ...],
    ) -> None:
        """Check the mixed-radix map against an explicit table."""
        logical_dim = prod(dims)
        assert len(table) == logical_dim

        for logical, encoded_index in enumerate(table):
            vector = np.zeros(logical_dim, dtype=np.complex128)
            vector[logical] = 1.0
            embedded = embed_state(dims, vector)
            expected = np.zeros(embedded.size, dtype=np.complex128)
            expected[encoded_index] = 1.0

            assert_allclose(
                embedded,
                expected,
                message=(
                    f"logical index {logical} of {dims} should be encoded "
                    f"at qubit index {encoded_index}"
                ),
            )

    @parametrize_layouts
    def test_embedded_amplitudes_vanish_outside_the_code_space(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """Check the embedding is exactly zero on leakage indices."""
        rng = np.random.default_rng(SEED + prod(dims))
        state = _random_state(prod(dims), rng)

        embedded = embed_state(dims, state)

        valid = {
            _encoded_index(dims, logical) for logical in range(prod(dims))
        }
        leakage = np.array(
            [
                embedded[index]
                for index in range(embedded.size)
                if index not in valid
            ],
            dtype=np.complex128,
        )

        assert embedded.size == 1 << _num_qubits(dims)
        assert_allclose(
            leakage,
            np.zeros(leakage.size, dtype=np.complex128),
            message="the embedded state has weight outside the code space",
        )


class TestIsometry:
    """``embed_state``/``project_state`` as an isometry pair."""

    @parametrize_layouts
    def test_round_trip_is_exact(self, dims: tuple[int, ...]) -> None:
        """Check ``project(embed(psi)) = psi`` exactly."""
        rng = np.random.default_rng(SEED + prod(dims))
        state = _random_state(prod(dims), rng)

        assert_allclose(
            project_state(dims, embed_state(dims, state)),
            state,
            message="the encode/decode round trip lost information",
        )

    @parametrize_layouts
    def test_embedding_preserves_norms_and_inner_products(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """Check overlaps survive the embedding, so do probabilities."""
        rng = np.random.default_rng(SEED + prod(dims))
        first = _random_state(prod(dims), rng)
        second = _random_state(prod(dims), rng)

        embedded_first = embed_state(dims, first)
        embedded_second = embed_state(dims, second)

        assert float(np.linalg.norm(embedded_first)) == pytest.approx(
            1.0,
            abs=ATOL,
        )
        assert complex(
            np.vdot(embedded_first, embedded_second),
        ) == pytest.approx(complex(np.vdot(first, second)), abs=ATOL)

    @parametrize_layouts
    def test_embedding_columns_are_orthonormal(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """Check ``E^dagger E = 1``: the embedding is an isometry."""
        isometry = _isometry(dims)

        assert isometry.shape == (1 << _num_qubits(dims), prod(dims))
        assert_allclose(
            isometry.conj().T @ isometry,
            np.eye(prod(dims), dtype=np.complex128),
            message="the embedding does not have orthonormal columns",
        )

    @parametrize_layouts
    def test_code_space_projector_is_an_orthogonal_projector(
        self,
        dims: tuple[int, ...],
    ) -> None:
        """Check ``E E^dagger`` is Hermitian, idempotent, of rank d."""
        isometry = _isometry(dims)
        projector = isometry @ isometry.conj().T

        assert_allclose(
            projector @ projector,
            projector,
            message="the code-space projector is not idempotent",
        )
        assert_allclose(
            projector.conj().T,
            projector,
            message="the code-space projector is not Hermitian",
        )
        assert complex(np.trace(projector)).real == pytest.approx(
            float(prod(dims)),
            abs=ATOL,
        )

    def test_project_state_rejects_a_leaked_amplitude(self) -> None:
        """Check a state leaking out of the code space is refused."""
        leaked = np.zeros(4, dtype=np.complex128)
        leaked[0] = 1.0 / np.sqrt(2.0)
        leaked[3] = 1.0 / np.sqrt(2.0)

        with pytest.raises(ValueError, match="outside the qudit subspace"):
            project_state((3,), leaked)


class TestGateEncodingConsistency:
    """Encoded evolution must agree with the qudit-level operator."""

    @parametrize_gates
    @parametrize_dims()
    def test_gate_action_commutes_with_the_encoding(
        self,
        make_gate: Callable[[int], QuditGate],
        dim: int,
    ) -> None:
        """Check ``project(U embed(psi)) = U_block psi``."""
        rng = np.random.default_rng(SEED + dim)
        state = _random_state(dim, rng)
        gate = make_gate(dim)

        evolved = gate_matrix(gate) @ embed_state((dim,), state)

        assert_allclose(
            project_state((dim,), evolved),
            subspace_block(gate) @ state,
            message=(
                "evolving in the encoded space and projecting back does "
                "not reproduce the qudit-level operator"
            ),
        )

    @pytest.mark.parametrize(
        ("dims", "position", "gate_class"),
        [
            pytest.param((3, 5), 0, QuditXGate, id="3x5-on-qudit0"),
            pytest.param((3, 5), 1, QuditZGate, id="3x5-on-qudit1"),
            pytest.param((2, 3), 1, QuditHGate, id="2x3-on-qudit1"),
            pytest.param((3, 2), 0, QuditKGate, id="3x2-on-qudit0"),
        ],
    )
    def test_encoding_factorises_over_qudits(
        self,
        dims: tuple[int, ...],
        position: int,
        gate_class: Callable[[int], QuditGate],
    ) -> None:
        """Check a one-qudit gate stays a tensor factor once encoded."""
        gate = gate_class(dims[position])
        low_qubits = _num_qubits(dims[:position])
        high_qubits = _num_qubits(dims[position + 1 :])
        encoded_operator = np.kron(
            np.eye(1 << high_qubits, dtype=np.complex128),
            np.kron(
                gate_matrix(gate),
                np.eye(1 << low_qubits, dtype=np.complex128),
            ),
        )
        logical_operator = np.kron(
            np.eye(prod(dims[position + 1 :]), dtype=np.complex128),
            np.kron(
                subspace_block(gate),
                np.eye(prod(dims[:position]), dtype=np.complex128),
            ),
        )
        rng = np.random.default_rng(SEED + prod(dims))
        state = _random_state(prod(dims), rng)

        evolved = encoded_operator @ embed_state(dims, state)

        assert_allclose(
            project_state(dims, evolved),
            logical_operator @ state,
            message=(
                f"a gate on qudit {position} of {dims} is not the "
                "expected tensor factor of the encoded operator"
            ),
        )
