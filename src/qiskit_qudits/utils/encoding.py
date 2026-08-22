r"""Encoding and decoding helpers for qudit-in-qubit simulation.

Conventions
===========

Throughout this module (and the whole library) Qiskit's **little-
endian** ordering is used:

* Inside a qudit, qubit ``j`` carries weight :math:`2^j`, therefore a
  level :math:`\ell` is stored as the bits of ``format(ell, '0{n}b')``
  with the *rightmost* character on qubit ``0``.
* Inside a classical bit-string returned by a backend, clbit ``i`` is at
  string position ``len(bitstring) - 1 - i``, and registers are
  separated by spaces with the *last* register leftmost. Consequently,
  dropping the spaces yields one global, right-to-left clbit string.

A useful corollary: the slice of a counts key belonging to a single
:class:`~qiskit_qudits.circuit.ClDigit` is MSB-first, so
``int(slice, 2)`` *is* the measured level.
"""

from __future__ import annotations

from collections import Counter
from math import prod
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from qiskit_qudits.utils.dims import qubits_per_qudit
from qiskit_qudits.utils.validation import validate_integer_range

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from qiskit_qudits.utils.typeguards import IntLike

#: What to do with a decoded value that falls outside the qudit subspace
#: (i.e. ``value >= dim``), which is a signature of leakage.
InvalidPolicy: TypeAlias = Literal["keep", "drop", "raise"]

#: A decoded shot: one level per :class:`.ClDigit`, in clbit-index order.
Levels: TypeAlias = tuple[int, ...]

_DEFAULT_ATOL = 1e-8


def level_to_bitstring(level: int, width: int) -> str:
    r"""Encode a qudit level as an MSB-first bit-string.

    Args:
        level: The computational level to encode.
        width: Width of the returned string.

    Returns:
        A string of ``width`` characters, MSB first. Feeding this to
        :meth:`qiskit.circuit.QuantumCircuit.initialize` together with
        the qubit list ``[q_0, ..., q_{n-1}]`` (LSB first) reproduces
        :math:`\lvert \text{level} \rangle`, because Qiskit applies the
        *leftmost* character to the *last* qubit.

    Raises:
        ValueError: If ``level`` does not fit into ``width``.

    Examples:
        >>> level_to_bitstring(5, 3)
        '101'
    """
    if level < 0 or level >= (1 << width):
        raise ValueError(f"level {level} does not fit into {width} bit(s).")
    return format(level, f"0{width}b")


def _bitstring_to_level(bits: str) -> int:
    """Decode an MSB-first bit-string into a qudit level.

    Args:
        bits: A non-empty string of ``'0'``/``'1'`` characters,
            MSB first.

    Returns:
        The corresponding integer level.

    Raises:
        ValueError: If ``bits`` is empty or contains
            other characters.
    """
    if not bits or any(char not in "01" for char in bits):
        raise ValueError(f"'{bits}' is not a binary string.")
    return int(bits, 2)


def _split_bitstring(
    bitstring: str,
    widths: Sequence[int],
) -> tuple[str, ...]:
    """Split a backend bit-string into per-:class:`.ClDigit` chunks.

    Args:
        bitstring: A counts key, e.g. ``'0110 11'``. Whitespace
            (Qiskit's register separator) is ignored.
        widths: Number of clbits of each digit, in **clbit-index order**
            (digit 0 first).

    Returns:
        One MSB-first chunk per width, in the same order as ``widths``.

    Raises:
        ValueError: If the total width does not match the string.

    Examples:
        The rightmost characters belong to the lowest clbit indices:

        >>> split_bitstring('10 01', [2, 2])
        ('01', '10')
    """
    raw = "".join(bitstring.split())
    total = sum(widths)
    if len(raw) != total:
        raise ValueError(
            f"bit-string '{bitstring}' has {len(raw)} bit(s) but the layout "
            f"requires {total}.",
        )

    chunks: list[str] = []
    position = len(raw)
    for width in widths:
        size = int(width)
        chunks.append(raw[position - size : position])
        position -= size
    return tuple(chunks)


def decode_bitstring(
    bitstring: str,
    widths: Sequence[int],
    dims: Sequence[int] | None = None,
    *,
    on_invalid: InvalidPolicy = "keep",
) -> Levels | None:
    """Decode a single counts key into one level per digit.

    Args:
        bitstring: A counts key.
        widths: Clbit width of each digit, in clbit-index order.
        dims: Optional number of levels of each digit, used to detect
            leakage out of the qudit subspace.
        on_invalid: Behaviour when a decoded value is ``>= dims[i]``:
            ``'keep'`` returns it unchanged, ``'drop'`` returns ``None``
            and ``'raise'`` raises.

    Returns:
        The decoded levels in digit order, or ``None`` when the shot is
        discarded because of ``on_invalid='drop'``.

    Raises:
        ValueError: On a width mismatch, or on leakage when
            ``on_invalid='raise'``.
    """
    levels = tuple(
        _bitstring_to_level(chunk)
        for chunk in _split_bitstring(bitstring, widths)
    )
    if dims is None:
        return levels

    if len(dims) != len(levels):
        raise ValueError(
            f"got {len(dims)} dimension(s) for {len(levels)} digit(s).",
        )
    for index, (level, dim) in enumerate(zip(levels, dims, strict=True)):
        if level >= int(dim):
            if on_invalid == "raise":
                raise ValueError(
                    f"digit {index} decoded to level {level}, which is outside "
                    f"the {int(dim)}-level qudit subspace (leakage?).",
                )
            if on_invalid == "drop":
                return None
    return levels


def decode_counts(
    counts: Mapping[str, int],
    widths: Sequence[int],
    dims: Sequence[int] | None = None,
    *,
    on_invalid: InvalidPolicy = "keep",
) -> dict[Levels, int]:
    """Decode a full counts mapping into qudit levels.

    Args:
        counts: Mapping from bit-string to number of shots, as returned
            by ``result.get_counts()`` or ``SamplerResult``-style
            containers.
        widths: Clbit width of each digit, in clbit-index order.
        dims: Optional levels per digit, for leakage detection.
        on_invalid: See :func:`decode_bitstring`.

    Returns:
        Mapping from a tuple of levels (**digit order**, digit 0 first) to
        the aggregated number of shots. Use :func:`format_levels` to
        get a Qiskit-ordered, human-readable key.
    """
    decoded: Counter[Levels] = Counter()
    for bitstring, shots in counts.items():
        levels = decode_bitstring(
            bitstring,
            widths,
            dims,
            on_invalid=on_invalid,
        )
        if levels is not None:
            decoded[levels] += int(shots)
    return dict(decoded)


def format_levels(levels: Iterable[int], *, separator: str = " ") -> str:
    """Render decoded levels the way Qiskit renders bit-strings.

    The **leftmost** token is the **last** digit, mirroring Qiskit's
    little-endian string convention, and tokens are separated so the
    result is never ambiguous for :math:`d > 10`.

    Args:
        levels: Levels in digit order (digit 0 first).
        separator: Token separator.

    Returns:
        e.g. ``'11 3 0'`` for ``levels=(0, 3, 11)``.
    """
    return separator.join(str(level) for level in reversed(list(levels)))


def _encoded_index(dims: Sequence[int], logical_index: int) -> int:
    r"""Map a mixed-radix qudit index onto an encoded qubit index.

    Args:
        dims: Level counts of the qudits, least significant first.
        logical_index: Index into the :math:`\prod_i d_i` dimensional
            qudit space.

    Returns:
        The corresponding index in the :math:`2^N` dimensional encoded
        space, with :math:`N = \sum_i \lceil \log_2 d_i \rceil`.
    """
    remainder = logical_index
    encoded = 0
    shift = 0
    for dim in dims:
        encoded |= (remainder % dim) << shift
        remainder //= dim
        shift += qubits_per_qudit(dim)
    return encoded


def embed_state(
    dims: Sequence[int],
    amplitudes: np.typing.NDArray[np.complex128],
    *,
    atol: float = _DEFAULT_ATOL,
) -> np.typing.NDArray[np.complex128]:
    r"""Embed a qudit state-vector into the encoded qubit space.

    Args:
        dims: Level counts of the target qudits, least significant
            first.
        amplitudes: :math:`\prod_i d_i` amplitudes over the qudit space.
        atol: Absolute tolerance of the normalisation check.

    Returns:
        A :math:`2^N` dimensional vector whose out-of-subspace entries
        are exactly zero.

    Raises:
        ValueError: On a length mismatch or a non-normalised
            input.
    """
    logical_dim = prod(dims)
    if amplitudes.size != logical_dim:
        raise ValueError(
            f"expected {logical_dim} amplitude(s) for dimensions "
            f"{tuple(dims)}, got {amplitudes.size}.",
        )
    norm = float(
        np.linalg.norm(amplitudes),  # pyright: ignore[reportUnknownMemberType]
    )
    if abs(norm - 1.0) > atol:
        raise ValueError(f"state-vector is not normalised (norm={norm}).")

    num_qubits = sum(qubits_per_qudit(dim) for dim in dims)
    encoded = np.zeros(1 << num_qubits, dtype=np.complex128)
    for logical in range(logical_dim):
        encoded[_encoded_index(dims, logical)] = amplitudes[logical]
    return encoded


def project_state(
    dims: Sequence[int],
    amplitudes: np.typing.NDArray[np.complex128],
    *,
    atol: float = _DEFAULT_ATOL,
) -> np.typing.NDArray[np.complex128]:
    r"""Project an encoded state-vector back onto the qudit space.

    Args:
        dims: Level counts of the qudits, least significant first.
        amplitudes: :math:`2^N` encoded amplitudes.
        atol: Tolerance used to check that no amplitude leaks outside
            the qudit subspace.

    Returns:
        The :math:`\prod_i d_i` dimensional qudit state-vector.

    Raises:
        ValueError: On a length mismatch, or if a non-negligible
            amplitude sits outside the qudit subspace.
    """
    vector = amplitudes.ravel()
    num_qubits = sum(qubits_per_qudit(int(dim)) for dim in dims)
    if vector.size != (1 << num_qubits):
        raise ValueError(
            f"expected {1 << num_qubits} amplitude(s), got {vector.size}.",
        )

    logical_dim = prod(int(dim) for dim in dims)
    logical = np.zeros(logical_dim, dtype=np.complex128)
    valid: set[int] = set()
    for index in range(logical_dim):
        encoded = _encoded_index(dims, index)
        logical[index] = vector[encoded]
        valid.add(encoded)

    leaked = float(
        np.linalg.norm(  # pyright: ignore[reportUnknownMemberType]
            np.array(
                [vector[i] for i in range(vector.size) if i not in valid],
                dtype=np.complex128,
            ),
        ),
    )
    if leaked > atol:
        raise ValueError(
            f"state-vector has amplitude {leaked} outside the qudit subspace.",
        )
    return logical


def parse_level_tokens(text: str, num_qudits: int) -> tuple[int, ...]:
    """Parse a whitespace-separated level specification.

    The string is read with **Qiskit's ordering**: the rightmost token
    belongs to the *first* (lowest-index) target qudit. Whitespace is
    mandatory because ``'1111'`` would be ambiguous as soon as any qudit
    has more than two levels.

    Args:
        text: e.g. ``'11 3 0'``. Tokens are parsed with
            ``int(token, 0)`` so ``'0b1011'`` and ``'0xb'`` also work.
        num_qudits: Expected number of tokens.

    Returns:
        The levels in **target order** (first target first).

    Raises:
        ValueError: On a token count mismatch or an unparsable
            token.

    Examples:
        >>> parse_level_tokens('11 3 0', 3)
        (0, 3, 11)
    """
    tokens = text.split()
    if len(tokens) != num_qudits:
        raise ValueError(
            f"expected {num_qudits} whitespace-separated level(s) in "
            f"'{text}', got {len(tokens)}. Levels must be separated by spaces "
            "because concatenated digits are ambiguous for d > 2.",
        )
    levels: list[int] = []
    for token in reversed(tokens):
        try:
            levels.append(int(token, 0))
        except ValueError as exc:
            raise ValueError(f"'{token}' is not an integer level.") from exc
    return tuple(levels)


def validate_basis_states(
    states: Sequence[IntLike],
    dims: Sequence[int],
) -> tuple[int, ...]:
    """Validate state indices against qudit dimensions.

    Args:
        states: Target state indices to prepare, in target order.
        dims: Dimension of each target qudit, in target order.

    Returns:
        The validated state indices as plain ``int``.

    Raises:
        ValueError: On length mismatch or out-of-range
            state index.
    """
    if len(states) != len(dims):
        raise ValueError(
            f"got {len(states)} state(s) for {len(dims)} qudit(s).",
        )

    return tuple(
        validate_integer_range(
            state,
            minimum=0,
            maximum=dim - 1,
            name=f"state for qudit {index}",
        )
        for index, (state, dim) in enumerate(zip(states, dims, strict=True))
    )
