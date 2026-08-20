"""Classical bytes and their registers."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, TypeAlias, final, overload

from qiskit._accelerate.circuit import ClassicalRegister, Clbit

from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.utils.consts import IntLike
from qiskit_qudits.utils.dims import qubits_per_qudit
from qiskit_qudits.utils.validation import validate_dim

if TYPE_CHECKING:
    from collections.abc import Iterator


@final
class ClByte:
    r"""A group of :class:`~qiskit.circuit.Clbit`.

    It holds one qudit outcome.

    A single classical bit cannot store the outcome of a measurement on
    a :math:`d`-level qudit, so measurements target a *clbyte*: an
    ordered tuple of :math:`\lceil \log_2 d \rceil` clbits. Clbit ``j``
    of the byte receives qubit ``j`` of the qudit, hence

    .. math::

        \ell = \sum_j \mathrm{bit}_j \, 2^j ,

    and in a backend counts key the byte's character slice is MSB-first,
    so ``int(slice, 2)`` is directly the measured level. See
    :mod:`qiskit_qudits.utils.encoding` for the decoding helpers.

    Instances compare and hash by identity.
    """

    __slots__ = ("_clbits", "_dim", "_index", "_register")

    def __init__(
        self,
        dim: IntLike,
        clbits: Sequence[Clbit] | None = None,
        *,
        register: ClByteRegister | None = None,
        index: int | None = None,
    ) -> None:
        r"""Create a clbyte.

        Args:
            dim: Dimension of the qudit this byte is meant to
                hold. Determines the width and enables leakage detection
                when decoding.
            clbits: The clbits, least significant first. Fresh loose
                clbits are created when ``None``.
            register: Owning register, set by :class:`ClByteRegister`.
            index: Position inside ``register``.

        Raises:
            QuditCircuitError: If the number of supplied clbits does
                not match :math:`\lceil \log_2 d \rceil`, or if
                duplicate clbits are supplied.
        """
        self._dim: int = validate_dim(dim)
        width = qubits_per_qudit(self._dim)

        if clbits is None:
            self._clbits: tuple[Clbit, ...] = tuple(
                Clbit() for _ in range(width)
            )
        else:
            self._clbits = tuple(clbits)
            if len(self._clbits) != width:
                raise QuditCircuitError(
                    f"a {self._dim}-level clbyte needs exactly {width} "
                    f"clbit(s), got {len(self._clbits)}.",
                )
            if len(set(self._clbits)) != width:
                raise QuditCircuitError("duplicate clbits in a single clbyte.")

        self._register: ClByteRegister | None = register
        self._index: int | None = index

    @property
    def dim(self) -> int:
        """Dimension of the qudit this byte is sized for."""
        return self._dim

    @property
    def num_clbits(self) -> int:
        """Number of clbits in this byte."""
        return len(self._clbits)

    @property
    def clbits(self) -> tuple[Clbit, ...]:
        """The clbits, **least significant first**."""
        return self._clbits

    @property
    def register(self) -> ClByteRegister | None:
        """Owning register, or ``None`` for a loose byte."""
        return self._register

    @property
    def index(self) -> int | None:
        """Position inside :attr:`register`, or ``None`` if loose."""
        return self._index

    def __repr__(self) -> str:
        """Return a short, unambiguous representation."""
        if self._register is None or self._index is None:
            return f"ClByte(d={self._dim})"
        return f"ClByte({self._register.name}[{self._index}], d={self._dim})"


class ClByteRegister:
    r"""An ordered collection of :class:`ClByte`\ s.

    Backed by a single :class:`~qiskit.circuit.ClassicalRegister` of
    :math:`\sum_k \lceil \log_2 d_k \rceil` clbits which is added to the
    encoded circuit, so results come back through the standard Qiskit
    machinery. Default names are ``C0``, ``C1``, ...

    Examples:
        .. code-block:: python

            from qiskit_qudits.circuit.clbyte import ClByteRegister

            out = ClByteRegister(2, 4, "out")       # 2 bytes x 2 clbits
            mixed = ClByteRegister.from_dims([3, 16])
    """

    prefix: ClassVar[str] = "C"
    _instances_counter: ClassVar[Iterator[int]] = itertools.count()

    __slots__ = ("_clbytes", "_creg", "_dims", "_name", "_widths")

    def __init__(
        self,
        size: int,
        dim: IntLike,
        name: str | None = None,
    ) -> None:
        """Create a homogeneous clbyte register.

        Args:
            size: Number of bytes.
            dim: Dimension each byte is sized for.
            name: Register name; auto-generated when ``None``.

        Raises:
            QuditCircuitError: If ``name`` is an empty string.
        """
        self._init(
            (validate_dim(dim),) * size,
            name,
        )

    @classmethod
    def from_dims(
        cls,
        dims: Sequence[IntLike],
        name: str | None = None,
    ) -> ClByteRegister:
        """Create a register whose bytes have individual widths.

        This is what :meth:`.QuditQuantumCircuit.measure_all` uses so
        that mixed-dimension circuits can be measured into a single
        register.

        Args:
            dims: Dimension of each byte, in register order.
            name: Register name; auto-generated when ``None``.

        Returns:
            The new heterogeneous register.

        """
        register = cls.__new__(cls)
        cls._init(
            register,
            tuple(validate_dim(dim) for dim in dims),
            name,
        )
        return register

    def _init(self, dims: tuple[int, ...], name: str | None) -> None:
        """Shared initialisation helper (see :meth:`__init__`)."""
        if name is None:
            name = f"{type(self).prefix}{next(self._instances_counter)}"
        elif not name:
            raise QuditCircuitError(
                "register name must be a non-empty string.",
            )

        self._dims: tuple[int, ...] = dims
        self._widths: tuple[int, ...] = tuple(
            qubits_per_qudit(dim) for dim in dims
        )
        self._name: str = name
        self._creg: ClassicalRegister = ClassicalRegister(
            sum(self._widths),
            name,
        )

        clbytes: list[ClByte] = []
        offset = 0
        for index, (dim, width) in enumerate(
            zip(dims, self._widths, strict=True),
        ):
            clbytes.append(
                ClByte(
                    dim,
                    self._creg[offset : offset + width],
                    register=self,
                    index=index,
                ),
            )
            offset += width
        self._clbytes: tuple[ClByte, ...] = tuple(clbytes)

    @property
    def name(self) -> str:
        """Register name (shared with the backing classical regs.)."""
        return self._name

    @property
    def size(self) -> int:
        """Number of bytes."""
        return len(self._clbytes)

    @property
    def dims(self) -> tuple[int, ...]:
        """Dimension of each byte."""
        return self._dims

    @property
    def widths(self) -> tuple[int, ...]:
        """Clbit width of each byte."""
        return self._widths

    @property
    def num_clbits(self) -> int:
        """Total number of clbits."""
        return self._creg.size

    @property
    def clbytes(self) -> tuple[ClByte, ...]:
        """The bytes of this register, in order."""
        return self._clbytes

    @property
    def creg(self) -> ClassicalRegister:
        """The backing :class:`~qiskit.circuit.ClassicalRegister`."""
        return self._creg

    def __len__(self) -> int:
        """Return the number of bytes."""
        return len(self._clbytes)

    def __iter__(self) -> Iterator[ClByte]:
        """Iterate over the bytes in register order."""
        return iter(self._clbytes)

    def __contains__(self, clbyte: object) -> bool:
        """Return whether ``clbyte`` belongs to this register."""
        return any(clbyte is member for member in self._clbytes)

    @overload
    def __getitem__(self, key: IntLike) -> ClByte: ...

    @overload
    def __getitem__(self, key: slice) -> list[ClByte]: ...

    def __getitem__(self, key: IntLike | slice) -> ClByte | list[ClByte]:
        """Index or slice the register."""
        if isinstance(key, slice):
            return list(self._clbytes[key])
        return self._clbytes[int(key)]

    def __repr__(self) -> str:
        """Return a short, unambiguous representation."""
        return (
            f"ClByteRegister({self.size}, dims={self._dims}, "
            f"'{self._name}')"
        )


#: Anything accepted where a clbyte (or several) is expected.
ClByteSpecifier: TypeAlias = (
    ClByte | ClByteRegister | IntLike | slice | Sequence[ClByte | IntLike]
)
