"""Classical digits and their registers."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self, TypeAlias, final, overload

from qiskit._accelerate.circuit import ClassicalRegister, Clbit

from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.utils.consts import IntLike
from qiskit_qudits.utils.dims import qubits_per_qudit
from qiskit_qudits.utils.validation import validate_dim

if TYPE_CHECKING:
    from collections.abc import Iterator


@final
class ClDigit:
    r"""A group of :class:`~qiskit.circuit.Clbit`.

    It holds one qudit outcome.

    A single classical bit cannot store the outcome of a measurement on
    a :math:`d`-level qudit, so measurements target a *cldigit*: an
    ordered tuple of :math:`\lceil \log_2 d \rceil` clbits. Clbit ``j``
    of the digit receives qubit ``j`` of the qudit, hence

    .. math::

        \ell = \sum_j \mathrm{bit}_j \, 2^j ,

    and in a backend counts key the digit's character slice is
    MSB-first, so ``int(slice, 2)`` is directly the measured level. See
    :mod:`qiskit_qudits.utils.encoding` for the decoding helpers.

    Instances compare and hash by identity.
    """

    __slots__ = ("_clbits", "_dim", "_index", "_register")

    def __init__(
        self,
        dim: IntLike,
        clbits: Sequence[Clbit] | None = None,
        *,
        register: ClDigitRegister | None = None,
        index: int | None = None,
    ) -> None:
        r"""Create a cldigit.

        Args:
            dim: Dimension of the qudit this digit is meant to
                hold. Determines the width and enables leakage detection
                when decoding.
            clbits: The clbits, least significant first. Fresh loose
                clbits are created when ``None``.
            register: Owning register, set by :class:`ClDigitRegister`.
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
                    f"a {self._dim}-level cldigit needs exactly {width} "
                    f"clbit(s), got {len(self._clbits)}.",
                )
            if len(set(self._clbits)) != width:
                raise QuditCircuitError(
                    "duplicate clbits in a single cldigit.",
                )

        self._register: ClDigitRegister | None = register
        self._index: int | None = index

    @property
    def dim(self) -> int:
        """Dimension of the qudit this digit is sized for."""
        return self._dim

    @property
    def num_clbits(self) -> int:
        """Number of clbits in this digit."""
        return len(self._clbits)

    @property
    def clbits(self) -> tuple[Clbit, ...]:
        """The clbits, **least significant first**."""
        return self._clbits

    @property
    def register(self) -> ClDigitRegister | None:
        """Owning register, or ``None`` for a loose digit."""
        return self._register

    @property
    def index(self) -> int | None:
        """Position inside :attr:`register`, or ``None`` if loose."""
        return self._index

    def __repr__(self) -> str:
        """Return a short, unambiguous representation."""
        if self._register is None or self._index is None:
            return f"ClDigit(d={self._dim})"
        return f"ClDigit({self._register.name}[{self._index}], d={self._dim})"

    def __copy__(self) -> Self:
        """Return a copy of the object."""
        return self

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> Self:
        """Return a copy of the object."""
        return self


class ClDigitRegister:
    r"""An ordered collection of :class:`ClDigit`\ s.

    Backed by a single :class:`~qiskit.circuit.ClassicalRegister` of
    :math:`\sum_k \lceil \log_2 d_k \rceil` clbits which is added to the
    encoded circuit, so results come back through the standard Qiskit
    machinery. Default names are ``C0``, ``C1``, ...

    Examples:
        .. code-block:: python

            from qiskit_qudits.circuit.cldigit import ClDigitRegister

            out = ClDigitRegister(2, 4, "out")     # 2 digits x 2 clbits
            mixed = ClDigitRegister.from_dims([3, 16])
    """

    prefix: ClassVar[str] = "C"
    _instances_counter: ClassVar[Iterator[int]] = itertools.count()

    __slots__ = ("_cldigits", "_creg", "_dims", "_name", "_widths")

    def __init__(
        self,
        size: int,
        dim: IntLike,
        name: str | None = None,
    ) -> None:
        """Create a homogeneous cldigit register.

        Args:
            size: Number of digits.
            dim: Dimension each digit is sized for.
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
    ) -> ClDigitRegister:
        """Create a register whose digits have individual widths.

        This is what :meth:`.QuditQuantumCircuit.measure_all` uses so
        that mixed-dimension circuits can be measured into a single
        register.

        Args:
            dims: Dimension of each digit, in register order.
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

        cldigits: list[ClDigit] = []
        offset = 0
        for index, (dim, width) in enumerate(
            zip(dims, self._widths, strict=True),
        ):
            cldigits.append(
                ClDigit(
                    dim,
                    self._creg[offset : offset + width],
                    register=self,
                    index=index,
                ),
            )
            offset += width
        self._cldigits: tuple[ClDigit, ...] = tuple(cldigits)

    @property
    def name(self) -> str:
        """Register name (shared with the backing classical regs.)."""
        return self._name

    @property
    def size(self) -> int:
        """Number of digits."""
        return len(self._cldigits)

    @property
    def dims(self) -> tuple[int, ...]:
        """Dimension of each digit."""
        return self._dims

    @property
    def widths(self) -> tuple[int, ...]:
        """Clbit width of each digit."""
        return self._widths

    @property
    def num_clbits(self) -> int:
        """Total number of clbits."""
        return self._creg.size

    @property
    def cldigits(self) -> tuple[ClDigit, ...]:
        """The digits of this register, in order."""
        return self._cldigits

    @property
    def creg(self) -> ClassicalRegister:
        """The backing :class:`~qiskit.circuit.ClassicalRegister`."""
        return self._creg

    def __len__(self) -> int:
        """Return the number of digits."""
        return len(self._cldigits)

    def __iter__(self) -> Iterator[ClDigit]:
        """Iterate over the digits in register order."""
        return iter(self._cldigits)

    def __contains__(self, cldigit: object) -> bool:
        """Return whether ``cldigit`` belongs to this register."""
        return any(cldigit is member for member in self._cldigits)

    @overload
    def __getitem__(self, key: IntLike) -> ClDigit: ...

    @overload
    def __getitem__(self, key: slice) -> list[ClDigit]: ...

    def __getitem__(self, key: IntLike | slice) -> ClDigit | list[ClDigit]:
        """Index or slice the register."""
        if isinstance(key, slice):
            return list(self._cldigits[key])
        return self._cldigits[int(key)]

    def __repr__(self) -> str:
        """Return a short, unambiguous representation."""
        return (
            f"ClDigitRegister({self.size}, dims={self._dims}, "
            f"'{self._name}')"
        )

    def __copy__(self) -> Self:
        """Return a copy of the object."""
        return self

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> Self:
        """Return a copy of the object."""
        return self


#: Anything accepted where a cldigit (or several) is expected.
ClDigitSpecifier: TypeAlias = (
    ClDigit | ClDigitRegister | IntLike | slice | Sequence[ClDigit | IntLike]
)
