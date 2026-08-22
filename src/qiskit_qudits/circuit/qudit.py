"""Qudits and their registers."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self, TypeAlias, final, overload

from qiskit._accelerate.circuit import QuantumRegister, Qubit

from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.utils.consts import IntLike
from qiskit_qudits.utils.dims import qubits_per_qudit
from qiskit_qudits.utils.validation import validate_dim

if TYPE_CHECKING:
    from collections.abc import Iterator


@final
class Qudit:
    r"""A :math:`d`-level qudit.

    Encoded in :math:`\lceil \log_2 d \rceil` qubits.

    A :class:`Qudit` plays the role that :class:`~qiskit.circuit.Qubit`
    plays in Qiskit: it is an opaque, hashable *handle* that circuits
    use to address a wire. In addition it carries

    * its dimension :attr:`dim`, so gate helpers such as
      :meth:`.QuditQuantumCircuit.x` can infer :math:`d` from the
      target, making mixed-dimension circuits work with no extra
      bookkeeping; and
    * the tuple of encoding :attr:`qubits`, so no index arithmetic is
      ever needed to map a qudit operation onto the underlying circuit.

    Instances compare and hash by **identity** (like modern Qiskit
    bits); two distinct :class:`Qudit` objects are never equal even if
    they wrap the same qubits.

    Examples:
        .. code-block:: python

            from qiskit_qudits.circuit.qudit import Qudit

            loose = Qudit(3)             # a registerless qutrit
            loose.num_qubits             # 2
            loose.qubits                 # (Qubit(), Qubit())
    """

    __slots__ = ("_dim", "_index", "_qubits", "_register")

    def __init__(
        self,
        dim: IntLike,
        qubits: Sequence[Qubit] | None = None,
        *,
        register: QuditRegister | None = None,
        index: int | None = None,
    ) -> None:
        r"""Create a qudit.

        Args:
            dim: Dimension :math:`d` of the qudit
                (:math:`d \geq 2`).
            qubits: The encoding qubits, least significant first. When
                ``None``, fresh registerless
                :class:`~qiskit.circuit.Qubit` objects are created.
            register: Owning register, set by :class:`QuditRegister`.
            index: Position inside ``register``.

        Raises:
            QuditCircuitError: If the number of supplied qubits does
                not match :math:`\lceil \log_2 d \rceil`, or if
                duplicate qubits are supplied.
        """
        self._dim: int = validate_dim(dim)
        width = qubits_per_qudit(self._dim)

        if qubits is None:
            self._qubits: tuple[Qubit, ...] = tuple(
                Qubit() for _ in range(width)
            )
        else:
            self._qubits = tuple(qubits)
            if len(self._qubits) != width:
                raise QuditCircuitError(
                    f"a {self._dim}-level qudit needs exactly {width} "
                    f"qubit(s), got {len(self._qubits)}.",
                )
            if len(set(self._qubits)) != width:
                raise QuditCircuitError("duplicate qubits in a single qudit.")

        self._register: QuditRegister | None = register
        self._index: int | None = index

    @property
    def dim(self) -> int:
        r"""Dimension :math:`d` of the qudit."""
        return self._dim

    @property
    def num_qubits(self) -> int:
        r"""Number of encoding qubits.

        This is equal to :math:`\lceil \log_2 d \rceil`.
        """
        return len(self._qubits)

    @property
    def qubits(self) -> tuple[Qubit, ...]:
        """The encoding qubits, **least significant first**."""
        return self._qubits

    @property
    def register(self) -> QuditRegister | None:
        """Owning :class:`QuditRegister`.

        ``None`` for a loose qudit.
        """
        return self._register

    @property
    def index(self) -> int | None:
        """Position inside :attr:`register`.

        ``None`` for a loose qudit.
        """
        return self._index

    @property
    def fills_hilbert_space(self) -> bool:
        r"""Whether :math:`d` is a power of two."""
        return self._dim == (1 << self.num_qubits)

    def __repr__(self) -> str:
        """Return a short, unambiguous representation."""
        if self._register is None or self._index is None:
            return f"Qudit(d={self._dim})"
        return f"Qudit({self._register.name}[{self._index}], d={self._dim})"

    def __copy__(self) -> Self:
        """Return a copy of the object."""
        return self

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> Self:
        """Return a copy of the object."""
        return self


class QuditRegister:
    r"""An ordered collection of :class:`Qudit`\ s.

    The register owns a backing :class:`~qiskit.circuit.QuantumRegister`
    of :math:`\sum_k \lceil \log_2 d_k \rceil` qubits, which is what
    gets added to the encoded :class:`~qiskit.circuit.QuantumCircuit`.
    The backing register shares the qudit register's :attr:`name`, so
    the "real" circuit view shows wires named ``qd0_0``, ``qd0_1``, ...

    Registers are homogeneous by default; use :meth:`from_dims` for
    mixed dimensions. Instances compare by identity.

    Examples:
        .. code-block:: python

            from qiskit_qudits.circuit.qudit import QuditRegister

            reg = QuditRegister(3, 4, "alice")   # 3 ququarts ->
                                                   6 qubits
            reg.num_qubits                       # 6
            reg[1].qubits                        # (alice_2, alice_3)

            mixed = QuditRegister.from_dims([2, 3, 16], "bob")
            mixed.dims                           # (2, 3, 16)
    """

    #: Prefix used when auto-generating register names.
    prefix: ClassVar[str] = "Q"
    _instances_counter: ClassVar[Iterator[int]] = itertools.count()

    __slots__ = ("_dims", "_name", "_qreg", "_qudits", "_widths")

    def __init__(
        self,
        size: int,
        dim: IntLike,
        name: str | None = None,
    ) -> None:
        r"""Create a homogeneous qudit register.

        Args:
            size: Number of qudits.
            dim: Dimension :math:`d` shared by all qudits.
            name: Register name. Auto-generated (``Q0``, ``Q1``, ...)
                when ``None``.

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
    ) -> QuditRegister:
        """Create a register with per-qudit dimensions.

        Args:
            dims: Dimension of each qudit, in register order.
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
        self._qreg: QuantumRegister = QuantumRegister(sum(self._widths), name)

        qudits: list[Qudit] = []
        offset = 0
        for index, (dim, width) in enumerate(
            zip(dims, self._widths, strict=True),
        ):
            qudits.append(
                Qudit(
                    dim,
                    self._qreg[offset : offset + width],
                    register=self,
                    index=index,
                ),
            )
            offset += width
        self._qudits: tuple[Qudit, ...] = tuple(qudits)

    @property
    def name(self) -> str:
        """Register name (shared with the backing quantum register)."""
        return self._name

    @property
    def size(self) -> int:
        """Number of qudits in the register."""
        return len(self._qudits)

    @property
    def dims(self) -> tuple[int, ...]:
        """Dimension of each qudit, in register order."""
        return self._dims

    @property
    def dim(self) -> int:
        r"""The common dimension :math:`d` of a homogeneous register.

        Raises:
            QuditCircuitError: If the register is heterogeneous; use
                :attr:`dims` instead.
        """
        unique = set(self._dims)
        if len(unique) > 1:
            raise QuditCircuitError(
                f"register '{self._name}' is heterogeneous ({self._dims}); "
                "use `dims` instead of `dim`.",
            )
        return next(iter(unique), 0)

    @property
    def widths(self) -> tuple[int, ...]:
        """Number of encoding qubits of each qudit."""
        return self._widths

    @property
    def num_qubits(self) -> int:
        """Total number of encoding qubits."""
        return self._qreg.size

    @property
    def qudits(self) -> tuple[Qudit, ...]:
        """The qudits of this register, in order."""
        return self._qudits

    @property
    def qreg(self) -> QuantumRegister:
        """The backing :class:`~qiskit.circuit.QuantumRegister`."""
        return self._qreg

    def __len__(self) -> int:
        """Return the number of qudits."""
        return len(self._qudits)

    def __iter__(self) -> Iterator[Qudit]:
        """Iterate over the qudits in register order."""
        return iter(self._qudits)

    def __contains__(self, qudit: object) -> bool:
        """Return whether ``qudit`` belongs to this register."""
        return any(qudit is member for member in self._qudits)

    @overload
    def __getitem__(self, key: IntLike) -> Qudit: ...

    @overload
    def __getitem__(self, key: slice) -> list[Qudit]: ...

    def __getitem__(self, key: IntLike | slice) -> Qudit | list[Qudit]:
        """Index or slice the register."""
        if isinstance(key, slice):
            return list(self._qudits[key])
        return self._qudits[int(key)]

    def __repr__(self) -> str:
        """Return a short, unambiguous representation."""
        if len(set(self._dims)) <= 1:
            dims_repr = f"d={self._dims[0] if self._dims else 0}"
        else:
            dims_repr = f"dims={self._dims}"
        return f"QuditRegister({self.size}, {dims_repr}, '{self._name}')"

    def __copy__(self) -> Self:
        """Return a copy of the object."""
        return self

    def __deepcopy__(self, memo: dict[int, object] | None = None) -> Self:
        """Return a copy of the object."""
        return self


#: Anything accepted where a qudit (or several) is expected.
#:
#: * :class:`Qudit` - one specific qudit;
#: * :class:`QuditRegister` - every qudit of the register, in order;
#: * ``int`` / ``numpy.integer`` - a circuit-wide qudit index;
#: * ``slice`` - a range of circuit-wide indices;
#: * a sequence mixing :class:`Qudit` objects and indices.
QuditSpecifier: TypeAlias = (
    Qudit | QuditRegister | IntLike | slice | Sequence[Qudit | IntLike]
)
