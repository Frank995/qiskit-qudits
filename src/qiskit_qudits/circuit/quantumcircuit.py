"""The qudit analogue of :class:`qiskit.circuit.QuantumCircuit`."""

from __future__ import annotations

import itertools
from collections import OrderedDict
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    cast,
)

import numpy as np
from qiskit.circuit import (
    Barrier,
    ClassicalRegister,
    Clbit,
    Gate,
    Instruction,
    Measure,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    Reset,
)

from qiskit_qudits.circuit.clbyte import ClByte, ClByteRegister
from qiskit_qudits.circuit.directives import (
    QuditBarrier,
    QuditDirective,
    QuditInitializeLevels,
    QuditMeasure,
    QuditReset,
    QuditStatePreparation,
)
from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.circuit.instruction import QuditCircuitInstruction
from qiskit_qudits.circuit.qudit import Qudit, QuditRegister
from qiskit_qudits.gates import (
    QuditHdgGate,
    QuditHGate,
    QuditIGate,
    QuditKGate,
    QuditNOTGate,
    QuditPGate,
    QuditQFTGate,
    QuditSdgGate,
    QuditSGate,
    QuditSUMPGate,
    QuditSUMXdgGate,
    QuditSUMXGate,
    QuditSWAPGate,
    QuditTdgGate,
    QuditTGate,
    QuditXdgGate,
    QuditXGate,
    QuditZdgGate,
    QuditZGate,
)
from qiskit_qudits.utils.encoding import (
    decode_bitstring,
    decode_counts,
    parse_level_tokens,
    project_state,
)
from qiskit_qudits.utils.typeguards import is_integral, is_vector
from qiskit_qudits.utils.validation import validate_dim, validate_float_finite

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from qiskit.circuit.parametertable import ParameterView

    from qiskit_qudits.circuit.clbyte import ClByteSpecifier
    from qiskit_qudits.circuit.instruction import QuditInstructionSet
    from qiskit_qudits.circuit.qudit import QuditSpecifier
    from qiskit_qudits.gates.base.controlledgate import QuditControlledGate
    from qiskit_qudits.gates.base.gate import QuditGate
    from qiskit_qudits.utils.consts import FloatLike, IntLike, VectorLike
    from qiskit_qudits.utils.encoding import InvalidPolicy, Levels

#: Which rendering of the circuit to produce.
#:
#: * ``'ideal'`` - one wire per qudit, one classical wire per clbyte;
#: * ``'real'`` - the encoded circuit, qudit gates over qubit wires;
#: * ``'decomposed'`` - the encoded circuit, unrolled one level.
CircuitView: TypeAlias = Literal["ideal", "real", "decomposed"]

#: A register accepted by :meth:`QuditQuantumCircuit.add_register`.
QuditCircuitRegister: TypeAlias = QuditRegister | ClByteRegister

#: Level specification accepted by
#: :meth:`QuditQuantumCircuit.initialize_levels`.
LevelsSpecifier: TypeAlias = "IntLike | str | Sequence[IntLike]"


class _QuditGateFactory(Protocol):
    """Constructor shape of a parameter-free single-qudit gate."""

    def __call__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> QuditGate:
        """Instantiate the gate for a :math:`d`-dimensional qudit."""
        ...


class _QuditPhaseGateFactory(Protocol):
    """Constructor shape of a single-qudit gate taking one angle."""

    def __call__(
        self,
        dim: IntLike,
        theta: FloatLike,
        *,
        label: str | None = None,
    ) -> QuditGate:
        """Instantiate the gate for a :math:`d`-dimensional qudit."""
        ...


class _QuditControlledGateFactory(Protocol):
    """Constructor shape of a parameter-free controlled qudit gate."""

    def __call__(
        self,
        target_dim: IntLike,
        control_dims: VectorLike,
        *,
        label: str | None = None,
    ) -> QuditControlledGate:
        """Instantiate the gate for the given qudit dimensions."""
        ...


class _QuditControlledPhaseGateFactory(Protocol):
    """Constructor shape of a controlled qudit gate taking one angle."""

    def __call__(
        self,
        target_dim: IntLike,
        control_dims: VectorLike,
        theta: FloatLike,
        *,
        label: str | None = None,
    ) -> QuditControlledGate:
        """Instantiate the gate for the given qudit dimensions."""
        ...


def _is_not_directive(instruction: QuditCircuitInstruction) -> bool:
    """Return ``True`` for anything that is not a circuit directive."""
    return not getattr(instruction.operation, "_directive", False)


class QuditQuantumCircuit:
    r"""A quantum circuit expressed in terms of qudits.

    :class:`QuditQuantumCircuit` is the qudit analogue of
    :class:`qiskit.circuit.QuantumCircuit`. It does **not** subclass
    it: instead it *owns* an encoded
    :class:`~qiskit.circuit.QuantumCircuit` (available as
    :attr:`circuit`) in which every :math:`d`-level qudit is
    represented by :math:`\lceil \log_2 d \rceil` qubits. This keeps
    the qudit API honest (there is no ``h(0)`` that would silently
    address a single encoding qubit) while giving full Qiskit
    interoperability::

        from qiskit import transpile
        transpile(qudit_circuit.circuit, backend)

    Two complementary views are available:

    * the **ideal** view (:meth:`to_ideal_circuit`,
      ``draw(view='ideal')``) renders one wire per qudit and one
      classical wire per clbyte, as if the hardware were natively
      qudit-based;
    * the **real** view (:attr:`circuit`, ``draw(view='real')``)
      renders the encoded circuit, with each qudit gate as one box
      spanning its encoding qubits; ``draw(view='decomposed')``
      unrolls one level further.

    Conventions
    ===========

    Everything is little-endian, exactly like Qiskit:

    * qubit ``j`` of a qudit carries weight :math:`2^j`;
    * clbit ``j`` of a :class:`.ClByte` receives qubit ``j``;
    * in a counts key, clbit ``i`` sits at position ``len - 1 - i``,
      so :meth:`decode_counts` consumes the clbytes right-to-left;
    * level strings passed to :meth:`initialize_levels` are read the
      same way (rightmost token = first target qudit), while
      ``Sequence[int]`` arguments are in plain target order.

    Invariants
    ==========

    All mutation of the encoded circuit happens through this class, so
    :attr:`data` (the qudit-level log) and :attr:`circuit` (the
    encoded circuit) can never drift apart. Do not mutate
    :attr:`circuit` directly; use :meth:`to_qubit_circuit` when an
    independent copy is needed.

    Examples:
        .. code-block:: python

            from qiskit_qudits.circuit import QuditQuantumCircuit

            # 2 qutrits + 2 clbytes sized for a qutrit outcome
            qc = QuditQuantumCircuit(2, 2, dim=3)
            qc.initialize_levels("2 0")   # qudit 0 -> |0>, 1 -> |2>
            qc.h(0)
            qc.sumx(0, 1)                 # qudit CX: |j,k> -> |j,k+j>
            qc.measure([0, 1], [0, 1])
            print(qc.draw())              # ideal view
    """

    #: Prefix used when auto-generating circuit names.
    prefix: ClassVar[str] = "quditcircuit"
    _instances_counter: ClassVar[Iterator[int]] = itertools.count()

    _MAX_INT_REGISTERS: Final[int] = 2

    # ---------------------------------------------------------------- #
    # Construction
    # ---------------------------------------------------------------- #
    def __init__(
        self,
        *regs: QuditCircuitRegister | IntLike,
        dim: IntLike | None = None,
        name: str | None = None,
        global_phase: FloatLike = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        r"""Create a qudit circuit.

        Args:
            regs: Either registers (:class:`.QuditRegister` and/or
                :class:`.ClByteRegister`) or up to two integers. With
                integers, the first is the number of qudits and the
                second the number of clbytes; ``dim`` is then
                mandatory and the auto-created registers are named
                ``'qd'`` and ``'cb'`` (the qudit counterparts of
                Qiskit's ``'q'`` and ``'c'``).
            dim: Dimension :math:`d` used by the integer form.
            name: Circuit name; auto-generated when ``None``.
            global_phase: Global phase in radians, stored on the
                encoded circuit.
            metadata: Free-form user metadata, stored on the encoded
                circuit so that it survives transpilation.

        Raises:
            QuditCircuitError: If `name` is an empty string.

        Examples:
            .. code-block:: python

                QuditQuantumCircuit(3, dim=4)         # 3 ququarts
                QuditQuantumCircuit(3, 3, dim=4)      # + 3 clbytes
                QuditQuantumCircuit(QuditRegister(2, 3, "alice"))
        """
        if name is not None and not name:
            raise QuditCircuitError(
                "circuit name must be a non-empty string.",
            )
        self._name: str = (
            f"{self.prefix}-{next(self._instances_counter)}"
            if name is None
            else name
        )

        self._qudits: list[Qudit] = []
        self._qdregs: list[QuditRegister] = []
        self._clbytes: list[ClByte] = []
        self._cbregs: list[ClByteRegister] = []
        self._qudit_indices: dict[Qudit, int] = {}
        self._clbyte_indices: dict[ClByte, int] = {}
        self._register_names: set[str] = set()
        self._data: list[QuditCircuitInstruction] = []

        # The encoded circuit; its name, metadata and global phase
        # mirror ours.
        self._circuit: QuantumCircuit = QuantumCircuit(
            name=self._name,
            global_phase=validate_float_finite(global_phase),
            metadata={} if metadata is None else metadata,
        )

        self.add_register(
            *self._resolve_constructor_args(
                regs,
                validate_dim(dim) if dim else None,
            ),
        )

    @staticmethod
    def _resolve_constructor_args(
        regs: tuple[QuditCircuitRegister | IntLike, ...],
        dim: int | None,
    ) -> tuple[QuditCircuitRegister, ...]:
        """Turn the ``*regs`` constructor arguments into registers.

        Args:
            regs: Raw constructor arguments.
            dim: Dimension for the integer form.

        Returns:
            The registers to add to the circuit, in order.

        Raises:
            QuditCircuitError: See :meth:`__init__`.
        """
        if not regs:
            return ()

        if all(
            isinstance(reg, (QuditRegister, ClByteRegister)) for reg in regs
        ):
            # `cast`-free narrowing: the check above is exhaustive.
            return tuple(
                reg
                for reg in regs
                if isinstance(reg, (QuditRegister, ClByteRegister))
            )

        sizes: list[int] = []
        for reg in regs:
            if not is_integral(reg):
                raise QuditCircuitError(
                    "QuditQuantumCircuit arguments must be either registers "
                    f"or integers, got "
                    f"{[type(item).__name__ for item in regs]}.",
                )
            sizes.append(int(reg))
        if dim is None:
            raise QuditCircuitError(
                "the integer form of QuditQuantumCircuit requires "
                "`dim`, e.g. QuditQuantumCircuit(3, dim=4).",
            )
        if len(sizes) > QuditQuantumCircuit._MAX_INT_REGISTERS:
            raise QuditCircuitError(
                "expected at most 2 integer arguments (qudits, "
                f"clbytes), got {len(sizes)}.",
            )

        created: list[QuditCircuitRegister] = []
        if sizes[0] > 0:
            created.append(QuditRegister(sizes[0], dim, "qd"))
        if (
            len(sizes) == QuditQuantumCircuit._MAX_INT_REGISTERS
            and sizes[1] > 0
        ):
            created.append(ClByteRegister(sizes[1], dim, "cb"))
        return tuple(created)

    # ---------------------------------------------------------------- #
    # Metadata
    # ---------------------------------------------------------------- #
    @property
    def name(self) -> str:
        """Circuit name (shared with the encoded circuit)."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Rename the circuit and its encoded counterpart."""
        if not value:
            raise QuditCircuitError(
                "circuit name must be a non-empty string.",
            )
        self._name = value
        self._circuit.name = value

    @property
    def metadata(self) -> dict[str, object]:
        """User metadata, stored on the encoded circuit."""
        return self._circuit.metadata

    @metadata.setter
    def metadata(self, value: dict[str, object]) -> None:
        """Replace the user metadata."""
        self._circuit.metadata = value

    @property
    def global_phase(self) -> float:
        """Global phase in radians, stored on the encoded circuit."""
        return cast("float", self._circuit.global_phase)

    @global_phase.setter
    def global_phase(self, value: FloatLike) -> None:
        """Set the global phase in radians."""
        self._circuit.global_phase = validate_float_finite(value)

    # ---------------------------------------------------------------- #
    # Data objects
    # ---------------------------------------------------------------- #
    @property
    def circuit(self) -> QuantumCircuit:
        """The encoded :class:`~qiskit.circuit.QuantumCircuit`.

        This is the object to hand to the transpiler, a simulator or a
        primitive. It is returned *live* (no copy) for efficiency and
        must be treated as read-only: mutating it behind this class's
        back breaks the correspondence with :attr:`data`. Use
        :meth:`to_qubit_circuit` for an independent copy.
        """
        return self._circuit

    @property
    def data(self) -> tuple[QuditCircuitInstruction, ...]:
        """The qudit-level instructions, in application order."""
        return tuple(self._data)

    @property
    def qudits(self) -> tuple[Qudit, ...]:
        r"""All :class:`.Qudit`\ s, in the order they were added."""
        return tuple(self._qudits)

    @property
    def qdregs(self) -> tuple[QuditRegister, ...]:
        r"""All :class:`.QuditRegister`\ s, in insertion order."""
        return tuple(self._qdregs)

    @property
    def clbytes(self) -> tuple[ClByte, ...]:
        r"""All :class:`.ClByte`\ s, in the order they were added.

        The order coincides with increasing clbit index, which is what
        :meth:`decode_counts` relies on.
        """
        return tuple(self._clbytes)

    @property
    def cbregs(self) -> tuple[ClByteRegister, ...]:
        r"""All :class:`.ClByteRegister`\ s, in insertion order."""
        return tuple(self._cbregs)

    @property
    def qubits(self) -> list[Qubit]:
        """The encoding qubits (delegates to the encoded circuit)."""
        return self._circuit.qubits

    @property
    def clbits(self) -> list[Clbit]:
        """The encoding clbits (delegates to the encoded circuit)."""
        return self._circuit.clbits

    @property
    def qregs(self) -> list[QuantumRegister]:
        """The backing quantum registers of the encoded circuit."""
        return self._circuit.qregs

    @property
    def cregs(self) -> list[ClassicalRegister]:
        """The backing classical registers of the encoded circuit."""
        return self._circuit.cregs

    @property
    def num_qudits(self) -> int:
        """Number of qudits."""
        return len(self._qudits)

    @property
    def num_clbytes(self) -> int:
        """Number of clbytes."""
        return len(self._clbytes)

    @property
    def num_qubits(self) -> int:
        """Number of encoding qubits."""
        return self._circuit.num_qubits

    @property
    def num_clbits(self) -> int:
        """Number of encoding clbits."""
        return self._circuit.num_clbits

    @property
    def dims(self) -> tuple[int, ...]:
        r"""Dimension :math:`d` of each qudit, in qudit order."""
        return tuple(qudit.dim for qudit in self._qudits)

    @property
    def dim(self) -> int:
        r"""The common dimension :math:`d` of a homogeneous circuit.

        Raises:
            QuditCircuitError: If the circuit mixes dimensions; use
                :attr:`dims` instead.
        """
        unique = set(self.dims)
        if len(unique) > 1:
            raise QuditCircuitError(
                f"this circuit is heterogeneous ({sorted(unique)}); use "
                "`dims` instead of `dim`.",
            )
        return next(iter(unique), 0)

    @property
    def clbyte_widths(self) -> tuple[int, ...]:
        """Clbit width of each clbyte, in clbit-index order."""
        return tuple(clbyte.num_clbits for clbyte in self._clbytes)

    @property
    def clbyte_dims(self) -> tuple[int, ...]:
        """Dimension each clbyte is sized for, in clbit-index order."""
        return tuple(clbyte.dim for clbyte in self._clbytes)

    @property
    def parameters(self) -> ParameterView[Any]:
        """Compile-time parameters of the encoded circuit."""
        return self._circuit.parameters

    @property
    def num_parameters(self) -> int:
        """Number of compile-time parameters."""
        return self._circuit.num_parameters

    def __len__(self) -> int:
        """Return the number of qudit-level instructions."""
        return len(self._data)

    def __getitem__(self, key: IntLike) -> QuditCircuitInstruction:
        """Return the instruction at ``key``."""
        return self._data[int(key)]

    # ---------------------------------------------------------------- #
    # Adding data objects
    # ---------------------------------------------------------------- #
    def add_register(self, *regs: QuditCircuitRegister) -> None:
        """Add qudit and/or clbyte registers to the circuit.

        The register's backing quantum/classical register is added to
        the encoded circuit, so its name must be unique across *both*
        kinds.

        Args:
            regs: Registers to add.

        Raises:
            QuditCircuitError: On an unsupported type, a duplicate
                register, or a name clash.
        """
        for register in regs:
            if register.name in self._register_names:
                raise QuditCircuitError(
                    f"register name '{register.name}' already exists in "
                    "this circuit.",
                )

            if isinstance(register, QuditRegister):
                if any(existing is register for existing in self._qdregs):
                    raise QuditCircuitError(
                        f"register '{register.name}' is already in this "
                        "circuit.",
                    )
                self._circuit.add_register(register.qreg)
                self._qdregs.append(register)
                self._register_qudits(register.qudits)
            elif isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                register,
                ClByteRegister,
            ):
                if any(existing is register for existing in self._cbregs):
                    raise QuditCircuitError(
                        f"register '{register.name}' is already in this "
                        "circuit.",
                    )
                self._circuit.add_register(register.creg)
                self._cbregs.append(register)
                self._register_clbytes(register.clbytes)
            else:
                raise QuditCircuitError(
                    "expected a QuditRegister or a ClByteRegister, got "
                    f"{type(register).__name__}.",
                )
            self._register_names.add(register.name)

    def add_qudits(self, qudits: Sequence[Qudit]) -> None:
        r"""Add loose (registerless) qudits.

        Args:
            qudits: The qudits to add. Their encoding qubits are added
                to the encoded circuit as loose
                :class:`~qiskit.circuit.Qubit`\ s.

        Raises:
            QuditCircuitError: If a qudit is already present.
        """
        for qudit in qudits:
            if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                qudit,
                Qudit,
            ):
                raise QuditCircuitError(
                    f"expected a Qudit, got {type(qudit).__name__}.",
                )
            self._circuit.add_bits(qudit.qubits)
        self._register_qudits(qudits)

    def add_clbytes(self, clbytes: Sequence[ClByte]) -> None:
        """Add loose (registerless) clbytes.

        Args:
            clbytes: The clbytes to add. Their clbits are added to the
                encoded circuit as loose clbits.

        Raises:
            QuditCircuitError: If a clbyte is already present.
        """
        for clbyte in clbytes:
            if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
                clbyte,
                ClByte,
            ):
                raise QuditCircuitError(
                    f"expected a ClByte, got {type(clbyte).__name__}.",
                )
            self._circuit.add_bits(clbyte.clbits)
        self._register_clbytes(clbytes)

    def _register_qudits(self, qudits: Sequence[Qudit]) -> None:
        """Record ``qudits`` in the circuit's index tables.

        Args:
            qudits: The qudits to record, in order.

        Raises:
            QuditCircuitError: If a qudit is already present.
        """
        for qudit in qudits:
            if qudit in self._qudit_indices:
                raise QuditCircuitError(
                    f"{qudit!r} is already in this circuit.",
                )
            self._qudit_indices[qudit] = len(self._qudits)
            self._qudits.append(qudit)

    def _register_clbytes(self, clbytes: Sequence[ClByte]) -> None:
        """Record ``clbytes`` in the circuit's index tables.

        Args:
            clbytes: The clbytes to record, in order.

        Raises:
            QuditCircuitError: If a clbyte is already present.
        """
        for clbyte in clbytes:
            if clbyte in self._clbyte_indices:
                raise QuditCircuitError(
                    f"{clbyte!r} is already in this circuit.",
                )
            self._clbyte_indices[clbyte] = len(self._clbytes)
            self._clbytes.append(clbyte)

    def has_register(self, register: QuditCircuitRegister) -> bool:
        """Return whether ``register`` belongs to this circuit.

        Args:
            register: The register to look for.

        Returns:
            ``True`` if the very same register object was added.
        """
        if isinstance(register, QuditRegister):
            return any(existing is register for existing in self._qdregs)
        return any(existing is register for existing in self._cbregs)

    def find_qudit(self, qudit: Qudit) -> int:
        """Return the circuit-wide index of ``qudit``.

        Args:
            qudit: The qudit to locate.

        Returns:
            Its index in :attr:`qudits`.

        Raises:
            QuditCircuitError: If the qudit is not in this circuit.
        """
        try:
            return self._qudit_indices[qudit]
        except KeyError as exc:
            raise QuditCircuitError(
                f"{qudit!r} is not in this circuit.",
            ) from exc

    def find_clbyte(self, clbyte: ClByte) -> int:
        """Return the circuit-wide index of ``clbyte``.

        Args:
            clbyte: The clbyte to locate.

        Returns:
            Its index in :attr:`clbytes`.

        Raises:
            QuditCircuitError: If the clbyte is not in this circuit.
        """
        try:
            return self._clbyte_indices[clbyte]
        except KeyError as exc:
            raise QuditCircuitError(
                f"{clbyte!r} is not in this circuit.",
            ) from exc

    # ---------------------------------------------------------------- #
    # Argument resolution
    # ---------------------------------------------------------------- #
    def _qudit_argument_conversion(
        self,
        specifier: QuditSpecifier,
    ) -> list[Qudit]:
        r"""Resolve a :data:`.QuditSpecifier` into concrete qudits.

        Args:
            specifier: A qudit, a register, an index, a slice, or a
                sequence of qudits/indices.

        Returns:
            The resolved qudits, in the order implied by the
            specifier.

        Raises:
            QuditCircuitError: On an unsupported specifier, an
                out-of-range index, or a foreign qudit.
        """
        if isinstance(specifier, Qudit):
            self.find_qudit(specifier)  # membership check
            return [specifier]
        if isinstance(specifier, QuditRegister):
            if not self.has_register(specifier):
                raise QuditCircuitError(
                    f"register '{specifier.name}' is not in this circuit.",
                )
            return list(specifier)
        if is_integral(specifier):
            index = int(specifier)
            if not -self.num_qudits <= index < self.num_qudits:
                raise QuditCircuitError(
                    f"qudit index {index} is out of range for a circuit "
                    f"with {self.num_qudits} qudit(s).",
                )
            return [self._qudits[index]]
        if isinstance(specifier, slice):
            return list(self._qudits[specifier])
        if isinstance(specifier, str):
            raise QuditCircuitError(
                "strings are not valid qudit specifiers.",
            )
        if isinstance(specifier, Sequence) or is_vector(specifier):
            resolved: list[Qudit] = []
            for item in specifier:
                if isinstance(item, Qudit) or is_integral(item):
                    resolved.extend(self._qudit_argument_conversion(item))
                else:
                    raise QuditCircuitError(
                        f"invalid qudit specifier element {item!r}.",
                    )
            return resolved
        raise QuditCircuitError(f"invalid qudit specifier {specifier!r}.")

    def _clbyte_argument_conversion(
        self,
        specifier: ClByteSpecifier,
    ) -> list[ClByte]:
        """Resolve a :data:`.ClByteSpecifier` into concrete clbytes.

        Args:
            specifier: A clbyte, a register, an index, a slice, or a
                sequence of clbytes/indices.

        Returns:
            The resolved clbytes, in the order implied by the
            specifier.

        Raises:
            QuditCircuitError: On an unsupported specifier, an
                out-of-range index, or a foreign clbyte.
        """
        if isinstance(specifier, ClByte):
            self.find_clbyte(specifier)  # membership check
            return [specifier]
        if isinstance(specifier, ClByteRegister):
            if not self.has_register(specifier):
                raise QuditCircuitError(
                    f"register '{specifier.name}' is not in this circuit.",
                )
            return list(specifier)
        if is_integral(specifier):
            index = int(specifier)
            if not -self.num_clbytes <= index < self.num_clbytes:
                raise QuditCircuitError(
                    f"clbyte index {index} is out of range for a circuit "
                    f"with {self.num_clbytes} clbyte(s).",
                )
            return [self._clbytes[index]]
        if isinstance(specifier, slice):
            return list(self._clbytes[specifier])
        if isinstance(specifier, str):
            raise QuditCircuitError(
                "strings are not valid clbyte specifiers.",
            )
        if isinstance(specifier, Sequence):
            resolved: list[ClByte] = []
            for item in specifier:
                if isinstance(item, ClByte) or is_integral(item):
                    resolved.extend(self._clbyte_argument_conversion(item))
                else:
                    raise QuditCircuitError(
                        f"invalid clbyte specifier element {item!r}.",
                    )
            return resolved
        raise QuditCircuitError(f"invalid clbyte specifier {specifier!r}.")

    @staticmethod
    def _check_duplicates(objects: Sequence[object], kind: str) -> None:
        """Raise if ``objects`` contains duplicates.

        Args:
            objects: Resolved operands.
            kind: Human-readable operand kind for the error message.

        Raises:
            QuditCircuitError: On duplicates.
        """
        if len(objects) != len({id(obj) for obj in objects}):
            raise QuditCircuitError(f"duplicate {kind} arguments.")

    # ---------------------------------------------------------------- #
    # Appending operations
    # ---------------------------------------------------------------- #
    def append(
        self,
        operation: Instruction,
        qudits: QuditSpecifier | None = None,
        clbytes: ClByteSpecifier | None = None,
        *,
        copy: bool = True,
    ) -> QuditCircuitInstruction:
        r"""Append one operation to the circuit.

        Unlike :meth:`qiskit.circuit.QuantumCircuit.append`, this
        method performs **no broadcasting**: the resolved operands
        must match the operation's arity exactly. Broadcasting is
        provided by the convenience helpers (:meth:`x`,
        :meth:`measure`, ...).

        Args:
            operation: A
                :class:`~qiskit_qudits.gates.base.gate.QuditGate`, a
                :class:`.QuditDirective`, or any
                :class:`~qiskit.circuit.Instruction` whose qubit count
                matches the encoding width of the targets (useful for
                raw unitaries).
            qudits: The target qudits.
            clbytes: The target clbytes.
            copy: Copy the operation when it carries parameters, so
                later mutation of the argument cannot affect this
                circuit. Mirrors Qiskit's ``append(copy=...)``.

        Returns:
            The recorded :class:`.QuditCircuitInstruction`.

        Raises:
            QuditCircuitError: On an arity, dimension or width
                mismatch, or on duplicate operands.
        """
        if not isinstance(  # pyright: ignore[reportUnnecessaryIsInstance]
            operation,
            Instruction,
        ):
            raise QuditCircuitError(
                f"expected an Instruction, got {type(operation).__name__}.",
            )

        qudit_targets = tuple(
            self._qudit_argument_conversion(
                () if qudits is None else qudits,
            ),
        )
        clbyte_targets = tuple(
            self._clbyte_argument_conversion(
                () if clbytes is None else clbytes,
            ),
        )
        self._check_duplicates(qudit_targets, "qudit")
        self._check_duplicates(clbyte_targets, "clbyte")

        self._validate_operands(operation, qudit_targets, clbyte_targets)

        if copy and operation.params:
            operation = operation.copy()

        # Mutate the encoded circuit first: if the expansion fails,
        # nothing is recorded in `_data` and the two views stay
        # consistent.
        self._apply_operation(operation, qudit_targets, clbyte_targets)

        instruction = QuditCircuitInstruction(
            operation,
            qudit_targets,
            clbyte_targets,
        )
        self._data.append(instruction)
        return instruction

    def _validate_operands(
        self,
        operation: Instruction,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Check that ``operation`` matches its operands.

        Args:
            operation: The operation about to be appended.
            qudits: Resolved target qudits.
            clbytes: Resolved target clbytes.

        Raises:
            QuditCircuitError: On any mismatch.
        """
        dims = self._operation_dims(operation)
        if dims is not None:
            if len(dims) != len(qudits):
                raise QuditCircuitError(
                    f"'{operation.name}' acts on {len(dims)} qudit(s), "
                    f"got {len(qudits)}.",
                )
            for position, (dim, qudit) in enumerate(
                zip(dims, qudits, strict=True),
            ):
                if dim != qudit.dim:
                    raise QuditCircuitError(
                        f"'{operation.name}' expects a {dim}-level qudit "
                        f"at position {position}, got {qudit.dim} levels.",
                    )

        expected_qubits = sum(qudit.num_qubits for qudit in qudits)
        if operation.num_qubits != expected_qubits:
            raise QuditCircuitError(
                f"'{operation.name}' acts on {operation.num_qubits} "
                f"qubit(s) but the target qudits encode into "
                f"{expected_qubits} qubit(s).",
            )

        if isinstance(operation, QuditDirective):
            if operation.num_clbytes != len(clbytes):
                raise QuditCircuitError(
                    f"'{operation.name}' acts on {operation.num_clbytes} "
                    f"clbyte(s), got {len(clbytes)}.",
                )
        else:
            expected_clbits = sum(clbyte.num_clbits for clbyte in clbytes)
            if operation.num_clbits != expected_clbits:
                raise QuditCircuitError(
                    f"'{operation.name}' acts on {operation.num_clbits} "
                    f"clbit(s) but the target clbytes provide "
                    f"{expected_clbits}.",
                )

    @staticmethod
    def _operation_dims(operation: Instruction) -> tuple[int, ...] | None:
        """Extract the qudit dimensions an operation was built for.

        Args:
            operation: The operation to inspect.

        Returns:
            One dimension per operand qudit, or ``None`` when the
            operation carries no qudit metadata (e.g. a raw Qiskit
            gate appended onto a qudit's encoding qubits).

        Note:
            Multi-qudit and controlled gates expose a ``dims`` tuple
            (control qudits first, target qudit last for
            :class:`~qiskit_qudits.gates.base.controlledgate.QuditControlledGate`),
            which takes precedence over the single-qudit ``dim``
            attribute.
        """
        if isinstance(operation, QuditDirective):
            return operation.dims

        candidate: object = getattr(operation, "dims", None)
        if isinstance(candidate, tuple) and candidate:
            collected: list[int] = []
            for item in cast("tuple[object, ...]", candidate):
                if not is_integral(item):
                    break
                collected.append(int(item))
            else:
                return tuple(collected)

        candidate = getattr(operation, "dim", None)
        if is_integral(candidate):
            return (int(candidate),)
        return None

    def _apply_operation(
        self,
        operation: Instruction,
        qudits: Sequence[Qudit],
        clbytes: Sequence[ClByte],
    ) -> None:
        """Expand one qudit operation onto the encoded circuit.

        Args:
            operation: The operation to expand.
            qudits: Resolved target qudits.
            clbytes: Resolved target clbytes.
        """
        if isinstance(operation, QuditDirective):
            operation.apply(self._circuit, qudits, clbytes)
            return

        qubits = [qubit for qudit in qudits for qubit in qudit.qubits]
        clbits = [clbit for clbyte in clbytes for clbit in clbyte.clbits]
        self._circuit.append(operation, qubits, clbits, copy=False)

    # ---------------------------------------------------------------- #
    # Gate helpers
    # ---------------------------------------------------------------- #
    def _append_qudit_gate(
        self,
        factory: _QuditGateFactory,
        target: QuditSpecifier,
        label: str | None,
    ) -> QuditInstructionSet:
        """Broadcast a parameter-free single-qudit gate over targets.

        The gate is instantiated **per qudit** with that qudit's own
        dimension, which is what makes mixed-dimension circuits work.

        Args:
            factory: The gate class.
            target: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            One instruction per target qudit.
        """
        return tuple(
            self.append(
                factory(qudit.dim, label=label),
                (qudit,),
                copy=False,
            )
            for qudit in self._qudit_argument_conversion(target)
        )

    def _append_qudit_phase_gate(
        self,
        factory: _QuditPhaseGateFactory,
        theta: FloatLike,
        target: QuditSpecifier,
        label: str | None,
    ) -> QuditInstructionSet:
        """Broadcast a one-angle single-qudit gate over targets.

        Args:
            factory: The gate class.
            theta: Rotation angle in radians (validated by the gate).
            target: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            One instruction per target qudit.
        """
        return tuple(
            self.append(
                factory(qudit.dim, float(theta), label=label),
                (qudit,),
                copy=False,
            )
            for qudit in self._qudit_argument_conversion(target)
        )

    def i(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        """Apply the qudit identity gate.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditIGate, qudit, label)

    def h(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the qudit Hadamard gate.

        See :class:`~qiskit_qudits.gates.QuditHGate` for the exact
        definition on a :math:`d`-level qudit.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditHGate, qudit, label)

    def hdg(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the inverse qudit Hadamard gate.

        See :class:`~qiskit_qudits.gates.QuditHdgGate` for the exact
        definition on a :math:`d`-level qudit.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditHdgGate, qudit, label)

    def k(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        """Apply the qudit K gate.

        See :class:`~qiskit_qudits.gates.QuditKGate` for the
        definition.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditKGate, qudit, label)

    def not_(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        """Apply the qudit NOT gate.

        See :class:`~qiskit_qudits.gates.QuditNOTGate` for the
        definition.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.

        Note:
            The trailing underscore is required because ``not`` is a
            Python keyword; :meth:`qnot` is provided as an alias.
        """
        return self._append_qudit_gate(QuditNOTGate, qudit, label)

    #: Readable alias of :meth:`not_`.
    qnot = not_

    def x(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the qudit shift gate :math:`X_d`.

        :math:`X_d \lvert k \rangle = \lvert (k+1) \bmod d \rangle`.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditXGate, qudit, label)

    def xdg(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the inverse shift gate :math:`X_d^\dagger`.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditXdgGate, qudit, label)

    def z(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the qudit clock gate :math:`Z_d`.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditZGate, qudit, label)

    def zdg(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the inversequdit clock gate :math:`Z_d`.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditZdgGate, qudit, label)

    def s(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        """Apply the qudit S gate.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditSGate, qudit, label)

    def sdg(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        """Apply the inverse qudit S gate.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditSdgGate, qudit, label)

    def t(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        """Apply the qudit T gate.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditTGate, qudit, label)

    def tdg(
        self,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        """Apply the qudit T gate.

        Args:
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_gate(QuditTdgGate, qudit, label)

    def p(
        self,
        theta: FloatLike,
        qudit: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the qudit phase gate.

        The argument order mirrors
        :meth:`qiskit.circuit.QuantumCircuit.p`: the angle comes
        first.

        Args:
            theta: Rotation angle in radians. Must be a concrete real
                number; symbolic :class:`~qiskit.circuit.Parameter`\ s
                are not yet supported by the qudit phase gates.
            qudit: The qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_qudit_phase_gate(
            QuditPGate,
            theta,
            qudit,
            label,
        )

    # ---------------------------------------------------------------- #
    # Controlled gate helpers
    # ---------------------------------------------------------------- #
    def _resolve_control_targets(
        self,
        control: QuditSpecifier,
        target: QuditSpecifier,
    ) -> tuple[tuple[Qudit, ...], tuple[Qudit, ...]]:
        r"""Resolve the operands of a controlled qudit gate.

        Args:
            control: The control qudit(s), in register order
                :math:`(c_0, c_1, \\dots, c_{m-1})`.
            target: The target qudit(s) the gate is broadcast over.

        Returns:
            A ``(controls, targets)`` pair of resolved qudits.

        Raises:
            QuditCircuitError: If either operand resolves to no qudit,
                or if a control qudit is repeated.
        """
        controls = tuple(self._qudit_argument_conversion(control))
        targets = tuple(self._qudit_argument_conversion(target))
        if not controls:
            raise QuditCircuitError(
                "a controlled qudit gate needs at least one control qudit.",
            )
        if not targets:
            raise QuditCircuitError(
                "a controlled qudit gate needs at least one target qudit.",
            )
        self._check_duplicates(controls, "control qudit")
        return controls, targets

    def _append_controlled_qudit_gate(
        self,
        factory: _QuditControlledGateFactory,
        control: QuditSpecifier,
        target: QuditSpecifier,
        label: str | None,
    ) -> QuditInstructionSet:
        r"""Broadcast a parameter-free controlled gate over targets.

        The gate is instantiated **per target qudit** with that
        qudit's own dimension and the dimensions of the shared
        controls, which is what makes mixed-dimension circuits work.
        Operands are passed in the gate's register order,
        ``(c_0, ..., c_{m-1}, t)``.

        Args:
            factory: The gate class.
            control: The control qudit(s), in register order.
            target: The target qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            One instruction per target qudit.
        """
        controls, targets = self._resolve_control_targets(control, target)
        control_dims = [qudit.dim for qudit in controls]
        return tuple(
            self.append(
                factory(qudit.dim, control_dims, label=label),
                (*controls, qudit),
                copy=False,
            )
            for qudit in targets
        )

    def _append_controlled_qudit_phase_gate(
        self,
        factory: _QuditControlledPhaseGateFactory,
        theta: FloatLike,
        control: QuditSpecifier,
        target: QuditSpecifier,
        label: str | None,
    ) -> QuditInstructionSet:
        """Broadcast a one-angle controlled gate over targets.

        Args:
            factory: The gate class.
            theta: Rotation angle in radians (validated by the gate).
            control: The control qudit(s), in register order.
            target: The target qudit(s) to apply the gate to.
            label: Optional display label.

        Returns:
            One instruction per target qudit.
        """
        controls, targets = self._resolve_control_targets(control, target)
        control_dims = [qudit.dim for qudit in controls]
        return tuple(
            self.append(
                factory(
                    qudit.dim,
                    control_dims,
                    float(theta),
                    label=label,
                ),
                (*controls, qudit),
                copy=False,
            )
            for qudit in targets
        )

    def sumx(
        self,
        control: QuditSpecifier,
        target: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the qudit sum gate :math:`SUMX_d`.

        :math:`SUMX_d \lvert j_0 \rangle \cdots \lvert j_{m-1} \rangle
        \lvert k \rangle = \lvert j_0 \rangle \cdots
        \lvert j_{m-1} \rangle \lvert (k + j_0 + \cdots + j_{m-1})
        \bmod d_t \rangle`, the qudit generalisation of ``cx``.

        Args:
            control: The control qudit(s), in register order
                :math:`(c_0, c_1, \dots, c_{m-1})`.
            target: The target qudit(s). The gate is applied once per
                target, every instance sharing the same controls.
            label: Optional display label.

        Returns:
            A handle to the instructions created.

        Examples:
            .. code-block:: python

                qc.sumx(0, 1)          # single control
                qc.sumx([0, 1], 2)     # two controls, one target
        """
        return self._append_controlled_qudit_gate(
            QuditSUMXGate,
            control,
            target,
            label,
        )

    def sumxdg(
        self,
        control: QuditSpecifier,
        target: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the inverse qudit sum gate :math:`SUMX_d^\dagger`.

        :math:`SUMX_d^\dagger` shifts the target qudit *down* by the
        sum of the control qudit values.

        Args:
            control: The control qudit(s), in register order.
            target: The target qudit(s). The gate is applied once per
                target, every instance sharing the same controls.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_controlled_qudit_gate(
            QuditSUMXdgGate,
            control,
            target,
            label,
        )

    def sump(
        self,
        theta: FloatLike,
        control: QuditSpecifier,
        target: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Apply the qudit controlled-phase gate :math:`SUMP_d`.

        The argument order mirrors :meth:`p` and
        :meth:`qiskit.circuit.QuantumCircuit.cp`: the angle comes
        first. The gate imprints the phase
        :math:`\omega^{(j_0 + \cdots + j_{m-1}) k \theta / \pi}` with
        :math:`\omega = e^{i 2\pi / d_t}`.

        Args:
            theta: Rotation angle in radians. Must be a concrete real
                number; symbolic :class:`~qiskit.circuit.Parameter`\ s
                are not yet supported by the qudit phase gates.
            control: The control qudit(s), in register order.
            target: The target qudit(s). The gate is applied once per
                target, every instance sharing the same controls.
            label: Optional display label.

        Returns:
            A handle to the instructions created.
        """
        return self._append_controlled_qudit_phase_gate(
            QuditSUMPGate,
            theta,
            control,
            target,
            label,
        )

    # ---------------------------------------------------------------- #
    # Multi-qudit gate helpers
    # ---------------------------------------------------------------- #
    @staticmethod
    def _common_dim(qudits: Sequence[Qudit], name: str) -> int:
        r"""Return the dimension shared by ``qudits``.

        Args:
            qudits: The resolved operands, which must be non-empty and
                homogeneous.
            name: Operation name, used in the error message.

        Returns:
            The common dimension :math:`d`.

        Raises:
            QuditCircuitError: If ``qudits`` is empty or mixes
                dimensions.
        """
        dims = tuple(qudit.dim for qudit in qudits)
        if not dims:
            raise QuditCircuitError(f"'{name}' needs at least one qudit.")
        if len(set(dims)) > 1:
            raise QuditCircuitError(
                f"'{name}' needs qudits of equal dimension, got {dims}.",
            )
        return dims[0]

    def swap(
        self,
        qudit1: QuditSpecifier,
        qudit2: QuditSpecifier,
        *,
        label: str | None = None,
    ) -> QuditInstructionSet:
        r"""Swap the states of qudit pairs.

        :math:`SWAP_d \lvert j \rangle \lvert k \rangle =
        \lvert k \rangle \lvert j \rangle`. The two qudits of a pair
        must have the same dimension.

        Args:
            qudit1: The first qudit(s) of each pair.
            qudit2: The second qudit(s) of each pair. Must have the
                same length as ``qudit1`` (one-to-one, as in Qiskit).
            label: Optional display label.

        Returns:
            One instruction per swapped pair.

        Raises:
            QuditCircuitError: If the operand counts differ or a pair
                mixes dimensions.

        Examples:
            .. code-block:: python

                qc.swap(0, 1)              # one pair
                qc.swap([0, 1], [2, 3])    # 0 <-> 2 and 1 <-> 3
        """
        first = self._qudit_argument_conversion(qudit1)
        second = self._qudit_argument_conversion(qudit2)
        if len(first) != len(second):
            raise QuditCircuitError(
                "swap needs one qudit per qudit, got "
                f"{len(first)} and {len(second)} qudit(s).",
            )
        return tuple(
            self.append(
                QuditSWAPGate(
                    self._common_dim(
                        (left, right),
                        QuditSWAPGate.gate_name,
                    ),
                    label=label,
                ),
                (left, right),
                copy=False,
            )
            for left, right in zip(first, second, strict=True)
        )

    def qft(
        self,
        qudits: QuditSpecifier | None = None,
        *,
        label: str | None = None,
    ) -> QuditCircuitInstruction:
        r"""Apply the quantum Fourier transform over a qudit register.

        The transform acts on the :math:`d^n`-dimensional subspace
        spanned by the targets, which must all share the same
        dimension :math:`d`:

        .. math::

            QFT \lvert x \rangle = \frac{1}{\sqrt{d^n}}
                \sum_{y=0}^{d^n - 1} \omega^{xy} \lvert y \rangle,
                \quad \omega = e^{i 2\pi / d^n}

        where :math:`x = \sum_i x_i d^i` is read in **target order**:
        the first target holds the least-significant digit, exactly
        like the encoding qubits inside a single qudit.

        Args:
            qudits: The target qudit(s), in transform order. Defaults
                to every qudit of the circuit, in order.
            label: Optional display label.

        Returns:
            The recorded instruction.

        Raises:
            QuditCircuitError: If fewer than two qudits are targeted
                or they do not all share the same dimension.

        Examples:
            .. code-block:: python

                qc = QuditQuantumCircuit(3, dim=4)
                qc.qft()                 # over the whole register
                qc.qft([0, 1])           # over a sub-register
        """
        targets = tuple(
            (
                self._qudits
                if qudits is None
                else self._qudit_argument_conversion(qudits)
            ),
        )
        if len(targets) < QuditQFTGate.MIN_QUDITS:
            raise QuditCircuitError(
                f"qft needs at least {QuditQFTGate.MIN_QUDITS} qudits, "
                f"got {len(targets)}.",
            )
        self._check_duplicates(targets, "qudit")

        return self.append(
            QuditQFTGate(
                len(targets),
                self._common_dim(targets, QuditQFTGate.gate_name),
                label=label,
            ),
            targets,
            copy=False,
        )

    # ---------------------------------------------------------------- #
    # Directives
    # ---------------------------------------------------------------- #
    def barrier(
        self,
        *qudits: QuditSpecifier,
        label: str | None = None,
    ) -> QuditCircuitInstruction:
        """Apply a barrier across the given qudits.

        Args:
            qudits: The qudits to include. When empty, every qudit of
                the circuit is included (Qiskit parity).
            label: Optional barrier label.

        Returns:
            The recorded instruction.
        """
        if qudits:
            # A dict preserves order while removing duplicates.
            unique: dict[int, Qudit] = {}
            for specifier in qudits:
                for qudit in self._qudit_argument_conversion(specifier):
                    unique.setdefault(id(qudit), qudit)
            targets = tuple(unique.values())
        else:
            targets = tuple(self._qudits)

        directive = QuditBarrier(
            [qudit.dim for qudit in targets],
            label=label,
        )
        return self.append(directive, targets, copy=False)

    def reset(self, qudit: QuditSpecifier) -> QuditInstructionSet:
        r"""Reset qudits to :math:`\lvert 0 \rangle`.

        Args:
            qudit: The qudit(s) to reset.

        Returns:
            One instruction per reset qudit.
        """
        return tuple(
            self.append(QuditReset([target.dim]), (target,), copy=False)
            for target in self._qudit_argument_conversion(qudit)
        )

    def measure(
        self,
        qudit: QuditSpecifier,
        clbyte: ClByteSpecifier,
    ) -> QuditInstructionSet:
        """Measure qudits into clbytes in the computational basis.

        Qubit ``j`` of each qudit is measured into clbit ``j`` of the
        paired clbyte, so the stored bits are the little-endian binary
        expansion of the measured level. Use :meth:`decode_counts` (or
        :func:`~qiskit_qudits.utils.encoding.decode_counts`) to turn
        raw bit-strings back into levels.

        Args:
            qudit: The qudit(s) to measure.
            clbyte: The clbyte(s) to store the outcome(s) in. Must
                have the same length as ``qudit`` (one-to-one, as in
                Qiskit).

        Returns:
            One instruction per measured qudit.

        Raises:
            QuditCircuitError: If the operand counts differ or a
                clbyte is too narrow for its qudit.

        Examples:
            .. code-block:: python

                qc.measure([0, 1], [0, 1])   # qudit i -> clbyte i
        """
        qudit_targets = self._qudit_argument_conversion(qudit)
        clbyte_targets = self._clbyte_argument_conversion(clbyte)
        if len(qudit_targets) != len(clbyte_targets):
            raise QuditCircuitError(
                "measure needs one clbyte per qudit, got "
                f"{len(qudit_targets)} qudit(s) and "
                f"{len(clbyte_targets)} clbyte(s).",
            )
        return tuple(
            self.append(
                QuditMeasure([qudit_target.dim]),
                (qudit_target,),
                (clbyte_target,),
                copy=False,
            )
            for qudit_target, clbyte_target in zip(
                qudit_targets,
                clbyte_targets,
                strict=True,
            )
        )

    def measure_all(
        self,
        *,
        inplace: bool = True,
        add_bytes: bool = True,
    ) -> QuditQuantumCircuit | None:
        """Measure every qudit, adding a ``'meas'`` register.

        Args:
            inplace: Modify this circuit (default) or return a new
                one.
            add_bytes: Create a new :class:`.ClByteRegister` sized for
                the circuit's dimensions (heterogeneous circuits are
                handled via :meth:`.ClByteRegister.from_dims`). When
                ``False``, the existing clbytes are used, clbyte ``i``
                receiving qudit ``i``.

        Returns:
            ``None`` when ``inplace=True``, otherwise the new circuit.

        Raises:
            QuditCircuitError: If ``add_bytes=False`` and there are
                fewer clbytes than qudits.
        """
        circuit = self if inplace else self.copy()

        if add_bytes:
            register = ClByteRegister.from_dims(
                circuit.dims,
                circuit._unique_register_name("meas"),  # noqa: SLF001
            )
            circuit.add_register(register)
            targets = list(register)
        else:
            if circuit.num_clbytes < circuit.num_qudits:
                raise QuditCircuitError(
                    "the number of clbytes must be at least the number "
                    "of qudits.",
                )
            targets = list(circuit.clbytes[: circuit.num_qudits])

        circuit.barrier()
        circuit.measure(list(circuit.qudits), targets)
        return None if inplace else circuit

    def _unique_register_name(self, base: str) -> str:
        """Return ``base``, suffixed if needed, so it is unused.

        Args:
            base: Preferred register name.

        Returns:
            A name that does not clash with an existing register.
        """
        if base not in self._register_names:
            return base
        for suffix in itertools.count():
            candidate = f"{base}{suffix}"
            if candidate not in self._register_names:
                return candidate
        raise AssertionError("unreachable")  # pragma: no cover

    def initialize_levels(
        self,
        levels: LevelsSpecifier,
        qudits: QuditSpecifier | None = None,
    ) -> QuditCircuitInstruction:
        """Initialise qudits to computational basis states.

        Basis states are specified **by level**, never by a
        concatenated bit-string, because ``'1111'`` would be ambiguous
        the moment a qudit has more than two levels.

        Args:
            levels: One of

                * an ``int`` - the level of a single target qudit;
                * a **whitespace-separated string** - one token per
                  target qudit, read with Qiskit's ordering, i.e. the
                  *rightmost* token belongs to the *first*
                  (lowest-index) target. Tokens are parsed with
                  ``int(token, 0)``, so ``'0b101'`` and ``'0x1f'``
                  work too;
                * a ``Sequence[int]`` - one level per target qudit, in
                  **target order** (not reversed).

            qudits: The target qudit(s). Defaults to every qudit of
                the circuit, in order.

        Returns:
            The recorded instruction.

        Raises:
            QuditCircuitError: On an operand-count mismatch, an
                unseparated multi-qudit string, or an out-of-range
                level.

        Examples:
            .. code-block:: python

                qc = QuditQuantumCircuit(3, dim=16)
                # qudit 0 -> |0>, qudit 1 -> |3>, qudit 2 -> |11>
                qc.initialize_levels("11 3 0")
                # identical, in target order:
                qc.initialize_levels([0, 3, 11])
        """
        targets = tuple(
            (
                self._qudits
                if qudits is None
                else self._qudit_argument_conversion(qudits)
            ),
        )
        if not targets:
            raise QuditCircuitError(
                "initialize_levels needs at least one qudit.",
            )

        values: tuple[int, ...]
        if isinstance(levels, str):
            values = parse_level_tokens(levels, len(targets))
        elif is_integral(levels):
            if len(targets) != 1:
                raise QuditCircuitError(
                    "a bare integer level is only allowed for a single "
                    f"target qudit, got {len(targets)}. Pass a "
                    "whitespace-separated string or a sequence of "
                    "levels instead.",
                )
            values = (int(levels),)
        elif isinstance(levels, Sequence):
            collected: list[int] = []
            for level in levels:
                if not is_integral(level):
                    raise QuditCircuitError(
                        f"non-integer level in {levels!r}.",
                    )
                collected.append(int(level))
            values = tuple(collected)
        else:
            raise QuditCircuitError(
                f"invalid level specification {levels!r}.",
            )

        directive = QuditInitializeLevels(
            [qudit.dim for qudit in targets],
            values,
        )
        return self.append(directive, targets, copy=False)

    def initialize(
        self,
        state: LevelsSpecifier | VectorLike,
        qudits: QuditSpecifier | None = None,
    ) -> QuditCircuitInstruction:
        r"""Initialise qudits to an arbitrary state.

        The dispatch mirrors
        :meth:`qiskit.circuit.QuantumCircuit.initialize`: *strings and
        integers are labels*, *sequences are amplitudes*.

        Args:
            state: One of

                * a ``str`` or ``int`` - forwarded to
                  :meth:`initialize_levels`;
                * a :class:`~qiskit.quantum_info.Statevector` or a 1-D
                  array of :math:`\prod_i d_i` amplitudes over the
                  **logical qudit space** (first target = least
                  significant, Qiskit-style);
                * a 1-D array of :math:`2^N` amplitudes in the
                  **already-encoded** space; it is projected back onto
                  the qudit subspace and it is an error for it to
                  carry amplitude on an invalid basis state.

            qudits: The target qudit(s). Defaults to every qudit.

        Returns:
            The recorded instruction.

        Raises:
            QuditCircuitError: On a length mismatch, a non-normalised
                vector, or amplitude outside the qudit subspace.

        Note:
            Like Qiskit's, this operation is **not unitary**: the
            encoding qubits are reset first, so it cannot be inverted.

        Examples:
            .. code-block:: python

                import numpy as np

                qc = QuditQuantumCircuit(1, dim=3)
                # uniform qutrit state
                qc.initialize(np.ones(3) / np.sqrt(3))
        """
        if isinstance(state, str) or is_integral(state):
            return self.initialize_levels(state, qudits)
        if not is_vector(state):
            raise QuditCircuitError(
                f"invalid state specification {state!r}.",
            )

        targets = tuple(
            (
                self._qudits
                if qudits is None
                else self._qudit_argument_conversion(qudits)
            ),
        )
        if not targets:
            raise QuditCircuitError("initialize needs at least one qudit.")

        dims = tuple(qudit.dim for qudit in targets)
        vector = np.asarray(state, dtype=np.complex128).ravel()

        logical_dim = int(np.prod(dims))
        encoded_dim = 1 << sum(qudit.num_qubits for qudit in targets)
        if vector.size == encoded_dim and logical_dim != encoded_dim:
            # An already-encoded vector: projecting it also verifies
            # that it carries no amplitude on invalid basis states.
            vector = project_state(dims, vector)
        elif vector.size != logical_dim:
            raise QuditCircuitError(
                f"expected {logical_dim} (logical) or {encoded_dim} "
                f"(encoded) amplitude(s) for dimensions {dims}, got "
                f"{vector.size}.",
            )

        directive = QuditStatePreparation(dims, vector)
        return self.append(directive, targets, copy=False)

    # ---------------------------------------------------------------- #
    # Views
    # ---------------------------------------------------------------- #
    def to_qubit_circuit(self, *, copy: bool = True) -> QuantumCircuit:
        """Return the encoded qubit circuit.

        Args:
            copy: Return an independent copy (default) instead of the
                live object exposed by :attr:`circuit`.

        Returns:
            The encoded :class:`~qiskit.circuit.QuantumCircuit`.
        """
        return self._circuit.copy() if copy else self._circuit

    def to_ideal_circuit(
        self,
        *,
        annotate_levels: bool = True,
    ) -> QuantumCircuit:
        """Build the *ideal* (one wire per qudit) circuit.

        The result is a presentation-only
        :class:`~qiskit.circuit.QuantumCircuit` with one
        :class:`~qiskit.circuit.Qubit` per qudit and one
        :class:`~qiskit.circuit.Clbit` per clbyte, where each qudit
        instruction becomes a single opaque
        :class:`~qiskit.circuit.Instruction`. It is meant for drawing
        and inspection - it is *not* executable, because a qubit wire
        cannot carry a qudit.

        Args:
            annotate_levels: Append ``d=<dims>`` to the gate labels,
                so the dimensions are visible in mixed-dimension
                circuits.

        Returns:
            The ideal-view circuit, with registers named exactly like
            the qudit/clbyte registers of this circuit.
        """
        ideal = QuantumCircuit(
            name=self._name,
            global_phase=self._circuit.global_phase,
        )

        qubit_of: dict[Qudit, Qubit] = {}
        seen_qdregs: set[int] = set()
        for qudit in self._qudits:
            qdreg = qudit.register
            if qdreg is None:
                qubit_wire = Qubit()
                ideal.add_bits([qubit_wire])
                qubit_of[qudit] = qubit_wire
            elif id(qdreg) not in seen_qdregs:
                # Register-owned qudits are always added contiguously,
                # so the first member met is index 0 of that register.
                seen_qdregs.add(id(qdreg))
                qreg = QuantumRegister(qdreg.size, qdreg.name)
                ideal.add_register(qreg)
                for index, member in enumerate(qdreg):
                    qubit_of[member] = qreg[index]

        clbit_of: dict[ClByte, Clbit] = {}
        seen_cbregs: set[int] = set()
        for clbyte in self._clbytes:
            cbreg = clbyte.register
            if cbreg is None:
                clbit_wire = Clbit()
                ideal.add_bits([clbit_wire])
                clbit_of[clbyte] = clbit_wire
            elif id(cbreg) not in seen_cbregs:
                seen_cbregs.add(id(cbreg))
                creg = ClassicalRegister(cbreg.size, cbreg.name)
                ideal.add_register(creg)
                for index, clbyte_member in enumerate(cbreg):
                    clbit_of[clbyte_member] = creg[index]

        for instruction in self._data:
            qargs = [qubit_of[qudit] for qudit in instruction.qudits]
            cargs = [clbit_of[clbyte] for clbyte in instruction.clbytes]
            ideal.append(
                self._ideal_operation(
                    instruction,
                    annotate_levels=annotate_levels,
                ),
                qargs,
                cargs,
            )
        return ideal

    @staticmethod
    def _ideal_operation(
        instruction: QuditCircuitInstruction,
        *,
        annotate_levels: bool,
    ) -> Instruction:
        """Build the ideal-view placeholder for one instruction.

        Args:
            instruction: The qudit-level instruction to render.
            annotate_levels: Include the qudit dimensions in the
                label.

        Returns:
            A one-wire-per-qudit
            :class:`~qiskit.circuit.Instruction`.
        """
        operation = instruction.operation
        num_qudits = instruction.num_qudits
        num_clbytes = instruction.num_clbytes

        # Native Qiskit operations render much better than
        # placeholders.
        if isinstance(operation, QuditBarrier):
            return Barrier(num_qudits, label=operation.label)
        if isinstance(operation, QuditReset):
            return Reset()
        if isinstance(operation, QuditMeasure):
            return Measure()

        arguments: list[str] = []
        for param in operation.params:
            if isinstance(param, (int, float)):
                arguments.append(f"{float(param):.4g}")
            elif isinstance(param, np.ndarray):
                arguments.append(f"ndarray[{param.size}]")
            else:
                arguments.append(str(param))
        if annotate_levels and instruction.dims:
            arguments.append(
                "d=" + ",".join(str(dim) for dim in instruction.dims),
            )

        base = (
            operation.label if operation.label is not None else operation.name
        )
        label = f"{base}({', '.join(arguments)})" if arguments else base

        # `Instruction` (unlike `Gate`) does not validate parameters;
        # the placeholder carries none anyway, everything is in the
        # label.
        return Instruction(
            operation.name,
            num_qudits,
            num_clbytes,
            [],
            label=label,
        )

    def draw(  # noqa: PLR0913 - mirrors `QuantumCircuit.draw`
        self,
        output: str | None = None,
        *,
        view: CircuitView = "ideal",
        annotate_levels: bool = True,
        scale: float | None = None,
        filename: str | None = None,
        style: dict[str, object] | str | None = None,
        fold: int | None = None,
        reverse_bits: bool | None = None,
        plot_barriers: bool = True,
        idle_wires: bool | str | None = None,
        initial_state: bool = False,
        justify: str | None = None,
        ax: object | None = None,
    ) -> Any:  # noqa: ANN401 - Qiskit's drawers are polymorphic
        """Draw the circuit.

        Args:
            output: Drawer backend (``'text'``, ``'mpl'``,
                ``'latex'``, ``'latex_source'``); Qiskit's default is
                used when ``None``.
            view: ``'ideal'`` (one wire per qudit, the default),
                ``'real'`` (the encoded circuit) or ``'decomposed'``
                (the encoded circuit, unrolled one level).
            annotate_levels: Only for ``view='ideal'``: show the qudit
                dimensions in the gate labels.
            scale: Image scale, for the ``mpl``/``latex`` drawers.
            filename: Write the drawing to this path.
            style: Drawer style name, path or dictionary.
            fold: Pagination width.
            reverse_bits: Reverse the wire order.
            plot_barriers: Draw barriers.
            idle_wires: Show wires without operations.
            initial_state: Annotate wires with their initial state.
            justify: ``'left'``, ``'right'`` or ``'none'``.
            ax: Matplotlib axes to draw into (``mpl`` only).

        Returns:
            Whatever :func:`qiskit.visualization.circuit_drawer`
            returns for the chosen ``output``.

        Raises:
            QuditCircuitError: If ``view`` is unknown.
        """
        if view == "ideal":
            target = self.to_ideal_circuit(annotate_levels=annotate_levels)
        elif view == "real":
            target = self._circuit
        elif view == "decomposed":
            target = self._circuit.decompose()
        else:
            raise QuditCircuitError(
                f"unknown view '{view}'; expected 'ideal', 'real' or "
                "'decomposed'.",
            )

        return target.draw(
            output=output,
            scale=scale,
            filename=filename,
            style=style,
            fold=fold,
            reverse_bits=reverse_bits,
            plot_barriers=plot_barriers,
            idle_wires=idle_wires,
            initial_state=initial_state,
            justify=justify,
            ax=ax,
        )

    def decompose(self, reps: int = 1) -> QuantumCircuit:
        """Return the encoded circuit with qudit gates unrolled.

        Args:
            reps: How many decomposition passes to run.

        Returns:
            The decomposed :class:`~qiskit.circuit.QuantumCircuit`.
        """
        return self._circuit.decompose(reps=reps)

    def __str__(self) -> str:
        """Return the ideal-view text drawing."""
        return str(self.draw(output="text"))

    def __repr__(self) -> str:
        """Return a short, unambiguous representation."""
        return (
            f"<QuditQuantumCircuit '{self._name}': {self.num_qudits} "
            f"qudit(s) dims={self.dims}, {self.num_clbytes} clbyte(s), "
            f"{len(self._data)} instruction(s), {self.num_qubits} "
            "qubit(s)>"
        )

    def __eq__(self, other: object) -> bool:
        """Compare dimensions and the encoded circuits."""
        if not isinstance(other, QuditQuantumCircuit):
            return NotImplemented
        return self.dims == other.dims and self._circuit == other._circuit

    # Mutable, exactly like `QuantumCircuit`, hence unhashable.
    __hash__ = None  # type: ignore[assignment]

    # ---------------------------------------------------------------- #
    # Structural operations
    # ---------------------------------------------------------------- #
    def copy_empty_like(self, name: str | None = None) -> QuditQuantumCircuit:
        """Return a circuit with the same wires but no instructions.

        Registers, loose qudits/clbytes, global phase and metadata are
        carried over. Bit objects are **shared** (exactly like
        Qiskit's
        :meth:`~qiskit.circuit.QuantumCircuit.copy_empty_like`), which
        is what makes :meth:`compose` able to map operands by
        position.

        Args:
            name: Name of the copy; defaults to this circuit's name.

        Returns:
            The empty copy.
        """
        out = QuditQuantumCircuit(name=self._name if name is None else name)

        added: set[int] = set()
        for qudit in self._qudits:
            qdreg = qudit.register
            if qdreg is None:
                out.add_qudits([qudit])
            elif id(qdreg) not in added:
                added.add(id(qdreg))
                out.add_register(qdreg)

        added.clear()
        for clbyte in self._clbytes:
            cbreg = clbyte.register
            if cbreg is None:
                out.add_clbytes([clbyte])
            elif id(cbreg) not in added:
                added.add(id(cbreg))
                out.add_register(cbreg)

        out.global_phase = self.global_phase
        out.metadata = dict(self.metadata)
        return out

    def copy(self, name: str | None = None) -> QuditQuantumCircuit:
        """Return a full copy of the circuit.

        Operations are shared, not deep-copied (they are treated as
        immutable by this class), while the instruction log and the
        encoded circuit are rebuilt.

        Args:
            name: Name of the copy; defaults to this circuit's name.

        Returns:
            The copy.
        """
        out = self.copy_empty_like(name)
        for instruction in self._data:
            out.append(
                instruction.operation,
                instruction.qudits,
                instruction.clbytes,
                copy=False,
            )
        return out

    def _adopt(self, other: QuditQuantumCircuit) -> None:
        """Take over ``other``'s state, used by the in-place ops.

        Args:
            other: The circuit whose internals are moved into ``self``.
        """
        self._name = other._name
        self._qudits = other._qudits
        self._qdregs = other._qdregs
        self._clbytes = other._clbytes
        self._cbregs = other._cbregs
        self._qudit_indices = other._qudit_indices
        self._clbyte_indices = other._clbyte_indices
        self._register_names = other._register_names
        self._data = other._data
        self._circuit = other._circuit

    def compose(
        self,
        other: QuditQuantumCircuit,
        qudits: QuditSpecifier | None = None,
        clbytes: ClByteSpecifier | None = None,
        *,
        front: bool = False,
        inplace: bool = False,
    ) -> QuditQuantumCircuit | None:
        """Inline another qudit circuit onto this one.

        Args:
            other: The circuit to inline. Its qudits are mapped onto
                ``qudits`` (or the first :attr:`num_qudits` of this
                circuit) **in order**, and dimensions must agree
                pairwise.
            qudits: Where to map ``other``'s qudits.
            clbytes: Where to map ``other``'s clbytes.
            front: Inline ``other`` *before* the existing
                instructions. This rebuilds the encoded circuit.
            inplace: Modify this circuit instead of returning a new
                one.

        Returns:
            ``None`` when ``inplace=True``, otherwise the composed
            circuit.

        Raises:
            QuditCircuitError: On a width or dimension mismatch.
        """
        mapped_qudits = (
            self._qudits[: other.num_qudits]
            if qudits is None
            else self._qudit_argument_conversion(qudits)
        )
        mapped_clbytes = (
            self._clbytes[: other.num_clbytes]
            if clbytes is None
            else self._clbyte_argument_conversion(clbytes)
        )
        if len(mapped_qudits) != other.num_qudits:
            raise QuditCircuitError(
                f"cannot compose a circuit with {other.num_qudits} "
                f"qudit(s) onto {len(mapped_qudits)} target(s).",
            )
        if len(mapped_clbytes) != other.num_clbytes:
            raise QuditCircuitError(
                f"cannot compose a circuit with {other.num_clbytes} "
                f"clbyte(s) onto {len(mapped_clbytes)} target(s).",
            )
        for source, target in zip(other.qudits, mapped_qudits, strict=True):
            if source.dim != target.dim:
                raise QuditCircuitError(
                    f"dimension mismatch while composing: {source.dim} "
                    f"levels onto {target.dim} levels.",
                )

        qudit_map = {
            id(source): target
            for source, target in zip(
                other.qudits,
                mapped_qudits,
                strict=True,
            )
        }
        clbyte_map = {
            id(source): target
            for source, target in zip(
                other.clbytes,
                mapped_clbytes,
                strict=True,
            )
        }

        def replay(
            destination: QuditQuantumCircuit,
            source: QuditQuantumCircuit,
        ) -> None:
            """Re-apply ``source``'s instructions onto ``destination``.

            Args:
                destination: The circuit to append onto.
                source: The circuit whose log is replayed.
            """
            for instruction in source.data:
                destination.append(
                    instruction.operation,
                    tuple(
                        qudit_map[id(qudit)] for qudit in instruction.qudits
                    ),
                    tuple(
                        clbyte_map[id(clbyte)]
                        for clbyte in instruction.clbytes
                    ),
                    copy=False,
                )

        if front:
            rebuilt = self.copy_empty_like(self._name)
            replay(rebuilt, other)
            for instruction in self._data:
                rebuilt.append(
                    instruction.operation,
                    instruction.qudits,
                    instruction.clbytes,
                    copy=False,
                )
            rebuilt.global_phase = self.global_phase + other.global_phase
            if inplace:
                self._adopt(rebuilt)
                return None
            return rebuilt

        destination = self if inplace else self.copy()
        replay(destination, other)
        destination.global_phase = (
            destination.global_phase + other.global_phase
        )
        return None if inplace else destination

    def inverse(self) -> QuditQuantumCircuit:
        """Return the adjoint circuit.

        Every instruction must be a :class:`~qiskit.circuit.Gate` (so
        no measurements, resets or initialisations), and each gate
        must implement :meth:`~qiskit.circuit.Gate.inverse`.

        Returns:
            The inverted circuit, named ``'<name>_dg'``.

        Raises:
            QuditCircuitError: If the circuit contains a non-unitary
                operation.
        """
        out = self.copy_empty_like(f"{self._name}_dg")
        out.global_phase = -self.global_phase
        for instruction in reversed(self._data):
            operation = instruction.operation
            if not isinstance(operation, Gate):
                raise QuditCircuitError(
                    f"cannot invert '{operation.name}': the circuit "
                    "contains non-unitary operations.",
                )
            out.append(
                cast("Instruction", operation.inverse()),
                instruction.qudits,
                instruction.clbytes,
                copy=False,
            )
        return out

    # ---------------------------------------------------------------- #
    # Metrics
    # ---------------------------------------------------------------- #
    def size(
        self,
        filter_function: Callable[
            [QuditCircuitInstruction],
            bool,
        ] = _is_not_directive,
    ) -> int:
        """Return the number of qudit-level operations.

        Args:
            filter_function: Predicate deciding which instructions
                count. Directives (barriers) are excluded by default.

        Returns:
            The number of counted instructions.
        """
        return sum(
            1 for instruction in self._data if filter_function(instruction)
        )

    def depth(
        self,
        filter_function: Callable[
            [QuditCircuitInstruction],
            bool,
        ] = _is_not_directive,
    ) -> int:
        """Return the qudit-level circuit depth.

        Wires are qudits and clbytes; barriers act as synchronisation
        points without adding depth (Qiskit semantics).

        Args:
            filter_function: Predicate deciding which instructions add
                depth.

        Returns:
            The critical-path length in qudit operations.
        """
        depths: dict[int, int] = {
            id(wire): 0 for wire in (*self._qudits, *self._clbytes)
        }
        for instruction in self._data:
            wires = [
                id(wire)
                for wire in (*instruction.qudits, *instruction.clbytes)
            ]
            if not wires:
                continue
            level = max(depths.get(wire, 0) for wire in wires)
            if filter_function(instruction):
                level += 1
            for wire in wires:
                depths[wire] = level
        return max(depths.values(), default=0)

    def width(self) -> int:
        """Return the number of qudits plus clbytes."""
        return self.num_qudits + self.num_clbytes

    def count_ops(self) -> OrderedDict[str, int]:
        """Count qudit-level operations by name.

        Returns:
            Names mapped to occurrences, most frequent first.
        """
        counts: dict[str, int] = {}
        for instruction in self._data:
            counts[instruction.name] = counts.get(instruction.name, 0) + 1
        return OrderedDict(
            sorted(counts.items(), key=lambda item: (-item[1], item[0])),
        )

    # ---------------------------------------------------------------- #
    # Result post-processing
    # ---------------------------------------------------------------- #
    def decode_bitstring(
        self,
        bitstring: str,
        *,
        on_invalid: InvalidPolicy = "keep",
    ) -> Levels | None:
        """Decode one counts key using this circuit's clbyte layout.

        Args:
            bitstring: A counts key, e.g. ``'01 10'``.
            on_invalid: What to do with values outside the qudit
                subspace (see
                :func:`~qiskit_qudits.utils.encoding.decode_bitstring`).

        Returns:
            One level per clbyte, in clbyte order, or ``None`` if the
            shot was dropped.
        """
        return decode_bitstring(
            bitstring,
            self.clbyte_widths,
            self.clbyte_dims,
            on_invalid=on_invalid,
        )

    def decode_counts(
        self,
        counts: Mapping[str, int],
        *,
        on_invalid: InvalidPolicy = "keep",
    ) -> dict[Levels, int]:
        """Decode a counts mapping using this circuit's clbyte layout.

        Args:
            counts: Mapping from bit-string to shots.
            on_invalid: What to do with leaked outcomes.

        Returns:
            Mapping from a tuple of levels (clbyte order, byte 0
            first) to shots. Use
            :func:`~qiskit_qudits.utils.encoding.format_levels` for a
            Qiskit-ordered display string.

        Examples:
            .. code-block:: python

                raw = result.get_counts()
                for levels, shots in qc.decode_counts(raw).items():
                    print(format_levels(levels), shots)
        """
        return decode_counts(
            counts,
            self.clbyte_widths,
            self.clbyte_dims,
            on_invalid=on_invalid,
        )
