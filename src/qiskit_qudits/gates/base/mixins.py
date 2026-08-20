"""Shared mixin for qudit gate classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import prod
from typing import TYPE_CHECKING, Any, ClassVar

from qiskit_qudits.utils.consts import MIN_QUDIT_DIM
from qiskit_qudits.utils.validation import (
    validate_float_finite,
    validate_integer_range,
    validate_vector,
)

if TYPE_CHECKING:
    import numpy as np

    from qiskit_qudits.utils.consts import (
        FloatLike,
        IntLike,
        VectorLike,
    )


class QuditGateMixin(ABC):
    """Mixin class providing shared utilities for qudit gates.

    Concrete subclasses must implement:

    * :meth:`_define` - build and assign the qubit-level circuit.
    * :meth:`_build_unitary` - return the gate's unitary matrix.

    The mixin assumes ``self.num_qubits`` is available, which is
    guaranteed by Qiskit's :class:`~qiskit.circuit.Gate` base class.
    """

    MIN_DIM: ClassVar[int] = MIN_QUDIT_DIM
    MAX_DIM: ClassVar[int] = 16

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits required to represent this gate.

        The qubit count is derived automatically from the qudit
        dimensions by the Qiskit base class and should not be set
        directly.
        """

    @classmethod
    def _validate_dim(cls, dim: IntLike) -> int:
        """Validate a qudit dimension and return it as an ``int``.

        Args:
            dim: The qudit dimension to validate.

        Returns:
            The validated value cast to a plain ``int``.

        Raises:
            TypeError: If ``dim`` is not an integer-like object.
            ValueError: If ``dim`` is outside the range defined
                by :attr:`MIN_DIM` and :attr:`MAX_DIM`.
        """
        return validate_integer_range(
            dim,
            minimum=cls.MIN_DIM,
            maximum=cls.MAX_DIM,
            name="dim",
        )

    @property
    def hilbert_dim(self) -> int:
        r"""Full Hilbert-space dimension.

        Derived from ``self.num_qubits`` as :math:`2^n`, which is set by
        the Qiskit base class. Always :math:`\geq` the product of all
        qudit dimensions; equal when every dimension is a power of two.
        """
        return 1 << self.num_qubits

    @abstractmethod
    def _define(self) -> None:
        """Build and assign the gate's qubit-level circuit.

        Implementors must set ``self.definition`` to a
        :class:`~qiskit.circuit.QuantumCircuit` that realises this
        qudit gate using standard qubit operations.

        Raises:
            NotImplementedError: If called from the base class without
                being overridden by a concrete subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _define() and set "
            "self.definition to a QuantumCircuit.",
        )

    @abstractmethod
    def _build_unitary(self) -> np.typing.NDArray[np.complex128]:
        r"""Return this gate's unitary matrix.

        Returns:
            Complex128 array of shape ``(hilbert_dim, hilbert_dim)``.

        Raises:
            NotImplementedError: If called from the base class without
                being overridden by a concrete subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_unitary().",
        )

    def __array__(
        self,
        dtype: np.typing.DTypeLike | None = None,
        *,
        copy: bool | None = None,
    ) -> np.typing.NDArray[Any]:
        r"""Return the gate's unitary as a NumPy array.

        Called automatically by ``numpy.array(gate)`` and similar
        constructors.

        Args:
            dtype: Desired element type. Defaults to ``complex128``.
            copy: When ``True``, guarantee the returned array is a
                fresh copy. When ``False``, raise a ValueError if a
                copy cannot be avoided. When ``None``, a copy is made
                only if required (e.g., for dtype conversion).

        Returns:
            The 2^n x 2^n unitary matrix, cast to ``dtype``
            when supplied.

        Raises:
            ValueError: If ``copy=False`` because a new copy is always
                instantiated.
        """
        # 1. Handle the copy=False constraint
        if copy is False:
            raise ValueError(
                "copy=False is not supported because the unitary "
                "matrix is generated on the fly and always requires "
                "a new allocation.",
            )

        mat = self._build_unitary()

        # 2. Map `bool | None` to a strict boolean for the `astype` call
        # copy=True  -> astype(..., copy=True)
        #   (always copy)
        # copy=None  -> astype(..., copy=False)
        #   (copy only if dtype changes)
        astype_copy = bool(copy)

        if dtype is not None:
            mat = mat.astype(dtype, copy=astype_copy)
        elif copy is True:
            # If no dtype casting occurred,
            # but a copy was explicitly requested
            mat = mat.copy()

        return mat


class QuditMultiGateMixin(QuditGateMixin):
    """Mixin class providing shared utilities for multi-qudit gates.

    Concrete subclasses must implement:

    * :attr:`_dims` - dimension of each qudit, in register order.
    * :attr:`_qubits_per_qudit` - number of qubits per each qudit,
        in register order.

    The mixin assumes :attr:`_dims` and :attr:`_qubits_per_qudit`
    are available.
    """

    MIN_QUDITS: ClassVar[int] = 2
    MAX_QUDITS: ClassVar[int] = 8

    _dims: tuple[int, ...]
    _qubits_per_qudit: tuple[int, ...]

    @classmethod
    def _validate_dims(
        cls,
        dims: VectorLike,
    ) -> tuple[int, ...]:
        """Validate multiple qudit dimensions.

        Args:
            dims: Sequence of qudit dimensions to validate, in
                register order.

        Returns:
            The validated values cast to a tuple of integers.

        Raises:
            TypeError: If ``dims`` is not a vector-like object, or if
                any element is not integer-like.
            ValueError: If ``dims`` is empty, any element is outside
                :attr:`~QuditGateMixin.MIN_DIM` /
                :attr:`~QuditGateMixin.MAX_DIM`, or the number of
                qudits is greater than :attr:`MAX_QUDITS`.
        """
        # `dtype=None` keeps the array's natural (integer) dtype so
        # each element still satisfies the integer-like check in
        # `_validate_dim` below.
        validated_dims = validate_vector(dims, dtype=None, name="dims")

        if validated_dims.size == 0:
            raise ValueError("dims must not be empty.")

        levels: tuple[int, ...] = tuple(
            cls._validate_dim(d) for d in validated_dims
        )
        num_qudits: int = len(levels)
        if num_qudits > cls.MAX_QUDITS:
            raise ValueError(
                f"Number of qudits must be at most "
                f"{cls.MAX_QUDITS}, got {num_qudits}.",
            )
        return levels

    @property
    def dims(self) -> tuple[int, ...]:
        """All qudit dimensions in register order."""
        return self._dims

    @property
    def num_qudits(self) -> int:
        """Total number of qudits."""
        return len(self._dims)

    def _qudit_range(self, qudit_index: int) -> range:
        """Return the qubit-index range for the given qudit.

        Args:
            qudit_index: Zero-based qudit index in
                ``[0, num_qudits)``. Can also accept negative indices.

        Returns:
            A :class:`range` of qubit indices for that qudit.
        """
        start = sum(self._qubits_per_qudit[:qudit_index])
        return range(start, start + self._qubits_per_qudit[qudit_index])

    @property
    def fills_hilbert_space(self) -> bool:
        """Return ``True`` when every qudit dimension is a power of two.

        When ``True``, no invalid states exist and the full Hilbert
        space is used by the qudit subspace.
        """
        return all(
            d == (1 << n)
            for d, n in zip(
                self._dims,
                self._qubits_per_qudit,
                strict=True,
            )
        )

    @property
    def num_invalid_states(self) -> int:
        r"""Total number of invalid states in the joint Hilbert space.

        A state is invalid when any constituent qudit value is
        :math:`\geq d_i`. Equals :math:`2^n - \prod_i d_i`.
        """
        return self.hilbert_dim - prod(self._dims)

    def _compute_strides(self) -> list[int]:
        r"""Compute the stride of each qudit in the register.

        Each qudit's stride is the product of the Hilbert-space sizes
        of all qudits that precede it (i.e. occupy lower-index qubits),
        matching Qiskit's convention that the first qubit in the
        argument list is the least-significant bit of the matrix index.

        **Register layout** (LSB → MSB)::

            [ qudit_0 | qudit_1 | ... | qudit_{n-1} ]

        Returns:
            A list of length :attr:`num_qudits` with each qudit's
            stride.
        """
        strides: list[int] = []
        running_stride = 1
        for n_qubits in self._qubits_per_qudit:
            strides.append(running_stride)
            running_stride <<= n_qubits
        return strides


class QuditPhaseGateMixin(ABC):
    """Mixin class providing shared utilities for phase qudit gates.

    **Subclassing contract** - concrete gates combining this mixin
    must store the rotation angle as the *first* entry of
    :attr:`params` (i.e. ``params[0]``), since :attr:`theta` reads it
    from that fixed position.
    """

    @staticmethod
    def _validate_theta(theta: FloatLike) -> float:
        """Validate the theta parameter and return it as a ``float``.

        Args:
            theta: Rotation angle in radians.

        Returns:
            The validated value cast to a plain ``float``.

        Raises:
            TypeError: If ``theta`` is not a real-number-like object.
            ValueError: If ``theta`` is not finite (``nan``/``inf``).
        """
        return validate_float_finite(
            theta,
            name="theta",
        )

    @property
    @abstractmethod
    def params(self) -> list[Any]:
        """Gate parameters."""

    @property
    def theta(self) -> float:
        """Rotation angle in radians.

        Assumes the subclassing contract that ``theta`` is stored as
        ``params[0]``.
        """
        return self.params[0]
