"""Implementation of the base Gate class for multi-qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit.circuit import Gate
from typing_extensions import override

from qiskit_qudits.gates.base.mixins import QuditMultiGateMixin
from qiskit_qudits.utils.dims import qubits_per_qudit

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import VectorLike


class QuditMultiGate(Gate, QuditMultiGateMixin):
    r"""Qudit base class for multi-qudit operations.

    A multi-qudit gate operates on two or more qudits. Each qudit may
    have a **different** dimension :math:`d_i`, encoded into
    :math:`n_i = \lceil \log_2(d_i) \rceil` qubits.

    This class inherits from :class:`qiskit.circuit.Gate` for
    compatibility with the Qiskit transpiler and circuit model.
    Qubit-level attributes are derived automatically from the qudit
    dimensions and should not be set directly.

    **Qubit-register layout** (LSB → MSB)::

        [ qudit_0 qubits | qudit_1 qubits | ... | qudit_{n-1} qubits ]

    Qudit values are therefore extracted / packed starting from the
    least-significant bits (qudit_0 first). This ordering is used
    consistently by all index-helper methods.

    **Subclassing contract** - override :meth:`_define` and
    :meth:`_build_unitary`.
    """

    def __init__(
        self,
        name: str,
        dims: VectorLike,
        params: list[Any],
        *,
        label: str | None = None,
    ) -> None:
        r"""Initialise a multi-qudit gate.

        Args:
            name: The Qiskit name of the gate.
            dims: Dimension :math:`d_i` of each qudit, given in
                register order :math:`(d_0, d_1, \dots, d_{n-1})`.
                Must contain at least two entries.
            params: Gate parameters forwarded to the Qiskit base class.
            label: Optional display label.

        Raises:
            ValueError: If fewer than two qudit dimensions are given.
        """
        self._dims: tuple[int, ...] = super()._validate_dims(dims)
        if len(self._dims) < type(self).MIN_QUDITS:
            raise ValueError(
                f"QuditMultiGate requires at least 2 qudits, "
                f"got {len(self._dims)}.",
            )

        self._qubits_per_qudit: tuple[int, ...] = tuple(
            qubits_per_qudit(d) for d in self._dims
        )

        super().__init__(
            name,
            sum(self._qubits_per_qudit),
            params,
            label=label,
        )

    @override
    def inverse(self, annotated: bool = False) -> QuditMultiGate:
        """Return the inverse gate.

        Subclasses must override this method to return an instance
        of the same (or a compatible) gate type.

        Args:
            annotated: forwarded for interface compatibility with
                :meth:`qiskit.circuit.Gate.inverse`. Subclasses with a
                cheap, well-known concrete inverse typically ignore it.

        Raises:
            NotImplementedError: If called from the base class without
                being overridden by a concrete subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement inverse().",
        )
