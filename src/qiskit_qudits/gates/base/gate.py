"""Implementation of the base Gate class for single-qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit.circuit import Gate
from typing_extensions import override

from qiskit_qudits.gates.base.mixins import QuditGateMixin
from qiskit_qudits.utils.dims import qubits_per_qudit

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditGate(Gate, QuditGateMixin):
    r"""Qudit base class for single-qudit operations.

    A d-level qudit is encoded into :math:`n = \lceil \log_2(d) \rceil`
    qubits, yielding a Hilbert space of dimension :math:`2^n`.
    When :math:`d` is not a power of two, the remaining :math:`2^n - d`
    basis states are *invalid* (outside the qudit subspace). Every
    concrete gate must leave those states unchanged so the full matrix
    stays unitary.

    This class inherits from :class:`qiskit.circuit.Gate` for
    compatibility with the Qiskit transpiler and circuit model.
    The qubit count is derived automatically from the qudit dimension
    and should not be set directly.

    **Subclassing contract** - override :meth:`_define` and
    :meth:`_build_unitary`.
    """

    def __init__(
        self,
        name: str,
        dim: IntLike,
        params: list[Any],
        *,
        label: str | None = None,
    ) -> None:
        r"""Create a single-qudit gate.

        Args:
            name: The Qiskit name of the gate.
            dim: Dimension :math:`d` of the qudit.
            params: Gate parameters forwarded to the Qiskit base class.
            label: Optional display label.
        """
        self._dim: int = super()._validate_dim(dim)

        super().__init__(
            name,
            qubits_per_qudit(self._dim),
            params,
            label=label,
        )

    @property
    def dim(self) -> int:
        r"""Dimension :math:`d` of the qudit."""
        return self._dim

    @property
    def fills_hilbert_space(self) -> bool:
        r"""Whether :math:`d` is a power of two.

        When ``True``, no invalid states exist and the full Hilbert
        space is used by the qudit subspace.
        """
        return self._dim == self.hilbert_dim

    @property
    def num_invalid_states(self) -> int:
        r"""Number of states outside the qudit subspace.

        These states must map to themselves so the full matrix stays
        unitary. The number is :math:`2^n - d`.
        """
        return self.hilbert_dim - self._dim

    @override
    def inverse(self, annotated: bool = False) -> QuditGate:
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
