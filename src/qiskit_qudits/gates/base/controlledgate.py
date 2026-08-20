"""Implementation of the base ControlledGate class for controlled qudit operations."""  # noqa: E501, W505

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiskit.circuit import ControlledGate
from typing_extensions import override

from qiskit_qudits.gates.base.mixins import QuditMultiGateMixin
from qiskit_qudits.utils.dims import qubits_per_qudit

if TYPE_CHECKING:
    from qiskit_qudits.gates.base.gate import QuditGate
    from qiskit_qudits.utils.consts import IntLike, VectorLike


class QuditControlledGate(ControlledGate, QuditMultiGateMixin):
    r"""Qudit base class for controlled qudit operations.

    A controlled qudit gate operates on one or more control qudits
    and a single target qudit. Each qudit may have a **different**
    dimension :math:`d_i`, encoded into
    :math:`n_i = \lceil \log_2(d_i) \rceil` qubits.

    This class inherits from :class:`qiskit.circuit.ControlledGate`
    for compatibility with the Qiskit transpiler and circuit model.
    Qubit-level attributes (``num_ctrl_qubits``, ``ctrl_state``) are
    derived automatically from the qudit dimensions and should not be
    set directly.

    **Qubit-register layout** (LSB → MSB)::

        [ control_0 qubits | control_1 qubits | ... |
            control_{m-1} qubits | target qubits ]

    Qudit values are therefore extracted / packed starting from the
    least-significant bits (control_0 first, target last). This ordering
    is used consistently by all index-helper methods.

    .. note::
        ``ctrl_state`` customization is not yet supported - every
        instance uses Qiskit's default (active-high, all control
        qubits ``1``) control convention. This is a qubit-level
        convention; it does not necessarily coincide with "control
        qudit value :math:`d_i - 1`" when :math:`d_i` is not a power
        of two. Qudit-level control semantics, if any, are the
        responsibility of :meth:`_build_unitary`.

    **Subclassing contract** - override :meth:`_define` and
    :meth:`_build_unitary`.
    """

    base_gate: (  # pyright: ignore[reportIncompatibleVariableOverride]
        QuditGate | None
    )

    def __init__(
        self,
        name: str,
        target_dim: IntLike,
        control_dims: VectorLike,
        params: list[Any],
        *,
        base_gate: QuditGate | None = None,
        label: str | None = None,
    ) -> None:
        r"""Initialise a multi-controlled qudit gate.

        Args:
            name: The Qiskit name of the gate.
            target_dim: Dimension :math:`d_t` of the **target** qudit.
            control_dims: Dimension :math:`d_{c_i}` of each **control**
                qudit, given in register order
                :math:`(d_{c_0}, d_{c_1}, \dots, d_{c_{m-1}})`. Must
                contain at least one entry.
            params: Gate parameters forwarded to the Qiskit base class.
            base_gate: Optional uncontrolled qudit gate that this gate
                controls. Stored for Qiskit compatibility.
            label: Optional display label.

        Raises:
            ValueError: If ``control_dims`` is empty. A controlled
                gate requires at least one control qudit.
        """
        validated_dims = super()._validate_dims(
            (*control_dims, target_dim),
        )
        if len(validated_dims) < type(self).MIN_QUDITS:
            raise ValueError(
                "QuditControlledGate requires at least one control "
                "qudit; got an empty control_dims.",
            )

        self._target_dim: int = validated_dims[-1]
        self._control_dims: tuple[int, ...] = validated_dims[:-1]

        self._dims: tuple[int, ...] = (
            *self._control_dims,
            self._target_dim,
        )
        self._qubits_per_qudit: tuple[int, ...] = tuple(
            qubits_per_qudit(d) for d in self._dims
        )

        super().__init__(
            name,
            sum(self._qubits_per_qudit),
            params,
            label=label,
            num_ctrl_qubits=sum(self._qubits_per_qudit[:-1]),
            ctrl_state=None,
            base_gate=base_gate,
        )

    @property
    def target_dim(self) -> int:
        """Target qudit dimension :math:`d_t`."""
        return self._target_dim

    @property
    def control_dims(self) -> tuple[int, ...]:
        """Control qudit dimensions, in register order."""
        return self._control_dims

    @property
    def num_control_qudits(self) -> int:
        r"""Number of control qudits :math:`m`."""
        return len(self._control_dims)

    @property
    def target_hilbert_dim(self) -> int:
        r"""Hilbert-space dimension of the target qudit alone.

        Derived from the target's qubit count as :math:`2^{n_t}`,
        mirroring :attr:`QuditGateMixin.hilbert_dim` for the
        single-qudit case. Always :math:`\geq d_t`; equal when
        :math:`d_t` is a power of two.
        """
        return 1 << self._qubits_per_qudit[-1]

    def _control_qudit_range(self, control_index: int) -> range:
        """Return the qubit-index range for the given control qudit.

        Args:
            control_index: Zero-based control qudit index
                in ``[0, num_control_qudits)``.

        Returns:
            A :class:`range` of qubit indices for that control qudit.
        """
        return self._qudit_range(control_index)

    def _target_qudit_range(self) -> range:
        """Return the qubit-index range for the target qudit.

        Returns:
            A :class:`range` of qubit indices for the target qudit.
        """
        return self._qudit_range(-1)

    def _compute_control_target_strides(self) -> tuple[list[int], int]:
        r"""Compute strides for the control qudits and the target qudit.

        Splits the general per-qudit strides (see
        :meth:`QuditMultiGateMixin._compute_strides`) into the control
        strides and the single target stride, following the register
        layout (LSB → MSB)::

            [ control_0 | control_1 | ... | control_{m-1} | target ]

        Returns:
            A ``(control_strides, target_stride)`` pair where
            ``control_strides`` has length :attr:`num_control_qudits`.
        """
        *control_strides, target_stride = super()._compute_strides()
        return control_strides, target_stride

    @override
    def inverse(self, annotated: bool = False) -> QuditControlledGate:
        """Return the inverse gate.

        Subclasses must override this method to return an instance
        of the same (or a compatible) gate type.

        Args:
            annotated: forwarded for interface compatibility with
                :meth:`qiskit.circuit.ControlledGate.inverse`.
                Subclasses with a cheap, well-known concrete inverse
                typically ignore it.

        Raises:
            NotImplementedError: If called from the base class without
                being overridden by a concrete subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement inverse().",
        )
