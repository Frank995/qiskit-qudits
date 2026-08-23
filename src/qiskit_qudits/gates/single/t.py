"""Implementation of the T-Gate class for qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import override

from qiskit_qudits.gates.single.p import QuditPGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditTGate(QuditPGate):
    r"""Qudit T-gate class.

    This gate applies the T operation on qudit systems, rotating
    the phase around the Z-axis of each quantum state level by
    an amount proportional to the level index. The T-gate is defined
    such that :math:`TT = S` and :math:`TTTT = Z`.

    .. math::

        T_d \lvert k \rangle = \omega^{\frac{k}{4}}
        \lvert k \rangle, \quad \omega = e^{\frac{i 2 \pi}{d}}, \quad k
        \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "T"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit T-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(
            dim,
            np.pi / 4,
            label=label,
        )

    @override
    def inverse(self, annotated: bool = False) -> QuditTdgGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditTdgGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditTdgGate(self.dim)


class QuditTdgGate(QuditPGate):
    r"""Qudit Tdg-gate class.

    This gate applies the inverse T operation on qudit systems,
    rotating the phase around the Z-axis of each quantum state level by
    an amount proportional to the level index. The Tdg-gate is defined
    such that :math:`T^\dagger T^\dagger = S^\dagger` and
    :math:`T^\dagger T^\dagger T^\dagger T^\dagger = Z^\dagger`.

    .. math::

        T_d^\dagger \lvert k \rangle = \omega^{-\frac{k}{4}}
        \lvert k \rangle, \quad \omega = e^{\frac{i 2 \pi}{d}}, \quad k
        \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "Tdg"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit Tdg-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(
            dim,
            -np.pi / 4,
            label=label,
        )

    @override
    def inverse(self, annotated: bool = False) -> QuditTGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditTGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditTGate(self.dim)
