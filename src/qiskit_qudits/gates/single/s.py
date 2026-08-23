"""Implementation of the S-Gate class for qudit operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from typing_extensions import override

from qiskit_qudits.gates.single.p import QuditPGate

if TYPE_CHECKING:
    from qiskit_qudits.utils.consts import IntLike


class QuditSGate(QuditPGate):
    r"""Qudit S-gate class.

    This gate applies the S operation on qudit systems, rotating
    the phase around the Z-axis of each quantum state level by
    an amount proportional to the level index. The S-gate is defined
    such that :math:`SS = Z`.

    .. math::

        S_d \lvert k \rangle = \omega^{\frac{k}{2}}
        \lvert k \rangle, \quad \omega = e^{\frac{i 2 \pi}{d}}, \quad k
        \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "S"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit S-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(
            dim,
            np.pi / 2,
            label=label,
        )

    @override
    def inverse(self, annotated: bool = False) -> QuditSdgGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditSdgGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditSdgGate(self.dim)


class QuditSdgGate(QuditPGate):
    r"""Qudit Sdg-gate class.

    This gate applies the inverse S operation on qudit systems,
    rotating the phase around the Z-axis of each quantum state level by
    an amount proportional to the level index. The Sdg-gate is defined
    such that :math:`S^\dagger S^\dagger = Z^\dagger`.

    .. math::

        S_d^\dagger \lvert k \rangle = \omega^{-\frac{k}{2}}
        \lvert k \rangle, \quad \omega = e^{\frac{i 2 \pi}{d}}, \quad k
        \in \{0, \dots, d-1\}

    Basis states outside the valid qudit subspace are fixed points,
    keeping the full :math:`2^n \times 2^n` matrix unitary.

    When d is a power of two, the decomposition uses a simpler
    predefined algorithm rather than unitary synthesis, so that
    transpiler and optimisation passes can recognise it.
    """

    gate_name: ClassVar[str] = "Sdg"

    def __init__(
        self,
        dim: IntLike,
        *,
        label: str | None = None,
    ) -> None:
        """Create a qudit Sdg-gate.

        Args:
            dim: Dimension of the qudit the gate operates on.
            label: An optional label for the gate.
        """
        super().__init__(
            dim,
            -np.pi / 2,
            label=label,
        )

    @override
    def inverse(self, annotated: bool = False) -> QuditSGate:
        """Return the inverse gate.

        Args:
            annotated: when ``True``, return an
                :class:`~qiskit.circuit.AnnotatedOperation`
                (not yet supported).

        Returns:
            A fresh :class:`QuditSGate` with the same number of
            levels.

        Raises:
            NotImplementedError: if ``annotated=True`` is requested.
        """
        if annotated:
            raise NotImplementedError(
                "annotated inverse is not yet supported for "
                f"{type(self).__name__}.",
            )
        return QuditSGate(self.dim)
