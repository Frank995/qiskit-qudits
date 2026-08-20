"""Exceptions for errors raised while handling Quantum Circuits."""

from __future__ import annotations

from qiskit.circuit.exceptions import CircuitError

from qiskit_qudits.exceptions import QuditError


class QuditCircuitError(QuditError, CircuitError):
    """Raised when an invalid qudit-circuit operation is attempted.

    This inherits from both :class:`QuditError` and Qiskit's
    :class:`~qiskit.circuit.exceptions.CircuitError` so that user code
    written against either exception hierarchy keeps working::

        try:
            circuit.x(99)
        except CircuitError:      # Qiskit-style handling
            ...
        except QuditError:        # library-specific handling
            ...
    """
