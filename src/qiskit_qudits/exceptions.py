"""Base exception class for :mod:`qiskit_qudits`."""

from __future__ import annotations

from qiskit.exceptions import QiskitError


class QuditError(QiskitError):
    """Base class for every error raised by :mod:`qiskit_qudits`."""
