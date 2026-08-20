"""Unit tests for the :mod:`qiskit_qudits` exception hierarchy.

Two promises are pinned down here:

* :class:`~qiskit_qudits.exceptions.QuditError` is a
  :class:`~qiskit.exceptions.QiskitError`, so a user who already
  handles Qiskit errors keeps working; and
* :class:`~qiskit_qudits.circuit.exceptions.QuditCircuitError` is
  *both* a ``QuditError`` and a Qiskit ``CircuitError``, so either
  ``except`` clause catches it.
"""

from __future__ import annotations

import pytest
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError

import qiskit_qudits
import qiskit_qudits.circuit
from qiskit_qudits.circuit.exceptions import QuditCircuitError
from qiskit_qudits.exceptions import QuditError

MESSAGE = "something went wrong"


def raise_circuit_error() -> None:
    """Raise a :class:`QuditCircuitError` from a helper.

    Raises:
        QuditCircuitError: Always.
    """
    raise QuditCircuitError(MESSAGE)


class TestHierarchy:
    """Inheritance relations between the exception classes."""

    @pytest.mark.parametrize(
        ("error", "base"),
        [
            (QuditError, QiskitError),
            (QuditError, Exception),
            (QuditCircuitError, QuditError),
            (QuditCircuitError, CircuitError),
            (QuditCircuitError, QiskitError),
        ],
    )
    def test_subclass_relations(
        self,
        error: type[Exception],
        base: type[Exception],
    ) -> None:
        """Each error subclasses the documented base."""
        assert issubclass(error, base)

    @pytest.mark.parametrize(
        ("error", "other"),
        [
            (QiskitError, QuditError),
            (CircuitError, QuditCircuitError),
            (QuditError, QuditCircuitError),
        ],
    )
    def test_relations_are_one_way(
        self,
        error: type[Exception],
        other: type[Exception],
    ) -> None:
        """The Qiskit classes know nothing about the qudit ones."""
        assert not issubclass(error, other)

    def test_the_library_base_comes_first_in_the_mro(self) -> None:
        """``QuditError`` wins over ``CircuitError`` on lookup."""
        mro = QuditCircuitError.__mro__
        assert mro[0] is QuditCircuitError
        assert mro.index(QuditError) < mro.index(CircuitError)


class TestCatching:
    """Both ``except`` paths reach the same exception."""

    @pytest.mark.parametrize(
        "caught_as",
        [QuditCircuitError, QuditError, CircuitError, QiskitError],
    )
    def test_a_circuit_error_is_catchable_as(
        self,
        caught_as: type[Exception],
    ) -> None:
        """Every base in the hierarchy catches the concrete error."""
        with pytest.raises(caught_as, match=MESSAGE):
            raise_circuit_error()

    def test_the_qudit_except_clause_matches(self) -> None:
        """A hand-written ``except QuditError`` catches it."""
        caught: Exception | None = None
        try:
            raise_circuit_error()
        except QuditError as exc:
            caught = exc
        assert isinstance(caught, QuditCircuitError)

    def test_the_qiskit_except_clause_matches(self) -> None:
        """A hand-written ``except CircuitError`` catches it too."""
        caught: Exception | None = None
        try:
            raise_circuit_error()
        except CircuitError as exc:
            caught = exc
        assert isinstance(caught, QuditCircuitError)

    def test_the_message_is_preserved(self) -> None:
        """The text handed to the constructor survives."""
        error = QuditCircuitError(MESSAGE)
        assert error.message == MESSAGE
        assert MESSAGE in str(error)


class TestExports:
    """The public re-exports of the two packages."""

    def test_the_top_level_package_exports_qudit_error(self) -> None:
        """``qiskit_qudits.QuditError`` is part of the public API."""
        assert "QuditError" in qiskit_qudits.__all__
        assert qiskit_qudits.QuditError is QuditError

    def test_the_circuit_package_exports_the_circuit_error(self) -> None:
        """``qiskit_qudits.circuit.QuditCircuitError`` is public."""
        assert "QuditCircuitError" in qiskit_qudits.circuit.__all__
        assert qiskit_qudits.circuit.QuditCircuitError is QuditCircuitError
