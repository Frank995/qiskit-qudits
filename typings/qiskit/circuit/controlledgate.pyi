# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root
# directory directory of this source tree or at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections.abc import Iterable
from typing import Any, Self

from qiskit.circuit.annotated_operation import AnnotatedOperation
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit

class ControlledGate(Gate):
    base_gate: Gate | None
    _num_ctrl_qubits: int
    _definition: QuantumCircuit | None
    _ctrl_state: int | None
    _open_ctrl: bool | None
    _name: str

    def __init__(
        self,
        name: str,
        num_qubits: int,
        params: list[Any],
        label: str | None = ...,
        num_ctrl_qubits: int = ...,
        definition: QuantumCircuit | None = ...,
        ctrl_state: int | str | None = ...,
        base_gate: Gate | None = ...,
        *,
        _base_label: Any = ...,
    ) -> None: ...
    @property
    def definition(self) -> QuantumCircuit: ...
    @definition.setter
    def definition(self, array: QuantumCircuit | None) -> None: ...
    @property
    def name(self) -> str: ...
    @name.setter
    def name(self, name: str) -> None: ...
    @property
    def num_ctrl_qubits(self) -> int: ...
    @num_ctrl_qubits.setter
    def num_ctrl_qubits(self, num_ctrl_qubits: int) -> None: ...
    @property
    def ctrl_state(self) -> int: ...
    @ctrl_state.setter
    def ctrl_state(self, ctrl_state: int | str | None) -> None: ...
    @property
    def params(self) -> list[Any]: ...
    @params.setter
    def params(self, parameters: Iterable[Any]) -> None: ...
    def __deepcopy__(self, memo: dict[int, Any] | None = ...) -> Self: ...
    def __eq__(self, other: object) -> bool: ...
    def inverse(
        self,
        annotated: bool = ...,
    ) -> ControlledGate | AnnotatedOperation: ...
