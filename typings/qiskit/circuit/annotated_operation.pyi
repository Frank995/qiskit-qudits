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

import dataclasses

import numpy as np
from qiskit.circuit.operation import Operation
from qiskit.circuit.parameterexpression import ParameterValueType

class Modifier: ...

@dataclasses.dataclass
class InverseModifier(Modifier): ...

@dataclasses.dataclass
class ControlModifier(Modifier):
    num_ctrl_qubits: int = ...
    ctrl_state: int | str | None = ...

    def __init__(
        self,
        num_ctrl_qubits: int = ...,
        ctrl_state: int | str | None = ...,
    ) -> None: ...

@dataclasses.dataclass
class PowerModifier(Modifier):
    power: float

class AnnotatedOperation(Operation):
    base_op: Operation
    modifiers: list[Modifier]

    def __init__(
        self,
        base_op: Operation,
        modifiers: Modifier | list[Modifier],
    ) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_clbits(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def copy(self) -> AnnotatedOperation: ...
    def to_matrix(self) -> np.ndarray: ...
    def control(
        self,
        num_ctrl_qubits: int = ...,
        label: str | None = ...,
        ctrl_state: int | str | None = ...,
        annotated: bool | None = ...,
    ) -> AnnotatedOperation: ...
    def inverse(self, annotated: bool = ...) -> AnnotatedOperation: ...
    def power(
        self,
        exponent: float,
        annotated: bool = ...,
    ) -> AnnotatedOperation: ...
    @property
    def params(self) -> list[ParameterValueType]: ...
    @params.setter
    def params(self, value: list[ParameterValueType]) -> None: ...
    def validate_parameter(
        self,
        parameter: ParameterValueType,
    ) -> ParameterValueType: ...
