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

from collections.abc import Iterator, Sequence
from enum import Enum
from typing import Any, overload

from qiskit.circuit.operation import Operation

from .classical.types import Duration

class Bit: ...

class Qubit(Bit):
    def __init__(
        self,
        register: QuantumRegister | None = ...,
        index: int | None = ...,
    ) -> None: ...

class Clbit(Bit):
    def __init__(
        self,
        register: ClassicalRegister | None = ...,
        index: int | None = ...,
    ) -> None: ...

class Register:
    @overload
    def __getitem__(self, key: int) -> Bit: ...
    @overload
    def __getitem__(self, key: slice) -> Sequence[Bit]: ...
    @property
    def size(self) -> int: ...

class QuantumRegister(Register):
    def __init__(
        self,
        size: int | None = ...,
        name: str | None = ...,
        bits: Sequence[Qubit] | None = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[Qubit]: ...
    @overload
    def __getitem__(self, key: int) -> Qubit: ...
    @overload
    def __getitem__(self, key: slice) -> list[Qubit]: ...

class ClassicalRegister(Register):
    def __init__(
        self,
        size: int | None = ...,
        name: str | None = ...,
        bits: Sequence[Clbit] | None = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[Clbit]: ...
    @overload
    def __getitem__(self, key: int) -> Clbit: ...
    @overload
    def __getitem__(self, key: slice) -> list[Clbit]: ...

class CircuitInstruction:
    def __init__(
        self,
        operation: Operation,
        qubits: Sequence[Qubit] | None = ...,
        clbits: Sequence[Clbit] | None = ...,
    ) -> None: ...
    @property
    def operation(self) -> Operation: ...
    @property
    def qubits(self) -> tuple[Qubit, ...]: ...
    @property
    def clbits(self) -> tuple[Clbit, ...]: ...

class Parameter:
    def __init__(self, uuid: str | None = ...) -> None: ...

class ParameterExpression:
    def __init__(
        self,
        name_map: dict[str, Any] | None = ...,
        expr: Any = ...,
    ) -> None: ...

class StandardInstructionType(Enum):
    Barrier = 0
    Delay = 1
    Measure = 2
    Reset = 3

__all__ = ["Duration"]
