# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from collections.abc import Callable, Iterator, MutableSequence, Sequence

from qiskit.circuit import ClassicalRegister, Clbit, Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.quantumcircuitdata import CircuitInstruction

class InstructionSet:
    __slots__ = ("_instructions", "_requester")

    _instructions: list[
        CircuitInstruction | tuple[MutableSequence[CircuitInstruction], int]
    ]
    _requester: Callable[..., ClassicalRegister | Clbit] | None

    def __init__(
        self,
        *,
        resource_requester: (
            Callable[..., ClassicalRegister | Clbit] | None
        ) = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> CircuitInstruction: ...
    def add(
        self,
        instruction: CircuitInstruction | Operation,
        qargs: Sequence[Qubit] | None = ...,
        cargs: Sequence[Clbit] | None = ...,
    ) -> None: ...
    def _add_ref(
        self,
        data: MutableSequence[CircuitInstruction],
        pos: int,
    ) -> None: ...
    def inverse(self, annotated: bool = ...) -> InstructionSet: ...
    def _instructions_iter(self) -> Iterator[CircuitInstruction]: ...
    @property
    def instructions(self) -> list[Operation]: ...
    @property
    def qargs(self) -> list[list[Qubit]]: ...
    @property
    def cargs(self) -> list[list[Clbit]]: ...
