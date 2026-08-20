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

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Self, TypeAlias

import numpy as np
from qiskit._accelerate.circuit import (
    Bit,
    CircuitInstruction,
    ClassicalRegister,
    Clbit,
    Parameter,
    QuantumRegister,
    Qubit,
    Register,
)
from qiskit.circuit.classical import expr
from qiskit.circuit.gate import Gate
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.instructionset import InstructionSet
from qiskit.circuit.operation import Operation
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.parametertable import ParameterView
from qiskit.circuit.quantumcircuitdata import QuantumCircuitData
from qiskit.quantum_info.states.statevector import Statevector

QubitSpecifier: TypeAlias = (
    Qubit | QuantumRegister | int | slice | Sequence[Qubit | int]
)
ClbitSpecifier: TypeAlias = (
    Clbit | ClassicalRegister | int | slice | Sequence[Clbit | int]
)

class QuantumCircuit:
    def __init__(
        self,
        *regs: Register | int | Sequence[Bit],
        name: str | None = ...,
        global_phase: ParameterValueType = ...,
        metadata: dict[str, Any] | None = ...,
        inputs: Iterable[expr.Var] = ...,
        captures: Iterable[expr.Var | expr.Stretch] = ...,
        declarations: (
            Mapping[expr.Var, expr.Expr] | Iterable[tuple[expr.Var, expr.Expr]]
        ) = ...,
    ) -> None: ...
    def __iter__(self) -> Iterator[CircuitInstruction]: ...
    @property
    def num_qubits(self) -> int: ...
    @property
    def num_clbits(self) -> int: ...
    @property
    def clbits(self) -> list[Clbit]: ...
    @property
    def qubits(self) -> list[Qubit]: ...
    @property
    def cregs(self) -> list[ClassicalRegister]: ...
    @property
    def qregs(self) -> list[QuantumRegister]: ...
    @property
    def name(self) -> str | None: ...
    @name.setter
    def name(self, value: str | None) -> None: ...
    @property
    def data(self) -> QuantumCircuitData: ...
    def inverse(self, annotated: bool = ...) -> Self: ...
    def count_ops(self) -> OrderedDict[str, int]: ...
    def to_gate(
        self,
        parameter_map: dict[Parameter, ParameterValueType] | None = ...,
        label: str | None = ...,
    ) -> Gate: ...
    def append(
        self,
        instruction: Operation | CircuitInstruction,
        qargs: Sequence[QubitSpecifier] | None = ...,
        cargs: Sequence[ClbitSpecifier] | None = ...,
        *,
        copy: bool = ...,
    ) -> InstructionSet: ...
    def decompose(
        self,
        gates_to_decompose: (
            str | type[Instruction] | Sequence[str | type[Instruction]] | None
        ) = ...,
        reps: int = ...,
    ) -> Self: ...
    def draw(
        self,
        output: str | None = ...,
        scale: float | None = ...,
        filename: str | None = ...,
        style: dict[str, object] | str | None = ...,
        interactive: bool = ...,
        plot_barriers: bool = ...,
        reverse_bits: bool | None = ...,
        justify: str | None = ...,
        vertical_compression: str | None = ...,
        idle_wires: bool | str | None = ...,
        with_layout: bool = ...,
        fold: int | None = ...,
        ax: Any = ...,
        initial_state: bool = ...,
        cregbundle: bool | None = ...,
        wire_order: list[int] | None = ...,
        expr_len: int = ...,
        measure_arrows: bool | None = ...,
    ) -> Any: ...
    def unitary(
        self,
        obj: np.ndarray | Gate | Any,
        qubits: Sequence[QubitSpecifier],
        label: str | None = ...,
    ) -> InstructionSet: ...
    def id(
        self,
        qubit: QubitSpecifier,
    ) -> InstructionSet: ...
    def x(
        self,
        qubit: QubitSpecifier,
        label: str | None = ...,
    ) -> InstructionSet: ...
    def mcx(
        self,
        control_qubits: Sequence[QubitSpecifier],
        target_qubit: QubitSpecifier,
        ancilla_qubits: None = ...,
        mode: None = ...,
        ctrl_state: str | int | None = ...,
    ) -> InstructionSet: ...
    def p(self, theta: float, qubit: QubitSpecifier) -> InstructionSet: ...
    def cp(
        self,
        theta: float,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = ...,
        ctrl_state: str | int | None = ...,
    ) -> InstructionSet: ...
    def measure(
        self,
        qubit: QubitSpecifier,
        cbit: ClbitSpecifier,
    ) -> InstructionSet: ...
    def barrier(
        self,
        *qargs: QubitSpecifier,
        label: str | None = ...,
    ) -> InstructionSet: ...
    def reset(self, qubit: QubitSpecifier) -> InstructionSet: ...
    def initialize(
        self,
        params: Statevector | Sequence[complex] | str | int,
        qubits: Sequence[QubitSpecifier] | None = ...,
        normalize: bool = ...,
    ) -> InstructionSet: ...
    @property
    def metadata(self) -> dict[str, Any]: ...
    @metadata.setter
    def metadata(self, metadata: dict[str, Any]) -> None: ...
    @property
    def global_phase(self) -> ParameterValueType: ...
    @global_phase.setter
    def global_phase(self, angle: ParameterValueType) -> None: ...
    def add_register(self, *regs: Register | int | Sequence[Bit]) -> None: ...
    def add_bits(self, bits: Iterable[Bit]) -> None: ...
    @property
    def parameters(self) -> ParameterView[Any]: ...
    @property
    def num_parameters(self) -> int: ...
    def copy(self, name: str | None = ...) -> Self: ...
