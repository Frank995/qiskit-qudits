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

from collections.abc import Iterator, Sequence
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import (
    QuantumChannel,
)
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.states.quantum_state import QuantumState

_StatevectorLike: TypeAlias = (
    np.ndarray
    | list[Any]
    | Statevector
    | Operator
    | QuantumCircuit
    | Instruction
)

class Statevector(QuantumState, TolerancesMixin):
    def __init__(
        self,
        data: _StatevectorLike,
        dims: int | Sequence[int] | None = ...,
    ) -> None: ...
    @classmethod
    def from_circuit(
        cls,
        circuit: QuantumCircuit,
        ignore_set_layout: bool = ...,
    ) -> Statevector: ...
    def __array__(
        self,
        dtype: Any = ...,
        copy: bool | None = ...,
    ) -> npt.NDArray[np.complex128]: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    @property
    def settings(self) -> dict[str, Any]: ...
    def draw(self, output: str | None = ..., **drawer_args: Any) -> Any: ...
    def _ipython_display_(self) -> None: ...
    def __getitem__(self, key: int | str) -> np.complex128: ...
    def __iter__(self) -> Iterator[np.complex128]: ...
    def __len__(self) -> int: ...
    @property
    def data(self) -> npt.NDArray[np.complex128]: ...
    def is_valid(
        self,
        atol: float | None = ...,
        rtol: float | None = ...,
    ) -> bool: ...
    def to_operator(self) -> Operator: ...
    def conjugate(self) -> Statevector: ...
    def trace(self) -> np.float64: ...
    def purity(self) -> np.float64: ...
    def tensor(
        self,
        other: QuantumState | _StatevectorLike,
    ) -> Statevector: ...
    def inner(self, other: _StatevectorLike) -> np.complex128: ...
    def expand(
        self,
        other: QuantumState | _StatevectorLike,
    ) -> Statevector: ...
    def _add(self, other: QuantumState | _StatevectorLike) -> Statevector: ...
    def _multiply(self, other: complex) -> Statevector: ...
    def evolve(
        self,
        other: Operator | QuantumChannel | QuantumCircuit | Instruction,
        qargs: Sequence[int] | None = ...,
    ) -> Statevector: ...
    def equiv(
        self,
        other: _StatevectorLike,
        rtol: float | None = ...,
        atol: float | None = ...,
    ) -> bool: ...
    def reverse_qargs(self) -> Statevector: ...
    # TODO: Implementing correct Pauli's stubs is not worth now
    # just for this method
    # def _expectation_value_pauli(
    #     self, pauli: Pauli, qargs: Sequence[int] | None = ...
    # ) -> complex: ...
    def expectation_value(
        self,
        oper: BaseOperator | QuantumCircuit | Instruction,
        qargs: Sequence[int] | None = ...,
    ) -> complex: ...
    def probabilities(
        self,
        qargs: Sequence[int] | None = ...,
        decimals: int | None = ...,
    ) -> npt.NDArray[np.float64]: ...
    def reset(self, qargs: Sequence[int] | None = ...) -> Statevector: ...
    @classmethod
    def from_label(cls, label: str) -> Statevector: ...
    @staticmethod
    def from_int(i: int, dims: int | Sequence[int]) -> Statevector: ...
    @classmethod
    def from_instruction(
        cls,
        instruction: Instruction | QuantumCircuit,
    ) -> Statevector: ...
    def to_dict(self, decimals: int | None = ...) -> dict[str, complex]: ...
    @staticmethod
    def _evolve_operator(
        statevec: Statevector,
        oper: Operator,
        qargs: Sequence[int] | None = ...,
    ) -> Statevector: ...
    @staticmethod
    def _evolve_instruction(
        statevec: Statevector,
        obj: Instruction | QuantumCircuit | Any,
        qargs: Sequence[int] | None = ...,
    ) -> Statevector: ...
