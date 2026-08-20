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

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import (
    QuantumChannel,
)
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts

class QuantumState(ABC):
    _op_shape: OpShape | None
    _rng_generator: np.random.Generator | None
    __array_priority__: int

    def __init__(self, op_shape: OpShape | None = ...) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def dim(self) -> int: ...
    @property
    def num_qubits(self) -> int | None: ...
    @property
    def _rng(self) -> np.random.Generator: ...
    def dims(self, qargs: Sequence[int] | None = ...) -> tuple[int, ...]: ...
    def copy(self) -> QuantumState: ...
    def seed(self, value: int | np.random.Generator | None = ...) -> None: ...
    @abstractmethod
    def is_valid(
        self,
        atol: float | None = ...,
        rtol: float | None = ...,
    ) -> bool: ...
    @abstractmethod
    def to_operator(self) -> Operator: ...
    @abstractmethod
    def conjugate(self) -> QuantumState: ...
    @abstractmethod
    def trace(self) -> float | complex: ...
    @abstractmethod
    def purity(self) -> float: ...
    @abstractmethod
    def tensor(self, other: QuantumState) -> QuantumState: ...
    @abstractmethod
    def expand(self, other: QuantumState) -> QuantumState: ...
    def _add(self, other: QuantumState) -> QuantumState: ...
    def _multiply(self, other: complex) -> QuantumState: ...
    @abstractmethod
    def evolve(
        self,
        other: Operator | QuantumChannel | QuantumCircuit | Instruction,
        qargs: Sequence[int] | None = ...,
    ) -> QuantumState: ...
    @abstractmethod
    def expectation_value(
        self,
        oper: BaseOperator,
        qargs: Sequence[int] | None = ...,
    ) -> complex: ...
    @abstractmethod
    def probabilities(
        self,
        qargs: Sequence[int] | None = ...,
        decimals: int | None = ...,
    ) -> npt.NDArray[np.float64]: ...
    def probabilities_dict(
        self,
        qargs: Sequence[int] | None = ...,
        decimals: int | None = ...,
    ) -> dict[str, float]: ...
    def sample_memory(
        self,
        shots: int,
        qargs: Sequence[int] | None = ...,
    ) -> npt.NDArray[np.str_]: ...
    def sample_counts(
        self,
        shots: int,
        qargs: Sequence[int] | None = ...,
    ) -> Counts: ...
    def measure(
        self,
        qargs: Sequence[int] | None = ...,
    ) -> tuple[str, QuantumState]: ...
    @staticmethod
    def _index_to_ket_array(
        inds: npt.NDArray[np.integer[Any]],
        dims: tuple[int, ...],
        string_labels: bool = ...,
    ) -> npt.NDArray[Any]: ...
    @staticmethod
    def _vector_to_dict(
        vec: npt.NDArray[Any],
        dims: tuple[int, ...],
        decimals: int | None = ...,
        string_labels: bool = ...,
    ) -> dict[str | tuple[int, ...], Any]: ...
    @staticmethod
    def _matrix_to_dict(
        mat: npt.NDArray[Any],
        dims: tuple[int, ...],
        decimals: int | None = ...,
        string_labels: bool = ...,
    ) -> dict[str | tuple[tuple[int, ...], tuple[int, ...]], Any]: ...
    @staticmethod
    def _subsystem_probabilities(
        probs: npt.NDArray[np.float64],
        dims: tuple[int, ...],
        qargs: Sequence[int] | None = ...,
    ) -> npt.NDArray[np.float64]: ...

    # Operator Overloads
    def __and__(self, other: Operator | QuantumChannel) -> QuantumState: ...
    def __xor__(self, other: QuantumState) -> QuantumState: ...
    def __mul__(self, other: complex) -> QuantumState: ...
    def __truediv__(self, other: complex) -> QuantumState: ...
    def __rmul__(self, other: complex) -> QuantumState: ...
    def __add__(self, other: QuantumState) -> QuantumState: ...
    def __sub__(self, other: QuantumState) -> QuantumState: ...
    def __neg__(self) -> QuantumState: ...
