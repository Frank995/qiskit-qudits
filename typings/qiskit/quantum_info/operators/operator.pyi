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

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.operation import Operation
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.transpiler.layout import Layout

class Operator(LinearOp):
    _data: npt.NDArray[np.complex128]

    def __init__(
        self,
        data: (
            QuantumCircuit
            | Operation
            | BaseOperator
            | np.ndarray
            | list[Any]
            | Any
        ),
        input_dims: int | Sequence[int] | None = ...,
        output_dims: int | Sequence[int] | None = ...,
    ) -> None: ...
    def __array__(
        self,
        dtype: Any = ...,
        copy: bool | None = ...,
    ) -> npt.NDArray[np.complex128]: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def data(self) -> npt.NDArray[np.complex128]: ...
    @property
    def settings(self) -> dict[str, Any]: ...
    def draw(self, output: str | None = ..., **drawer_args: Any) -> Any: ...
    def _ipython_display_(self) -> None: ...
    @classmethod
    def from_label(cls, label: str) -> Operator: ...
    def apply_permutation(
        self,
        perm: Sequence[int],
        front: bool = ...,
    ) -> Operator: ...
    @classmethod
    def from_circuit(
        cls,
        circuit: QuantumCircuit,
        ignore_set_layout: bool = ...,
        layout: Layout | None = ...,
        final_layout: Layout | None = ...,
    ) -> Operator: ...
    def is_unitary(
        self,
        atol: float | None = ...,
        rtol: float | None = ...,
    ) -> bool: ...
    def to_operator(self) -> Self: ...
    def to_instruction(self) -> Instruction: ...
    def conjugate(self) -> Operator: ...
    def transpose(self) -> Operator: ...
    def compose(
        self,
        other: (
            Operator
            | QuantumCircuit
            | Operation
            | BaseOperator
            | np.ndarray
            | Any
        ),
        qargs: Sequence[int] | None = ...,
        front: bool = ...,
    ) -> Operator: ...
    def power(
        self,
        n: float,
        branch_cut_rotation: float = ...,
        assume_unitary: bool = ...,
    ) -> Operator: ...
    def tensor(self, other: Operator | Any) -> Operator: ...
    def expand(self, other: Operator | Any) -> Operator: ...
    @classmethod
    def _tensor(cls, a: Operator, b: Operator) -> Operator: ...
    def _add(
        self,
        other: Operator | Any,
        qargs: Sequence[int] | None = ...,
    ) -> Operator: ...
    def _multiply(self, other: complex) -> Operator: ...
    def equiv(
        self,
        other: Operator | Any,
        rtol: float | None = ...,
        atol: float | None = ...,
    ) -> bool: ...
    def reverse_qargs(self) -> Operator: ...
    def to_matrix(self) -> npt.NDArray[np.complex128]: ...
    @classmethod
    def _einsum_matmul(
        cls,
        tensor: npt.NDArray[np.complex128],
        mat: npt.NDArray[np.complex128],
        indices: Sequence[int],
        shift: int = ...,
        right_mul: bool = ...,
    ) -> npt.NDArray[np.complex128]: ...
    @classmethod
    def _init_instruction(
        cls,
        instruction: QuantumCircuit | Operation | Any,
    ) -> Operator: ...
    @classmethod
    def _instruction_to_matrix(
        cls,
        obj: Any,
    ) -> npt.NDArray[np.complex128] | None: ...
    def _append_instruction(
        self,
        obj: Any,
        qargs: Sequence[int] | None = ...,
    ) -> None: ...
