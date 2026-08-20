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

from abc import ABC
from collections.abc import Sequence

from qiskit.quantum_info.operators.mixins import GroupMixin
from qiskit.quantum_info.operators.op_shape import OpShape

class BaseOperator(GroupMixin, ABC):
    __array_priority__: int

    _qargs: tuple[int, ...] | None
    _op_shape: OpShape

    def __init__(
        self,
        input_dims: int | Sequence[int] | None = ...,
        output_dims: int | Sequence[int] | None = ...,
        num_qubits: int | None = ...,
        shape: Sequence[int] | None = ...,
        op_shape: OpShape | None = ...,
    ) -> None: ...
    def __call__(self, *qargs: int) -> BaseOperator: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def qargs(self) -> tuple[int, ...] | None: ...
    @property
    def dim(self) -> tuple[int, int]: ...
    @property
    def num_qubits(self) -> int | None: ...
    @property
    def _input_dim(self) -> int: ...
    @property
    def _output_dim(self) -> int: ...
    def reshape(
        self,
        input_dims: int | Sequence[int] | None = ...,
        output_dims: int | Sequence[int] | None = ...,
        num_qubits: int | None = ...,
    ) -> BaseOperator: ...
    def input_dims(
        self,
        qargs: Sequence[int] | None = ...,
    ) -> tuple[int, ...]: ...
    def output_dims(
        self,
        qargs: Sequence[int] | None = ...,
    ) -> tuple[int, ...]: ...
    def copy(self) -> BaseOperator: ...
