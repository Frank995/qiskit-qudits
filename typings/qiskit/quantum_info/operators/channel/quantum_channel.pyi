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

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator

class QuantumChannel(LinearOp):
    _data: Any
    def __init__(
        self,
        data: list[Any] | np.ndarray,
        num_qubits: int | None = ...,
        op_shape: OpShape | None = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def data(self) -> Any: ...
    @property
    def _channel_rep(self) -> str: ...
    @property
    def settings(self) -> dict[str, Any]: ...
    @abstractmethod
    def conjugate(self) -> Self: ...
    @abstractmethod
    def transpose(self) -> Self: ...
    def adjoint(self) -> Self: ...
    def power(self, n: float) -> Self: ...
    def __sub__(self, other: Any) -> Self: ...
    def _add(self, other: Any, qargs: Sequence[int] | None = ...) -> Self: ...
    def _multiply(self, other: complex) -> Self: ...
    def is_cptp(
        self,
        atol: float | None = ...,
        rtol: float | None = ...,
    ) -> bool: ...
    def is_tp(
        self,
        atol: float | None = ...,
        rtol: float | None = ...,
    ) -> bool: ...
    def is_cp(
        self,
        atol: float | None = ...,
        rtol: float | None = ...,
    ) -> bool: ...
    def is_unitary(
        self,
        atol: float | None = ...,
        rtol: float | None = ...,
    ) -> bool: ...
    def to_operator(self) -> Operator: ...
    def to_instruction(self) -> Instruction: ...
    def _is_cp_helper(
        self,
        choi: np.ndarray,
        atol: float | None,
        rtol: float | None,
    ) -> bool: ...
    def _is_tp_helper(
        self,
        choi: np.ndarray,
        atol: float | None,
        rtol: float | None,
    ) -> bool: ...
    def _format_state(
        self,
        state: Any,
        density_matrix: bool = False,
    ) -> np.ndarray: ...
    @abstractmethod
    def _evolve(
        self,
        state: Any,
        qargs: Sequence[int] | None = ...,
    ) -> Any: ...
    @classmethod
    def _init_transformer(cls, data: Any) -> QuantumChannel | Operator: ...
